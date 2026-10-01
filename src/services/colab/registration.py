"""``POST /internal/colab/register``: the VM reports in (ADR-0010).

The route is public (the VM has no Google identity to present); the
signature is the authentication. :meth:`ColabRegistrar.handle` checks, in
order, and answers with the first failure:

======================================================  ======
Check                                                   Status
======================================================  ======
body larger than ``MAX_REGISTER_BYTES``                 413
signature header missing or malformed                   403
timestamp more than ``MAX_CLOCK_SKEW_SECONDS`` away     403
body not a JSON object, unknown event, no nbh           400
signature does not match ``reg_key(nbh)``               403
signature already used (replay)                         403
the state document cannot be read (retry)               503
nbh is not the current session's (state document)       403
event not possible in the session's status              409
``ready``: tunnel_url breaks the rule or resolves to a  422
non-public address
``ready``: the readiness probe failed (retry later)     503
======================================================  ======

Events: ``ready`` (with a passing probe) moves ASSIGNING/BOOTING/READY to
READY, unless ``!colab off`` came first: then the session drains and the VM
gets 409 (it reports ``failed`` and releases itself). A router that is
not listed or loaded, at another digest than its pin (a republished tag)
or not wholly in VRAM passes the probe; the session keeps the digests and sizes the probe saw, and the
router skips it (:meth:`ColabEndpoint.unpinned`,
:meth:`ColabEndpoint.off_gpu`). ``failed`` moves an
active session to DRAINING and, from ASSIGNING or BOOTING, counts a boot
failure (three in a row disable the tier); ``stopping`` (the VM's watchdog
is about to release itself) moves it to DRAINING. On a session already
DRAINING either one only notes the error. The controller then releases the
runtime. Nothing the VM sends is logged beyond the event, the step and a
clipped error.

Every check up to the signature is pure computation (``reg_key`` depends
only on the secret and the body's ``nbh``), so an unsigned request costs no
Firestore read and logs only at DEBUG.
"""

from __future__ import annotations

import json
import time
from collections.abc import Callable
from dataclasses import dataclass, replace
from typing import Any

from src.services.colab import protocol
from src.services.colab.endpoint import ColabEndpoint
from src.services.colab.state import (
    ASSIGNING,
    BOOTING,
    DRAINING,
    OFF,
    READY,
    ColabState,
    ColabStateError,
    clip,
)
from src.services.search.page_fetcher import FetchError, Resolver, resolve_public
from src.utils.dedupe import RecentKeys
from src.utils.logger import logger

_READY_FROM = frozenset({ASSIGNING, BOOTING, READY})
_STOP_FROM = frozenset({ASSIGNING, BOOTING, READY, DRAINING})
# A ``failed`` from these is a boot failure.
_BOOTING = frozenset({ASSIGNING, BOOTING})
_MAX_NBH_CHARS = 64
_MAX_REPORTED_ITEMS = 8
_MAX_REPORTED_CHARS = 100


@dataclass(frozen=True)
class RegistrationResult:
    status_code: int
    body: dict[str, Any]


def _refuse(status_code: int, error: str) -> RegistrationResult:
    return RegistrationResult(status_code, {"ok": False, "error": error})


def _unsigned(status_code: int, error: str) -> RegistrationResult:
    """A refusal before the signature matched: noise on a public route, so DEBUG only."""
    logger.debug(f"Colab register: {error}; refused")
    return _refuse(status_code, error)


class ColabRegistrar:
    """Verifies and applies the VM's registration calls."""

    def __init__(
        self,
        endpoint: ColabEndpoint,
        *,
        secret: str,
        tunnel_host: str = "",
        clock: Callable[[], float] = time.time,
        resolver: Resolver | None = None,
    ) -> None:
        """``resolver`` is for tests; production resolves with getaddrinfo."""
        self._endpoint = endpoint
        self._secret = secret
        self._tunnel_host = tunnel_host
        self._clock = clock
        self._resolver = resolver
        # A signature is accepted once: a replay inside the skew window
        # (a captured ``stopping``, say) is refused.
        self._used = RecentKeys(ttl_seconds=2 * protocol.MAX_CLOCK_SKEW_SECONDS, max_size=256)

    async def handle(self, raw_body: bytes, signature_header: str | None) -> RegistrationResult:
        if len(raw_body) > protocol.MAX_REGISTER_BYTES:
            return _unsigned(413, "body too large")
        signature = protocol.parse_signature(signature_header)
        if signature is None:
            return _unsigned(403, "missing or malformed signature")
        now = self._clock()
        if abs(now - signature.timestamp) > protocol.MAX_CLOCK_SKEW_SECONDS:
            return _unsigned(403, "timestamp outside the allowed clock skew")
        try:
            payload = json.loads(raw_body)
        except (ValueError, UnicodeDecodeError):
            return _unsigned(400, "body is not JSON")
        if not isinstance(payload, dict):
            return _unsigned(400, "body is not a JSON object")
        event = payload.get("event")
        nbh = payload.get("nbh")
        # Unverified JSON: an event may be a list (unhashable) and an nbh a
        # lone surrogate (no UTF-8 for reg_key); an nbh is a UUID, so ASCII.
        if not isinstance(event, str) or event not in protocol.EVENTS:
            return _unsigned(400, "unknown event")
        if not isinstance(nbh, str) or not nbh or len(nbh) > _MAX_NBH_CHARS or not nbh.isascii():
            return _unsigned(400, "nbh missing or malformed")

        # The signature before any I/O: the route is public, and reg_key
        # needs only the secret and the body's nbh. Unauthenticated noise is
        # logged at DEBUG, so it can neither flood the log nor cost a read.
        reg_key = protocol.derive_reg_key(self._secret, nbh)
        if not protocol.signature_matches(reg_key, signature, raw_body):
            return _unsigned(403, "bad signature")
        if not self._used.add(signature.v1):
            logger.warning(f"Colab register: replayed {event}; refused")
            return _refuse(403, "signature already used")

        try:
            state = await self._endpoint.store.load()
        except ColabStateError as exc:
            # Not applied: the VM may send the same signature again.
            self._used.discard(signature.v1)
            logger.error(f"Colab register: state unreadable: {exc}")
            return _refuse(503, "state unavailable, retry")
        if not state.nbh or state.nbh != nbh:
            logger.warning(f"Colab register: {event} for a session that is not current; refused")
            return _refuse(403, "unknown session")

        if event == protocol.EVENT_READY:
            return await self._ready(payload, state, now)
        return await self._stop(event, payload, state, now)

    async def _ready(self, payload: dict[str, Any], state: ColabState, now: float) -> RegistrationResult:
        nbh = state.nbh
        if state.status not in _READY_FROM:
            return _refuse(409, f"session is {state.status}")
        if state.desired == OFF:
            return await self._switched_off(nbh, now)
        try:
            host = protocol.check_tunnel_url(payload.get("tunnel_url"), self._tunnel_host)
            await resolve_public(host, 443, **({"resolver": self._resolver} if self._resolver else {}))
        except (protocol.TunnelURLError, FetchError) as exc:
            logger.warning(f"Colab register: tunnel_url refused: {exc}")
            await self._note(nbh, f"tunnel_url refused: {exc}", step="register")
            return _refuse(422, f"tunnel_url refused: {exc}")
        tunnel_url = protocol.tunnel_base_url(host)

        report = await self._endpoint.readiness(tunnel_url, nbh)
        problems = report.problems
        if problems:
            detail = clip("; ".join(problems))
            logger.warning(f"Colab register: readiness probe failed: {detail}")
            await self._note(nbh, f"readiness probe: {detail}", step="probe")
            return RegistrationResult(503, {"ok": False, "error": "readiness probe failed", "problems": problems[:10]})

        reported = {
            "versions": _reported(payload.get("versions")),
            # What the probe saw rather than what the VM says: a router at
            # another digest or partly on the CPU stays out of this session.
            "digests": dict(report.digests),
            "vram": dict(report.vram),
            "gpu": _reported(payload.get("gpu")),
        }

        def to_ready(current: ColabState) -> ColabState | None:
            if current.nbh != nbh or current.status not in _READY_FROM:
                return None
            if current.desired == OFF:
                return _drained_for_off(current, now)
            return current.with_status(
                READY,
                now,
                tunnel_url=tunnel_url,
                ready_at=now if current.status != READY else current.ready_at,
                last_health_at=now,
                boot_step="ready",
                last_error="",
                failures=0,
                health_failures=0,
                cooldown_step=0,
                **reported,
            )

        try:
            updated = await self._endpoint.store.update(to_ready)
        except ColabStateError as exc:
            logger.error(f"Colab register: READY not saved: {exc}")
            return _refuse(503, "state unavailable, retry")
        if updated.nbh != nbh or updated.status != READY:
            if updated.nbh == nbh and updated.status == DRAINING:
                self._endpoint.mark_unhealthy("!colab off")
            return _refuse(409, f"session is {updated.status}")
        self._endpoint.record_success()
        logger.info(f"Colab session READY ({protocol.OLLAMA_VERSION}, gpu={reported['gpu'].get('name', '?')})")
        return RegistrationResult(200, {"ok": True, "status": READY})

    async def _stop(self, event: str, payload: dict[str, Any], state: ColabState, now: float) -> RegistrationResult:
        nbh = state.nbh
        step = clip(str(payload.get("step") or ""), 60)
        error = clip(str(payload.get("error") or ""))
        if event == protocol.EVENT_FAILED:
            reason = f"VM bootstrap failed at {step or '?'}: {error or 'no detail'}"
        else:
            reason = f"VM stopping: {error or 'watchdog'}"
        if state.status not in _STOP_FROM:
            # Already released or never started: nothing to drain.
            return RegistrationResult(200, {"ok": True, "status": state.status})

        def to_draining(current: ColabState) -> ColabState | None:
            if current.nbh != nbh or current.status not in _STOP_FROM:
                return None
            if current.status == DRAINING:
                # Already on its way out (``!colab off``, an idle release):
                # not a boot failure, only the reason is kept.
                return replace(current, last_error=clip(reason))
            boot_failure = event == protocol.EVENT_FAILED and current.status in _BOOTING
            return current.with_status(
                DRAINING,
                now,
                last_error=clip(reason),
                boot_step=step or current.boot_step,
                failures=current.failures + (1 if boot_failure else 0),
            )

        try:
            updated = await self._endpoint.store.update(to_draining)
        except ColabStateError as exc:
            logger.error(f"Colab register: {event} not saved: {exc}")
            return _refuse(503, "state unavailable, retry")
        self._endpoint.mark_unhealthy(reason)
        logger.warning(f"Colab register: {reason}")
        return RegistrationResult(200, {"ok": True, "status": updated.status})

    async def _switched_off(self, nbh: str, now: float) -> RegistrationResult:
        """A ``ready`` after ``!colab off``: drain instead (409 tells the VM to stop)."""

        def change(current: ColabState) -> ColabState | None:
            if current.nbh != nbh or current.status not in _READY_FROM:
                return None
            return _drained_for_off(current, now)

        try:
            await self._endpoint.store.update(change)
        except ColabStateError as exc:
            logger.error(f"Colab register: DRAINING after !colab off not saved: {exc}")
            return _refuse(503, "state unavailable, retry")
        self._endpoint.mark_unhealthy("!colab off")
        logger.info("Colab register: ready refused, Colab is switched off (!colab off)")
        return _refuse(409, "colab switched off")

    async def _note(self, nbh: str, error: str, *, step: str) -> None:
        """Keep why a ``ready`` was refused on the state, for ``!colab status``."""

        def change(current: ColabState) -> ColabState | None:
            if current.nbh != nbh:
                return None
            return replace(current, last_error=clip(error), boot_step=step)

        try:
            await self._endpoint.store.update(change)
        except ColabStateError as exc:
            logger.warning(f"Colab register: could not note the refusal: {exc}")


def _drained_for_off(state: ColabState, now: float) -> ColabState:
    return state.with_status(DRAINING, now, last_error="!colab off")


def _reported(value: Any) -> dict[str, Any]:
    """A small flat map from the VM, for display: short keys, short scalars."""
    if not isinstance(value, dict):
        return {}
    reported: dict[str, Any] = {}
    for key, item in list(value.items())[:_MAX_REPORTED_ITEMS]:
        if not isinstance(key, str):
            continue
        if isinstance(item, bool) or not isinstance(item, (str, int, float)):
            continue
        reported[key[:_MAX_REPORTED_CHARS]] = item[:_MAX_REPORTED_CHARS] if isinstance(item, str) else item
    return reported
