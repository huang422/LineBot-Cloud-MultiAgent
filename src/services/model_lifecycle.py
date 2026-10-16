"""Daily end-of-life check for every model the role table names.

NVIDIA retires hosted models on a published date and then answers HTTP 410;
OpenRouter drops free endpoints. Waiting for live traffic to hit the 410
costs a user's reply window, so once per Asia/Taipei day the per-minute
``/internal/cron`` queues an ``eol`` task (:meth:`ModelLifecycle.claim_day`)
and its handler runs :meth:`ModelLifecycle.run`. See ADR-0007.

What is checked, per id (no API key is ever sent; these are public):

- NVIDIA chat and embedding ids: ``GET /v1/models/{id}`` on
  integrate.api.nvidia.com. 200 is live (and clears a dead mark in
  :class:`~src.services.model_health.ModelHealth`), 410 is retired with the
  end-of-life date from the body (the model is marked dead), 404 means it
  is not in the catalog.
- NVIDIA FLUX ids are not in that catalog, so their build.nvidia.com page
  (the profile's ``page_slug``) is read for ``"isDeprecated":true`` or a
  ``deprecationNotice``.
- OpenRouter ids: ``GET /api/v1/models/{id}/endpoints``, plus the id's
  entry in ``GET /api/v1/models``, fetched once per run for every id:
  OpenRouter publishes ``expiration_date`` only in that list (live
  2026-10-16: 2026-12-31 for the dots-3 ``:free`` preview, and no such key
  in its endpoints answer). No endpoints (reported ahead of any date) is
  not ok; an expiration date (in the list, on the model or on an endpoint)
  reads ``expiring``. A list that cannot be read fails an OpenRouter check
  that would otherwise read ok, so it keeps its last state rather than lose
  an expiry; no endpoints, or a date in the endpoints answer, is still
  reported. ``openrouter/free`` is a router with no endpoints of its own and
  is not checked; neither are Colab ids (the VM's fixed list) or image ids
  without a page.

Results stay in memory for ``/health/detail``. A change of state is logged
(the first check logs only what is not ok): to ``ok`` or ``expiring`` at
INFO, to any other state at ERROR. An expiration date alone is no alert,
since the model still answers (dots-3's 2026-12-31 is a backup's, and the
operator wants no reminder); the ERROR comes when it is actually gone (no
endpoints, not in the catalog). A check that cannot run (network error,
5xx) keeps the last known state and records the error.
"""

from __future__ import annotations

import asyncio
import re
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, replace
from datetime import date
from typing import Any

import httpx

from src.providers.model_profiles import IMAGE_PROTOCOLS
from src.providers.openrouter_provider import FREE_ROUTER_MODEL
from src.utils.logger import logger

NVIDIA_MODELS_URL = "https://integrate.api.nvidia.com/v1/models/{model}"
BUILD_PAGE_URL = "https://build.nvidia.com/{slug}"
OPENROUTER_ENDPOINTS_URL = "https://openrouter.ai/api/v1/models/{model}/endpoints"
OPENROUTER_MODELS_URL = "https://openrouter.ai/api/v1/models"

# Seconds for one check; they run concurrently, well inside the task budget.
CHECK_TIMEOUT_SECONDS = 10.0

OK = "ok"
RETIRED = "retired"
NOT_IN_CATALOG = "not_in_catalog"
DEPRECATED = "deprecated"
PAGE_MISSING = "page_missing"
NO_ENDPOINTS = "no_endpoints"
EXPIRING = "expiring"
# No check has succeeded yet (only failures so far).
UNKNOWN = "unknown"

CATALOG = "catalog"
PAGE = "page"
ENDPOINTS = "endpoints"

# 410 bodies read "... has reached its end of life on 2026-07-27T00:00:00Z ...".
_EOL_RE = re.compile(r"end of life on (\S+Z)")
# build.nvidia.com embeds its page data as JSON inside script strings, so the
# quotes may arrive escaped; they are unescaped before matching.
_DEPRECATED_RE = re.compile(r'"isDeprecated"\s*:\s*true')
_NOTICE_RE = re.compile(r'"deprecationNotice"\s*:\s*"([^"]{0,300})')
_DETAIL_LIMIT = 200


@dataclass(frozen=True)
class LifecycleTarget:
    """One id to check and how."""

    provider: str
    model: str
    kind: str  # CATALOG, PAGE or ENDPOINTS
    page_slug: str = ""

    @property
    def label(self) -> str:
        return f"{self.provider}:{self.model}"


@dataclass(frozen=True)
class LifecycleResult:
    """The last known lifecycle state of one id."""

    state: str
    detail: str = ""
    eol: str | None = None
    checked_at: float | None = None
    # Why the last check could not run, if it could not.
    error: str | None = None

    def view(self) -> dict[str, Any]:
        return {
            "state": self.state,
            "detail": self.detail or None,
            "eol": self.eol,
            "checked_at": self.checked_at,
            "error": self.error,
        }


class _CheckFailed(Exception):
    """The check itself could not run; says nothing about the model."""


def lifecycle_targets(role_table: Any) -> list[LifecycleTarget]:
    """Every checkable id in *role_table* (enabled or not), once each."""
    targets: dict[str, LifecycleTarget] = {}
    for role in role_table.roles():
        for row in role_table.rows(role):
            target = _target_for(row.spec.provider, row.spec.model, row.profile)
            if target is not None:
                targets.setdefault(target.label, target)
    return list(targets.values())


def _target_for(provider: str, model: str, profile: Any) -> LifecycleTarget | None:
    if provider == "nvidia":
        if profile.protocol in IMAGE_PROTOCOLS:
            return LifecycleTarget(provider, model, PAGE, profile.page_slug) if profile.page_slug else None
        return LifecycleTarget(provider, model, CATALOG)
    if provider == "openrouter" and model != FREE_ROUTER_MODEL:
        return LifecycleTarget(provider, model, ENDPOINTS)
    return None


class ModelLifecycle:
    """Runs the daily check and keeps its results for ``/health/detail``."""

    def __init__(
        self,
        targets: list[LifecycleTarget],
        health: Any,
        *,
        transport: httpx.AsyncBaseTransport | None = None,
        clock: Callable[[], float] = time.time,
        timeout: float = CHECK_TIMEOUT_SECONDS,
    ) -> None:
        self._targets = list(targets)
        self._health = health
        self._transport = transport
        self._clock = clock
        self._timeout = timeout
        self._results: dict[str, LifecycleResult] = {}
        self._claimed_day: date | None = None
        self.last_run_at: float | None = None

    @classmethod
    def for_role_table(cls, role_table: Any, health: Any, **kwargs: Any) -> ModelLifecycle:
        return cls(lifecycle_targets(role_table), health, **kwargs)

    @property
    def targets(self) -> list[LifecycleTarget]:
        return list(self._targets)

    def claim_day(self, today: date) -> bool:
        """True once per *today*: the caller should queue the check.

        In memory only, so a restart may check twice in a day, which is
        harmless.
        """
        if self._claimed_day == today:
            return False
        self._claimed_day = today
        return True

    def results(self) -> dict[str, dict[str, Any]]:
        """``provider:model`` → the last result, for the role table view."""
        return {label: result.view() for label, result in self._results.items()}

    def view(self) -> dict[str, Any]:
        return {
            "last_claimed_day": self._claimed_day.isoformat() if self._claimed_day else None,
            "last_run_at": self.last_run_at,
            "results": self.results(),
        }

    async def run(self) -> dict[str, LifecycleResult]:
        """Check every target concurrently; returns the results by label."""
        async with httpx.AsyncClient(
            timeout=self._timeout,
            transport=self._transport,
            follow_redirects=True,
            headers={"Accept": "application/json, text/html;q=0.9"},
        ) as client:
            # One model list for every OpenRouter id of the run.
            catalog = (
                asyncio.ensure_future(_openrouter_catalog(client))
                if any(target.kind == ENDPOINTS for target in self._targets)
                else None
            )
            outcomes = await asyncio.gather(
                *(self._check(client, target, catalog) for target in self._targets),
                return_exceptions=True,
            )
            if catalog is not None:
                # Settled even when no check got as far as reading it.
                await asyncio.gather(catalog, return_exceptions=True)
        now = self._clock()
        for target, outcome in zip(self._targets, outcomes, strict=True):
            self._apply(target, outcome, now)
        self.last_run_at = now
        not_ok = sum(1 for result in self._results.values() if result.state != OK)
        logger.info(
            f"Model lifecycle: checked {len(self._targets)} model(s), {not_ok} not ok"
        )
        return dict(self._results)

    # ── Checks ───────────────────────────────────────────────

    async def _check(
        self,
        client: httpx.AsyncClient,
        target: LifecycleTarget,
        catalog: Awaitable[dict[str, dict]] | None,
    ) -> LifecycleResult:
        if target.kind == CATALOG:
            return await _check_nvidia_catalog(client, target.model)
        if target.kind == PAGE:
            return await _check_build_page(client, target.page_slug)
        assert catalog is not None  # run() fetches it whenever an id needs it
        return await _check_openrouter_endpoints(client, target.model, catalog)

    def _apply(self, target: LifecycleTarget, outcome: LifecycleResult | BaseException, now: float) -> None:
        label = target.label
        previous = self._results.get(label)
        if isinstance(outcome, BaseException):
            error = _describe(outcome)
            logger.warning(f"Model lifecycle: could not check {label}: {error}")
            base = previous or LifecycleResult(UNKNOWN)
            self._results[label] = replace(base, error=error, checked_at=now)
            return

        result = replace(outcome, checked_at=now)
        self._results[label] = result
        if result.state == OK and target.kind in (CATALOG, ENDPOINTS):
            # The catalog says the model is live: end a dead mark early.
            self._health.revive(target.provider, target.model)
        elif result.state == RETIRED:
            self._health.mark_dead(target.provider, target.model, result.detail, eol=result.eol)

        before = previous.state if previous is not None else None
        if before != result.state and not (before in (None, UNKNOWN) and result.state == OK):
            # A model that still answers (ok, or only an expiry date) is
            # information, not an alert.
            log = logger.info if result.state in (OK, EXPIRING) else logger.error
            log(
                f"Model lifecycle: {label} is {result.state} (was {before or 'unchecked'}): "
                f"{result.detail}"
            )


async def _check_nvidia_catalog(client: httpx.AsyncClient, model: str) -> LifecycleResult:
    response = await client.get(NVIDIA_MODELS_URL.format(model=model))
    if response.status_code == 200:
        return LifecycleResult(OK, "listed in the NVIDIA model catalog")
    if response.status_code == 410:
        match = _EOL_RE.search(response.text)
        eol = match.group(1) if match else None
        return LifecycleResult(RETIRED, f"HTTP 410 from the NVIDIA model catalog, end of life {eol or 'unknown'}", eol)
    if response.status_code == 404:
        return LifecycleResult(NOT_IN_CATALOG, "HTTP 404: not in the NVIDIA model catalog")
    raise _CheckFailed(f"HTTP {response.status_code} from the NVIDIA model catalog")


async def _check_build_page(client: httpx.AsyncClient, slug: str) -> LifecycleResult:
    response = await client.get(BUILD_PAGE_URL.format(slug=slug))
    if response.status_code == 404:
        return LifecycleResult(PAGE_MISSING, f"HTTP 404 for build.nvidia.com/{slug}")
    if response.status_code != 200:
        raise _CheckFailed(f"HTTP {response.status_code} for build.nvidia.com/{slug}")
    page = response.text.replace('\\"', '"')
    notice = _NOTICE_RE.search(page)
    if notice or _DEPRECATED_RE.search(page):
        text = notice.group(1).strip() if notice else "isDeprecated is true"
        return LifecycleResult(DEPRECATED, f"build.nvidia.com/{slug}: {text}"[:_DETAIL_LIMIT])
    return LifecycleResult(OK, f"build.nvidia.com/{slug} carries no deprecation notice")


async def _openrouter_catalog(client: httpx.AsyncClient) -> dict[str, dict]:
    """Id → entry of OpenRouter's public model list."""
    response = await client.get(OPENROUTER_MODELS_URL)
    if response.status_code != 200:
        raise _CheckFailed(f"HTTP {response.status_code} from the OpenRouter model list")
    try:
        data = response.json().get("data")
    except (ValueError, AttributeError) as exc:
        raise _CheckFailed("the OpenRouter model list is not a JSON object") from exc
    if not isinstance(data, list):
        raise _CheckFailed("the OpenRouter model list has no data list")
    return {str(item["id"]): item for item in data if isinstance(item, dict) and item.get("id")}


async def _check_openrouter_endpoints(
    client: httpx.AsyncClient, model: str, catalog: Awaitable[dict[str, dict]]
) -> LifecycleResult:
    response = await client.get(OPENROUTER_ENDPOINTS_URL.format(model=model))
    if response.status_code == 404:
        return LifecycleResult(NOT_IN_CATALOG, "HTTP 404: not in the OpenRouter catalog")
    if response.status_code != 200:
        raise _CheckFailed(f"HTTP {response.status_code} from OpenRouter")
    try:
        data = response.json().get("data")
    except (ValueError, AttributeError) as exc:
        raise _CheckFailed("OpenRouter answered without a JSON object") from exc
    if not isinstance(data, dict):
        raise _CheckFailed("OpenRouter answered without a data object")
    endpoints = [item for item in data.get("endpoints") or [] if isinstance(item, dict)]
    own_dates = [data.get("expiration_date"), *(item.get("expiration_date") for item in endpoints)]
    try:
        listed = (await catalog).get(model) or {}
    except Exception:
        # The list only adds an expiry. What this answer shows by itself (no
        # endpoint left, or a date of its own) still stands; only a would-be
        # ok fails the check, since the list might have held a date.
        if endpoints and not any(own_dates):
            raise
        listed = {}
    expirations = sorted(str(value) for value in (listed.get("expiration_date"), *own_dates) if value)
    eol = expirations[0] if expirations else None
    if not endpoints:
        # Before the expiry: a preview's list entry carries its date all
        # along, and a withdrawn one (qwen3.8 :free from 2026-10-13) must not
        # keep reading as merely expiring.
        return LifecycleResult(NO_ENDPOINTS, "OpenRouter lists no endpoint serving it", eol)
    if eol:
        return LifecycleResult(EXPIRING, f"OpenRouter lists an expiration date: {eol}", eol)
    providers = ", ".join(sorted({str(item.get("provider_name") or "?") for item in endpoints}))
    return LifecycleResult(OK, f"{len(endpoints)} endpoint(s): {providers}"[:_DETAIL_LIMIT])


def _describe(error: BaseException) -> str:
    if isinstance(error, _CheckFailed):
        return str(error)
    return f"{type(error).__name__}: {error}"[:_DETAIL_LIMIT]
