"""The role table: which models answer each kind of request, in what order.

A *role* is a job the bot gives a model (routing, chatting, thinking, looking
at images, background utility work, embedding, drawing). Each role is an
ordered list of tiers (:class:`TierSpec`): the code defaults below, with any
role replaced wholesale by the ``MODEL_ROLES`` JSON object::

    MODEL_ROLES='{"chat": [{"provider": "nvidia", "model": "nvidia/nemotron-3-ultra-550b-a55b",
                            "thinking": false, "timeout_s": 40}]}'

:meth:`RoleTable.from_settings` checks everything at startup and never drops
a tier silently. A bad table (invalid JSON, unknown role or provider, a paid
OpenRouter id, a model that cannot do the role) raises
:class:`RoleConfigError`, so the revision never becomes ready. A tier whose
provider is not configured (no key; Colab until it registers) stays in the
table, disabled with its reason, for ``/health/detail``. An id without a
model profile gets the provider's generic protocol and a config warning,
except on Colab: the VM pulls only the pinned (profiled) models, and any
other id would answer 404, which marks the whole Colab endpoint unhealthy,
so an unprofiled Colab id is a RoleConfigError.

``MODEL_PATH_PREFERENCE`` orders every role: ``colab`` (the default) puts
its Colab tiers first, ``api`` puts them last; the order within each group
is the configured one. ``FORCE_TIER`` ("provider" or "provider:model",
test-only) disables every tier that does not match it, so one path can be
exercised end to end; a FORCE_TIER that matches nothing, or leaves a role
that answers users without a usable tier, fails startup.

How the tier options are read: ``thinking`` switches thinking on or off;
``think`` (Colab) and ``effort`` (OpenRouter) set how hard, and switch it
too when ``thinking`` is absent ("false" and "none" mean off). A tier with
neither follows the request; in the thinking role, which only requests
that need thinking use, it is marked as thinking, so the chain holds it to
the thinking tiers' half of the time like the others.

Sampling: a tier's ``temperature`` wins, then the model card's default
from the model profile (Nemotron 3 on NIM: 1.0 / top_p 0.95), then the
call's (CHAT_TEMPERATURE, ...). The NIM router tiers (Super and Ultra) and
the utility tier on Super set a low temperature on purpose: routing JSON,
search keywords and memory summaries should come out the same for the same
input, and the card's 1.0 is untested for them (the planned routing eval
decides).

Colab never holds up a reply: its tiers are enabled only with
COLAB_ENABLED (the "colab" provider), and until the on-demand Colab
session is READY they are skipped at once (SkipTierError, no network), so
every message is answered by the API tiers meanwhile. Once it is READY, a
Colab tier with a usable API tier after it stops
``COLAB_API_RESERVE_SECONDS`` before the deadline, or halfway to it when
that is later (``Tier.shares_time``, whatever its thinking setting), so a
slow session cannot use up the window before the API tiers get a turn. The
router's Colab tier (nimble, protocol systemone) is not a chain tier: the
orchestrator asks it directly (src/providers/systemone.py), and
COLAB_ROUTER_ENABLED=false disables it here.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator

from src.config import RETIRED_MODEL_IDS
from src.providers.fallback_chain import Tier
from src.providers.model_profiles import (
    CHAT,
    CHAT_PROTOCOLS,
    EMBED,
    EMBED_PROTOCOLS,
    IMAGE,
    IMAGE_PROTOCOLS,
    MAX_REASONING_BUDGET,
    PROVIDERS,
    SYSTEMONE,
    THINK_LEVELS,
    ModelProfile,
    generic_profile,
    get_profile,
)
from src.providers.openrouter_provider import is_free_model_id

ROLES: tuple[str, ...] = ("router", "chat", "thinking", "vision", "utility", "embed", "image")
COLAB = "colab"

_ROLE_KIND: dict[str, str] = {
    "router": CHAT,
    "chat": CHAT,
    "thinking": CHAT,
    "vision": CHAT,
    "utility": CHAT,
    "embed": EMBED,
    "image": IMAGE,
}
# A cold Colab must never block a reply, so these need an API tier too
# (and FORCE_TIER must leave each of them a usable tier).
_NEEDS_API_TIER: tuple[str, ...] = ("router", "chat", "vision", "utility")
# Seconds a tier may run when its spec gives no timeout_s.
_DEFAULT_TIMEOUT_S: dict[str, float] = {
    "router": 12,
    "chat": 40,
    "thinking": 100,
    "vision": 40,
    "utility": 30,
    "embed": 10,
    "image": 60,
}
_MISSING_PROVIDER: dict[str, str] = {
    "colab": "colab tier not enabled",
    "nvidia": "NVIDIA_API_KEY not set",
    "openrouter": "OPENROUTER_API_KEY not set",
}
# Spec fields a provider reads from Tier.options (plus the derived "thinking").
_OPTION_FIELDS = frozenset(
    {"think", "effort", "reasoning_budget", "json_mode", "max_tokens", "temperature", "dim", "steps"}
)

# The v3 line-up (decision table rows 6, 7, 16, 17; since 2026-10-01
# Nemotron 3 Ultra backs up router, chat and utility, see ADR-0007).
COLAB_QWEN = "qwen3.8:27b-mtp-q4_K_M"
COLAB_NIMBLE = "nimble:9b"
NIM_ULTRA = "nvidia/nemotron-3-ultra-550b-a55b"
NIM_SUPER = "nvidia/nemotron-3-super-120b-a12b"
NIM_MUSE = "meta/muse-glimmer-30b"
NIM_OMNI = "nvidia/nemotron-3-nano-omni-30b-a3b-reasoning"
NIM_EMBED_1B = "nvidia/nemotron-3-embed-1b"
FLUX_KLEIN = "black-forest-labs/flux.2-klein-4b"
FLUX_DEV = "black-forest-labs/flux.1-dev"
OR_QWEN = "qwen/qwen3.8-27b:free"
# Served by NVIDIA behind OpenRouter: last, since they fail together with NIM.
OR_ULTRA = "nvidia/nemotron-3-ultra-550b-a55b:free"
OR_SUPER = "nvidia/nemotron-3-super-120b-a12b:free"


class RoleConfigError(ValueError):
    """The role table cannot be used; startup must fail."""


class TierSpec(BaseModel):
    """One tier as configured: which model, how long, and with what options."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    provider: Literal[PROVIDERS]
    model: str
    # None: the role's default (router 12 s, chat 40 s, thinking 100 s, ...).
    timeout_s: float | None = Field(default=None, gt=0)
    thinking: bool | None = None
    think: Literal[THINK_LEVELS] | None = None
    effort: Literal["none", "low", "medium", "high", "xhigh"] | None = None
    reasoning_budget: int | None = Field(default=None, gt=0, le=MAX_REASONING_BUDGET)
    json_mode: Literal["json_schema", "json_object", "prompt"] | None = None
    max_tokens: int | None = Field(default=None, gt=0)
    temperature: float | None = Field(default=None, ge=0)
    dim: int | None = Field(default=None, gt=0)
    steps: int | None = Field(default=None, gt=0)

    @field_validator("model")
    @classmethod
    def _strip_model(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("must not be empty")
        return value

    @field_validator("think", mode="before")
    @classmethod
    def _think_false(cls, value: Any) -> Any:
        # JSON false means "false"; true is refused (Ollama reads it as medium).
        if value is False:
            return "false"
        if value is True:
            raise ValueError("use a level: false, low, medium or xhigh")
        return value

    @property
    def label(self) -> str:
        return f"{self.provider}:{self.model}"

    def wants_thinking(self) -> bool | None:
        """True/False when the spec decides thinking, None when the request does."""
        if self.thinking is not None:
            return self.thinking
        if self.think is not None:
            return self.think != "false"
        if self.effort is not None:
            return self.effort != "none"
        return None

    def options(self) -> dict[str, Any]:
        """What the chain and the providers read from ``Tier.options``."""
        options = self.model_dump(include=_OPTION_FIELDS, exclude_none=True)
        thinking = self.wants_thinking()
        if thinking is not None:
            options["thinking"] = thinking
        return options


def default_role_specs(settings: Any) -> dict[str, list[TierSpec]]:
    """The code defaults, with the Colab think levels and budgets from settings."""

    def tier(provider: str, model: str, **options: Any) -> TierSpec:
        return TierSpec(provider=provider, model=model, **options)

    chat_think = settings.colab_chat_think
    return {
        # The NIM tiers' low temperature is deliberate (see the module
        # docstring). Ultra has no structured output, so it is asked for
        # JSON in the prompt alone (json_mode "prompt").
        "router": [
            tier("colab", COLAB_NIMBLE, timeout_s=6),
            tier(
                "nvidia",
                NIM_SUPER,
                thinking=False,
                json_mode="json_schema",
                max_tokens=256,
                temperature=0.1,
                timeout_s=12,
            ),
            tier(
                "nvidia",
                NIM_ULTRA,
                thinking=False,
                json_mode="prompt",
                max_tokens=256,
                temperature=0.1,
                timeout_s=12,
            ),
            tier("openrouter", OR_SUPER, effort="none", json_mode="json_object", timeout_s=12),
        ],
        # A Colab answer tier runs to about 70 s: the reply window
        # (REPLY_WINDOW_SECONDS, 90 s) minus the API tiers' reserve
        # (fallback_chain.COLAB_API_RESERVE_SECONDS).
        "chat": [
            tier("colab", COLAB_QWEN, think=chat_think, timeout_s=70),
            tier("nvidia", NIM_ULTRA, thinking=False, timeout_s=40),
            tier("openrouter", OR_QWEN, effort="none", timeout_s=40),
            tier("openrouter", OR_ULTRA, effort="none", timeout_s=40),
        ],
        # Callers append the chat tiers (thinking off) after these. Ultra
        # on NIM takes no reasoning_budget (HTTP 400): it thinks at
        # medium_effort and NVIDIA_THINKING_BUDGET only adds max_tokens.
        "thinking": [
            tier("colab", COLAB_QWEN, think=settings.colab_deep_think, timeout_s=110),
            tier("nvidia", NIM_ULTRA, thinking=True, timeout_s=100),
            tier("openrouter", OR_QWEN, effort="medium", timeout_s=100),
            tier("openrouter", OR_ULTRA, effort="medium", timeout_s=100),
        ],
        # The 2026-10-01 vision probe: Nano Omni answers in 2-6 s but is
        # often busy (HTTP 503 in under a second); muse-glimmer takes 8-19 s
        # at the ~4k-token prompt VisionAgent keeps to. The API tiers never
        # think, not even for a request that needs it (only Colab does).
        # Ultra cannot see images, so it has no vision tier.
        "vision": [
            tier("colab", COLAB_QWEN, think=chat_think, timeout_s=70),
            tier("nvidia", NIM_OMNI, thinking=False, temperature=0.2, max_tokens=1024, timeout_s=15),
            tier("nvidia", NIM_MUSE, thinking=False, timeout_s=35),
            tier("openrouter", OR_QWEN, effort="none", timeout_s=15),
        ],
        # No Colab: its single slot belongs to replies.
        "utility": [
            tier("nvidia", NIM_SUPER, thinking=False, temperature=0.2, timeout_s=30),
            tier("nvidia", NIM_ULTRA, thinking=False, timeout_s=30),
            tier("openrouter", OR_ULTRA, effort="none", timeout_s=30),
        ],
        "embed": [tier("nvidia", NIM_EMBED_1B, dim=settings.embedding_dimensions, timeout_s=10)],
        "image": [
            tier("nvidia", FLUX_KLEIN, timeout_s=60),
            tier("nvidia", FLUX_DEV, steps=50, timeout_s=90),
        ],
    }


def parse_model_roles(raw: str) -> dict[str, list[TierSpec]]:
    """The roles ``MODEL_ROLES`` replaces; {} when it is blank."""
    raw = (raw or "").strip()
    if not raw:
        return {}
    try:
        data = json.loads(raw)
    except ValueError as exc:
        raise RoleConfigError(f"MODEL_ROLES is not valid JSON ({exc.msg} at char {exc.pos})") from None
    if not isinstance(data, dict):
        raise RoleConfigError("MODEL_ROLES must be a JSON object: {role: [tier, ...]}")

    errors: list[str] = []
    roles: dict[str, list[TierSpec]] = {}
    for role, tiers in data.items():
        if role not in ROLES:
            errors.append(f"unknown role {role!r} (roles: {', '.join(ROLES)})")
            continue
        if not isinstance(tiers, list) or not tiers:
            errors.append(f"{role}: expected a non-empty list of tiers")
            continue
        specs: list[TierSpec] = []
        for index, item in enumerate(tiers):
            try:
                specs.append(TierSpec.model_validate(item))
            except ValidationError as exc:
                errors.append(f"{role}[{index}]: {_validation_summary(exc)}")
        roles[role] = specs
    if errors:
        raise RoleConfigError("MODEL_ROLES: " + "; ".join(errors))
    return roles


def role_specs(settings: Any) -> dict[str, list[TierSpec]]:
    """Every role's validated tiers in call order, defaults filled in."""
    return {role: [spec for spec, _ in rows] for role, rows in _resolve(settings)[0].items()}


@dataclass(frozen=True)
class RoleTier:
    """One tier of a role: usable (``tier`` set) or disabled with a reason."""

    role: str
    spec: TierSpec
    profile: ModelProfile
    tier: Tier | None = None
    disabled_reason: str = ""

    @property
    def enabled(self) -> bool:
        return self.tier is not None

    @property
    def label(self) -> str:
        return self.spec.label

    def view(
        self,
        health: Mapping[str, Any] | None = None,
        lifecycle: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        """This tier for ``/health/detail``.

        *health* is the model's entry in ``ModelHealth.get_status(
        include_errors=True)["models"]`` (None before its first call) and
        *lifecycle* its last daily end-of-life check (None before one ran).
        """
        health = health or {}
        return {
            "provider": self.spec.provider,
            "model": self.spec.model,
            "protocol": self.profile.protocol,
            "profiled": self.profile.known,
            "timeout_s": self.spec.timeout_s,
            "enabled": self.enabled,
            "disabled_reason": self.disabled_reason or None,
            "health": health.get("state", "untried"),
            "health_reason": health.get("reason"),
            "last_error": health.get("last_error"),
            "dead": health.get("state") == "dead",
            "eol": health.get("eol") or (lifecycle or {}).get("eol"),
            "lifecycle": dict(lifecycle) if lifecycle else None,
        }


class RoleTable:
    """Validated role tiers, bound to the providers this process built."""

    def __init__(
        self,
        rows: Mapping[str, Sequence[RoleTier]],
        *,
        warnings: Sequence[str] = (),
    ) -> None:
        self._rows = {role: list(rows.get(role, ())) for role in ROLES}
        self.warnings: tuple[str, ...] = tuple(warnings)

    @classmethod
    def from_settings(cls, settings: Any, available_providers: Mapping[str, Any]) -> RoleTable:
        """The table for *settings*, with tiers bound to *available_providers*.

        ``available_providers`` maps a provider name ("nvidia", "openrouter",
        "colab") to its client; a missing or None entry disables that
        provider's tiers.

        Raises:
            RoleConfigError: The configuration cannot be used.
        """
        resolved, warnings = _resolve(settings)
        force = (getattr(settings, "force_tier", "") or "").strip()
        router_off = not getattr(settings, "colab_router_enabled", True)
        rows: dict[str, list[RoleTier]] = {}
        for role, specs in resolved.items():
            rows[role] = [
                RoleTier(role, spec, profile, disabled_reason="COLAB_ROUTER_ENABLED=false")
                if router_off and profile.protocol == SYSTEMONE
                else _bind(role, spec, profile, available_providers.get(spec.provider), force)
                for spec, profile in specs
            ]
        if force:
            _check_force_tier(force, rows)
        return cls(rows, warnings=warnings)

    def roles(self) -> tuple[str, ...]:
        return tuple(self._rows)

    def rows(self, role: str) -> list[RoleTier]:
        """Every tier of *role*, enabled or not, in call order."""
        if role not in self._rows:
            raise ValueError(f"unknown role {role!r}")
        return list(self._rows[role])

    def tiers(self, role: str) -> list[Tier]:
        """The enabled tiers of *role*, in call order."""
        return [row.tier for row in self.rows(role) if row.tier is not None]

    def has_api_tier(self, role: str) -> bool:
        """True when *role* has an enabled tier that does not wait on Colab."""
        return any(row.enabled and row.spec.provider != COLAB for row in self.rows(role))

    def describe(self, role: str) -> str:
        """``nvidia:a → openrouter:b (off: colab:c)`` for the startup log."""
        rows = self.rows(role)
        enabled = " → ".join(row.label for row in rows if row.enabled) or "(none)"
        disabled = [row.label for row in rows if not row.enabled]
        return enabled + (f" (off: {', '.join(disabled)})" if disabled else "")

    def view(self, health: Any = None, lifecycle: Any = None) -> dict[str, list[dict[str, Any]]]:
        """Every role's tiers for ``/health/detail``.

        *health* (a ``ModelHealth``) adds each model's state and last error,
        *lifecycle* (a ``ModelLifecycle``) its last end-of-life check.
        """
        models = health.get_status(include_errors=True)["models"] if health is not None else {}
        checks = lifecycle.results() if lifecycle is not None else {}
        return {
            role: [row.view(models.get(row.label), checks.get(row.label)) for row in rows]
            for role, rows in self._rows.items()
        }


# ── Resolution and validation ────────────────────────────────


def _resolve(settings: Any) -> tuple[dict[str, list[tuple[TierSpec, ModelProfile]]], list[str]]:
    """Defaults + MODEL_ROLES, checked, ordered by MODEL_PATH_PREFERENCE.

    Raises:
        RoleConfigError: Every problem found, in one message.
    """
    specs = {**default_role_specs(settings), **parse_model_roles(getattr(settings, "model_roles", ""))}
    # Settings validates the value: "colab" or "api".
    colab_last = getattr(settings, "model_path_preference", "colab") == "api"

    errors: list[str] = []
    warnings: list[str] = []
    resolved: dict[str, list[tuple[TierSpec, ModelProfile]]] = {}
    for role in ROLES:
        rows: list[tuple[TierSpec, ModelProfile]] = []
        for spec in specs[role]:
            spec = _with_defaults(role, spec, settings)
            profile = _profile(role, spec, errors, warnings)
            if profile is not None:
                _check(role, spec, profile, settings, errors)
                rows.append((spec, profile))
        _check_role(role, specs[role], errors)
        # A stable sort: "api" uses Colab only if it happens to be up, never
        # ahead of the API path; "colab" tries it first in every role.
        rows.sort(key=lambda row: (row[0].provider == COLAB) == colab_last)
        resolved[role] = rows
    if errors:
        raise RoleConfigError("Invalid model role table: " + "; ".join(errors))
    return resolved, warnings


def _with_defaults(role: str, spec: TierSpec, settings: Any) -> TierSpec:
    update: dict[str, Any] = {}
    if spec.timeout_s is None:
        update["timeout_s"] = _DEFAULT_TIMEOUT_S[role]
    if role == "embed" and spec.dim is None:
        update["dim"] = settings.embedding_dimensions
    return spec.model_copy(update=update) if update else spec


def _profile(role: str, spec: TierSpec, errors: list[str], warnings: list[str]) -> ModelProfile | None:
    profile = get_profile(spec.provider, spec.model)
    if profile is not None:
        return profile
    profile = generic_profile(spec.provider, spec.model, _ROLE_KIND[role])
    if profile is None:
        if spec.provider == COLAB:
            errors.append(
                f"{role}: {spec.label} has no model profile; Colab serves only the pinned models "
                "(model profiles and colab/versions.json)"
            )
        else:
            errors.append(f"{role}: {spec.provider} cannot serve the {role} role ({spec.model})")
        return None
    warnings.append(
        f"{role}: {spec.label} has no model profile; it gets the generic "
        f"{profile.protocol} protocol (check the id)"
    )
    return profile


def _check(role: str, spec: TierSpec, profile: ModelProfile, settings: Any, errors: list[str]) -> None:
    """Capability and option checks for one tier."""

    def fail(problem: str) -> None:
        errors.append(f"{role}: {spec.label} {problem}")

    kind = _ROLE_KIND[role]
    if spec.model in RETIRED_MODEL_IDS:
        fail("is retired (HTTP 410)")
    allowed = {
        CHAT: CHAT_PROTOCOLS | ({SYSTEMONE} if role == "router" else set()),
        EMBED: EMBED_PROTOCOLS,
        IMAGE: IMAGE_PROTOCOLS,
    }[kind]
    if profile.protocol not in allowed:
        fail(f"is a {profile.protocol} model and cannot serve the {role} role")
    if (
        spec.provider == "openrouter"
        and not getattr(settings, "openrouter_allow_paid", False)
        and not is_free_model_id(spec.model)
    ):
        fail("is a paid OpenRouter id; use a ':free' id or set OPENROUTER_ALLOW_PAID=true")
    if role == "vision" and "image" not in profile.modalities:
        fail("does not take image input")

    if spec.wants_thinking() and not profile.can_think:
        fail("cannot think")
    if spec.think is not None:
        if spec.provider != "colab":
            fail("sets think, which only Colab tiers read (use thinking or effort)")
        elif profile.think_levels and spec.think not in profile.think_levels:
            fail(f"does not accept think={spec.think}")
    if spec.effort is not None:
        if spec.provider != "openrouter":
            fail("sets effort, which only OpenRouter tiers read (use thinking)")
        elif spec.effort != "none" and profile.efforts and spec.effort not in profile.efforts:
            fail(f"does not accept effort={spec.effort} (it lists {', '.join(profile.efforts)})")
    if spec.reasoning_budget is not None and not (spec.provider == "openrouter" or profile.reasoning_budget):
        fail("does not take a reasoning_budget")
    if spec.json_mode is not None and (kind != CHAT or spec.json_mode not in profile.json_modes):
        supported = ", ".join(sorted(profile.json_modes)) or "none"
        fail(f"does not support json_mode={spec.json_mode} (supported: {supported})")
    if kind != CHAT and (spec.max_tokens is not None or spec.temperature is not None):
        fail("sets max_tokens/temperature, which only chat roles read")
    if spec.steps is not None and role != "image":
        fail("sets steps, which only the image role reads")
    if role == "embed":
        if spec.dim != settings.embedding_dimensions:
            fail(f"dim {spec.dim} must equal EMBEDDING_DIMENSIONS ({settings.embedding_dimensions}), the vector index width")
        if profile.native_dim is not None and spec.dim is not None and spec.dim > profile.native_dim:
            fail(f"dim {spec.dim} is wider than the model's {profile.native_dim}")
    elif spec.dim is not None:
        fail("sets dim, which only the embed role reads")


def _check_role(role: str, specs: Sequence[TierSpec], errors: list[str]) -> None:
    if role in _NEEDS_API_TIER and all(spec.provider == COLAB for spec in specs):
        errors.append(f"{role}: needs at least one tier that is not Colab")
    if role == "embed" and len(specs) != 1:
        # Vectors from two models cannot be compared, so no fallback.
        errors.append("embed: needs exactly one tier")


def _bind(role: str, spec: TierSpec, profile: ModelProfile, provider: Any, force: str) -> RoleTier:
    if force and not _matches(force, spec):
        return RoleTier(role, spec, profile, disabled_reason=f"FORCE_TIER={force}")
    if provider is None:
        return RoleTier(role, spec, profile, disabled_reason=_MISSING_PROVIDER[spec.provider])
    options = spec.options()
    if role == "thinking":
        # It thinks anyway (only thinking requests use this role); marked,
        # it counts against the thinking tiers' half of the time left.
        options.setdefault("thinking", True)
    # _with_defaults has filled timeout_s.
    tier = Tier(
        provider,
        spec.model,
        timeout_s=float(spec.timeout_s),
        options=options,
        modalities=profile.modalities,
        shares_time=spec.provider == COLAB,
    )
    return RoleTier(role, spec, profile, tier=tier)


def _matches(force: str, spec: TierSpec) -> bool:
    provider, _, model = force.partition(":")
    return spec.provider == provider and (not model or spec.model == model)


def _check_force_tier(force: str, rows: Mapping[str, Sequence[RoleTier]]) -> None:
    """FORCE_TIER is test-only; one that silently disables replies must fail.

    Raises:
        RoleConfigError: *force* matches no tier, or leaves a role that
            answers users (router, chat, vision, utility) without one.
    """
    if not any(_matches(force, row.spec) for role_rows in rows.values() for row in role_rows):
        raise RoleConfigError(f"FORCE_TIER={force} matches no tier of the model role table")
    empty = [role for role in _NEEDS_API_TIER if not any(row.enabled for row in rows[role])]
    if empty:
        reasons = sorted(
            {
                row.disabled_reason
                for role in empty
                for row in rows[role]
                if _matches(force, row.spec)
            }
        )
        why = f" ({'; '.join(reasons)})" if reasons else ""
        raise RoleConfigError(
            f"FORCE_TIER={force} leaves no usable tier for: {', '.join(empty)}{why}"
        )


def _validation_summary(exc: ValidationError) -> str:
    """Field paths and messages only; the input values are not echoed."""
    return ", ".join(
        f"{'.'.join(str(part) for part in error['loc']) or 'tier'}: {error['msg']}"
        for error in exc.errors()
    )
