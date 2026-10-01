"""The model role table: defaults, overrides, ordering, FORCE_TIER, validation.

Also how agents pick tiers from it (``BaseAgent.targets_for``). Providers are
plain fakes: binding only needs something to hold, and nothing is called.
"""

from __future__ import annotations

import json
import re
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

from src.agents.chat_agent import ChatAgent
from src.agents.vision_agent import VisionAgent
from src.config import Settings
from src.models.agent_request import AgentRequest
from src.providers.fallback_chain import Tier
from src.providers.model_profiles import get_profile
from src.runtime.model_roles import (
    COLAB_NIMBLE,
    COLAB_QWEN,
    FLUX_DEV,
    FLUX_KLEIN,
    NIM_EMBED_1B,
    NIM_MUSE,
    NIM_OMNI,
    NIM_SUPER,
    NIM_ULTRA,
    OR_QWEN,
    OR_SUPER,
    OR_ULTRA,
    ROLES,
    RoleConfigError,
    RoleTable,
    default_role_specs,
    parse_model_roles,
    role_specs,
)
from src.services.model_health import ModelHealth


def _provider(name: str) -> SimpleNamespace:
    return SimpleNamespace(name=name)


def _providers(*names: str) -> dict[str, SimpleNamespace]:
    return {name: _provider(name) for name in names}


def _settings(**overrides) -> Settings:
    if isinstance(overrides.get("model_roles"), dict):
        overrides["model_roles"] = json.dumps(overrides["model_roles"])
    return Settings(_env_file=None, **overrides)


def _table(providers=("nvidia", "openrouter"), **overrides) -> RoleTable:
    return RoleTable.from_settings(_settings(**overrides), _providers(*providers))


def _labels(table: RoleTable, role: str) -> list[str]:
    return [row.label for row in table.rows(role)]


def _enabled(table: RoleTable, role: str) -> list[str]:
    return [tier.label for tier in table.tiers(role)]


class DefaultTableTests(unittest.TestCase):
    def test_every_role_has_the_decided_tiers_in_order(self) -> None:
        table = _table(providers=("nvidia", "openrouter", "colab"))

        self.assertEqual(
            {role: _labels(table, role) for role in ROLES},
            {
                "router": [
                    f"colab:{COLAB_NIMBLE}",
                    f"nvidia:{NIM_SUPER}",
                    f"nvidia:{NIM_ULTRA}",
                    f"openrouter:{OR_SUPER}",
                ],
                "chat": [
                    f"colab:{COLAB_QWEN}",
                    f"nvidia:{NIM_ULTRA}",
                    f"openrouter:{OR_QWEN}",
                    f"openrouter:{OR_ULTRA}",
                ],
                "thinking": [
                    f"colab:{COLAB_QWEN}",
                    f"nvidia:{NIM_ULTRA}",
                    f"openrouter:{OR_QWEN}",
                    f"openrouter:{OR_ULTRA}",
                ],
                # The 2026-10-01 vision probe's pick (D3).
                "vision": [
                    f"colab:{COLAB_QWEN}",
                    f"nvidia:{NIM_OMNI}",
                    f"nvidia:{NIM_MUSE}",
                    f"openrouter:{OR_QWEN}",
                ],
                "utility": [f"nvidia:{NIM_SUPER}", f"nvidia:{NIM_ULTRA}", f"openrouter:{OR_ULTRA}"],
                "embed": [f"nvidia:{NIM_EMBED_1B}"],
                "image": [f"nvidia:{FLUX_KLEIN}", f"nvidia:{FLUX_DEV}"],
            },
        )
        self.assertEqual(table.warnings, ())

    def test_the_default_tier_options_and_timeouts(self) -> None:
        table = _table(nvidia_thinking_budget=2048, colab_chat_think="false", colab_deep_think="xhigh")

        def row(role, index):
            found = table.rows(role)[index]
            return found.spec.timeout_s, found.spec.options()

        self.assertEqual(row("router", 0), (6, {}))
        self.assertEqual(
            row("router", 1),
            (12, {"thinking": False, "json_mode": "json_schema", "max_tokens": 256, "temperature": 0.1}),
        )
        # Ultra has no structured output: the prompt alone asks for the JSON.
        self.assertEqual(
            row("router", 2),
            (12, {"thinking": False, "json_mode": "prompt", "max_tokens": 256, "temperature": 0.1}),
        )
        self.assertEqual(
            row("router", 3), (12, {"effort": "none", "json_mode": "json_object", "thinking": False})
        )
        # The Colab window (90 s) minus the API reserve (20 s).
        self.assertEqual(row("chat", 0), (70, {"think": "false", "thinking": False}))
        self.assertEqual(row("chat", 1), (40, {"thinking": False}))
        self.assertEqual(row("chat", 2), (40, {"effort": "none", "thinking": False}))
        self.assertEqual(row("chat", 3), (40, {"effort": "none", "thinking": False}))
        self.assertEqual(row("thinking", 0), (110, {"think": "xhigh", "thinking": True}))
        # Ultra takes no reasoning_budget on NIM (HTTP 400 in the live gate).
        self.assertEqual(row("thinking", 1), (100, {"thinking": True}))
        self.assertEqual(row("thinking", 2), (100, {"effort": "medium", "thinking": True}))
        self.assertEqual(row("thinking", 3), (100, {"effort": "medium", "thinking": True}))
        self.assertEqual(row("vision", 0), (70, {"think": "false", "thinking": False}))
        # Nano Omni as the probe called it: thinking off, 0.2, 1024 tokens.
        self.assertEqual(
            row("vision", 1), (15, {"thinking": False, "max_tokens": 1024, "temperature": 0.2})
        )
        # The API vision tiers never think, not even for a thinking request.
        self.assertEqual(row("vision", 2), (35, {"thinking": False}))
        self.assertEqual(row("vision", 3), (15, {"effort": "none", "thinking": False}))
        # A deliberately low temperature on Super for utility work.
        self.assertEqual(row("utility", 0), (30, {"thinking": False, "temperature": 0.2}))
        self.assertEqual(row("utility", 1), (30, {"thinking": False}))
        self.assertEqual(row("utility", 2), (30, {"effort": "none", "thinking": False}))
        for role, count in (("router", 4), ("chat", 4), ("thinking", 4), ("vision", 4), ("utility", 3)):
            with self.subTest(role=role):
                self.assertEqual(len(table.rows(role)), count)
        self.assertEqual(row("embed", 0), (10, {"dim": 1024}))
        self.assertEqual(row("image", 0), (60, {}))
        self.assertEqual(row("image", 1), (90, {"steps": 50}))

    def test_colab_tiers_stay_disabled_until_a_colab_provider_registers(self) -> None:
        table = _table()

        for role in ("router", "chat", "thinking", "vision"):
            with self.subTest(role=role):
                first = table.rows(role)[0]
                self.assertEqual(first.spec.provider, "colab")
                self.assertFalse(first.enabled)
                self.assertEqual(first.disabled_reason, "colab tier not enabled")
        self.assertEqual(_enabled(table, "chat")[0], f"nvidia:{NIM_ULTRA}")

    def test_a_provider_registered_as_colab_is_picked_up(self) -> None:
        colab = _provider("colab")
        table = RoleTable.from_settings(_settings(), {"colab": colab, "openrouter": _provider("openrouter")})

        chat = table.tiers("chat")
        self.assertIs(chat[0].provider, colab)
        self.assertEqual((chat[0].model, chat[0].timeout_s), (COLAB_QWEN, 70))
        self.assertEqual(chat[0].modalities, frozenset({"text", "image"}))
        self.assertEqual(table.tiers("router")[0].model, COLAB_NIMBLE)
        # A Colab tier leaves the API tiers after it their share of the time
        # left, whatever its thinking setting (5a review: immediate replies).
        for role in ("router", "chat", "thinking", "vision"):
            with self.subTest(role=role):
                tiers = table.tiers(role)
                self.assertTrue(tiers[0].shares_time)
                self.assertFalse(any(tier.shares_time for tier in tiers[1:]))

    def test_a_missing_key_disables_that_providers_tiers_with_a_reason(self) -> None:
        table = _table(providers=("openrouter",))

        self.assertEqual(
            _enabled(table, "chat"),
            [f"openrouter:{OR_QWEN}", f"openrouter:{OR_ULTRA}"],
        )
        self.assertEqual(table.tiers("embed"), [])
        self.assertEqual(
            [row.disabled_reason for row in table.rows("image")], ["NVIDIA_API_KEY not set"] * 2
        )
        table = _table(providers=("nvidia",))
        self.assertEqual(
            [row.disabled_reason for row in table.rows("utility")], ["", "", "OPENROUTER_API_KEY not set"]
        )

    def test_tiers_carry_the_profile_modalities(self) -> None:
        table = _table()

        modalities = {tier.label: tier.modalities for tier in table.tiers("chat")}
        # Ultra is text only, so an image request skips it.
        self.assertEqual(modalities[f"nvidia:{NIM_ULTRA}"], frozenset({"text"}))
        self.assertIn("image", modalities[f"openrouter:{OR_QWEN}"])

    def test_default_vision_tiers_see_images_and_answering_roles_keep_an_api_tier(self) -> None:
        table = _table(providers=("nvidia", "openrouter", "colab"))

        for row in table.rows("vision"):
            with self.subTest(tier=row.label):
                self.assertIn("image", row.profile.modalities)
        for role in ("router", "chat", "vision", "utility"):
            with self.subTest(role=role):
                self.assertTrue(table.has_api_tier(role))
        # No default tier is a Gemma model any more (2026-10-01).
        for role in ROLES:
            for row in table.rows(role):
                with self.subTest(tier=row.label):
                    self.assertNotIn("gemma", row.spec.model)

    def test_has_api_tier_ignores_colab_and_disabled_tiers(self) -> None:
        self.assertTrue(_table().has_api_tier("chat"))
        colab_only = RoleTable.from_settings(_settings(), _providers("colab"))
        self.assertTrue(colab_only.tiers("chat"))
        self.assertFalse(colab_only.has_api_tier("chat"))

    def test_role_specs_needs_no_providers(self) -> None:
        specs = role_specs(_settings(embedding_dimensions=512))

        self.assertEqual(specs["embed"][0].model, NIM_EMBED_1B)
        self.assertEqual((specs["embed"][0].dim, specs["embed"][0].timeout_s), (512, 10))


class OverrideTests(unittest.TestCase):
    def test_model_roles_replaces_only_the_roles_it_names(self) -> None:
        table = _table(
            model_roles={
                "utility": [
                    {"provider": "openrouter", "model": OR_QWEN, "effort": "none", "timeout_s": 20},
                ]
            }
        )

        self.assertEqual(_labels(table, "utility"), [f"openrouter:{OR_QWEN}"])
        self.assertEqual(table.tiers("utility")[0].timeout_s, 20)
        # Every other role keeps its default.
        self.assertEqual(_labels(table, "router"), _labels(_table(), "router"))

    def test_a_tier_without_timeout_gets_the_roles_default(self) -> None:
        table = _table(model_roles={"thinking": [{"provider": "nvidia", "model": NIM_ULTRA, "thinking": True}]})

        self.assertEqual(table.tiers("thinking")[0].timeout_s, 100)

    def test_a_thinking_tier_without_a_thinking_flag_counts_as_thinking(self) -> None:
        # Review: with no thinking/think/effort it followed the request (and
        # thought), but the chain saw no "thinking" option, so it escaped the
        # thinking tiers' half of the time and could use the whole window.
        table = _table(
            model_roles={
                "thinking": [
                    {"provider": "nvidia", "model": NIM_SUPER},
                    {"provider": "openrouter", "model": OR_QWEN, "effort": "none"},
                ],
                "vision": [{"provider": "nvidia", "model": NIM_MUSE}],
            }
        )

        unflagged, off = table.tiers("thinking")
        self.assertIs(unflagged.options["thinking"], True)
        # An explicit setting stays as configured; other roles still follow the request.
        self.assertIs(off.options["thinking"], False)
        self.assertNotIn("thinking", table.tiers("vision")[0].options)

    def test_an_unknown_id_gets_the_generic_protocol_and_a_warning(self) -> None:
        table = _table(model_roles={"chat": [{"provider": "nvidia", "model": "example/new-model"}]})

        [row] = table.rows("chat")
        self.assertTrue(row.enabled)
        self.assertEqual(row.profile.protocol, "generic")
        self.assertFalse(row.profile.known)
        self.assertEqual(len(table.warnings), 1)
        self.assertIn("nvidia:example/new-model has no model profile", table.warnings[0])

    def test_an_unprofiled_colab_id_fails_startup(self) -> None:
        # The VM pulls only the pinned models: any other id would answer 404
        # and mark the whole Colab endpoint (qwen3.8 chat, nimble) unhealthy.
        roles = {
            "chat": [
                {"provider": "colab", "model": "qwen3.8:27b", "think": "low"},
                {"provider": "nvidia", "model": NIM_ULTRA},
            ]
        }
        with self.assertRaisesRegex(RoleConfigError, "chat: colab:qwen3.8:27b has no model profile; Colab serves only the pinned models"):
            _table(providers=("colab", "nvidia"), model_roles=roles)

    def test_json_false_is_the_think_level_false(self) -> None:
        table = _table(
            providers=("colab", "nvidia"),
            model_roles={
                "chat": [
                    {"provider": "colab", "model": COLAB_QWEN, "think": False},
                    {"provider": "nvidia", "model": NIM_ULTRA},
                ]
            },
        )

        self.assertEqual(dict(table.tiers("chat")[0].options), {"think": "false", "thinking": False})

    def test_paid_openrouter_ids_need_the_opt_in(self) -> None:
        roles = {"chat": [{"provider": "openrouter", "model": "qwen/qwen3.8-27b"}]}
        with self.assertRaisesRegex(RoleConfigError, "paid OpenRouter id"):
            _table(model_roles=roles)

        table = _table(model_roles=roles, openrouter_allow_paid=True)
        self.assertEqual(_enabled(table, "chat"), ["openrouter:qwen/qwen3.8-27b"])
        # The free router is billed like a ":free" id.
        _table(model_roles={"chat": [{"provider": "openrouter", "model": "openrouter/free"}]})


class PathPreferenceTests(unittest.TestCase):
    _ROLES = {
        "chat": [
            {"provider": "nvidia", "model": NIM_ULTRA},
            {"provider": "colab", "model": COLAB_QWEN},
            {"provider": "openrouter", "model": OR_QWEN},
        ]
    }

    def test_colab_puts_colab_tiers_first_in_every_role(self) -> None:
        table = _table(providers=("nvidia", "openrouter", "colab"), model_roles=self._ROLES)

        self.assertEqual(
            _labels(table, "chat"),
            [f"colab:{COLAB_QWEN}", f"nvidia:{NIM_ULTRA}", f"openrouter:{OR_QWEN}"],
        )

    def test_api_puts_colab_tiers_last_and_keeps_the_rest_in_order(self) -> None:
        table = _table(
            providers=("nvidia", "openrouter", "colab"), model_path_preference="api"
        )

        for role in ("router", "chat", "thinking", "vision"):
            with self.subTest(role=role):
                labels = _labels(table, role)
                self.assertTrue(labels[-1].startswith("colab:"))
                default = [label for label in _labels(_table(), role) if not label.startswith("colab:")]
                self.assertEqual(labels[:-1], default)

        table = _table(model_path_preference="api", model_roles=self._ROLES)
        self.assertEqual(
            _labels(table, "chat"),
            [f"nvidia:{NIM_ULTRA}", f"openrouter:{OR_QWEN}", f"colab:{COLAB_QWEN}"],
        )


class ForceTierTests(unittest.TestCase):
    def test_force_tier_keeps_one_provider(self) -> None:
        table = _table(force_tier="openrouter")

        for role in ("router", "chat", "thinking", "vision", "utility"):
            with self.subTest(role=role):
                self.assertTrue(table.tiers(role))
                self.assertTrue(all(tier.provider_name == "openrouter" for tier in table.tiers(role)))
        self.assertEqual(table.tiers("embed"), [])
        self.assertEqual(
            {row.disabled_reason for row in table.rows("image")}, {"FORCE_TIER=openrouter"}
        )

    def test_force_tier_can_pin_one_model(self) -> None:
        # qwen3.8 free is in chat, thinking and vision; give the router and
        # utility one too.
        table = _table(
            force_tier=f"openrouter:{OR_QWEN}",
            model_roles={
                "router": [
                    {"provider": "nvidia", "model": NIM_SUPER, "json_mode": "json_schema"},
                    {"provider": "openrouter", "model": OR_QWEN, "effort": "none", "json_mode": "json_schema"},
                ],
                "utility": [
                    {"provider": "nvidia", "model": NIM_SUPER},
                    {"provider": "openrouter", "model": OR_QWEN, "effort": "none"},
                ],
            },
        )

        for role in ("router", "chat", "thinking", "vision", "utility"):
            with self.subTest(role=role):
                self.assertEqual(_enabled(table, role), [f"openrouter:{OR_QWEN}"])
        self.assertEqual(table.tiers("embed"), [])
        self.assertEqual(table.tiers("image"), [])

    def test_pinning_a_model_some_answering_role_lacks_fails_startup(self) -> None:
        # Super is not a chat tier by default, and is text only: nothing
        # would answer a message or an image.
        with self.assertRaisesRegex(RoleConfigError, "leaves no usable tier for: chat, vision$"):
            _table(force_tier=f"nvidia:{NIM_SUPER}")

    def test_a_force_tier_that_matches_nothing_fails_startup(self) -> None:
        with self.assertRaisesRegex(RoleConfigError, "FORCE_TIER=nvidia:nvidia/typo matches no tier"):
            _table(force_tier="nvidia:nvidia/typo")

    def test_a_force_tier_that_leaves_a_user_facing_role_empty_fails_startup(self) -> None:
        # Colab has no provider before Phase 8: every reply would fail.
        with self.assertRaisesRegex(RoleConfigError, r"leaves no usable tier for: router, chat, vision, utility \(colab tier not enabled\)"):
            _table(force_tier="colab")
        # A forced provider without its key is just as silent.
        with self.assertRaisesRegex(RoleConfigError, r"NVIDIA_API_KEY not set"):
            _table(providers=("openrouter",), force_tier="nvidia")


class ValidationTests(unittest.TestCase):
    def assertRejected(self, roles, pattern: str, **settings) -> None:
        with self.assertRaisesRegex(RoleConfigError, pattern):
            _table(model_roles=roles, **settings)

    def test_bad_json_and_bad_shapes(self) -> None:
        for raw, pattern in (
            ("{not json", "not valid JSON"),
            ('["chat"]', "must be a JSON object"),
            ('{"chat": []}', "chat: expected a non-empty list"),
            ('{"chat": {"provider": "nvidia"}}', "chat: expected a non-empty list"),
            ('{"chitchat": [{"provider": "nvidia", "model": "x"}]}', "unknown role 'chitchat'"),
            ('{"chat": [{"provider": "gemini", "model": "x"}]}', r"chat\[0\]: provider"),
            ('{"chat": [{"provider": "nvidia"}]}', r"chat\[0\]: model"),
            ('{"chat": [{"provider": "nvidia", "model": "x", "colour": "red"}]}', r"chat\[0\]: colour"),
            ('{"chat": [{"provider": "nvidia", "model": "x", "timeout_s": 0}]}', r"chat\[0\]: timeout_s"),
            ('{"chat": [{"provider": "colab", "model": "x", "think": true}]}', r"chat\[0\]: think"),
        ):
            with self.subTest(raw=raw), self.assertRaisesRegex(RoleConfigError, pattern):
                _table(model_roles=raw)

    def test_error_messages_never_echo_values(self) -> None:
        with self.assertRaises(RoleConfigError) as ctx:
            _table(model_roles='{"chat": [{"provider": "secret-looking-value", "model": "x"}]}')

        self.assertNotIn("secret-looking-value", str(ctx.exception))

    def test_capability_checks(self) -> None:
        cases = (
            # Ultra is text only.
            ({"vision": [{"provider": "nvidia", "model": NIM_ULTRA}]}, "does not take image input"),
            ({"chat": [{"provider": "nvidia", "model": NIM_ULTRA, "json_mode": "json_schema"}]},
             r"does not support json_mode=json_schema \(supported: prompt\)"),
            # Neither Ultra takes structured output: their router JSON is asked for in the prompt.
            ({"router": [{"provider": "nvidia", "model": NIM_ULTRA, "json_mode": "json_object"}]},
             r"does not support json_mode=json_object \(supported: prompt\)"),
            ({"router": [{"provider": "openrouter", "model": OR_ULTRA, "json_mode": "json_object"}]},
             r"does not support json_mode=json_object \(supported: prompt\)"),
            ({"chat": [{"provider": "openrouter", "model": "openrouter/free", "thinking": True}]}, "cannot think"),
            ({"chat": [{"provider": "nvidia", "model": NIM_ULTRA, "think": "low"}]}, "only Colab tiers read"),
            ({"chat": [{"provider": "nvidia", "model": NIM_ULTRA, "effort": "low"}]}, "only OpenRouter tiers read"),
            ({"thinking": [{"provider": "openrouter", "model": OR_QWEN, "effort": "high"}]},
             "does not accept effort=high"),
            ({"vision": [{"provider": "nvidia", "model": NIM_MUSE, "reasoning_budget": 1024}]},
             "does not take a reasoning_budget"),
            # Any reasoning_budget is HTTP 400 on Ultra (live gate 2026-09-30).
            ({"thinking": [{"provider": "nvidia", "model": NIM_ULTRA, "thinking": True, "reasoning_budget": 1024}]},
             "does not take a reasoning_budget"),
            ({"vision": [{"provider": "nvidia", "model": NIM_MUSE, "effort": "low"}]}, "only OpenRouter tiers read"),
            ({"embed": [{"provider": "nvidia", "model": NIM_EMBED_1B, "max_tokens": 10}]},
             "only chat roles read"),
            ({"chat": [{"provider": "nvidia", "model": NIM_ULTRA, "steps": 4}]}, "only the image role reads"),
            ({"chat": [{"provider": "nvidia", "model": NIM_ULTRA, "dim": 4}]}, "only the embed role reads"),
            ({"chat": [{"provider": "nvidia", "model": FLUX_DEV}]}, "cannot serve the chat role"),
            ({"embed": [{"provider": "openrouter", "model": OR_QWEN}]}, "cannot serve the embed role"),
            ({"router": [{"provider": "colab", "model": COLAB_NIMBLE}]}, "router: needs at least one tier that is not Colab"),
            ({"chat": [{"provider": "colab", "model": COLAB_QWEN}]}, "chat: needs at least one tier that is not Colab"),
            ({"chat": [{"provider": "nvidia", "model": "qwen/qwen3.5-397b-a17b"}]}, r"is retired \(HTTP 410\)"),
            ({"embed": [
                {"provider": "nvidia", "model": NIM_EMBED_1B},
                {"provider": "nvidia", "model": "example/other-embed"},
            ]}, "embed: needs exactly one tier"),
        )
        for roles, pattern in cases:
            with self.subTest(pattern=pattern):
                self.assertRejected(roles, pattern)

    def test_the_embed_width_must_be_the_index_width_and_fit_the_model(self) -> None:
        self.assertRejected(
            {"embed": [{"provider": "nvidia", "model": NIM_EMBED_1B, "dim": 512}]},
            r"dim 512 must equal EMBEDDING_DIMENSIONS \(1024\)",
        )
        self.assertRejected(
            {"embed": [{"provider": "nvidia", "model": NIM_EMBED_1B, "dim": 4096}]},
            "wider than the model's 2048",
            embedding_dimensions=4096,
        )
        table = _table(embedding_dimensions=512)
        self.assertEqual(dict(table.tiers("embed")[0].options), {"dim": 512})

    def test_every_problem_is_reported_at_once(self) -> None:
        with self.assertRaises(RoleConfigError) as ctx:
            _table(
                model_roles={
                    "vision": [{"provider": "nvidia", "model": NIM_ULTRA}],
                    "chat": [{"provider": "openrouter", "model": "qwen/qwen3.8-27b"}],
                }
            )

        self.assertIn("does not take image input", str(ctx.exception))
        self.assertIn("paid OpenRouter id", str(ctx.exception))

    def test_defaults_are_valid_for_every_think_level(self) -> None:
        for level in ("false", "low", "medium", "xhigh"):
            with self.subTest(level=level):
                specs = default_role_specs(_settings(colab_chat_think=level, colab_deep_think=level))
                self.assertEqual(specs["chat"][0].think, level)
                _table(colab_chat_think=level, colab_deep_think=level)


class DetailViewTests(unittest.TestCase):
    def test_view_merges_health_and_lifecycle_without_traffic_details(self) -> None:
        health = ModelHealth()
        health.mark_dead("nvidia", NIM_ULTRA, "HTTP 410, end of life 2027-01-01T00:00:00Z", eol="2027-01-01T00:00:00Z")
        lifecycle = SimpleNamespace(
            results=lambda: {f"nvidia:{NIM_SUPER}": {"state": "ok", "eol": None}}
        )
        table = _table()

        view = table.view(health, lifecycle)

        colab, ultra = view["chat"][:2]
        self.assertEqual(
            (colab["enabled"], colab["disabled_reason"], colab["health"]),
            (False, "colab tier not enabled", "untried"),
        )
        self.assertEqual((ultra["health"], ultra["dead"], ultra["eol"]), ("dead", True, "2027-01-01T00:00:00Z"))
        self.assertEqual(view["utility"][0]["lifecycle"], {"state": "ok", "eol": None})
        self.assertEqual(view["utility"][0]["protocol"], "nim_nemotron3")
        self.assertEqual(set(view), set(ROLES))


class AgentTierSelectionTests(unittest.IsolatedAsyncioTestCase):
    """BaseAgent.targets_for: which role's tiers answer which request."""

    def _agents(self, **settings):
        table = _table(providers=("nvidia", "openrouter", "colab"), **settings)
        tiers = {
            "thinking_targets": table.tiers("thinking"),
            "vision_targets": table.tiers("vision"),
        }
        chat = ChatAgent(_settings(**settings), None, targets=table.tiers("chat"), **tiers)
        vision = VisionAgent(_settings(**settings), None, targets=table.tiers("vision"), **tiers)
        return table, chat, vision

    def test_a_plain_request_gets_the_chat_tiers_as_configured(self) -> None:
        table, chat, _ = self._agents()

        tiers = chat.targets_for(AgentRequest(text="你好", disable_thinking=True))

        self.assertEqual(tiers, table.tiers("chat"))

    def test_a_thinking_request_gets_the_thinking_role_then_chat_without_thinking(self) -> None:
        table, chat, _ = self._agents(colab_deep_think="xhigh")

        tiers = chat.targets_for(AgentRequest(text="請一步一步推導", disable_thinking=False))

        thinking = table.tiers("thinking")
        self.assertEqual(tiers[: len(thinking)], thinking)
        # Review: Colab qwen came again with thinking off, so a READY but
        # slow session took 1/2 then 1/4 of the window and the API 1/4. It
        # gets its one half-share; the API tiers keep the other half.
        self.assertEqual(
            [(tier.label, tier.options.get("thinking")) for tier in tiers[len(thinking):]],
            [(f"nvidia:{NIM_ULTRA}", False), (f"openrouter:{OR_QWEN}", False), (f"openrouter:{OR_ULTRA}", False)],
        )
        self.assertEqual(tiers[0].label, f"colab:{COLAB_QWEN}")
        self.assertEqual(dict(tiers[0].options), {"think": "xhigh", "thinking": True})
        self.assertEqual(dict(tiers[1].options), {"thinking": True})
        self.assertEqual(tiers[2].options["effort"], "medium")
        self.assertEqual((tiers[3].label, tiers[3].options["effort"]), (f"openrouter:{OR_ULTRA}", "medium"))

    def test_a_colab_tier_only_in_the_chat_role_still_runs_with_thinking_off(self) -> None:
        roles = {"thinking": [{"provider": "nvidia", "model": NIM_ULTRA, "thinking": True}]}
        table, chat, _ = self._agents(model_roles=roles)

        tiers = chat.targets_for(AgentRequest(text="證明", disable_thinking=False))

        self.assertEqual(
            [tier.label for tier in tiers],
            [f"nvidia:{NIM_ULTRA}", *(tier.label for tier in table.tiers("chat"))],
        )

    def test_a_thinking_override_in_model_roles_is_what_runs(self) -> None:
        roles = {"thinking": [{"provider": "nvidia", "model": NIM_SUPER, "thinking": True,
                               "reasoning_budget": 16000, "timeout_s": 30}]}
        _, chat, _ = self._agents(model_roles=roles)

        first = chat.targets_for(AgentRequest(text="證明", disable_thinking=False))[0]

        self.assertEqual((first.label, first.timeout_s), (f"nvidia:{NIM_SUPER}", 30))
        self.assertEqual(dict(first.options), {"thinking": True, "reasoning_budget": 16000})

    def test_an_image_request_gets_the_vision_tiers(self) -> None:
        table, chat, vision = self._agents(colab_deep_think="medium")
        image = "data:image/png;base64,AAAA"

        for agent in (chat, vision):
            with self.subTest(agent=agent.name):
                plain = agent.targets_for(
                    AgentRequest(text="這是什麼", image_base64=image, disable_thinking=True)
                )
                self.assertEqual(plain, table.tiers("vision"))
        # A quoted image counts too.
        quoted = chat.targets_for(
            AgentRequest(text="這張呢", quoted_image_url="https://x/a.png", disable_thinking=True)
        )
        self.assertEqual(quoted, table.tiers("vision"))

        thinking = vision.targets_for(
            AgentRequest(text="仔細分析這張圖", image_base64=image, disable_thinking=False)
        )
        # Only Colab thinks (once); every API tier is set to thinking off.
        self.assertEqual(
            [(tier.label, tier.options["thinking"]) for tier in thinking],
            [
                (f"colab:{COLAB_QWEN}", True),
                (f"nvidia:{NIM_OMNI}", False),
                (f"nvidia:{NIM_MUSE}", False),
                (f"openrouter:{OR_QWEN}", False),
            ],
        )
        # The Colab tier thinks at the deep level.
        self.assertEqual(thinking[0].options["think"], "medium")

    async def test_the_chain_receives_the_selected_tiers(self) -> None:
        table = _table()
        chain = SimpleNamespace(generate=AsyncMock(return_value=SimpleNamespace(text="ok", model="m")))
        agent = ChatAgent(
            _settings(),
            chain,
            targets=table.tiers("chat"),
            thinking_targets=table.tiers("thinking"),
            vision_targets=table.tiers("vision"),
        )

        await agent.process(AgentRequest(text="請一步一步推導", disable_thinking=False))

        sent = chain.generate.await_args.kwargs["targets"]
        self.assertEqual(sent[0].label, f"nvidia:{NIM_ULTRA}")
        self.assertIs(sent[0].options["thinking"], True)
        self.assertIsInstance(sent[0], Tier)


class DocumentedExampleTests(unittest.TestCase):
    def test_documented_model_roles_examples_keep_router_and_utility_temperatures_low(self) -> None:
        # The model card's temperature (1.0 on Nemotron) outranks the call's,
        # so a router or utility tier on such a model must set its own, or
        # copying the example raises summaries and query plans to 1.0.
        root = Path(__file__).resolve().parents[1]
        found = 0
        for name in ("README.md", ".env.example"):
            path = root / name
            if not path.is_file():
                continue
            for raw in re.findall(r"MODEL_ROLES='(\{.*?\})'", path.read_text(encoding="utf-8")):
                for role, specs in parse_model_roles(raw).items():
                    for spec in specs:
                        profile = get_profile(spec.provider, spec.model)
                        if role in ("router", "utility") and profile and profile.default_temperature is not None:
                            found += 1
                            with self.subTest(doc=name, tier=spec.label):
                                self.assertIsNotNone(spec.temperature)
        if not found:
            self.skipTest("no documented router/utility example on a model with card sampling")


if __name__ == "__main__":
    unittest.main()
