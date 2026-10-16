"""!colab / !whoami, the group trigger, the Colab settings and their wiring."""

from __future__ import annotations

import asyncio
import tempfile
import unittest
from contextlib import ExitStack
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock, patch

from fastapi.testclient import TestClient
from pydantic import ValidationError

import main as main_module
from scripts.envfile import parse_env_file
from src.config import Settings
from src.handlers.webhook_handler import should_handle
from src.models.agent_request import AgentRequest
from src.pipeline.event_pipeline import PipelineDeps, process_event
from src.providers.ollama_provider import OllamaChatProvider
from src.providers.systemone import RouterGates, SystemOneRouter
from src.runtime.composition import (
    build_colab_controller,
    build_colab_endpoint,
    build_container,
    colab_config_problems,
    split_router_tiers,
)
from src.runtime.model_roles import RoleTable
from src.services.colab import protocol
from src.services.colab.admin import (
    ADMIN_ONLY,
    COLAB_OFF_CONFIG,
    COLAB_USAGE,
    STATE_UNAVAILABLE,
    WHOAMI_DM_ONLY,
    Command,
    format_status,
    parse_command,
    run_command,
    turn_off,
    turn_on,
)
from src.services.colab.control_plane import ColabControlPlane
from src.services.colab.controller import ColabController
from src.services.colab.endpoint import ColabEndpoint
from src.services.colab.registration import ColabRegistrar
from src.services.colab.state import (
    ASSIGNING,
    AUTO,
    BOOTING,
    COOLDOWN,
    DISABLED,
    DRAINING,
    IDLE,
    OFF,
    READY,
    ColabState,
    ColabStateError,
    FirestoreColabStateStore,
    InMemoryColabStateStore,
)
from src.services.model_health import ModelHealth
from src.services.model_lifecycle import ModelLifecycle
from tests.colab_fakes import (
    QWEN,
    ROUTER,
    SECRET,
    TUNNEL,
    Clock,
    FakeTunnel,
    make_endpoint,
    on_gpu,
    ready_state,
    write_adc,
)

_ADMIN = "U" + "a" * 32
_OTHER = "U" + "b" * 32
_ADC_DIR = tempfile.TemporaryDirectory()
_ADC = write_adc(_ADC_DIR.name)


def tearDownModule() -> None:
    _ADC_DIR.cleanup()


def _colab_settings(**overrides) -> Settings:
    values = dict(
        _env_file=None,
        colab_enabled=True,
        firestore_enabled=True,
        colab_shared_secret=SECRET,
        colab_adc_path=str(_ADC),
        internal_cron_secret="c" * 32,
        line_admin_user_ids=_ADMIN,
        openrouter_api_key="k",
    )
    values.update(overrides)
    return Settings(**values)


class ParseTests(unittest.TestCase):
    def test_commands(self) -> None:
        cases = {
            "!colab": Command("colab", ""),
            "!colab on": Command("colab", "on"),
            "  !COLAB   Status ": Command("colab", "status"),
            "!colab off now": Command("colab", "off now"),
            "!whoami": Command("whoami"),
            "!WhoAmI ": Command("whoami"),
            "!whoami please": None,
            "!colabs": None,
            "我想用 !colab": None,
            "": None,
        }
        for text, expected in cases.items():
            with self.subTest(text=text):
                self.assertEqual(parse_command(text), expected)


class RunCommandTests(unittest.IsolatedAsyncioTestCase):
    async def _run(self, text: str, *, user: str = _ADMIN, source: str = "user", colab="default", state=None):
        clock = Clock(1_790_000_000.0)
        if colab == "default":
            colab, _ = make_endpoint(state or ColabState(status=IDLE), clock=clock)
        request = AgentRequest(user_id=user, group_id="G1" if source != "user" else user, source_type=source)
        reply = await run_command(parse_command(text), request, colab=colab, admin_user_ids=frozenset({_ADMIN}), clock=clock)
        return reply, colab, clock

    async def test_whoami_answers_the_id_in_private_only(self) -> None:
        reply, *_ = await self._run("!whoami", user=_OTHER)
        self.assertEqual(reply, f"你的 LINE userId：{_OTHER}")
        for source in ("group", "room"):
            reply, *_ = await self._run("!whoami", user=_OTHER, source=source)
            self.assertEqual(reply, WHOAMI_DM_ONLY)
            self.assertNotIn(_OTHER, reply)

    async def test_colab_is_admin_only(self) -> None:
        for user in (_OTHER, ""):
            for text in ("!colab on", "!colab off", "!colab status", "!colab"):
                reply, colab, _ = await self._run(text, user=user)
                with self.subTest(user=user, text=text):
                    self.assertEqual(reply, ADMIN_ONLY)
                    self.assertEqual(colab.store.writes, 0)
        reply, *_ = await self._run("!colab status", user=_OTHER, colab=None)
        self.assertEqual(reply, ADMIN_ONLY)

    async def test_an_admin_without_colab_is_told_it_is_off(self) -> None:
        reply, *_ = await self._run("!colab on", colab=None)
        self.assertEqual(reply, COLAB_OFF_CONFIG)

    async def test_on_off_status_and_usage(self) -> None:
        reply, colab, clock = await self._run("!colab on", state=ColabState(status=DISABLED, disabled_reason="invalid_grant"))
        state = colab.store.cached
        self.assertEqual(
            (state.status, state.desired, state.manual_demand_at, state.last_demand_at, state.disabled_reason),
            (IDLE, AUTO, clock(), None, ""),
        )
        self.assertIn("已開啟 Colab", reply)

        reply, colab, _ = await self._run("!colab off", state=ready_state())
        self.assertEqual((colab.store.cached.status, colab.store.cached.desired), (DRAINING, OFF))
        self.assertIn("已關閉 Colab", reply)

        reply, colab, _ = await self._run("!colab", state=ready_state(ccu_balance=88.2, ccu_rate=5.37, cu_used_today=3.1))
        self.assertIn("READY", reply)
        self.assertIn("CU 餘額：88.2", reply)
        self.assertIn("今日用量：3.1 / 30 CU", reply)
        self.assertNotIn(TUNNEL, reply)
        self.assertNotIn(ROUTER, reply)

        # A moved tag is news, not a reason to sit out (the models follow
        # their tags); no digest recorded is no reason either.
        verified = protocol.colab_models()
        moved = ready_state(
            digests={QWEN: verified[QWEN], ROUTER: "adfcdbb4070d303fb43fc11fc06a14269ec2bc7136a55369e86d412b86ba2831"}
        )
        reply, colab, _ = await self._run("!colab status", state=moved)
        self.assertIn(f"{ROUTER}：tag 已更新（digest adfcdbb4070d，驗證過的是 73e026ff389b）", reply)
        self.assertNotIn("這次不用它路由", reply)

        reply, colab, _ = await self._run("!colab status", state=ready_state(digests={QWEN: verified[QWEN]}))
        self.assertNotIn(ROUTER, reply)

        spilled = ready_state(vram={QWEN: on_gpu(QWEN), ROUTER: on_gpu(ROUTER, 6 << 30)})
        reply, colab, _ = await self._run("!colab status", state=spilled)
        self.assertIn(f"{ROUTER}：沒有完全載入 GPU（VRAM 6144 / 10110 MiB），這次不用它路由（改用 API）", reply)
        self.assertEqual(reply.count(ROUTER), 1)

        # Not loaded when the probe looked, or evicted since (a health check
        # found it gone): the endpoint's view, not only the stored one.
        reply, colab, _ = await self._run("!colab status", state=ready_state(vram={QWEN: on_gpu(QWEN)}))
        self.assertIn(f"{ROUTER}：沒有載入（/api/ps 找不到），這次不用它路由（改用 API）", reply)
        clock = Clock(1_790_000_000.0)
        evicted = FakeTunnel(**{"GET /api/ps": (200, {"models": [{"name": QWEN, **on_gpu(QWEN)}]})})
        colab, _ = make_endpoint(ready_state(clock()), evicted, clock=clock)
        await colab.ready()
        with self.assertLogs("linebot", "WARNING"):
            self.assertTrue(await colab.check_health())
        reply, *_ = await self._run("!colab status", colab=colab)
        self.assertIn(f"{ROUTER}：沒有載入（/api/ps 找不到），這次不用它路由（改用 API）", reply)

        reply, colab, _ = await self._run("!colab restart")
        self.assertEqual((reply, colab.store.writes), (COLAB_USAGE, 0))

    async def test_a_store_failure_is_a_friendly_reply(self) -> None:
        colab, _ = make_endpoint(ColabState())

        async def broken(*args, **kwargs):
            raise ColabStateError("down")

        colab.store.update = broken
        colab.store.load = broken
        for text in ("!colab on", "!colab off", "!colab status"):
            reply, *_ = await self._run(text, colab=colab)
            self.assertEqual(reply, STATE_UNAVAILABLE)


class TransitionTests(unittest.TestCase):
    def test_on(self) -> None:
        for status in (DISABLED, COOLDOWN):
            state = turn_on(
                ColabState(
                    status=status,
                    desired=OFF,
                    failures=3,
                    cooldown_until=99.0,
                    cooldown_step=2,
                    disabled_reason="x",
                    hold_reason="y",
                ),
                50.0,
            )
            with self.subTest(status=status):
                self.assertEqual(
                    (state.status, state.since, state.desired, state.failures, state.cooldown_until, state.disabled_reason),
                    (IDLE, 50.0, AUTO, 0, None, ""),
                )
                self.assertEqual((state.cooldown_step, state.hold_reason), (0, ""))
                # A manual start, told apart from a triggered message's demand.
                self.assertEqual((state.manual_demand_at, state.last_demand_at), (50.0, None))
        for status in (IDLE, ASSIGNING, BOOTING, READY, DRAINING):
            state = turn_on(ColabState(status=status, desired=OFF, last_demand_at=7.0), 50.0)
            with self.subTest(status=status):
                self.assertEqual(
                    (state.status, state.desired, state.manual_demand_at, state.last_demand_at),
                    (status, AUTO, 50.0, 7.0),
                )

    def test_off(self) -> None:
        for status in (ASSIGNING, BOOTING, READY):
            state = turn_off(ColabState(status=status, action="assign", action_at=40.0), 50.0)
            with self.subTest(status=status):
                self.assertEqual((state.status, state.desired, state.since), (DRAINING, OFF, 50.0))
                # An assign in flight stays known, so its runtime is released when it comes.
                self.assertEqual((state.action, state.action_at), ("assign", 40.0))
        for status in (IDLE, DRAINING, COOLDOWN, DISABLED):
            state = turn_off(ColabState(status=status), 50.0)
            with self.subTest(status=status):
                self.assertEqual((state.status, state.desired), (status, OFF))

    def test_status_lines(self) -> None:
        colab = SimpleNamespace(auto_start=False, is_ready=lambda: False, limits=SimpleNamespace(daily_cu_cap=30.0))
        text = format_status(
            ColabState(status=COOLDOWN, since=1.0, desired=OFF, cooldown_until=9e9, last_error="HTTP 412", disabled_reason=""),
            colab,
            100.0,
        )
        self.assertIn("COOLDOWN", text)
        self.assertIn("手動（!colab on）", text)
        self.assertIn("目前已關閉", text)
        self.assertIn("冷卻到", text)
        self.assertIn("最後錯誤：HTTP 412", text)
        self.assertNotIn("CU 餘額", text)
        self.assertNotIn("暫不開機", text)

        held = format_status(ColabState(status=IDLE, hold_reason="CU 餘額 5.0 低於下限 10", starts_today=2), colab, 100.0)
        self.assertIn("暫不開機：CU 餘額 5.0 低於下限 10", held)
        self.assertIn("（開機 2 次）", held)


class PipelineCommandTests(unittest.IsolatedAsyncioTestCase):
    async def _process(self, text: str, *, source: str = "user", user: str = _ADMIN, colab=None):
        request = AgentRequest(user_id=user, group_id=user if source == "user" else "G1", source_type=source, text=text, reply_token="r")
        line = SimpleNamespace(send_text=AsyncMock(), send_loading_animation=AsyncMock())
        orchestrator = SimpleNamespace(route=AsyncMock())
        enrich = AsyncMock(return_value=request)
        deps = PipelineDeps(
            line=line,
            memory=SimpleNamespace(),
            message_cache=SimpleNamespace(),
            orchestrator=orchestrator,
            get_agent=lambda name: None,
            process_input=AsyncMock(return_value=request),
            enrich_request=enrich,
            extract_text=Mock(return_value=text),
            colab=colab,
            admin_user_ids=frozenset({_ADMIN}),
        )
        event = {"type": "message", "replyToken": "r", "source": {"type": source, "userId": user, "groupId": "G1"}}
        await process_event(event, deps)
        return line, orchestrator, enrich

    async def test_commands_reply_without_routing_or_rate_limit(self) -> None:
        colab, _ = make_endpoint(ColabState(status=IDLE))
        for text, expected in (("!whoami", f"你的 LINE userId：{_ADMIN}"), ("!colab on", "已開啟 Colab")):
            line, orchestrator, enrich = await self._process(text, colab=colab)
            with self.subTest(text=text):
                self.assertIn(expected, line.send_text.await_args.args[2])
                orchestrator.route.assert_not_called()
                enrich.assert_not_called()
                line.send_loading_animation.assert_not_called()

    async def test_a_non_admin_in_a_group_is_refused(self) -> None:
        line, orchestrator, _ = await self._process("!colab off", source="group", user=_OTHER)
        line.send_text.assert_awaited_once_with("r", "G1", ADMIN_ONLY)
        orchestrator.route.assert_not_called()


class GroupTriggerTests(unittest.TestCase):
    def _group(self, text: str) -> dict:
        return {"type": "message", "source": {"type": "group", "groupId": "G1", "userId": "U1"}, "message": {"type": "text", "text": text}}

    def test_colab_commands_trigger_in_groups(self) -> None:
        with patch("src.handlers.webhook_handler.get_settings", return_value=Settings(_env_file=None)):
            for text, handled in (
                ("!colab status", True),
                ("!colab", True),
                ("!COLAB on", True),
                ("!hej !colab off", True),
                ("!colabs", False),
                ("!whoami", False),
                ("聊聊 !colab", False),
            ):
                with self.subTest(text=text):
                    self.assertEqual(should_handle(self._group(text)), handled)


class SettingsTests(unittest.TestCase):
    def test_defaults(self) -> None:
        settings = Settings(_env_file=None)
        self.assertEqual(
            (
                settings.colab_enabled,
                settings.colab_router_enabled,
                settings.colab_router_confidence,
                settings.colab_router_think_on,
                settings.colab_router_think_off,
                settings.colab_idle_ttl_seconds,
                settings.colab_max_session_seconds,
                settings.colab_min_cu_balance,
                settings.colab_daily_cu_cap,
                settings.admin_user_ids,
            ),
            (False, True, 0.0, 0.85, 0.15, 900, 39600, 10.0, 30.0, frozenset()),
        )
        self.assertEqual(
            (
                settings.colab_adc_path,
                settings.colab_tunnel_mode,
                settings.colab_tunnel_token,
                settings.colab_boot_timeout_seconds,
                settings.colab_heartbeat_seconds,
                settings.colab_account_dedicated,
                settings.colab_dry_run,
            ),
            ("/secrets/colab/adc.json", "quick", "", 900, 300, False, False),
        )

    def test_colab_requirements_never_stop_the_app(self) -> None:
        # Settings accept them all; composition keeps the tier off instead.
        with tempfile.TemporaryDirectory() as tmp:
            broken = Path(tmp) / "broken.json"
            broken.write_text('{"type": "service_account"}', encoding="utf-8")
            cases = {
                "COLAB_ENABLED needs FIRESTORE_ENABLED=true": {"firestore_enabled": False},
                # The controller runs only from the authenticated cron tick.
                "COLAB_ENABLED needs INTERNAL_CRON_SECRET": {"internal_cron_secret": ""},
                "COLAB_SHARED_SECRET of at least 32": {"colab_shared_secret": "x" * 31},
                "COLAB_ADC_PATH": {"colab_adc_path": str(Path(tmp) / "missing.json")},
                "COLAB_ADC_PATH: " + str(broken): {"colab_adc_path": str(broken)},
                "COLAB_TUNNEL_MODE=named needs COLAB_TUNNEL_HOST": {"colab_tunnel_mode": "named", "colab_tunnel_token": "t"},
                "COLAB_TUNNEL_MODE=named needs COLAB_TUNNEL_TOKEN": {
                    "colab_tunnel_mode": "named",
                    "colab_tunnel_host": "ollama.example.org",
                },
            }
            for expected, overrides in cases.items():
                settings = _colab_settings(**overrides)
                with self.subTest(expected=expected):
                    self.assertTrue(settings.colab_enabled)
                    self.assertTrue(any(expected in problem for problem in colab_config_problems(settings)))
                    with self.assertLogs("linebot", "WARNING") as logs:
                        self.assertIsNone(build_colab_endpoint(settings))
                    self.assertIn("Colab tier off", "\n".join(logs.output))
                    self.assertNotIn(SECRET, "\n".join(logs.output))
        self.assertEqual(colab_config_problems(_colab_settings()), [])
        self.assertEqual(
            colab_config_problems(
                _colab_settings(
                    colab_tunnel_mode="named", colab_tunnel_host="ollama.example.org", colab_tunnel_token="t"
                )
            ),
            [],
        )
        # Off, none is needed.
        Settings(_env_file=None, colab_shared_secret="")

    def test_on_cloud_run_colab_needs_cloud_tasks(self) -> None:
        # Inline, the controller's minutes-long calls would run in the cron request.
        cloud_run = {"K_SERVICE": "linebot"}
        needs = "COLAB_ENABLED on Cloud Run needs Cloud Tasks"
        for name, overrides in {
            "inline": {"tasks_mode": "inline", "internal_tasks_secret": "s" * 32},
            "no tasks secret": {"tasks_mode": "auto", "internal_tasks_secret": ""},
            "cloud without a secret": {"tasks_mode": "cloud", "internal_tasks_secret": ""},
        }.items():
            with self.subTest(name):
                problems = colab_config_problems(_colab_settings(**overrides), cloud_run)
                self.assertTrue(any(needs in problem for problem in problems), problems)
        for mode in ("auto", "cloud"):
            with self.subTest(mode=mode):
                settings = _colab_settings(tasks_mode=mode, internal_tasks_secret="s" * 32)
                self.assertEqual(colab_config_problems(settings, cloud_run), [])
        # Off Cloud Run (local development) the inline queue is fine.
        self.assertEqual(colab_config_problems(_colab_settings(tasks_mode="inline"), {}), [])

    def test_colab_tasks_need_a_queue_of_their_own(self) -> None:
        # Sharing the event queue, minutes-long Colab tasks hold its slots.
        shared = _colab_settings(cloud_tasks_queue="q1", cloud_tasks_colab_queue="q1")
        self.assertTrue(
            any("CLOUD_TASKS_COLAB_QUEUE must differ" in problem for problem in colab_config_problems(shared, {}))
        )
        # Empty is the default queue, as in the deploy script.
        self.assertEqual(_colab_settings(cloud_tasks_colab_queue=" ").cloud_tasks_colab_queue, "linebot-colab")

    def test_value_checks(self) -> None:
        for field, value in (
            ("colab_router_confidence", 1.0),
            ("colab_router_confidence", -0.1),
            ("colab_router_think_on", 0.49),
            ("colab_router_think_on", 1.0),
            ("colab_router_think_off", 0.0),
            ("colab_router_think_off", 0.51),
            ("colab_tunnel_host", "https://x.example.org"),
            ("colab_tunnel_host", "x.example.org:443"),
            ("colab_tunnel_host", "x.example.org/path"),
            ("colab_idle_ttl_seconds", 0),
            ("colab_daily_cu_cap", 0),
            ("colab_min_cu_balance", -1),
            ("colab_tunnel_mode", "ssh"),
            ("colab_boot_timeout_seconds", 0),
            ("colab_heartbeat_seconds", 59),
            ("colab_heartbeat_seconds", 901),
        ):
            with self.subTest(field=field, value=value), self.assertRaises(ValidationError):
                Settings(_env_file=None, **{field: value})
        self.assertEqual(Settings(_env_file=None, colab_tunnel_host=" Ollama.Example.org. ").colab_tunnel_host, "ollama.example.org")

    def test_router_gates_keep_their_ranges_and_on_above_off(self) -> None:
        edges = Settings(
            _env_file=None, colab_router_confidence=0.0, colab_router_think_on=0.99, colab_router_think_off=0.01
        )
        self.assertEqual(
            RouterGates.from_settings(edges), RouterGates(confidence=0.0, think_on=0.99, think_off=0.01)
        )
        Settings(_env_file=None, colab_router_confidence=0.99, colab_router_think_on=0.51, colab_router_think_off=0.5)
        # The confidence gate may sit under 0.5 (0 is argmax); THINK_ON may not.
        self.assertEqual(Settings(_env_file=None, colab_router_confidence=0.49).colab_router_confidence, 0.49)
        with self.assertRaisesRegex(ValidationError, "colab_router_think_on must be between 0.5 and 0.99"):
            Settings(_env_file=None, colab_router_think_on=0.49)
        with self.assertRaisesRegex(ValidationError, "COLAB_ROUTER_THINK_ON must be above COLAB_ROUTER_THINK_OFF"):
            Settings(_env_file=None, colab_router_think_on=0.5, colab_router_think_off=0.5)

    def test_env_example_documents_every_colab_setting_at_its_default(self) -> None:
        example_path = Path(__file__).resolve().parents[1] / ".env.example"
        if not example_path.is_file():
            self.skipTest(".env.example is not part of this checkout")
        example = parse_env_file(example_path)
        fields = [name for name in Settings.model_fields if name.startswith("colab_") or name == "line_admin_user_ids"]
        self.assertGreaterEqual(len(fields), 11)
        for name in fields:
            with self.subTest(field=name):
                self.assertIn(name.upper(), example)
                documented = Settings(_env_file=None, **{name: example[name.upper()]})
                self.assertEqual(getattr(documented, name), Settings.model_fields[name].default)

    def test_admin_ids_are_a_set(self) -> None:
        settings = Settings(_env_file=None, line_admin_user_ids=f" {_ADMIN}, {_OTHER}\n{_ADMIN} ")
        self.assertEqual(settings.admin_user_ids, frozenset({_ADMIN, _OTHER}))


class _Api:
    name = "nvidia"

    async def generate(self, model, messages, **kwargs):
        raise AssertionError("not called")

    async def close(self) -> None:
        return None


class CompositionTests(unittest.IsolatedAsyncioTestCase):
    async def test_colab_off_builds_nothing(self) -> None:
        settings = Settings(_env_file=None, openrouter_api_key="k")
        self.assertIsNone(build_colab_endpoint(settings))
        container = build_container(settings)
        self.assertIsNone(container.colab)
        self.assertIsNone(container.colab_registrar)
        self.assertNotIn("colab", container.providers)
        self.assertIsNone(container.orchestrator.system_one)
        self.assertTrue(all(row.spec.provider != "colab" for row in container.role_table.rows("chat") if row.enabled))
        await container.aclose()

    async def test_colab_on_registers_the_provider_router_and_registrar(self) -> None:
        settings = _colab_settings()
        endpoint = build_colab_endpoint(settings)
        self.assertIsInstance(endpoint, ColabEndpoint)
        self.assertIsInstance(endpoint.store, FirestoreColabStateStore)
        self.assertTrue(endpoint.auto_start)
        await endpoint.close()

        colab, _ = make_endpoint()
        container = build_container(settings, colab=colab)
        self.assertIs(container.colab, colab)
        self.assertIsInstance(container.colab_registrar, ColabRegistrar)
        self.assertIsInstance(container.providers["colab"], OllamaChatProvider)
        chat = container.role_table.rows("chat")
        self.assertEqual((chat[0].spec.provider, chat[0].enabled), ("colab", True))
        router = container.orchestrator
        self.assertIsInstance(router.system_one, SystemOneRouter)
        self.assertTrue(router.system_one_first)
        self.assertEqual((router.system_one.model, router.system_one.timeout_s), (ROUTER, 6))
        self.assertEqual(router.system_one.gates, RouterGates.from_settings(settings))
        # The Colab router answers on /v1/systemone, never through the JSON chain.
        self.assertNotIn("colab", {tier.provider_name for tier in router.targets})
        await container.aclose()

    async def test_api_preference_puts_the_colab_router_last_and_records_no_demand(self) -> None:
        settings = _colab_settings(model_path_preference="api")
        endpoint = build_colab_endpoint(settings)
        self.assertFalse(endpoint.auto_start)
        await endpoint.close()

        colab, _ = make_endpoint(auto_start=False)
        container = build_container(settings, colab=colab)
        self.assertFalse(container.orchestrator.system_one_first)
        self.assertEqual(container.role_table.rows("chat")[-1].spec.provider, "colab")
        await container.aclose()

    async def test_router_switch_off_disables_the_colab_router_tier(self) -> None:
        colab, _ = make_endpoint()
        container = build_container(_colab_settings(colab_router_enabled=False), colab=colab)

        rows = [row for row in container.role_table.rows("router") if row.spec.model == ROUTER]
        self.assertEqual([(row.enabled, row.disabled_reason) for row in rows], [(False, "COLAB_ROUTER_ENABLED=false")])
        self.assertIsNone(container.orchestrator.system_one)
        self.assertEqual(split_router_tiers(container.role_table)[1:], (None, False))
        await container.aclose()

    async def test_a_fake_colab_provider_never_gets_the_router_tier(self) -> None:
        settings = Settings(_env_file=None, openrouter_api_key="k")
        container = build_container(settings, extra_providers={"colab": _Api(), "nvidia": _Api()})
        tiers, system_one, first = split_router_tiers(container.role_table)
        self.assertEqual((system_one.model, first), (ROUTER, True))
        self.assertIsNone(container.orchestrator.system_one)
        self.assertNotIn(ROUTER, [tier.model for tier in container.orchestrator.targets])
        await container.aclose()

    async def test_unmet_requirements_leave_the_tier_off_with_the_reason(self) -> None:
        settings = _colab_settings(firestore_enabled=False)
        with self.assertLogs("linebot", "WARNING"):
            container = build_container(settings)

        self.assertIsNone(container.colab)
        self.assertIsNone(container.colab_registrar)
        self.assertNotIn("colab", container.providers)
        self.assertIn("FIRESTORE_ENABLED", container.colab_disabled_reason)
        self.assertTrue(any(w.startswith("Colab tier off: COLAB_ENABLED needs FIRESTORE") for w in container.config_warnings))
        chat = container.role_table.rows("chat")
        self.assertEqual((chat[0].spec.provider, chat[0].enabled), ("colab", False))
        self.assertTrue(container.role_table.has_api_tier("chat"))
        await container.aclose()

    async def test_without_the_cron_secret_the_tier_is_off_not_silently_idle(self) -> None:
        # /internal/cron refuses every call without it, so no tick would
        # ever start a session: say so instead of READY never coming.
        with self.assertLogs("linebot", "WARNING"):
            container = build_container(_colab_settings(internal_cron_secret=""))

        self.assertIsNone(container.colab)
        self.assertIn("INTERNAL_CRON_SECRET", container.colab_disabled_reason)
        self.assertTrue(any("Colab tier off" in w and "INTERNAL_CRON_SECRET" in w for w in container.config_warnings))
        await container.aclose()

    async def test_the_controller_shares_the_endpoint_store(self) -> None:
        async def submit(kind, payload):
            return None

        self.assertIsNone(build_colab_controller(_colab_settings(), None, submit=submit, bot_url=lambda: ""))
        colab, _ = make_endpoint()
        settings = _colab_settings(colab_dry_run=True, colab_account_dedicated=True, model_path_preference="api")

        controller = build_colab_controller(settings, colab, submit=submit, bot_url=lambda: "https://bot.example")

        self.assertIsInstance(controller, ColabController)
        self.assertIs(controller.store, colab.store)
        self.assertEqual(
            (controller.config.dry_run, controller.config.account_dedicated, controller.config.auto_start),
            (True, True, False),
        )
        self.assertIsInstance(controller._control_plane, ColabControlPlane)
        self.assertEqual(str(controller._control_plane._adc_path), str(_ADC))
        self.assertTrue(controller._launcher.settings.dry_run)
        await colab.close()

    async def test_the_container_closes_the_endpoint(self) -> None:
        colab, _ = make_endpoint()
        container = build_container(_colab_settings(), colab=colab)
        await container.aclose()
        self.assertTrue(colab._client.is_closed)


class MainWiringTests(unittest.TestCase):
    def test_the_pipeline_gets_the_endpoint_and_the_admin_ids(self) -> None:
        colab = SimpleNamespace(name="colab")
        container = SimpleNamespace(colab=colab, orchestrator=SimpleNamespace(), get_agent=lambda name: None)
        with ExitStack() as stack:
            stack.enter_context(patch.object(main_module, "container", container))
            stack.enter_context(patch.object(main_module, "get_settings", return_value=_colab_settings()))
            for name in ("get_line_service", "get_memory_service", "get_message_cache_service"):
                stack.enter_context(patch.object(main_module, name, return_value=SimpleNamespace()))
            deps = main_module._build_pipeline_deps()

        self.assertIs(deps.colab, colab)
        self.assertEqual(deps.admin_user_ids, frozenset({_ADMIN}))

        with ExitStack() as stack:
            stack.enter_context(patch.object(main_module, "container", None))
            stack.enter_context(patch.object(main_module, "get_settings", return_value=Settings(_env_file=None)))
            for name in ("get_line_service", "get_memory_service", "get_message_cache_service"):
                stack.enter_context(patch.object(main_module, name, return_value=SimpleNamespace()))
            deps = main_module._build_pipeline_deps()

        self.assertIsNone(deps.colab)
        self.assertEqual(deps.admin_user_ids, frozenset())


class HealthDetailTests(unittest.TestCase):
    def test_detail_shows_the_colab_section(self) -> None:
        settings = _colab_settings(internal_cron_secret="cron")
        health = ModelHealth()
        colab = ColabEndpoint(SECRET, InMemoryColabStateStore(ready_state(ccu_balance=42.0)))
        asyncio.run(colab.ensure_loaded())
        table = RoleTable.from_settings(settings, {"openrouter": SimpleNamespace(name="openrouter")})
        container = SimpleNamespace(
            colab=colab,
            colab_controller=None,
            colab_disabled_reason="",
            role_table=table,
            model_health=health,
            model_lifecycle=ModelLifecycle([], health),
            config_warnings=[],
            agents_stats=lambda: {},
            fallback_chain=SimpleNamespace(fallback_count=0),
            task_queue=None,
            search=None,
        )
        with ExitStack() as stack:
            stack.enter_context(patch.object(main_module, "container", container))
            stack.enter_context(patch.object(main_module, "get_settings", return_value=settings))
            stack.enter_context(patch.object(main_module, "get_memory_service", return_value=SimpleNamespace(get_stats=lambda: {})))
            stack.enter_context(patch("src.services.scheduler_service.peek_scheduler_service", return_value=None))
            stack.enter_context(
                patch("src.services.storage_service.get_storage_service", return_value=SimpleNamespace(get_usage_stats=lambda: {}))
            )
            response = TestClient(main_module.app).get("/health/detail", headers={"X-Cron-Token": "cron"})

        body = response.json()["colab"]
        self.assertEqual((body["enabled"], body["status"], body["ready"], body["ccu_balance"]), (True, READY, True, 42.0))
        self.assertNotIn(TUNNEL, response.text)

        container.colab_controller = SimpleNamespace(view=lambda: {"dry_run": True})
        body = self._detail(container, settings)["colab"]
        self.assertEqual(body["controller"], {"dry_run": True})

        off = SimpleNamespace(**{**vars(container), "colab": None, "colab_controller": None})
        off.colab_disabled_reason = "COLAB_ADC_PATH: /secrets/colab/adc.json cannot be read: FileNotFoundError"
        body = self._detail(off, settings)["colab"]
        self.assertEqual(body, {"enabled": False, "reason": off.colab_disabled_reason})
        asyncio.run(colab.close())

    def _detail(self, container, settings) -> dict:
        with ExitStack() as stack:
            stack.enter_context(patch.object(main_module, "container", container))
            stack.enter_context(patch.object(main_module, "get_settings", return_value=settings))
            stack.enter_context(patch.object(main_module, "get_memory_service", return_value=SimpleNamespace(get_stats=lambda: {})))
            stack.enter_context(patch("src.services.scheduler_service.peek_scheduler_service", return_value=None))
            stack.enter_context(
                patch("src.services.storage_service.get_storage_service", return_value=SimpleNamespace(get_usage_stats=lambda: {}))
            )
            return TestClient(main_module.app).get("/health/detail", headers={"X-Cron-Token": "cron"}).json()


if __name__ == "__main__":
    unittest.main()
