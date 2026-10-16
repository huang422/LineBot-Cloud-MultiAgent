"""The daily model end-of-life check (src/services/model_lifecycle.py).

Every request goes to an httpx.MockTransport; nothing reaches the network.
"""

from __future__ import annotations

import json
import unittest
from datetime import date
from types import SimpleNamespace
from unittest.mock import patch

import httpx

from src.config import Settings
from src.runtime.model_roles import (
    FLUX_DEV,
    FLUX_KLEIN,
    NIM_EMBED_1B,
    NIM_MUSE,
    NIM_OMNI,
    NIM_SUPER,
    NIM_ULTRA,
    OR_DOTS,
    OR_SUPER,
    OR_ULTRA,
    RoleTable,
)
from src.services import model_lifecycle as lifecycle_module
from src.services.model_health import ModelHealth
from src.services.model_lifecycle import (
    CATALOG,
    ENDPOINTS,
    PAGE,
    LifecycleTarget,
    ModelLifecycle,
    lifecycle_targets,
)

_GONE = (
    '{"type":"about:blank","title":"Gone","status":410,"detail":"The model '
    "'nvidia/nemotron-3-ultra-550b-a55b' has reached its end of life on "
    '2027-01-15T00:00:00Z and is no longer available."}'
)


def _models(*entries: tuple[str, str | None]) -> dict:
    """GET /api/v1/models: (id, expiration_date) per listed model."""
    return {"data": [{"id": model, "name": model, "expiration_date": expires} for model, expires in entries]}


_MODELS_PATH = "/api/v1/models"


def _endpoints(*providers: str, expiration: str | None = None, model_expiration: str | None = None) -> dict:
    return {
        "data": {
            "id": "x",
            "expiration_date": model_expiration,
            "endpoints": [
                {"provider_name": name, "status": 0, "expiration_date": expiration} for name in providers
            ],
        }
    }


class _Routes:
    """Answers per URL path; records every request."""

    def __init__(self, routes: dict[str, httpx.Response | Exception]) -> None:
        self.routes = routes
        self.requests: list[httpx.Request] = []

    def __call__(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        answer = self.routes.get(request.url.path, httpx.Response(200, json={}))
        if isinstance(answer, Exception):
            raise answer
        return answer


def _lifecycle(targets, routes: dict, health: ModelHealth | None = None) -> tuple[ModelLifecycle, _Routes, ModelHealth]:
    recorder = _Routes(routes)
    health = health or ModelHealth()
    return (
        ModelLifecycle(targets, health, transport=httpx.MockTransport(recorder), clock=lambda: 1_000.0),
        recorder,
        health,
    )


def _catalog(model: str) -> LifecycleTarget:
    return LifecycleTarget("nvidia", model, CATALOG)


async def _run_logged(lifecycle: ModelLifecycle, label: str) -> list[tuple[str, str]]:
    """One run; (level, message) of each line it logged about *label*."""
    levels = ("info", "warning", "error")
    with (
        patch.object(lifecycle_module.logger, "info") as info,
        patch.object(lifecycle_module.logger, "warning") as warning,
        patch.object(lifecycle_module.logger, "error") as error,
    ):
        await lifecycle.run()
    return [
        (level, call.args[0])
        for level, mock in zip(levels, (info, warning, error), strict=True)
        for call in mock.call_args_list
        if call.args[0].startswith(f"Model lifecycle: {label} ")
    ]


class TargetTests(unittest.TestCase):
    def test_every_checkable_id_of_the_role_table_once(self) -> None:
        table = RoleTable.from_settings(Settings(_env_file=None), {})

        targets = {target.label: target for target in lifecycle_targets(table)}

        self.assertEqual(
            set(targets),
            {
                f"nvidia:{NIM_SUPER}",
                f"nvidia:{NIM_ULTRA}",
                f"nvidia:{NIM_MUSE}",
                f"nvidia:{NIM_OMNI}",
                f"nvidia:{NIM_EMBED_1B}",
                f"nvidia:{FLUX_KLEIN}",
                f"nvidia:{FLUX_DEV}",
                f"openrouter:{OR_DOTS}",
                f"openrouter:{OR_ULTRA}",
                f"openrouter:{OR_SUPER}",
            },
        )
        self.assertEqual(targets[f"nvidia:{NIM_ULTRA}"].kind, CATALOG)
        self.assertEqual(targets[f"nvidia:{NIM_MUSE}"].kind, CATALOG)
        self.assertEqual(targets[f"nvidia:{NIM_OMNI}"].kind, CATALOG)
        self.assertEqual(targets[f"nvidia:{NIM_EMBED_1B}"].kind, CATALOG)
        # FLUX is not in /v1/models: its build.nvidia.com page is checked.
        self.assertEqual(
            (targets[f"nvidia:{FLUX_KLEIN}"].kind, targets[f"nvidia:{FLUX_KLEIN}"].page_slug),
            (PAGE, "black-forest-labs/flux_2-klein-4b"),
        )
        self.assertEqual(targets[f"openrouter:{OR_DOTS}"].kind, ENDPOINTS)
        self.assertEqual(targets[f"openrouter:{OR_ULTRA}"].kind, ENDPOINTS)

    def test_routers_colab_and_pageless_image_ids_are_not_checked(self) -> None:
        roles = {
            "chat": [{"provider": "openrouter", "model": "openrouter/free"}],
            "image": [{"provider": "nvidia", "model": "example/new-image-model"}],
        }
        table = RoleTable.from_settings(Settings(_env_file=None, model_roles=json.dumps(roles)), {})

        labels = {target.label for target in lifecycle_targets(table)}

        self.assertNotIn("openrouter:openrouter/free", labels)
        self.assertNotIn("nvidia:example/new-image-model", labels)
        self.assertFalse(any(label.startswith("colab:") for label in labels))


class NvidiaCatalogTests(unittest.IsolatedAsyncioTestCase):
    async def test_200_410_and_404(self) -> None:
        lifecycle, recorder, health = _lifecycle(
            [_catalog(NIM_SUPER), _catalog(NIM_ULTRA), _catalog("example/unlisted")],
            {
                f"/v1/models/{NIM_SUPER}": httpx.Response(200, json={"id": NIM_SUPER, "object": "model"}),
                f"/v1/models/{NIM_ULTRA}": httpx.Response(
                    410, text=_GONE, headers={"content-type": "application/problem+json"}
                ),
                "/v1/models/example/unlisted": httpx.Response(
                    404, text="Model does not exist by the name of example/unlisted"
                ),
            },
        )

        with patch.object(lifecycle_module.logger, "error"):
            results = await lifecycle.run()

        self.assertEqual(results[f"nvidia:{NIM_SUPER}"].state, "ok")
        retired = results[f"nvidia:{NIM_ULTRA}"]
        self.assertEqual((retired.state, retired.eol), ("retired", "2027-01-15T00:00:00Z"))
        self.assertEqual(results["nvidia:example/unlisted"].state, "not_in_catalog")
        # A retired model leaves rotation before live traffic hits the 410.
        self.assertTrue(health.is_dead("nvidia", NIM_ULTRA))
        self.assertEqual(health.get_status()["models"][f"nvidia:{NIM_ULTRA}"]["eol"], "2027-01-15T00:00:00Z")
        # A 404 from the catalog alone does not take a model out.
        self.assertFalse(health.is_dead("nvidia", "example/unlisted"))
        self.assertEqual(
            {str(request.url) for request in recorder.requests},
            {
                f"https://integrate.api.nvidia.com/v1/models/{NIM_SUPER}",
                f"https://integrate.api.nvidia.com/v1/models/{NIM_ULTRA}",
                "https://integrate.api.nvidia.com/v1/models/example/unlisted",
            },
        )

    async def test_a_listed_model_comes_back_from_a_dead_mark(self) -> None:
        health = ModelHealth()
        health.mark_dead("nvidia", NIM_SUPER, "HTTP 404, retry in 6h", retry_after=6 * 3600)
        lifecycle, _, _ = _lifecycle(
            [_catalog(NIM_SUPER)], {f"/v1/models/{NIM_SUPER}": httpx.Response(200, json={})}, health
        )

        await lifecycle.run()

        self.assertTrue(health.is_available("nvidia", NIM_SUPER))

    async def test_no_request_carries_an_api_key(self) -> None:
        lifecycle, recorder, _ = _lifecycle(
            [
                _catalog(NIM_SUPER),
                LifecycleTarget("nvidia", FLUX_DEV, PAGE, "black-forest-labs/flux_1-dev"),
                LifecycleTarget("openrouter", OR_DOTS, ENDPOINTS),
            ],
            {},
        )

        with patch.object(lifecycle_module.logger, "warning"):
            await lifecycle.run()

        # The OpenRouter id costs two: its endpoints and the model list.
        self.assertEqual(len(recorder.requests), 4)
        for request in recorder.requests:
            self.assertNotIn("authorization", {name.lower() for name in request.headers})


class BuildPageTests(unittest.IsolatedAsyncioTestCase):
    _SLUG = "black-forest-labs/flux_2-klein-4b"

    async def _run(self, response: httpx.Response):
        target = LifecycleTarget("nvidia", FLUX_KLEIN, PAGE, self._SLUG)
        lifecycle, recorder, health = _lifecycle([target], {f"/{self._SLUG}": response})
        with patch.object(lifecycle_module.logger, "error"):
            results = await lifecycle.run()
        self.assertEqual(str(recorder.requests[0].url), f"https://build.nvidia.com/{self._SLUG}")
        return results[f"nvidia:{FLUX_KLEIN}"], health

    async def test_a_clean_page_is_ok(self) -> None:
        result, _ = await self._run(
            httpx.Response(200, text='<script>self.__next_f.push([1,"{\\"showUnavailableBanner\\":false}"])</script>')
        )
        self.assertEqual(result.state, "ok")

    async def test_is_deprecated_even_in_escaped_page_json(self) -> None:
        result, health = await self._run(
            httpx.Response(200, text='<script>push("{\\"isDeprecated\\":true,\\"name\\":\\"flux\\"}")</script>')
        )
        self.assertEqual(result.state, "deprecated")
        self.assertIn("isDeprecated", result.detail)
        # A notice warns; the model keeps serving until it is gone.
        self.assertTrue(health.is_available("nvidia", FLUX_KLEIN))

    async def test_a_deprecation_notice_is_quoted(self) -> None:
        result, _ = await self._run(
            httpx.Response(
                200,
                text='{"isDeprecated":false,"deprecationNotice":"This API will be deprecated on 11/30/2026."}',
            )
        )
        self.assertEqual(result.state, "deprecated")
        self.assertIn("deprecated on 11/30/2026", result.detail)

    async def test_a_missing_page(self) -> None:
        result, _ = await self._run(httpx.Response(404, text="page not found"))
        self.assertEqual(result.state, "page_missing")


class OpenRouterEndpointTests(unittest.IsolatedAsyncioTestCase):
    async def _run(self, response: httpx.Response, models: httpx.Response | None = None):
        target = LifecycleTarget("openrouter", OR_DOTS, ENDPOINTS)
        lifecycle, recorder, _ = _lifecycle(
            [target],
            {
                f"/api/v1/models/{OR_DOTS}/endpoints": response,
                _MODELS_PATH: models or httpx.Response(200, json=_models((OR_DOTS, None))),
            },
        )
        with patch.object(lifecycle_module.logger, "error"):
            results = await lifecycle.run()
        self.assertEqual(
            sorted(str(request.url) for request in recorder.requests),
            ["https://openrouter.ai/api/v1/models", f"https://openrouter.ai/api/v1/models/{OR_DOTS}/endpoints"],
        )
        return results[f"openrouter:{OR_DOTS}"]

    async def test_live_endpoints_are_ok(self) -> None:
        result = await self._run(httpx.Response(200, json=_endpoints("ModelRun")))
        self.assertEqual((result.state, result.detail), ("ok", "1 endpoint(s): ModelRun"))

    async def test_no_endpoints_warns(self) -> None:
        result = await self._run(httpx.Response(200, json=_endpoints()))
        self.assertEqual(result.state, "no_endpoints")

    async def test_an_expiration_date_warns_with_the_earliest_date(self) -> None:
        for body in (
            _endpoints("ModelRun", expiration="2026-12-01"),
            _endpoints("ModelRun", model_expiration="2026-12-01"),
        ):
            with self.subTest(body=body):
                result = await self._run(httpx.Response(200, json=body))
                self.assertEqual((result.state, result.eol), ("expiring", "2026-12-01"))

    async def test_an_unknown_id(self) -> None:
        result = await self._run(httpx.Response(404, json={"error": {"message": "Model not found"}}))
        self.assertEqual(result.state, "not_in_catalog")

    async def test_the_model_lists_expiration_date_warns_ahead_of_time(self) -> None:
        # Live 2026-10-16: OpenRouter publishes dots-3's expiry only in
        # GET /api/v1/models; its endpoints answer has no expiration_date.
        result = await self._run(
            httpx.Response(200, json=_endpoints("AtlasCloud")),
            httpx.Response(200, json=_models(("other/model:free", "2026-10-31"), (OR_DOTS, "2026-12-31"))),
        )

        self.assertEqual((result.state, result.eol), ("expiring", "2026-12-31"))
        self.assertEqual(result.detail, "OpenRouter lists an expiration date: 2026-12-31")

    async def test_no_endpoints_wins_over_the_model_lists_expiration_date(self) -> None:
        # dots-3's list entry carries its date all along; losing every
        # endpoint (as qwen3.8 :free did on 2026-10-13) must still show.
        result = await self._run(
            httpx.Response(200, json=_endpoints()),
            httpx.Response(200, json=_models((OR_DOTS, "2026-12-31"))),
        )

        self.assertEqual((result.state, result.eol), ("no_endpoints", "2026-12-31"))
        self.assertEqual(result.detail, "OpenRouter lists no endpoint serving it")

    async def test_an_id_missing_from_the_model_list_is_judged_by_its_endpoints(self) -> None:
        result = await self._run(
            httpx.Response(200, json=_endpoints("AtlasCloud")),
            httpx.Response(200, json=_models(("other/model:free", "2026-10-31"))),
        )

        self.assertEqual((result.state, result.eol), ("ok", None))

    async def test_one_model_list_serves_every_openrouter_id_of_the_run(self) -> None:
        lifecycle, recorder, _ = _lifecycle(
            [
                LifecycleTarget("openrouter", OR_DOTS, ENDPOINTS),
                LifecycleTarget("openrouter", OR_ULTRA, ENDPOINTS),
                LifecycleTarget("openrouter", OR_SUPER, ENDPOINTS),
            ],
            {
                f"/api/v1/models/{OR_DOTS}/endpoints": httpx.Response(200, json=_endpoints("AtlasCloud")),
                f"/api/v1/models/{OR_ULTRA}/endpoints": httpx.Response(200, json=_endpoints("Nvidia")),
                f"/api/v1/models/{OR_SUPER}/endpoints": httpx.Response(200, json=_endpoints("Nvidia")),
                _MODELS_PATH: httpx.Response(200, json=_models((OR_DOTS, "2026-12-31"), (OR_ULTRA, None))),
            },
        )

        with patch.object(lifecycle_module.logger, "error"):
            results = await lifecycle.run()

        paths = [request.url.path for request in recorder.requests]
        self.assertEqual(paths.count(_MODELS_PATH), 1)
        self.assertEqual(len(paths), 4)
        self.assertEqual(results[f"openrouter:{OR_DOTS}"].state, "expiring")
        self.assertEqual(results[f"openrouter:{OR_ULTRA}"].state, "ok")
        self.assertEqual(results[f"openrouter:{OR_SUPER}"].state, "ok")

    async def test_no_openrouter_id_means_no_model_list_request(self) -> None:
        lifecycle, recorder, _ = _lifecycle([_catalog(NIM_SUPER)], {})

        await lifecycle.run()

        self.assertEqual([request.url.path for request in recorder.requests], [f"/v1/models/{NIM_SUPER}"])

    async def test_an_unreadable_model_list_keeps_the_last_state(self) -> None:
        # Without the list an expiry would read as ok: the check fails instead.
        routes: dict = {
            f"/api/v1/models/{OR_DOTS}/endpoints": httpx.Response(200, json=_endpoints("AtlasCloud")),
            _MODELS_PATH: httpx.Response(200, json=_models((OR_DOTS, "2026-12-31"))),
            f"/v1/models/{NIM_SUPER}": httpx.Response(200, json={}),
        }
        lifecycle, _, _ = _lifecycle(
            [LifecycleTarget("openrouter", OR_DOTS, ENDPOINTS), _catalog(NIM_SUPER)], routes
        )
        with patch.object(lifecycle_module.logger, "error"):
            await lifecycle.run()

        for failure, error in (
            (httpx.Response(503, text="busy"), "HTTP 503 from the OpenRouter model list"),
            (httpx.Response(200, json={"data": {}}), "the OpenRouter model list has no data list"),
            (httpx.Response(200, text="<html>"), "the OpenRouter model list is not a JSON object"),
            (httpx.ConnectError("no route"), "ConnectError: no route"),
        ):
            with self.subTest(error=error):
                routes[_MODELS_PATH] = failure
                with patch.object(lifecycle_module.logger, "warning"), patch.object(
                    lifecycle_module.logger, "error"
                ) as log_error:
                    results = await lifecycle.run()

                result = results[f"openrouter:{OR_DOTS}"]
                self.assertEqual((result.state, result.eol, result.error), ("expiring", "2026-12-31", error))
                log_error.assert_not_called()
                # The NVIDIA checks do not need the list.
                self.assertEqual(results[f"nvidia:{NIM_SUPER}"].state, "ok")
                self.assertIsNone(results[f"nvidia:{NIM_SUPER}"].error)

    async def test_an_unreadable_model_list_hides_nothing_the_endpoints_answer_shows(self) -> None:
        # Only a would-be ok needs the list; no endpoints, or a date in the
        # endpoints answer, is reported without it.
        for failure in (httpx.Response(503, text="busy"), httpx.ConnectError("no route")):
            for endpoints, expected in (
                (_endpoints(), ("no_endpoints", None)),
                (_endpoints("AtlasCloud", expiration="2026-12-01"), ("expiring", "2026-12-01")),
            ):
                with self.subTest(failure=failure, expected=expected):
                    lifecycle, _, _ = _lifecycle(
                        [LifecycleTarget("openrouter", OR_DOTS, ENDPOINTS)],
                        {
                            f"/api/v1/models/{OR_DOTS}/endpoints": httpx.Response(200, json=endpoints),
                            _MODELS_PATH: failure,
                        },
                    )
                    with patch.object(lifecycle_module.logger, "warning"), patch.object(
                        lifecycle_module.logger, "error"
                    ):
                        results = await lifecycle.run()

                    result = results[f"openrouter:{OR_DOTS}"]
                    self.assertEqual((result.state, result.eol), expected)
                    self.assertIsNone(result.error)


class StateTests(unittest.IsolatedAsyncioTestCase):
    async def test_state_changes_are_logged_at_error_and_steady_states_are_not(self) -> None:
        routes = {f"/v1/models/{NIM_SUPER}": httpx.Response(200, json={})}
        lifecycle, _, _ = _lifecycle([_catalog(NIM_SUPER)], routes)

        with patch.object(lifecycle_module.logger, "error") as log_error:
            await lifecycle.run()  # first check, ok: nothing to report
            await lifecycle.run()  # unchanged
            routes[f"/v1/models/{NIM_SUPER}"] = httpx.Response(410, text=_GONE)
            await lifecycle.run()  # ok -> retired
            await lifecycle.run()  # unchanged

        # (The logger is shared: ModelHealth reports the dead mark itself.)
        lifecycle_errors = [
            call.args[0] for call in log_error.call_args_list if call.args[0].startswith("Model lifecycle:")
        ]
        self.assertEqual(len(lifecycle_errors), 1)
        self.assertIn(f"nvidia:{NIM_SUPER} is retired (was ok)", lifecycle_errors[0])

    async def test_an_expiry_date_is_logged_at_info_and_never_alerts(self) -> None:
        # 2026-10-16: dots-3, a backup listed to 2026-12-31, logged an ERROR
        # on every new instance; the operator wants no reminder at all. The
        # clock reads 1970: no date, near, past or unreadable, is louder.
        for expires in ("2026-12-31", "1970-01-01", "someday"):
            with self.subTest(expires=expires):
                lifecycle, _, _ = _lifecycle(
                    [LifecycleTarget("openrouter", OR_DOTS, ENDPOINTS)],
                    {
                        f"/api/v1/models/{OR_DOTS}/endpoints": httpx.Response(200, json=_endpoints("AtlasCloud")),
                        _MODELS_PATH: httpx.Response(200, json=_models((OR_DOTS, expires))),
                    },
                )

                logged = await _run_logged(lifecycle, f"openrouter:{OR_DOTS}")  # newly seen
                logged += await _run_logged(lifecycle, f"openrouter:{OR_DOTS}")  # unchanged

                detail = f"OpenRouter lists an expiration date: {expires}"
                self.assertEqual(
                    logged,
                    [("info", f"Model lifecycle: openrouter:{OR_DOTS} is expiring (was unchecked): {detail}")],
                )
                # Still on /health/detail.
                self.assertEqual(lifecycle.view()["results"][f"openrouter:{OR_DOTS}"]["eol"], expires)

    async def test_a_date_that_comes_and_goes_is_logged_at_info(self) -> None:
        routes = {
            f"/api/v1/models/{OR_DOTS}/endpoints": httpx.Response(200, json=_endpoints("AtlasCloud")),
            _MODELS_PATH: httpx.Response(200, json=_models((OR_DOTS, None))),
        }
        lifecycle, _, _ = _lifecycle([LifecycleTarget("openrouter", OR_DOTS, ENDPOINTS)], routes)

        logged = await _run_logged(lifecycle, f"openrouter:{OR_DOTS}")  # first check, ok: nothing
        routes[_MODELS_PATH] = httpx.Response(200, json=_models((OR_DOTS, "2026-12-31")))
        logged += await _run_logged(lifecycle, f"openrouter:{OR_DOTS}")  # ok -> expiring
        routes[_MODELS_PATH] = httpx.Response(200, json=_models((OR_DOTS, None)))
        logged += await _run_logged(lifecycle, f"openrouter:{OR_DOTS}")  # expiring -> ok

        self.assertEqual([level for level, _ in logged], ["info", "info"])
        self.assertIn("is expiring (was ok)", logged[0][1])
        self.assertIn("is ok (was expiring)", logged[1][1])

    async def test_an_expiring_model_that_is_gone_is_logged_at_error(self) -> None:
        endpoints_path = f"/api/v1/models/{OR_DOTS}/endpoints"
        for gone, state in (
            (httpx.Response(200, json=_endpoints()), "no_endpoints"),
            (httpx.Response(404, text="not found"), "not_in_catalog"),
        ):
            with self.subTest(state=state):
                routes = {
                    endpoints_path: httpx.Response(200, json=_endpoints("AtlasCloud")),
                    _MODELS_PATH: httpx.Response(200, json=_models((OR_DOTS, "2026-12-31"))),
                }
                lifecycle, _, _ = _lifecycle([LifecycleTarget("openrouter", OR_DOTS, ENDPOINTS)], routes)

                logged = await _run_logged(lifecycle, f"openrouter:{OR_DOTS}")  # expiring
                routes[endpoints_path] = gone
                logged += await _run_logged(lifecycle, f"openrouter:{OR_DOTS}")  # gone

                self.assertEqual(lifecycle.results()[f"openrouter:{OR_DOTS}"]["state"], state)
                self.assertEqual([level for level, _ in logged], ["info", "error"])
                self.assertIn(f"openrouter:{OR_DOTS} is {state} (was expiring)", logged[1][1])

    async def test_a_failed_check_keeps_the_last_state_and_records_the_error(self) -> None:
        routes: dict = {f"/v1/models/{NIM_SUPER}": httpx.Response(410, text=_GONE)}
        lifecycle, _, _ = _lifecycle([_catalog(NIM_SUPER)], routes)
        with patch.object(lifecycle_module.logger, "error"):
            await lifecycle.run()

        for failure in (httpx.Response(503, text="busy"), httpx.ConnectError("no route")):
            with self.subTest(failure=failure):
                routes[f"/v1/models/{NIM_SUPER}"] = failure
                with patch.object(lifecycle_module.logger, "warning"), patch.object(
                    lifecycle_module.logger, "error"
                ) as log_error:
                    results = await lifecycle.run()

                result = results[f"nvidia:{NIM_SUPER}"]
                self.assertEqual(result.state, "retired")
                self.assertTrue(result.error)
                log_error.assert_not_called()

    async def test_a_first_check_that_fails_is_unknown(self) -> None:
        lifecycle, _, _ = _lifecycle(
            [_catalog(NIM_SUPER)], {f"/v1/models/{NIM_SUPER}": httpx.Response(500)}
        )

        with patch.object(lifecycle_module.logger, "warning"):
            results = await lifecycle.run()

        result = results[f"nvidia:{NIM_SUPER}"]
        self.assertEqual((result.state, result.error), ("unknown", "HTTP 500 from the NVIDIA model catalog"))

    async def test_view_for_health_detail(self) -> None:
        lifecycle, _, _ = _lifecycle([_catalog(NIM_SUPER)], {})
        self.assertTrue(lifecycle.claim_day(date(2026, 9, 30)))
        await lifecycle.run()

        view = lifecycle.view()

        self.assertEqual(view["last_claimed_day"], "2026-09-30")
        self.assertEqual(view["last_run_at"], 1_000.0)
        self.assertEqual(view["results"][f"nvidia:{NIM_SUPER}"]["state"], "ok")
        self.assertEqual(lifecycle.results(), view["results"])


class ClaimDayTests(unittest.TestCase):
    def test_one_claim_per_day(self) -> None:
        lifecycle = ModelLifecycle([], SimpleNamespace())

        self.assertTrue(lifecycle.claim_day(date(2026, 9, 30)))
        self.assertFalse(lifecycle.claim_day(date(2026, 9, 30)))
        self.assertTrue(lifecycle.claim_day(date(2026, 10, 1)))


if __name__ == "__main__":
    unittest.main()
