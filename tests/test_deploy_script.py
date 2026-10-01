"""scripts/deploy_cloud_run.sh v2 against the fake gcloud (tests/fake_gcloud.py).

Provisioned (scripts/provision_gcp.sh ran): Cloud Build runs as
linebot-builder with a us-west1 staging bucket, and Cloud Run runs as
linebot-runtime reading .env from the linebot-dotenv secret. Not provisioned:
the pre-provisioning path, with warnings. The smoke checks run against a
local stub of the service; nothing reaches Google or LINE. Every deploy also
checks that no gcloud call whose output or flags carry a secret (the
env-file deploy's --env-vars-file included) ran with gcloud's file logging
on.
"""

from __future__ import annotations

import base64
import hashlib
import hmac
import json
import os
import shutil
import subprocess
import sys
import threading
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from scripts import envfile
from tests.script_rehearsal import RehearsedScripts, start_rehearsals, stop_rehearsals

_ROOT = Path(__file__).resolve().parents[1]
_SCRIPT = _ROOT / "scripts" / "deploy_cloud_run.sh"
_FAKE = Path(__file__).resolve().with_name("fake_gcloud.py")

_PROJECT = "test-project"
_RUNTIME_SA = f"linebot-runtime@{_PROJECT}.iam.gserviceaccount.com"
_BUILDER_SA = f"linebot-builder@{_PROJECT}.iam.gserviceaccount.com"
_STAGING = f"gs://{_PROJECT}-cb-src-usw1"
_LINE_SECRET = "line-secret-do-not-print"
_TAG = "abc1234-20260930"

_BASE_ENV = (
    f"GCP_PROJECT_ID={_PROJECT}",
    f"LINE_CHANNEL_SECRET={_LINE_SECRET}",
    "LINE_CHANNEL_ACCESS_TOKEN=line-token-do-not-print",
    "OPENROUTER_API_KEY=sk-or-do-not-print",
    "TASKS_MODE=inline",
)


# `curl -sS -o /dev/null -w '%{{http_code}}' URL`: prints the status code.
_FAKE_CURL = """#!/bin/sh
exec "{python}" -c '
import sys, urllib.error, urllib.request
try:
    with urllib.request.urlopen(sys.argv[-1], timeout=5) as response:
        code = response.status
except urllib.error.HTTPError as exc:
    code = exc.code
except OSError:
    code = 0
print(code, end="")
' "$@"
"""


class _StubService(BaseHTTPRequestHandler):
    """/health and a signature-checking /webhook, like the real service."""

    ready = True

    def _reply(self, status: int, payload: dict | None = None) -> None:
        body = json.dumps(payload or {}).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:  # noqa: N802 - http.server API
        if self.path == "/health":
            self._reply(
                200, {"status": "healthy", "ready_for_webhook": self.ready, "version": _TAG}
            )
        else:
            self._reply(404)

    def do_POST(self) -> None:  # noqa: N802 - http.server API
        body = self.rfile.read(int(self.headers.get("Content-Length", "0")))
        expected = base64.b64encode(
            hmac.new(_LINE_SECRET.encode(), body, hashlib.sha256).digest()
        ).decode()
        self._reply(200 if self.headers.get("X-Line-Signature") == expected else 403)

    def log_message(self, *args) -> None:
        pass


def provisioned(**overrides) -> dict:
    state = {
        "service_accounts": [_RUNTIME_SA, _BUILDER_SA],
        "buckets": {_STAGING: {"location": "US-WEST1", "default_storage_class": "STANDARD"}},
        "secrets": {"linebot-dotenv": []},
        "repos": {"linebot-cloud-agent": {}},
    }
    state.update(overrides)
    return state


def version(number: int, data: str, state: str = "ENABLED") -> dict:
    return {"id": number, "state": state, "data": base64.b64encode(data.encode()).decode()}


def dotenv_mount(pinned: str) -> list[dict]:
    """A revision's volumes when it mounts linebot-dotenv version *pinned*."""
    item = {"key": pinned, "path": "app.env"}
    return [{"name": "secret-1", "secret": {"secretName": "linebot-dotenv", "items": [item]}}]


def revision(
    name: str,
    digest: str,
    pinned: str | None = None,
    *,
    service: str | None = None,
    region: str | None = None,
    ready: bool = True,
) -> dict:
    """A revision of *service* in *region* (without them: of whichever
    service and region lists it); ``ready=False`` for one that never became
    Ready (a failed deploy)."""
    found = {
        "name": name,
        "image": f"us-west1-docker.pkg.dev/{_PROJECT}/linebot-cloud-agent/app@{digest}",
        "volumes": dotenv_mount(pinned) if pinned else [],
        "ready": ready,
    }
    if service is not None:
        found["service"] = service
    if region is not None:
        found["region"] = region
    return found


# infra/gcp/ar-cleanup.json as the Artifact Registry API reports it.
_AR_POLICIES = {
    "keep-2-most-recent": {"action": {"type": "Keep"}, "mostRecentVersions": {"keepCount": 2}},
    "delete-older-than-1d": {
        "action": {"type": "Delete"},
        "condition": {"tagState": "any", "olderThan": "1d"},
    },
}


# The stub service, ready for webhooks or not, by readiness.
_SERVICES: dict[bool, ThreadingHTTPServer] = {}


def setUpModule() -> None:
    # The services are up before the rehearsals deploy to them.
    for ready in (True, False):
        handler = type("Stub", (_StubService,), {"ready": ready})
        _SERVICES[ready] = ThreadingHTTPServer(("127.0.0.1", 0), handler)
        threading.Thread(target=_SERVICES[ready].serve_forever, daemon=True).start()
    start_rehearsals(sys.modules[__name__])


def tearDownModule() -> None:
    stop_rehearsals()
    for server in _SERVICES.values():
        server.shutdown()
        server.server_close()
    _SERVICES.clear()


class DeployHarness(RehearsedScripts, unittest.TestCase):
    ready = True
    script = _SCRIPT

    def setUp(self) -> None:
        super().setUp()
        self.service_url = f"http://127.0.0.1:{_SERVICES[self.ready].server_address[1]}"
        self.tmp = self.work_dir()
        bin_dir = self.tmp / "bin"
        bin_dir.mkdir(exist_ok=True)
        gcloud = bin_dir / "gcloud"
        # -S: the fake is standard library only, and every call starts a python.
        gcloud.write_text(f'#!/bin/sh\nexec "{sys.executable}" -S "{_FAKE}" "$@"\n', encoding="utf-8")
        gcloud.chmod(0o755)
        # A wrapper, not a symlink: a venv's python only finds its packages
        # when started by its own path.
        python3 = bin_dir / "python3"
        python3.write_text(f'#!/bin/sh\nexec "{sys.executable}" "$@"\n', encoding="utf-8")
        python3.chmod(0o755)
        # The health wait's curl, without depending on one being installed
        # (python:3.12-slim, where Cloud Build runs these tests, has none).
        curl = bin_dir / "curl"
        curl.write_text(_FAKE_CURL.format(python=sys.executable), encoding="utf-8")
        curl.chmod(0o755)
        self.state_path = self.tmp / "state.json"
        self.log_path = self.tmp / "gcloud.log"
        self.env_file = self.tmp / "deploy.env"

        environ = {
            key: value
            for key, value in self.base_environ.items()
            if key.lower() not in {"http_proxy", "https_proxy", "all_proxy"}
            # The script must turn gcloud's file logging and prompts off itself.
            and key not in {"CLOUDSDK_CORE_DISABLE_FILE_LOGGING", "CLOUDSDK_CORE_DISABLE_PROMPTS"}
        }
        environ.update(
            {
                "PATH": f"{bin_dir}{os.pathsep}{self.base_environ.get('PATH', '')}",
                "FAKE_GCLOUD_STATE": str(self.state_path),
                "FAKE_GCLOUD_LOG": str(self.log_path),
                "TMPDIR": str(self.tmp),
                "NO_PROXY": "127.0.0.1,localhost",
            }
        )
        self.environ = environ

    def deploy(self, state: dict, env_lines=(), *args: str) -> subprocess.CompletedProcess:
        self.env_file.write_text("\n".join([*_BASE_ENV, *env_lines]) + "\n", encoding="utf-8")
        self.state_path.write_text(json.dumps({"service_url": self.service_url, **state}), encoding="utf-8")
        self.log_path.unlink(missing_ok=True)
        result = self.run_script(
            ["bash", str(self.script), "--env-file", str(self.env_file), "--image-tag", _TAG, *args],
            timeout=180,
        )
        # gcloud copies what it prints, and every parsed flag, into
        # ~/.config/gcloud/logs: a secret payload, a job's X-Cron-Token or
        # the env-vars file's values must never pass through it with it on.
        self.assertEqual(self.state.get("file_logged", []), [], result.stderr)
        return result

    @property
    def state(self) -> dict:
        return json.loads(self.state_path.read_text(encoding="utf-8"))

    def calls(self, prefix: str) -> list[str]:
        lines = self.log_path.read_text(encoding="utf-8").splitlines() if self.log_path.exists() else []
        return [line for line in lines if line.startswith(prefix)]

    def blob(self) -> str:
        return envfile.to_dotenv(envfile.parse_env_file(self.env_file))

    def assert_no_secret_printed(self, result: subprocess.CompletedProcess) -> None:
        for value in (_LINE_SECRET, "line-token-do-not-print", "sk-or-do-not-print"):
            self.assertNotIn(value, result.stdout + result.stderr)


class SecretsModeTests(DeployHarness):
    def test_a_provisioned_project_builds_as_builder_and_deploys_from_the_secret(self) -> None:
        state = provisioned(secrets={"linebot-dotenv": [version(1, "OLD='x'\n")]})

        result = self.deploy(state)

        self.assertEqual(result.returncode, 0, result.stderr)
        [build] = self.calls("builds submit")
        self.assertIn(f"--service-account=projects/{_PROJECT}/serviceAccounts/{_BUILDER_SA}", build)
        self.assertIn(f"--gcs-source-staging-dir={_STAGING}/source", build)
        # .env changed since version 1, so version 2 holds exactly the new blob.
        versions = self.state["secrets"]["linebot-dotenv"]
        self.assertEqual([v["id"] for v in versions], [1, 2])
        self.assertEqual(base64.b64decode(versions[1]["data"]).decode(), self.blob())
        [deploy] = self.calls("run deploy")
        self.assertIn(f"--service-account {_RUNTIME_SA}", deploy)
        self.assertIn("--set-secrets=/secrets/app/app.env=linebot-dotenv:2", deploy)
        self.assertIn(f"--set-env-vars=APP_VERSION={_TAG}", deploy)
        for flag in ("--env-vars-file", "--clear-env-vars", "--update-env-vars", "--clear-secrets"):
            self.assertNotIn(flag, deploy)
        self.assertIn("linebot-dotenv version 2 at /secrets/app/app.env", result.stdout)
        # The tokenCreator grant moved to provisioning.
        self.assertFalse(
            [c for c in self.calls("iam service-accounts add-iam-policy-binding") if "tokenCreator" in c]
        )
        self.assertNotIn("provision_gcp.sh first", result.stderr)
        self.assert_no_secret_printed(result)

    def test_an_unchanged_env_reuses_the_latest_version(self) -> None:
        self.env_file.write_text("\n".join(_BASE_ENV) + "\n", encoding="utf-8")
        state = provisioned(secrets={"linebot-dotenv": [version(1, "A='1'\n"), version(2, self.blob())]})

        result = self.deploy(state)

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(self.calls("secrets versions add"), [])
        self.assertIn("Unchanged since the latest version", result.stdout)
        [deploy] = self.calls("run deploy")
        self.assertIn("--set-secrets=/secrets/app/app.env=linebot-dotenv:2", deploy)

    def test_old_versions_are_destroyed_after_the_smoke_check_passes(self) -> None:
        state = provisioned(
            secrets={
                "linebot-dotenv": [
                    version(1, "A='1'\n", "DESTROYED"),
                    version(2, "A='2'\n", "DISABLED"),  # billed until destroyed
                    version(3, "A='3'\n"),
                    version(4, "A='4'\n"),
                ]
            }
        )

        result = self.deploy(state)

        self.assertEqual(result.returncode, 0, result.stderr)
        # Version 5 is deployed; 4 stays for rolling back to the previous revision.
        states = {v["id"]: v["state"] for v in self.state["secrets"]["linebot-dotenv"]}
        self.assertEqual(
            states, {1: "DESTROYED", 2: "DESTROYED", 3: "DESTROYED", 4: "ENABLED", 5: "ENABLED"}
        )
        self.assertIn("[ok] /webhook accepts a valid signed request", result.stdout)

    def test_keeping_more_revisions_keeps_their_secret_versions(self) -> None:
        state = provisioned(
            secrets={"linebot-dotenv": [version(n, f"A='{n}'\n") for n in (1, 2, 3)]}
        )

        result = self.deploy(state, ["DEPLOY_KEEP_REVISIONS=3"])

        self.assertEqual(result.returncode, 0, result.stderr)
        states = {v["id"]: v["state"] for v in self.state["secrets"]["linebot-dotenv"]}
        self.assertEqual(states, {1: "DESTROYED", 2: "ENABLED", 3: "ENABLED", 4: "ENABLED"})

    def test_skipping_the_smoke_check_keeps_every_version(self) -> None:
        state = provisioned(secrets={"linebot-dotenv": [version(n, f"A='{n}'\n") for n in (1, 2, 3)]})

        result = self.deploy(state, (), "--skip-smoke")

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(self.calls("secrets versions destroy"), [])
        self.assertIn("smoke checks were skipped", result.stdout)
        # Versions pile up while smoke checks are skipped; say so.
        self.assertIn("linebot-dotenv has 4 active versions", result.stderr)

    def test_a_version_a_kept_revision_pins_is_never_destroyed(self) -> None:
        # Version 6 came from a deploy that failed before making a revision,
        # then .env changed. Counting alone would keep 7 and 6 and destroy
        # 5, which the previous revision (the rollback target) still pins.
        state = provisioned(
            secrets={"linebot-dotenv": [version(n, f"A='{n}'\n") for n in (4, 5, 6)]},
            revisions=[revision("linebot-cloud-agent-previous", "sha256:prev", pinned="5")],
            images=["sha256:prev"],
        )

        result = self.deploy(state)

        self.assertEqual(result.returncode, 0, result.stderr)
        states = {v["id"]: v["state"] for v in self.state["secrets"]["linebot-dotenv"]}
        self.assertEqual(states, {4: "DESTROYED", 5: "ENABLED", 6: "ENABLED", 7: "ENABLED"})
        self.assertIn("Keeping linebot-dotenv version 5 (a kept revision uses it)", result.stdout)

    def test_a_revision_that_never_became_ready_takes_no_keep_slot(self) -> None:
        # Version 2 went to a deploy whose revision never became Ready; the
        # good revision before it is still the only one to roll back to.
        state = provisioned(
            secrets={"linebot-dotenv": [version(1, "A='1'\n"), version(2, "A='2'\n")]},
            revisions=[
                revision("linebot-cloud-agent-good", "sha256:good", pinned="1"),
                revision("linebot-cloud-agent-failed", "sha256:failed", pinned="2", ready=False),
            ],
            images=["sha256:failed", "sha256:good"],
        )

        fixed = self.deploy(state)

        self.assertEqual(fixed.returncode, 0, fixed.stderr)
        names = [r["name"] for r in self.state["revisions"]]
        self.assertIn("linebot-cloud-agent-good", names)
        self.assertIn("sha256:good", self.state["images"])
        states = {v["id"]: v["state"] for v in self.state["secrets"]["linebot-dotenv"]}
        self.assertEqual(states, {1: "ENABLED", 2: "ENABLED", 3: "ENABLED"})

        # Once two Ready revisions are newer, both old ones go.
        after = self.deploy(self.state, ["BOT_NAME=Next"])

        self.assertEqual(after.returncode, 0, after.stderr)
        names = [r["name"] for r in self.state["revisions"]]
        self.assertEqual(names, ["linebot-cloud-agent-deploy1", "linebot-cloud-agent-deploy2"])
        self.assertNotIn("sha256:good", self.state["images"])
        states = {v["id"]: v["state"] for v in self.state["secrets"]["linebot-dotenv"]}
        self.assertEqual(
            states, {1: "DESTROYED", 2: "DESTROYED", 3: "ENABLED", 4: "ENABLED"}
        )

    def test_what_another_service_uses_is_never_cleaned_up(self) -> None:
        # linebot-dotenv and the image repository are one per project: a
        # staging service's deploy must leave prod's versions and images.
        prod = "linebot-cloud-agent"
        state = provisioned(
            secrets={"linebot-dotenv": [version(n, f"A='{n}'\n") for n in (1, 2, 3)]},
            revisions=[
                revision("prod-previous", "sha256:prod-previous", pinned="1", service=prod),
                revision("prod-live", "sha256:prod-live", pinned="2", service=prod),
                revision("staging-previous", "sha256:staging", pinned="3", service="linebot-staging"),
            ],
            images=["sha256:staging", "sha256:prod-live", "sha256:prod-previous"],
        )

        result = self.deploy(state, ["BOT_NAME=Staging"], "--service-name", "linebot-staging")

        self.assertEqual(result.returncode, 0, result.stderr)
        states = {v["id"]: v["state"] for v in self.state["secrets"]["linebot-dotenv"]}
        self.assertEqual(states, {1: "ENABLED", 2: "ENABLED", 3: "ENABLED", 4: "ENABLED"})
        names = [r["name"] for r in self.state["revisions"]]
        self.assertEqual(names[:2], ["prod-previous", "prod-live"])
        for digest in ("sha256:prod-live", "sha256:prod-previous"):
            self.assertIn(digest, self.state["images"])

    def test_what_a_service_in_another_region_pins_is_never_destroyed(self) -> None:
        # linebot-dotenv is one per project, and revisions are listed per
        # region: every region with a service is read.
        state = provisioned(
            secrets={"linebot-dotenv": [version(n, f"A='{n}'\n") for n in (1, 2, 3)]},
            run_services=[
                {"name": "linebot-cloud-agent", "region": "us-west1"},
                {"name": "linebot-eu", "region": "europe-west1"},
            ],
            revisions=[
                revision("eu-live", "sha256:eu", pinned="1", service="linebot-eu", region="europe-west1")
            ],
        )

        result = self.deploy(state)

        self.assertEqual(result.returncode, 0, result.stderr)
        states = {v["id"]: v["state"] for v in self.state["secrets"]["linebot-dotenv"]}
        # 4 is deployed, 3 is within the newest two, 1 is europe-west1's.
        self.assertEqual(states, {1: "ENABLED", 2: "DESTROYED", 3: "ENABLED", 4: "ENABLED"})
        self.assertIn("Keeping linebot-dotenv version 1 (a kept revision uses it)", result.stdout)

    def test_an_unlistable_service_region_destroys_no_secret_version(self) -> None:
        state = provisioned(
            secrets={"linebot-dotenv": [version(n, f"A='{n}'\n") for n in (1, 2, 3)]},
            fail={"run services list": "UNAVAILABLE: try again"},
        )

        result = self.deploy(state)

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("Could not read which linebot-dotenv versions the revisions use", result.stderr)
        self.assertEqual(self.calls("secrets versions destroy"), [])

    def test_unreadable_revisions_destroy_no_secret_version(self) -> None:
        state = provisioned(
            secrets={"linebot-dotenv": [version(n, f"A='{n}'\n") for n in (1, 2, 3)]},
            fail={"run revisions list": "UNAVAILABLE: try again"},
        )

        result = self.deploy(state)

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("Could not read which linebot-dotenv versions the revisions use", result.stderr)
        self.assertEqual(self.calls("secrets versions destroy"), [])

    def test_the_payload_read_and_cron_job_lookups_skip_the_gcloud_log(self) -> None:
        # The harness fails any deploy that reads a payload or a job with
        # file logging on; these deploys make every such call.
        cron = "INTERNAL_CRON_SECRET=" + "c0" * 32
        state = provisioned(secrets={"linebot-dotenv": [version(1, "A='1'\n")]})

        created = self.deploy(state, [cron])
        self.assertEqual(created.returncode, 0, created.stderr)
        self.assertTrue(self.calls("secrets versions access"))
        self.assertTrue(self.calls("scheduler jobs create"))
        jobs = self.state["scheduler_jobs"]

        updated = self.deploy({**self.state, "scheduler_jobs": jobs}, [cron])
        self.assertEqual(updated.returncode, 0, updated.stderr)
        self.assertTrue(self.calls("scheduler jobs update"))

        removed = self.deploy({**self.state, "scheduler_jobs": jobs})
        self.assertEqual(removed.returncode, 0, removed.stderr)
        self.assertTrue(self.calls("scheduler jobs delete"))
        self.assertEqual(self.state["scheduler_jobs"], {})
        for result in (created, updated, removed):
            self.assertNotIn("c0" * 32, result.stdout + result.stderr)
        # The job's token is this deploy's, whichever revision serves: the
        # rollback advice says so while the job exists.
        hint = "X-Cron-Token stays this deploy's INTERNAL_CRON_SECRET"
        self.assertIn(hint, created.stdout)
        self.assertIn(hint, updated.stdout)
        self.assertNotIn(hint, removed.stdout)

    def test_an_unreadable_env_value_stops_the_deploy_before_the_build(self) -> None:
        result = self.deploy(provisioned(), ["BROKEN='do-not-print", "LATER=still read"])

        self.assertEqual(result.returncode, 1)
        self.assertIn("cannot parse the quoted value of: BROKEN", result.stdout)
        self.assertNotIn("do-not-print", result.stdout + result.stderr)
        self.assertEqual(self.calls("builds submit"), [])

    def test_an_explicit_runtime_service_account_is_used(self) -> None:
        custom = f"custom-runtime@{_PROJECT}.iam.gserviceaccount.com"
        state = provisioned(service_accounts=[custom, _BUILDER_SA])

        result = self.deploy(state, [f"CLOUD_RUN_SERVICE_ACCOUNT={custom}"])

        self.assertEqual(result.returncode, 0, result.stderr)
        [deploy] = self.calls("run deploy")
        self.assertIn(f"--service-account {custom}", deploy)
        self.assertIn("--set-secrets=", deploy)

    def test_values_python_dotenv_would_expand_stop_the_deploy_before_the_build(self) -> None:
        result = self.deploy(provisioned(), ["BOT_NAME=hi-${USER}-do-not-print"])

        self.assertEqual(result.returncode, 1)
        self.assertIn("BOT_NAME contains '${'", result.stderr)
        self.assertNotIn("do-not-print", result.stdout + result.stderr)
        self.assertEqual(self.calls("builds submit"), [])


class LookupFailureTests(DeployHarness):
    """Only NOT_FOUND means "not provisioned"; any other error stops the deploy."""

    def test_a_lookup_error_never_falls_back_to_plain_variables(self) -> None:
        secret = {"linebot-dotenv": [version(1, "A='1'\n")]}
        for prefix, secrets in (
            ("iam service-accounts describe", secret),
            ("secrets describe", secret),
            ("storage buckets list", secret),
            # The secret is really gone, but whether the service reads it
            # cannot be told.
            ("run services describe", {}),
        ):
            with self.subTest(prefix=prefix):
                state = provisioned(
                    secrets=secrets,
                    fail={prefix: "PERMISSION_DENIED: Permission denied (or it may not exist)."},
                )

                result = self.deploy(state)

                self.assertEqual(result.returncode, 1)
                self.assertIn("PERMISSION_DENIED", result.stderr)
                self.assertIn("not with NOT_FOUND", result.stderr)
                self.assertNotIn("does not exist", result.stderr)
                self.assertNotIn("so the deploy puts every .env value", result.stderr)
                self.assertEqual(self.calls("builds submit"), [])
                self.assertEqual(self.calls("run deploy"), [])
                self.assertNotIn("env_vars_file", self.state)

    def test_a_disabled_secret_manager_stops_the_deploy_without_a_hidden_prompt(self) -> None:
        # The lookup's stderr goes to a file: gcloud's "enable and retry?
        # (y/N)" would wait there unseen.
        state = provisioned(disabled_apis=["secretmanager.googleapis.com"])

        result = self.deploy(state)

        self.assertEqual(result.returncode, 1)
        self.assertIn("reason: SERVICE_DISABLED", result.stderr)
        self.assertIn("not with NOT_FOUND", result.stderr)
        self.assertEqual(self.state.get("enable_api_prompts", []), [])
        self.assertEqual(self.calls("builds submit"), [])

    def test_a_service_reading_the_secret_is_not_moved_to_plain_variables(self) -> None:
        # The secret is gone (NOT_FOUND) while the service still mounts it.
        state = provisioned(secrets={}, service_volumes=dotenv_mount("3"))

        refused = self.deploy(state)

        self.assertEqual(refused.returncode, 1)
        self.assertIn("reads its settings from secret linebot-dotenv", refused.stderr)
        self.assertIn("--allow-env-fallback", refused.stderr)
        self.assertEqual(self.calls("builds submit"), [])
        self.assertEqual(self.calls("run deploy"), [])

        allowed = self.deploy(state, (), "--allow-env-fallback")

        self.assertEqual(allowed.returncode, 0, allowed.stderr)
        self.assertIn("--allow-env-fallback: service 'linebot-cloud-agent' leaves", allowed.stderr)
        [deploy] = self.calls("run deploy")
        self.assertIn("--env-vars-file", deploy)
        self.assert_no_secret_printed(allowed)

    def test_a_staging_bucket_of_another_project_is_never_used(self) -> None:
        # Bucket names are global and this one is predictable.
        state = provisioned(buckets={_STAGING: {"location": "US-WEST1", "project": "squatter"}})

        result = self.deploy(state)

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn(f"{_STAGING} does not exist in project {_PROJECT}", result.stderr)
        [build] = self.calls("builds submit")
        self.assertNotIn("--gcs-source-staging-dir", build)
        self.assertNotIn("--service-account", build)


class ArtifactCleanupPolicyTests(DeployHarness):
    def _deploy(self, dry_run: bool) -> subprocess.CompletedProcess:
        # The previous revision's image is third newest: a failed deploy
        # pushed "sha256:failed" after it.
        state = provisioned(
            repos={
                "linebot-cloud-agent": {"cleanupPolicies": _AR_POLICIES, "cleanupPolicyDryRun": dry_run}
            },
            revisions=[revision("linebot-cloud-agent-previous", "sha256:prev")],
            images=["sha256:failed", "sha256:prev"],
        )
        return self.deploy(state)

    def test_an_enforced_policy_that_would_delete_a_kept_image_is_reported(self) -> None:
        result = self._deploy(dry_run=False)

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("1 kept revision(s) use an image outside the 2 newest", result.stderr)
        # The deploy's own cleanup still keeps the image a revision uses.
        self.assertIn("sha256:prev", self.state["images"])

    def test_a_dry_run_policy_deletes_nothing_and_is_not_reported(self) -> None:
        result = self._deploy(dry_run=True)

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertNotIn("outside the", result.stderr)


class FailedSmokeTests(DeployHarness):
    ready = False

    def test_a_failed_smoke_check_destroys_nothing(self) -> None:
        state = provisioned(secrets={"linebot-dotenv": [version(n, f"A='{n}'\n") for n in (1, 2, 3)]})

        result = self.deploy(state)

        self.assertEqual(result.returncode, 1)
        self.assertIn("ready_for_webhook=false", result.stderr)
        self.assertEqual(self.calls("secrets versions destroy"), [])
        self.assertEqual(self.calls("run revisions delete"), [])


class FallbackTests(DeployHarness):
    def test_without_provisioning_it_deploys_the_old_way_and_says_so(self) -> None:
        # A first deploy: the Cloud Run service does not exist yet either.
        result = self.deploy(
            {"repos": {"linebot-cloud-agent": {}}, "service_missing": True}, ["BOT_NAME=Bot"]
        )

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(result.stderr.count("Run ./scripts/provision_gcp.sh first"), 2)
        [build] = self.calls("builds submit")
        self.assertNotIn("--service-account", build)
        self.assertNotIn("--gcs-source-staging-dir", build)
        [deploy] = self.calls("run deploy")
        self.assertIn("--env-vars-file", deploy)
        self.assertIn("--clear-secrets", deploy)
        for flag in ("--set-secrets", "--set-env-vars", "--service-account"):
            self.assertNotIn(flag, deploy)
        env_yaml = self.state["env_vars_file"]
        self.assertIn(f'APP_VERSION: "{_TAG}"', env_yaml)
        self.assertIn('BOT_NAME: "Bot"', env_yaml)
        self.assertNotIn("GCP_PROJECT_ID", env_yaml)
        self.assertEqual(self.calls("secrets"), [])
        # The env-vars file never outlives the deploy.
        self.assertEqual(list(self.tmp.glob("linebot-deploy.*")), [])
        self.assert_no_secret_printed(result)

    def test_the_builder_needs_both_its_account_and_the_staging_bucket(self) -> None:
        state = provisioned(buckets={})

        result = self.deploy(state)

        self.assertEqual(result.returncode, 0, result.stderr)
        [build] = self.calls("builds submit")
        self.assertNotIn("--service-account", build)
        self.assertIn(f"{_STAGING} does not exist", result.stderr)
        # The runtime side is provisioned, so it still deploys from the secret.
        [deploy] = self.calls("run deploy")
        self.assertIn("--set-secrets=", deploy)

    def test_a_missing_secret_falls_back_to_the_env_file(self) -> None:
        state = provisioned(secrets={})

        result = self.deploy(state)

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("secret linebot-dotenv does not exist", result.stderr)
        [deploy] = self.calls("run deploy")
        self.assertIn("--env-vars-file", deploy)


class MediaBucketTests(DeployHarness):
    def _deploy_with_bucket(self, bucket: dict | None) -> subprocess.CompletedProcess:
        buckets = {_STAGING: {"location": "US-WEST1"}}
        if bucket is not None:
            buckets["gs://media"] = bucket
        return self.deploy(provisioned(buckets=buckets), ["GCS_BUCKET_NAME=media"])

    def test_a_bucket_outside_the_free_tier_stops_the_deploy_before_the_build(self) -> None:
        cases = (
            ({"location": "US", "default_storage_class": "STANDARD"}, "is in US"),
            ({"location": "ASIA-EAST1", "default_storage_class": "STANDARD"}, "is in ASIA-EAST1"),
            ({"location": "US-WEST1", "default_storage_class": "NEARLINE"}, "has default storage class NEARLINE"),
            (None, "could not be described"),
        )
        for bucket, problem in cases:
            with self.subTest(problem=problem):
                result = self._deploy_with_bucket(bucket)

                self.assertEqual(result.returncode, 1)
                self.assertIn(f"Media bucket gs://media {problem}", result.stderr)
                self.assertIn("--location=us-west1 --default-storage-class=STANDARD", result.stderr)
                self.assertEqual(self.calls("builds submit"), [])

    def test_a_free_tier_bucket_passes_and_keeps_its_lifecycle_rule(self) -> None:
        # Any of the three regions, in any case (one full deploy is enough).
        result = self._deploy_with_bucket({"location": "us-central1", "default_storage_class": "STANDARD"})

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("Bucket is STANDARD in an Always Free location", result.stdout)
        rules = self.state["buckets"]["gs://media"]["lifecycle_config"]["rule"]
        self.assertIn({"action": {"type": "Delete"}, "condition": {"age": 3}}, rules)
        # linebot-runtime's signing role is provisioning's (runtime-sa step).
        self.assertEqual(self.calls("iam service-accounts get-iam-policy"), [])
        self.assertEqual(self.calls("iam service-accounts add-iam-policy-binding"), [])

    def test_any_other_runtime_account_gets_the_signing_role_once(self) -> None:
        # Signed media URLs need roles/iam.serviceAccountTokenCreator on the
        # account itself; without it image and voice replies become text.
        custom = f"custom-runtime@{_PROJECT}.iam.gserviceaccount.com"
        compute = "843253221534-compute@developer.gserviceaccount.com"
        media = {"gs://media": {"location": "US-WEST1", "default_storage_class": "STANDARD"}}
        staging = {_STAGING: {"location": "US-WEST1", "default_storage_class": "STANDARD"}}
        cases = (
            (
                "custom account",
                provisioned(service_accounts=[custom, _BUILDER_SA], buckets={**staging, **media}),
                [f"CLOUD_RUN_SERVICE_ACCOUNT={custom}"],
                custom,
            ),
            (
                "not provisioned",
                {"repos": {"linebot-cloud-agent": {}}, "buckets": dict(media), "service_account_on_service": compute},
                [],
                compute,
            ),
        )
        for name, state, lines, account in cases:
            with self.subTest(name):
                granted = self.deploy(state, ["GCS_BUCKET_NAME=media", *lines])

                self.assertEqual(granted.returncode, 0, granted.stderr)
                binding = [account, "roles/iam.serviceAccountTokenCreator", f"serviceAccount:{account}"]
                self.assertIn(binding, self.state["bindings"])

                again = self.deploy(self.state, ["GCS_BUCKET_NAME=media", *lines])

                self.assertEqual(again.returncode, 0, again.stderr)
                self.assertEqual(self.calls("iam service-accounts add-iam-policy-binding"), [])

    def test_an_unknown_runtime_account_is_named_for_the_signing_role(self) -> None:
        media = {"gs://media": {"location": "US-WEST1", "default_storage_class": "STANDARD"}}
        state = {"repos": {"linebot-cloud-agent": {}}, "buckets": media}

        result = self.deploy(state, ["GCS_BUCKET_NAME=media"])

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("Runtime service account unknown; grant it roles/iam.serviceAccountTokenCreator", result.stderr)


class EgressTests(DeployHarness):
    def test_the_service_keeps_cloud_runs_own_egress(self) -> None:
        # Search is Tavily only (ADR-0009): no Direct VPC egress, no subnet
        # lookup. An old .env that still sets the removed self-hosted search
        # keys only gets a note.
        old_keys = ["SEARXNG_BASE_URL=http://10.138.0.10:8080", "SEARXNG_TIMEOUT_SECONDS=4.5"]
        for lines in ((), old_keys):
            with self.subTest(lines=lines):
                result = self.deploy(provisioned(), lines)

                self.assertEqual(result.returncode, 0, result.stderr)
                [deploy] = self.calls("run deploy")
                for flag in ("--network", "--subnet", "--vpc-egress", "--vpc-connector"):
                    self.assertNotIn(flag, deploy)
                self.assertEqual(self.calls("compute"), [])
                output = result.stdout + result.stderr
                if lines:
                    self.assertIn("no longer read", output)
                    self.assertIn("SEARXNG_BASE_URL, SEARXNG_TIMEOUT_SECONDS", output)


class ColabSecretTests(DeployHarness):
    """linebot-colab-adc (ADR-0010): mounted, pinned, only with COLAB_ENABLED=true."""

    _ON = ["COLAB_ENABLED=true"]
    _COLAB = "linebot-colab-adc"
    _READS = [_COLAB, "roles/secretmanager.secretAccessor", f"serviceAccount:{_RUNTIME_SA}"]

    def _state(self, colab_versions: list[dict], *, readable: bool = True, **overrides) -> dict:
        return provisioned(
            secrets={"linebot-dotenv": [version(1, "A='1'\n")], self._COLAB: colab_versions},
            bindings=[self._READS] if readable else [],
            **overrides,
        )

    def _colab_calls(self) -> list[str]:
        return [call for call in self.calls("") if self._COLAB in call]

    def test_the_newest_enabled_version_is_pinned_and_never_read(self) -> None:
        state = self._state([version(1, "{}"), version(2, "{}"), version(3, "{}", "DISABLED")])

        result = self.deploy(state, self._ON)

        self.assertEqual(result.returncode, 0, result.stderr)
        [deploy] = self.calls("run deploy")
        self.assertIn(
            "--set-secrets=/secrets/app/app.env=linebot-dotenv:2,/secrets/colab/adc.json=linebot-colab-adc:2",
            deploy,
        )
        self.assertIn("Mounting linebot-colab-adc version 2 at /secrets/colab/adc.json", result.stdout)
        self.assertIn("Colab credential: secret linebot-colab-adc version 2", result.stdout)
        volumes = self.state["revisions"][-1]["volumes"]
        self.assertEqual(
            [(v["secret"]["secretName"], v["secret"]["items"][0]["key"]) for v in volumes],
            [("linebot-dotenv", "2"), (self._COLAB, "2")],
        )
        # The deploy never reads the token itself.
        self.assertEqual([c for c in self._colab_calls() if "versions access" in c], [])
        self.assertNotIn("COLAB_ADC_PATH", result.stderr)

    def test_versions_no_kept_revision_pins_are_destroyed_after_the_smoke_check(self) -> None:
        previous = {
            "name": "linebot-cloud-agent-previous",
            "image": f"us-west1-docker.pkg.dev/{_PROJECT}/linebot-cloud-agent/app@sha256:prev",
            "volumes": [
                *dotenv_mount("1"),
                {"name": "secret-2", "secret": {"secretName": self._COLAB, "items": [{"key": "2", "path": "adc.json"}]}},
            ],
        }
        state = self._state(
            [version(n, "{}") for n in (1, 2, 3, 4)], revisions=[previous], images=["sha256:prev"]
        )

        result = self.deploy(state, [*self._ON, "COLAB_ADC_PATH=/elsewhere/adc.json"])

        self.assertEqual(result.returncode, 0, result.stderr)
        states = {v["id"]: v["state"] for v in self.state["secrets"][self._COLAB]}
        # 4 is deployed, 3 is within the newest two, 2 is the rollback target's.
        self.assertEqual(states, {1: "DESTROYED", 2: "ENABLED", 3: "ENABLED", 4: "ENABLED"})
        self.assertIn("Keeping linebot-colab-adc version 2 (a kept revision uses it)", result.stdout)
        # The app reads COLAB_ADC_PATH; elsewhere, the mounted file goes unread.
        self.assertIn("COLAB_ADC_PATH is /elsewhere/adc.json", result.stderr)

    def test_without_the_cron_secret_colab_warns_that_it_stays_off(self) -> None:
        state = self._state([version(1, "{}")])

        result = self.deploy(state, self._ON)

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("COLAB_ENABLED=true needs INTERNAL_CRON_SECRET", result.stderr)

        result = self.deploy(self._state([version(1, "{}")]), [*self._ON, "INTERNAL_CRON_SECRET=" + "c0" * 32])

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertNotIn("needs INTERNAL_CRON_SECRET", result.stderr)

    def test_without_colab_enabled_the_token_is_neither_mounted_nor_touched(self) -> None:
        state = self._state([version(n, "{}") for n in (1, 2, 3, 4)])

        result = self.deploy(state, ["COLAB_ENABLED=false"])

        self.assertEqual(result.returncode, 0, result.stderr)
        [deploy] = self.calls("run deploy")
        self.assertIn("--set-secrets=/secrets/app/app.env=linebot-dotenv:2 ", deploy + " ")
        self.assertEqual(self._colab_calls(), [])
        self.assertEqual({v["state"] for v in self.state["secrets"][self._COLAB]}, {"ENABLED"})

    def test_a_missing_secret_warns_and_deploys_without_colab(self) -> None:
        result = self.deploy(provisioned(secrets={"linebot-dotenv": [version(1, "A='1'\n")]}), self._ON)

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("secret linebot-colab-adc does not exist", result.stderr)
        self.assertIn("./scripts/mint_colab_adc.sh --apply", result.stderr)
        [deploy] = self.calls("run deploy")
        self.assertNotIn(self._COLAB, deploy)

    def test_a_runtime_account_that_cannot_read_it_stops_the_deploy_before_the_build(self) -> None:
        result = self.deploy(self._state([version(1, "{}")], readable=False), self._ON)

        self.assertEqual(result.returncode, 1)
        self.assertIn(f"{_RUNTIME_SA} cannot read secret linebot-colab-adc", result.stderr)
        self.assertIn("provision_gcp.sh --step colab-secret --apply", result.stderr)
        self.assertEqual(self.calls("builds submit"), [])
        self.assertEqual(self.calls("run deploy"), [])

    def test_colab_stays_off_without_an_enabled_version_or_provisioning(self) -> None:
        for state, warning in (
            (self._state([version(1, "{}", "DISABLED"), version(2, "{}", "DESTROYED")]), "has no enabled version"),
            ({"secrets": {self._COLAB: [version(1, "{}")]}}, "only a provisioned deploy"),
        ):
            with self.subTest(warning=warning):
                result = self.deploy(state, self._ON)

                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertIn(warning, result.stderr)
                [deploy] = self.calls("run deploy")
                self.assertNotIn(self._COLAB, deploy)

    def test_an_unreadable_colab_secret_stops_the_deploy(self) -> None:
        state = self._state(
            [version(1, "{}")], fail={"secrets versions list linebot-colab-adc": "PERMISSION_DENIED: nope"}
        )

        result = self.deploy(state, self._ON)

        self.assertEqual(result.returncode, 1)
        self.assertIn("not with NOT_FOUND", result.stderr)
        self.assertEqual(self.calls("builds submit"), [])


class ColabQueueTests(DeployHarness):
    """The ``colab`` tasks' own queue (ADR-0010), so they never hold the event slots."""

    _CLOUD = ["TASKS_MODE=auto", "INTERNAL_TASKS_SECRET=" + "a1" * 32]

    def _state(self) -> dict:
        return provisioned(
            secrets={"linebot-dotenv": [version(1, "A='1'\n")], ColabSecretTests._COLAB: [version(1, "{}")]},
            bindings=[ColabSecretTests._READS],
        )

    def _queue_calls(self, queue: str) -> list[str]:
        return [call for call in self.calls("tasks queues") if call.split()[3] == queue]

    def test_colab_gets_its_own_two_slot_queue_and_the_enqueuer_role(self) -> None:
        result = self.deploy(self._state(), [*self._CLOUD, "COLAB_ENABLED=true"])

        self.assertEqual(result.returncode, 0, result.stderr)
        [events] = [c for c in self._queue_calls("linebot-events") if c.startswith("tasks queues create")]
        self.assertIn("--max-concurrent-dispatches=4", events)
        [colab] = [c for c in self._queue_calls("linebot-colab") if c.startswith("tasks queues create")]
        self.assertIn("--max-attempts=1", colab)
        self.assertIn("--max-concurrent-dispatches=2", colab)
        self.assertIn("--max-dispatches-per-second=1", colab)
        self.assertNotIn("--max-retry-duration", colab)
        granted = sorted(
            call.split()[3]
            for call in self.calls("tasks queues add-iam-policy-binding")
            if f"serviceAccount:{_RUNTIME_SA}" in call and "roles/cloudtasks.enqueuer" in call
        )
        self.assertEqual(granted, ["linebot-colab", "linebot-events"])

    def test_without_colab_there_is_no_colab_queue(self) -> None:
        result = self.deploy(self._state(), [*self._CLOUD, "COLAB_ENABLED=false"])

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(self._queue_calls("linebot-colab"), [])
        self.assertTrue(self._queue_calls("linebot-events"))

    def test_the_colab_queue_must_not_be_the_event_queue(self) -> None:
        result = self.deploy(
            self._state(), [*self._CLOUD, "COLAB_ENABLED=true", "CLOUD_TASKS_COLAB_QUEUE=linebot-events"]
        )

        self.assertEqual(result.returncode, 1)
        self.assertIn("CLOUD_TASKS_COLAB_QUEUE must differ from CLOUD_TASKS_QUEUE", result.stderr)
        self.assertEqual(self.calls("tasks queues"), [])
        self.assertEqual(self.calls("builds submit"), [])

    def test_inline_tasks_leave_colab_off_with_a_warning(self) -> None:
        result = self.deploy(self._state(), ["COLAB_ENABLED=true"])

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("COLAB_ENABLED=true needs Cloud Tasks", result.stderr)
        self.assertEqual(self.calls("tasks queues"), [])


class EnvFileTests(DeployHarness):
    """The deploy reads the env file once (envfile.py dump), then checks it."""

    def test_required_keys_are_set_and_no_key_holds_a_placeholder(self) -> None:
        for line, error in (
            ("LINE_CHANNEL_SECRET=", "Required env key missing or empty in {}: LINE_CHANNEL_SECRET"),
            (
                "OPENROUTER_API_KEY=your_openrouter_api_key",
                "Required env key still uses an example placeholder in {}: OPENROUTER_API_KEY",
            ),
            ("NVIDIA_API_KEY='your_nvidia_api_key'", "Env key still uses an example placeholder in {}: NVIDIA_API_KEY"),
            ("GCP_PROJECT_ID=your_project_id", "GCP project is still an example placeholder"),
        ):
            with self.subTest(line=line):
                result = self.deploy(provisioned(), [line])

                self.assertEqual(result.returncode, 1)
                self.assertIn(error.format(self.env_file), result.stderr)
                self.assertEqual(self.calls("builds submit"), [])

    def test_a_line_missing_its_equals_sign_is_named_by_number_never_shown(self) -> None:
        # The '=' of the base64 padding makes the key "NVIDIA_API_KEY nvapi-...".
        result = self.deploy(provisioned(), ["NVIDIA_API_KEY nvapi-do-not-print=="])

        self.assertEqual(result.returncode, 1)
        self.assertIn(f"line {len(_BASE_ENV) + 1}: not KEY=VALUE", result.stdout + result.stderr)
        self.assertNotIn("nvapi-do-not-print", result.stdout + result.stderr)
        self.assertEqual(self.calls("builds submit"), [])
        self.assert_no_secret_printed(result)

    def test_the_tasks_queue_location_follows_the_app(self) -> None:
        # A missing key means us-west1 (the Settings default), an empty one
        # the Cloud Run region. A failing build ends each deploy early.
        state = provisioned(fail={"builds submit": "INTERNAL: stop after the queues"})
        cloud = ["TASKS_MODE=auto", "INTERNAL_TASKS_SECRET=" + "a1" * 32]
        for lines, location in (
            ([], "us-west1"),
            (["CLOUD_RUN_REGION=us-central1", "CLOUD_TASKS_LOCATION="], "us-central1"),
            # Keys after a multi-line value still read right.
            (["SCHEDULED_WEEKLY_MESSAGES='[\n  {\"id\": \"w\"}\n]'", "CLOUD_TASKS_LOCATION=asia-east1"], "asia-east1"),
        ):
            with self.subTest(location=location):
                result = self.deploy(state, [*cloud, *lines])

                self.assertEqual(result.returncode, 1)
                self.assertIn("stop after the queues", result.stderr)
                [create] = self.calls("tasks queues create linebot-events")
                self.assertIn(f"--location {location} ", create)


class EnvFileInRepositoryTests(DeployHarness):
    """--env-file inside the repository: gcloud would upload it (and the
    image copy it) whatever its name, unless .gcloudignore leaves it out."""

    def setUp(self) -> None:
        super().setUp()
        # A copy of the scripts is the repository, so its env file is inside.
        repo = self.tmp / "repo"
        (repo / "scripts").mkdir(parents=True, exist_ok=True)
        for name in ("deploy_cloud_run.sh", "envfile.py"):
            shutil.copy2(_ROOT / "scripts" / name, repo / "scripts" / name)
        self.script = repo / "scripts" / "deploy_cloud_run.sh"
        self.env_file = repo / "prod.env"
        # gcloud would upload the env file along with the code.
        (self.tmp / "bin" / "gcloud").write_text(
            "#!/bin/sh\n"
            'if [ "$1 $2" = "meta list-files-for-upload" ]; then printf \'main.py\\nprod.env\\n\'; exit 0; fi\n'
            f'exec "{sys.executable}" -S "{_FAKE}" "$@"\n',
            encoding="utf-8",
        )

    def test_an_env_file_gcloud_would_upload_stops_the_deploy_before_the_build(self) -> None:
        result = self.deploy(provisioned())

        self.assertEqual(result.returncode, 1, result.stderr)
        self.assertIn("  prod.env\n", result.stderr)
        self.assertIn("Refusing to submit", result.stderr)
        self.assertEqual(self.calls("builds submit"), [])
        self.assert_no_secret_printed(result)


class ScriptTextTests(unittest.TestCase):
    def test_the_deploy_script_parses_under_bash_n(self) -> None:
        result = subprocess.run(["bash", "-n", str(_SCRIPT)], capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_every_payload_read_turns_gcloud_file_logging_off(self) -> None:
        text = _SCRIPT.read_text(encoding="utf-8").replace("\\\n", " ")
        reads = [line.strip() for line in text.splitlines() if "gcloud secrets versions access" in line]

        self.assertTrue(reads)
        for line in reads:
            self.assertIn("CLOUDSDK_CORE_DISABLE_FILE_LOGGING=true gcloud secrets versions access", line)

    def test_secret_payloads_never_touch_a_file(self) -> None:
        text = _SCRIPT.read_text(encoding="utf-8").replace("\\\n", " ")
        adds = [line for line in text.splitlines() if "gcloud secrets versions add" in line]
        self.assertTrue(adds)
        for line in adds:
            self.assertIn("--data-file=-", line)
        self.assertNotIn("--out-file", text)


if __name__ == "__main__":
    unittest.main()
