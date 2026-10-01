"""scripts/mint_colab_adc.sh against the stateful fake gcloud (tests/fake_gcloud.py).

The fake's ``auth application-default login`` writes a fake authorized_user
credential into ``$CLOUDSDK_CONFIG`` (and refuses without it), so these
tests follow the token from the scratch directory into Secret Manager.
Nothing here reaches Google or opens a browser.
"""

from __future__ import annotations

import base64
import json
import os
import re
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from src.services.colab.control_plane import adc_file_problem
from tests.fake_gcloud import FAKE_ADC

_ROOT = Path(__file__).resolve().parents[1]
_SCRIPT = _ROOT / "scripts" / "mint_colab_adc.sh"
_FAKE = Path(__file__).resolve().with_name("fake_gcloud.py")

_PROJECT = "test-project"
_RUNTIME_SA = f"linebot-runtime@{_PROJECT}.iam.gserviceaccount.com"
_RUNTIME = f"serviceAccount:{_RUNTIME_SA}"
_SECRET = "linebot-colab-adc"
_SCOPES = (
    "openid,https://www.googleapis.com/auth/cloud-platform,"
    "https://www.googleapis.com/auth/userinfo.email,"
    "https://www.googleapis.com/auth/colaboratory"
)
_ACCESSOR = "roles/secretmanager.secretAccessor"
# Verbs that change something; read-only checks never use them.
_MUTATING = re.compile(r"\b(create|add|add-iam-policy-binding|login|destroy|delete)\b")


class MintHarness(unittest.TestCase):
    def setUp(self) -> None:
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.tmp = Path(tmp.name)
        bin_dir = self.tmp / "bin"
        bin_dir.mkdir()
        gcloud = bin_dir / "gcloud"
        gcloud.write_text(f'#!/bin/sh\nexec "{sys.executable}" "{_FAKE}" "$@"\n', encoding="utf-8")
        gcloud.chmod(0o755)
        # A wrapper, not a symlink: a venv's python only finds its packages
        # when started by its own path.
        python3 = bin_dir / "python3"
        python3.write_text(f'#!/bin/sh\nexec "{sys.executable}" "$@"\n', encoding="utf-8")
        python3.chmod(0o755)
        self.scratch_parent = self.tmp / "scratch"
        self.scratch_parent.mkdir()
        self.state_path = self.tmp / "state.json"
        self.log_path = self.tmp / "gcloud.log"
        self.env_file = self.tmp / "deploy.env"
        self.env_file.write_text(f"GCP_PROJECT_ID={_PROJECT}\n", encoding="utf-8")
        environ = {
            key: value
            for key, value in os.environ.items()
            # The script picks the scratch config itself; a person's own
            # CLOUDSDK_CONFIG must not leak into the fake's bookkeeping.
            if key not in {"CLOUDSDK_CONFIG", "CLOUDSDK_CORE_DISABLE_FILE_LOGGING"}
        }
        environ.update(
            {
                "PATH": f"{bin_dir}{os.pathsep}{os.environ.get('PATH', '')}",
                "FAKE_GCLOUD_STATE": str(self.state_path),
                "FAKE_GCLOUD_LOG": str(self.log_path),
                "TMPDIR": str(self.scratch_parent),
            }
        )
        self.environ = environ

    def mint(self, state: dict, *args: str) -> subprocess.CompletedProcess:
        self.state_path.write_text(json.dumps(state), encoding="utf-8")
        self.log_path.unlink(missing_ok=True)
        return subprocess.run(
            ["bash", str(_SCRIPT), "--env-file", str(self.env_file), *args],
            capture_output=True,
            text=True,
            env=self.environ,
            timeout=120,
        )

    @property
    def state(self) -> dict:
        return json.loads(self.state_path.read_text(encoding="utf-8"))

    def calls(self) -> list[str]:
        if not self.log_path.exists():
            return []
        return self.log_path.read_text(encoding="utf-8").splitlines()

    def mutating_calls(self) -> list[str]:
        return [call for call in self.calls() if _MUTATING.search(call.split(" --")[0])]

    def assert_scratch_gone(self) -> None:
        self.assertEqual(list(self.scratch_parent.iterdir()), [], "the scratch config must be deleted")

    def assert_token_never_printed(self, result: subprocess.CompletedProcess) -> None:
        for value in (FAKE_ADC["refresh_token"], FAKE_ADC["client_secret"]):
            self.assertNotIn(value, result.stdout + result.stderr)


def provisioned(**overrides) -> dict:
    state: dict = {"service_accounts": [_RUNTIME_SA]}
    state.update(overrides)
    return state


class CommandLineTests(MintHarness):
    def test_the_script_parses_under_bash_n(self) -> None:
        result = subprocess.run(["bash", "-n", str(_SCRIPT)], capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_help_and_unknown_options(self) -> None:
        help_result = self.mint({}, "--help")
        self.assertEqual(help_result.returncode, 0)
        self.assertIn("linebot-colab-adc", help_result.stdout)
        for args in (("--bogus",), ("--project-id",), ("--project-id", "Not_A_Project")):
            with self.subTest(args=args):
                result = self.mint({}, *args)
                self.assertEqual(result.returncode, 1)
                self.assertIn("[fail]", result.stderr)
        self.assertFalse([call for call in self.calls() if "secrets" in call or "login" in call])


class DryRunTests(MintHarness):
    def test_the_dry_run_mints_stores_and_grants_nothing(self) -> None:
        state = provisioned()

        result = self.mint(state)

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("[mode] dry run", result.stdout)
        self.assertEqual(self.mutating_calls(), [])
        for key in ("adc_logins", "secrets", "bindings"):
            self.assertNotIn(key, self.state)
        self.assertIn(
            "[dry-run] CLOUDSDK_CONFIG=<scratch dir> gcloud auth application-default login "
            f"--no-launch-browser --disable-quota-project --scopes={_SCOPES}",
            result.stdout,
        )
        self.assertIn(
            f"[dry-run] gcloud secrets create {_SECRET} --project {_PROJECT} "
            "--replication-policy=automatic --data-file=- < <the minted credential file>",
            result.stdout,
        )
        self.assertIn("[dry-run] ./scripts/provision_gcp.sh --step colab-secret --apply", result.stdout)
        # The advice a person needs before the browser opens, and after.
        self.assertIn("DEDICATED Google account", result.stdout)
        self.assertIn("no role in project", result.stdout)
        self.assertIn("100 live", result.stdout)
        self.assertIn("https://myaccount.google.com/connections", result.stdout)
        self.assertIn("unused for 6 months", result.stdout)
        self.assertIn("[done] dry run finished; nothing was minted, stored or granted", result.stdout)
        self.assert_scratch_gone()

    def test_an_existing_secret_is_planned_as_a_new_version(self) -> None:
        result = self.mint(provisioned(secrets={_SECRET: [{"id": 1, "state": "ENABLED", "data": ""}]}))

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn(
            f"[dry-run] gcloud secrets versions add {_SECRET} --project {_PROJECT} --data-file=-",
            result.stdout,
        )
        self.assertNotIn("secrets create", result.stdout)


class ApplyTests(MintHarness):
    def test_apply_mints_under_a_scratch_config_and_stores_the_token(self) -> None:
        result = self.mint(provisioned(), "--apply")

        self.assertEqual(result.returncode, 0, result.stderr)
        state = self.state
        # One sign-in, with the verified flags and scopes, in a scratch
        # config outside the repository that is gone afterwards.
        [login] = state["adc_logins"]
        self.assertEqual(login["scopes"], _SCOPES)
        self.assertEqual(login["flags"], ["--disable-quota-project", "--no-launch-browser"])
        self.assertTrue(login["config"].startswith(str(self.scratch_parent)), login["config"])
        self.assertFalse(login["config"].startswith(str(_ROOT)))
        self.assertFalse(Path(login["config"]).exists())
        self.assert_scratch_gone()
        # Only the sign-in used the scratch config: the Secret Manager and
        # IAM calls ran as the person's usual gcloud login.
        self.assertEqual(state.get("scratch_config_calls", []), [])
        # The secret holds exactly the file gcloud wrote, created with
        # automatic replication.
        [version] = state["secrets"][_SECRET]
        self.assertEqual(json.loads(base64.b64decode(version["data"])), FAKE_ADC)
        self.assertEqual(state["secret_creates"], ["automatic"])
        self.assertIn([_SECRET, _ACCESSOR, _RUNTIME], state["bindings"])
        self.assertIn(f"stored as {_SECRET} version 1", result.stdout)
        self.assertIn(f"mounts {_SECRET} version 1 at\n          /secrets/colab/adc.json", result.stdout)
        self.assertIn("DEDICATED Google account", result.stdout)
        self.assertIn("https://myaccount.google.com/connections", result.stdout)
        self.assert_token_never_printed(result)
        # The minted credential passes the bot's own rule.
        credential = self.tmp / "adc.json"
        credential.write_text(json.dumps(FAKE_ADC), encoding="utf-8")
        self.assertEqual(adc_file_problem(credential), "")

    def test_an_existing_secret_gets_a_new_version_and_keeps_its_binding(self) -> None:
        state = provisioned(
            secrets={_SECRET: [{"id": 1, "state": "ENABLED", "data": ""}]},
            bindings=[[_SECRET, _ACCESSOR, _RUNTIME]],
        )

        result = self.mint(state, "--apply")

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual([v["id"] for v in self.state["secrets"][_SECRET]], [1, 2])
        self.assertNotIn("secret_creates", self.state)
        self.assertEqual([c for c in self.calls() if "add-iam-policy-binding" in c], [])
        self.assertEqual(self.state["bindings"], [[_SECRET, _ACCESSOR, _RUNTIME]])
        self.assertIn(f"stored as {_SECRET} version 2", result.stdout)
        self.assert_token_never_printed(result)

    def test_without_the_runtime_account_it_stores_and_says_how_to_grant(self) -> None:
        result = self.mint({}, "--apply")

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(len(self.state["secrets"][_SECRET]), 1)
        self.assertNotIn("bindings", self.state)
        self.assertIn("run --step runtime-sa --apply", result.stderr)


class RefusalTests(MintHarness):
    def test_a_credential_that_cannot_drive_colab_is_never_stored(self) -> None:
        service_account = {"type": "service_account", "private_key": "-----BEGIN do-not-print"}
        no_refresh = {key: value for key, value in FAKE_ADC.items() if key != "refresh_token"}
        for payload, problem in (
            (service_account, "'service_account', not an authorized_user"),
            (no_refresh, "lacks refresh_token"),
            ("not json {do-not-print", "cannot be read (JSONDecodeError)"),
        ):
            with self.subTest(problem=problem):
                result = self.mint(provisioned(adc=payload), "--apply")

                self.assertEqual(result.returncode, 1)
                self.assertIn(problem, result.stdout)
                self.assertIn("nothing was stored", result.stderr)
                self.assertNotIn("secrets", self.state)
                self.assertNotIn("do-not-print", result.stdout + result.stderr)
                self.assert_scratch_gone()
                # The bot's control plane refuses the same file.
                credential = self.tmp / "adc.json"
                credential.write_text(
                    payload if isinstance(payload, str) else json.dumps(payload), encoding="utf-8"
                )
                self.assertNotEqual(adc_file_problem(credential), "")

    def test_a_sign_in_that_does_not_finish_stores_nothing(self) -> None:
        result = self.mint(provisioned(adc_login_fails=True), "--apply")

        self.assertEqual(result.returncode, 1)
        self.assertIn("The sign-in did not finish; nothing was stored", result.stderr)
        self.assertNotIn("secrets", self.state)
        self.assert_scratch_gone()

    def test_checks_that_fail_stop_before_any_token_is_minted(self) -> None:
        for state, error in (
            (provisioned(fail={"secrets describe": "PERMISSION_DENIED: no access"}), "not with NOT_FOUND"),
            (provisioned(active_account=""), "No active gcloud account"),
        ):
            with self.subTest(error=error):
                result = self.mint(state, "--apply")

                self.assertEqual(result.returncode, 1)
                self.assertIn(error, result.stderr)
                self.assertNotIn("adc_logins", self.state)
                self.assertNotIn("secrets", self.state)
                self.assert_scratch_gone()


if __name__ == "__main__":
    unittest.main()
