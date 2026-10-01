"""scripts/provision_gcp.sh against a stateful fake gcloud (tests/fake_gcloud.py).

Nothing here reaches Google: the fake keeps the "project" in a JSON file.
"""

from __future__ import annotations

import base64
import json
import os
import re
import subprocess
import sys
import unittest
from pathlib import Path

from scripts import envfile
from tests.script_rehearsal import RehearsedScripts, start_rehearsals, stop_rehearsals

_ROOT = Path(__file__).resolve().parents[1]
_SCRIPT = _ROOT / "scripts" / "provision_gcp.sh"
_MINT = _ROOT / "scripts" / "mint_colab_adc.sh"
_FAKE = Path(__file__).resolve().with_name("fake_gcloud.py")

_PROJECT = "test-project"
_RUNTIME = f"serviceAccount:linebot-runtime@{_PROJECT}.iam.gserviceaccount.com"
_BUILDER = f"serviceAccount:linebot-builder@{_PROJECT}.iam.gserviceaccount.com"
_STAGING = f"gs://{_PROJECT}-cb-src-usw1"
_LEGACY = f"gs://{_PROJECT}_cloudbuild"
_ENV = (
    f"GCP_PROJECT_ID={_PROJECT}\n"
    "GCS_BUCKET_NAME=media-bucket\n"
    "LINE_CHANNEL_SECRET=line-secret-do-not-print\n"
    "SCHEDULED_WEEKLY_MESSAGES='[\n  {\"id\":\"w\",\"message\":\"早安 #1\"}\n]'\n"
)
_OLD_VM = {
    "name": "instance-gcp-banana",
    "zone": "https://www.googleapis.com/compute/v1/projects/test-project/zones/us-west1-b",
    "status": "TERMINATED",
}
# Steps removed with the self-hosted search backend; now unknown.
_REMOVED_STEPS = ("firewall-pre", "verify", "searxng-net", "searxng-vm", "searxng-lockdown")
# Verbs that change something; read-only checks never use them.
_MUTATING = re.compile(
    r"\b(create|delete|update|enable|add-iam-policy-binding|remove-iam-policy-binding|set-cleanup-policies|rm)\b"
)


def fresh_project() -> dict:
    """terabanana before provisioning, in the fake's terms."""
    return {
        "buckets": {
            _LEGACY: {
                "location": "US",
                "soft_delete_policy": {"retentionDurationSeconds": "604800"},
            },
            "gs://media-bucket": {"location": "US-WEST1", "default_storage_class": "STANDARD"},
        },
        "cloudbuild_objects": 58,
        "repos": {"linebot-cloud-agent": {}},
        "queues": ["linebot-events"],
        "apis": ["run.googleapis.com", "iamcredentials.googleapis.com"],
        # The stopped v2 VM (Ubuntu 25.10, end of life).
        "instances": [dict(_OLD_VM)],
    }


def setUpModule() -> None:
    start_rehearsals(sys.modules[__name__])


def tearDownModule() -> None:
    stop_rehearsals()


class ProvisionHarness(RehearsedScripts, unittest.TestCase):
    def setUp(self) -> None:
        super().setUp()
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
        self.state_path = self.tmp / "state.json"
        self.log_path = self.tmp / "gcloud.log"
        self.env_file = self.tmp / "deploy.env"
        self.env_file.write_text(_ENV, encoding="utf-8")
        self.environ = {
            # The scripts must turn gcloud's prompts off themselves.
            **{key: value for key, value in self.base_environ.items() if key != "CLOUDSDK_CORE_DISABLE_PROMPTS"},
            "PATH": f"{bin_dir}{os.pathsep}{self.base_environ.get('PATH', '')}",
            "FAKE_GCLOUD_STATE": str(self.state_path),
            "FAKE_GCLOUD_LOG": str(self.log_path),
        }
        self.set_state(fresh_project())

    def set_state(self, state: dict) -> None:
        self.state_path.write_text(json.dumps(state), encoding="utf-8")

    @property
    def state(self) -> dict:
        return json.loads(self.state_path.read_text(encoding="utf-8"))

    def calls(self) -> list[str]:
        if not self.log_path.exists():
            return []
        return self.log_path.read_text(encoding="utf-8").splitlines()

    def mutating_calls(self) -> list[str]:
        return [call for call in self.calls() if _MUTATING.search(call.split(" --")[0])]

    def provision(self, *args: str) -> subprocess.CompletedProcess:
        self.log_path.unlink(missing_ok=True)
        return self.run_script(["bash", str(_SCRIPT), "--env-file", str(self.env_file), *args], timeout=120)

    @staticmethod
    def section(stdout: str, step: str) -> str:
        """The output of one step."""
        match = re.search(rf"^\[step\] {re.escape(step)}\n(.*?)(?=^\[step\]|^\[done\])", stdout, re.M | re.S)
        assert match, f"no [step] {step} in output"
        return match.group(1)


class CommandLineTests(ProvisionHarness):
    def test_list_names_every_step(self) -> None:
        result = self.provision("--list")

        self.assertEqual(result.returncode, 0, result.stderr)
        steps = re.findall(r"(?m)^([a-z-]+)\s", result.stdout)
        self.assertEqual(
            steps,
            [
                "apis",
                "runtime-sa",
                "staging-bucket",
                "builder-sa",
                "purge-cloudbuild",
                "secrets",
                "colab-secret",
                "ar-cleanup",
                "ar-cleanup-enforce",
                "drop-editor",
                "budget",
                "summary",
            ],
        )
        self.assertEqual(self.calls(), [])

    def test_unknown_steps_and_options_are_refused(self) -> None:
        # An empty --step must not mean every step, nor may two steps pass
        # as one. The removed network steps and their flag are unknown.
        for args in (
            ("--step", "everything"),
            ("--bogus",),
            ("--step",),
            ("--step", ""),
            ("--step", " "),
            ("--step", "apis runtime-sa"),
            ("--step", "", "--apply"),
            *(("--step", step) for step in _REMOVED_STEPS),
            ("--apply", "--yes-delete-old-vm"),
        ):
            with self.subTest(args=args):
                result = self.provision(*args)
                self.assertEqual(result.returncode, 1)
                self.assertIn("[fail]", result.stderr)
                self.assertNotIn("[step]", result.stdout)
        self.assertEqual(self.calls(), [])

    def test_the_script_parses_under_bash_n(self) -> None:
        result = subprocess.run(["bash", "-n", str(_SCRIPT)], capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)


class DryRunTests(ProvisionHarness):
    # One dry run of every step, shared by the tests below (so, live: a
    # rehearsal per test would repeat it).
    rehearse = False
    _run: tuple | None = None

    def setUp(self) -> None:
        super().setUp()
        if DryRunTests._run is None:
            result = self.provision()
            DryRunTests._run = (result, self.calls(), self.mutating_calls(), self.state)
        self.result, self.dry_calls, self.dry_mutating, self.dry_state = DryRunTests._run
        self.out = self.result.stdout

    def test_dry_run_changes_nothing(self) -> None:
        self.assertEqual(self.result.returncode, 0, self.result.stderr)
        self.assertIn("[mode] dry run", self.out)
        self.assertIn("[done] dry run finished; nothing was changed", self.out)
        self.assertNotIn("[run]", self.out)
        # Only reads reached gcloud; the fake project is untouched.
        self.assertEqual(self.dry_mutating, [])
        self.assertTrue(self.dry_calls)
        self.assertEqual(self.dry_state, fresh_project())
        # Nothing reads or plans Compute Engine: no network, no VM.
        self.assertFalse([call for call in self.dry_calls if call.startswith("compute ")])
        self.assertNotIn("gcloud compute", self.out)
        # drop-editor runs only when named.
        self.assertNotIn("[step] drop-editor", self.out)

    def test_apis_enables_tasks_and_secret_manager_but_never_scanning(self) -> None:
        section = self.section(self.out, "apis")

        self.assertIn(
            "[dry-run] gcloud services enable cloudtasks.googleapis.com "
            f"secretmanager.googleapis.com --project {_PROJECT}",
            section,
        )
        commands = [line for line in self.out.splitlines() if "[dry-run]" in line]
        self.assertFalse([line for line in commands if "scanning" in line])

    def test_runtime_sa_gets_exactly_its_roles(self) -> None:
        section = self.section(self.out, "runtime-sa")

        self.assertIn(f"gcloud iam service-accounts create linebot-runtime --project {_PROJECT}", section)
        self.assertIn(
            f"gcloud projects add-iam-policy-binding {_PROJECT} --member={_RUNTIME} "
            "--role=roles/datastore.user --condition=None",
            section,
        )
        self.assertIn(
            f"gcloud storage buckets add-iam-policy-binding gs://media-bucket --project {_PROJECT} "
            f"--member={_RUNTIME} --role=roles/storage.objectUser",
            section,
        )
        self.assertIn(
            f"gcloud iam service-accounts add-iam-policy-binding linebot-runtime@{_PROJECT}.iam.gserviceaccount.com "
            f"--project {_PROJECT} --member={_RUNTIME} --role=roles/iam.serviceAccountTokenCreator",
            section,
        )
        self.assertIn(
            f"gcloud tasks queues add-iam-policy-binding linebot-events --project {_PROJECT} "
            f"--location us-west1 --member={_RUNTIME} --role=roles/cloudtasks.enqueuer",
            section,
        )
        # The Colab queue comes with the first deploy with COLAB_ENABLED=true.
        self.assertIn("queue linebot-colab (us-west1) does not exist yet", section)
        # No secret exists yet: the secrets step grants the accessor role.
        self.assertIn("secret linebot-dotenv does not exist yet", section)
        grants = [line for line in self.out.splitlines() if "add-iam-policy-binding" in line]
        self.assertFalse([line for line in grants if "roles/editor" in line or "Admin" in line])

    def test_staging_bucket_is_regional_standard_private_and_short_lived(self) -> None:
        section = self.section(self.out, "staging-bucket")

        self.assertIn(f"gcloud storage buckets create {_STAGING} --project {_PROJECT}", section)
        for flag in (
            "--location=us-west1",
            "--default-storage-class=STANDARD",
            "--uniform-bucket-level-access",
            "--public-access-prevention",
            "--soft-delete-duration=0",
            "infra/gcp/lifecycle-delete-3d.json",
        ):
            self.assertIn(flag, section)

    def test_builder_sa_can_push_images_write_logs_and_read_the_source(self) -> None:
        section = self.section(self.out, "builder-sa")

        self.assertIn(f"gcloud iam service-accounts create linebot-builder --project {_PROJECT}", section)
        self.assertIn(
            "gcloud artifacts repositories add-iam-policy-binding linebot-cloud-agent "
            f"--project {_PROJECT} --location us-west1 --member={_BUILDER} "
            "--role=roles/artifactregistry.writer",
            section,
        )
        self.assertIn(f"--member={_BUILDER} --role=roles/logging.logWriter", section)
        self.assertIn(
            f"gcloud storage buckets add-iam-policy-binding {_STAGING} --project {_PROJECT} "
            f"--member={_BUILDER} --role=roles/storage.objectViewer",
            section,
        )

    def test_purge_clears_soft_delete_before_deleting_and_warns(self) -> None:
        commands = [
            line for line in self.section(self.out, "purge-cloudbuild").splitlines() if "[dry-run]" in line
        ]

        self.assertEqual(len(commands), 3, commands)
        self.assertIn(f"gcloud storage buckets update {_LEGACY} --project {_PROJECT} --clear-soft-delete", commands[0])
        self.assertIn(f"gcloud storage rm {_LEGACY}/source/\\*\\*", commands[1])
        self.assertIn("infra/gcp/lifecycle-delete-1d.json", commands[2])
        self.assertIn("PERMANENTLY deletes", self.result.stderr)
        self.assertIn("58 object(s)", self.out)

    def test_secrets_pipes_the_blob_into_an_automatic_secret(self) -> None:
        section = self.section(self.out, "secrets")

        self.assertRegex(
            section,
            r"\[dry-run\] python3 scripts/envfile.py to-dotenv --file \S+/deploy.env \| gcloud secrets "
            rf"create linebot-dotenv --project {_PROJECT} --replication-policy=automatic --data-file=-",
        )
        self.assertIn(
            f"gcloud secrets add-iam-policy-binding linebot-dotenv --project {_PROJECT} "
            f"--member={_RUNTIME} --role=roles/secretmanager.secretAccessor",
            section,
        )
        for forbidden in ("--locations", "--version-destroy-ttl", "user-managed"):
            self.assertNotIn(forbidden, self.out)
        self.assertNotIn("line-secret-do-not-print", self.out + self.result.stderr)

    def test_ar_cleanup_starts_in_dry_run_mode(self) -> None:
        section = self.section(self.out, "ar-cleanup")

        self.assertIn(
            "gcloud artifacts repositories set-cleanup-policies linebot-cloud-agent "
            f"--project {_PROJECT} --location us-west1",
            section,
        )
        self.assertIn("infra/gcp/ar-cleanup.json --dry-run", section)
        self.assertNotIn("--no-dry-run", self.out)
        # Enforcing it caps rollback at the 2 newest images; say so up front.
        self.assertIn("rollback reaches back 2 images", section)

    def test_budget_is_console_steps_only(self) -> None:
        section = self.section(self.out, "budget")

        self.assertNotIn("[dry-run]", section)
        self.assertNotIn("gcloud billing", section)
        for text in ("30 TWD", "50%, 90%, 100% of actual spend", "100% of forecasted spend", "no spend cap"):
            self.assertIn(text, section)

    def test_summary_lists_what_needs_a_person(self) -> None:
        section = self.section(self.out, "summary")

        for text in (
            "Rotate every secret",
            "Budget alert",
            "Webhook redelivery",
            "roles/editor",
        ):
            self.assertIn(text, section)


class ApplyTests(ProvisionHarness):
    def test_apply_provisions_the_fake_project(self) -> None:
        result = self.provision("--apply")

        self.assertEqual(result.returncode, 0, result.stderr)
        state = self.state
        self.assertLessEqual(
            {"cloudtasks.googleapis.com", "secretmanager.googleapis.com"}, set(state["apis"])
        )
        # No network or VM step: Compute Engine is never touched.
        self.assertEqual(state["instances"], fresh_project()["instances"])
        self.assertFalse([call for call in self.calls() if call.startswith("compute ")])
        self.assertEqual(
            sorted(state["service_accounts"]),
            [f"linebot-builder@{_PROJECT}.iam.gserviceaccount.com", f"linebot-runtime@{_PROJECT}.iam.gserviceaccount.com"],
        )
        bindings = sorted(tuple(binding) for binding in state["bindings"])
        self.assertEqual(
            bindings,
            sorted(
                [
                    (_PROJECT, "roles/datastore.user", _RUNTIME),
                    ("gs://media-bucket", "roles/storage.objectUser", _RUNTIME),
                    (
                        f"linebot-runtime@{_PROJECT}.iam.gserviceaccount.com",
                        "roles/iam.serviceAccountTokenCreator",
                        _RUNTIME,
                    ),
                    ("linebot-events", "roles/cloudtasks.enqueuer", _RUNTIME),
                    ("linebot-dotenv", "roles/secretmanager.secretAccessor", _RUNTIME),
                    ("linebot-cloud-agent", "roles/artifactregistry.writer", _BUILDER),
                    (_PROJECT, "roles/logging.logWriter", _BUILDER),
                    (_STAGING, "roles/storage.objectViewer", _BUILDER),
                ]
            ),
        )
        staging = state["buckets"][_STAGING]
        self.assertEqual(staging["location"], "US-WEST1")
        self.assertEqual(staging["soft_delete_policy"]["retentionDurationSeconds"], "0")
        self.assertEqual(staging["lifecycle_config"]["rule"][0]["condition"], {"age": 3})
        # The secret holds exactly the blob, stored with automatic replication.
        [version] = state["secrets"]["linebot-dotenv"]
        self.assertEqual(
            base64.b64decode(version["data"]).decode("utf-8"),
            envfile.to_dotenv(envfile.parse_env_file(self.env_file)),
        )
        self.assertEqual(state["secret_creates"], ["automatic"])
        self.assertTrue(state["repos"]["linebot-cloud-agent"]["cleanupPolicyDryRun"])
        # Without --yes-purge the purge is skipped and said so.
        self.assertEqual(state["cloudbuild_objects"], 58)
        self.assertIn("skipped purge-cloudbuild", result.stderr)
        self.assertNotIn("line-secret-do-not-print", result.stdout + result.stderr)

    def test_a_second_apply_changes_nothing(self) -> None:
        self.assertEqual(self.provision("--apply", "--yes-purge").returncode, 0)
        before = self.state

        result = self.provision("--apply", "--yes-purge")

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(self.mutating_calls(), [])
        self.assertNotIn("[run]", result.stdout)
        self.assertEqual(self.state, before)

    def test_existing_resources_are_described_not_recreated(self) -> None:
        state = fresh_project()
        state["apis"] += ["cloudtasks.googleapis.com", "secretmanager.googleapis.com"]
        state["service_accounts"] = [
            f"linebot-runtime@{_PROJECT}.iam.gserviceaccount.com",
            f"linebot-builder@{_PROJECT}.iam.gserviceaccount.com",
        ]
        state["secrets"] = {"linebot-dotenv": [{"id": 1, "state": "ENABLED", "data": ""}]}
        self.set_state(state)

        result = self.provision("--apply")

        self.assertEqual(result.returncode, 0, result.stderr)
        mutating = self.mutating_calls()
        for resource in (
            "services enable",
            "service-accounts create",
            "secrets create",
        ):
            self.assertFalse([call for call in mutating if resource in call], mutating)
        self.assertIn("ok: secret linebot-dotenv exists", result.stdout)

    def test_drifted_staging_bucket_is_corrected(self) -> None:
        state = fresh_project()
        state["buckets"][_STAGING] = {
            "location": "US-WEST1",
            "soft_delete_policy": {"retentionDurationSeconds": "604800"},
        }
        self.set_state(state)

        result = self.provision("--apply", "--step", "staging-bucket")

        self.assertEqual(result.returncode, 0, result.stderr)
        bucket = self.state["buckets"][_STAGING]
        self.assertEqual(bucket["soft_delete_policy"]["retentionDurationSeconds"], "0")
        self.assertEqual(bucket["lifecycle_config"]["rule"][0]["condition"], {"age": 3})
        self.assertFalse([call for call in self.calls() if "buckets create" in call])

    def test_runtime_sa_can_read_every_secret_that_exists(self) -> None:
        state = fresh_project()
        state["secrets"] = {
            "linebot-dotenv": [{"id": 1, "state": "ENABLED", "data": ""}],
            "linebot-colab-adc": [{"id": 1, "state": "ENABLED", "data": ""}],
        }
        self.set_state(state)

        result = self.provision("--apply", "--step", "runtime-sa")

        self.assertEqual(result.returncode, 0, result.stderr)
        accessor = [call for call in self.calls() if "roles/secretmanager.secretAccessor" in call]
        self.assertEqual(
            sorted(call.split()[2] for call in accessor), ["linebot-colab-adc", "linebot-dotenv"]
        )

    def test_the_colab_queue_is_granted_once_the_deploy_made_it(self) -> None:
        state = fresh_project()
        state["queues"] = ["linebot-events", "linebot-colab"]
        self.set_state(state)

        result = self.provision("--apply", "--step", "runtime-sa")

        self.assertEqual(result.returncode, 0, result.stderr)
        enqueuer = [call for call in self.calls() if "roles/cloudtasks.enqueuer" in call and "add-iam" in call]
        self.assertEqual(sorted(call.split()[3] for call in enqueuer), ["linebot-colab", "linebot-events"])

    def test_the_queue_location_follows_the_app(self) -> None:
        # A missing key means us-west1, an empty one the Cloud Run region;
        # read after _ENV's multi-line value.
        for lines, location in (("", "us-west1"), ("CLOUD_RUN_REGION=us-central1\nCLOUD_TASKS_LOCATION=\n", "us-central1")):
            with self.subTest(location=location):
                self.env_file.write_text(_ENV + lines, encoding="utf-8")

                result = self.provision("--step", "runtime-sa")

                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertIn(
                    f"gcloud tasks queues add-iam-policy-binding linebot-events --project {_PROJECT} "
                    f"--location {location} --member={_RUNTIME}",
                    result.stdout,
                )
                self.assertIn(f"queue linebot-colab ({location}) does not exist yet", result.stdout)

    def test_no_media_bucket_means_no_bucket_grant(self) -> None:
        self.env_file.write_text(f"GCP_PROJECT_ID={_PROJECT}\nGCS_BUCKET_NAME=\n", encoding="utf-8")

        result = self.provision("--step", "runtime-sa")

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("GCS_BUCKET_NAME is empty", result.stdout)
        self.assertNotIn("roles/storage.objectUser", result.stdout)


class DisabledApiTests(ProvisionHarness):
    """A project whose Cloud Tasks and Secret Manager APIs are still off:
    the apis step turns them on, and until then every call to them fails
    with PERMISSION_DENIED, reason SERVICE_DISABLED."""

    _APIS = ["cloudtasks.googleapis.com", "secretmanager.googleapis.com"]

    def setUp(self) -> None:
        super().setUp()
        state = fresh_project()
        state["disabled_apis"] = list(self._APIS)
        self.set_state(state)

    def test_a_dry_run_reads_them_as_nothing_there_yet(self) -> None:
        result = self.provision()

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertNotIn("[fail]", result.stderr)
        self.assertIn(
            f"[dry-run] gcloud services enable {' '.join(self._APIS)} --project {_PROJECT}",
            self.section(result.stdout, "apis"),
        )
        runtime = self.section(result.stdout, "runtime-sa")
        for api in self._APIS:
            self.assertIn(
                f"{api} is not enabled yet, so this reads as: does not exist yet (the apis step enables the API)",
                runtime,
            )
        self.assertIn("queue linebot-events (us-west1) does not exist yet", runtime)
        self.assertIn("secret linebot-dotenv does not exist yet", runtime)
        self.assertIn("[dry-run] python3 scripts/envfile.py to-dotenv", self.section(result.stdout, "secrets"))
        # A lookup's stderr goes to a file: gcloud's "enable and retry?
        # (y/N)" would wait there unseen.
        self.assertEqual(self.state.get("enable_api_prompts", []), [])
        self.assertEqual(self.mutating_calls(), [])

    def test_apply_stops_until_the_apis_step_has_enabled_them(self) -> None:
        result = self.provision("--apply", "--step", "runtime-sa")

        self.assertEqual(result.returncode, 1)
        self.assertIn("reason: SERVICE_DISABLED", result.stderr)
        self.assertIn(
            f"cloudtasks.googleapis.com is not enabled in project {_PROJECT}; run --step apis --apply first",
            result.stderr,
        )
        self.assertNotIn("does not exist yet", result.stdout)
        self.assertEqual(self.state.get("enable_api_prompts", []), [])

        self.assertEqual(self.provision("--apply", "--step", "apis").returncode, 0)
        result = self.provision("--apply", "--step", "runtime-sa")

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn(
            ["linebot-events", "roles/cloudtasks.enqueuer", _RUNTIME], self.state["bindings"]
        )

    def test_the_mint_script_never_waits_on_a_hidden_prompt(self) -> None:
        result = self.run_script(["bash", str(_MINT), "--env-file", str(self.env_file)], timeout=120)

        self.assertEqual(result.returncode, 1)
        self.assertIn("reason: SERVICE_DISABLED", result.stderr)
        self.assertEqual(self.state.get("enable_api_prompts", []), [])


class ColabSecretTests(ProvisionHarness):
    _SECRET = "linebot-colab-adc"
    _ACCESSOR = "roles/secretmanager.secretAccessor"

    def _with_secret(self, **overrides) -> dict:
        state = fresh_project()
        state["service_accounts"] = [f"linebot-runtime@{_PROJECT}.iam.gserviceaccount.com"]
        state["secrets"] = {self._SECRET: [{"id": 1, "state": "ENABLED", "data": ""}]}
        state.update(overrides)
        return state

    def test_it_lets_the_runtime_account_read_the_colab_secret_once(self) -> None:
        self.set_state(self._with_secret())

        result = self.provision("--apply", "--step", "colab-secret")

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(self.state["bindings"], [[self._SECRET, self._ACCESSOR, _RUNTIME]])
        # Only the binding: the secret (a person's token) is never created here.
        self.assertEqual(
            self.mutating_calls(),
            [
                f"secrets add-iam-policy-binding {self._SECRET} --project {_PROJECT} "
                f"--member={_RUNTIME} --role={self._ACCESSOR}"
            ],
        )
        self.assertIn("/secrets/colab/adc.json while COLAB_ENABLED=true", result.stdout)

        again = self.provision("--apply", "--step", "colab-secret")

        self.assertEqual(again.returncode, 0, again.stderr)
        self.assertEqual(self.mutating_calls(), [])
        self.assertIn(f"ok: {self._ACCESSOR} for {_RUNTIME} on {self._SECRET}", again.stdout)

    def test_without_the_secret_it_points_to_the_mint_script(self) -> None:
        for args in (("--step", "colab-secret"), ("--apply", "--step", "colab-secret")):
            with self.subTest(args=args):
                result = self.provision(*args)

                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertIn("./scripts/mint_colab_adc.sh --apply creates it", result.stdout)
                self.assertEqual(self.mutating_calls(), [])
                self.assertNotIn(self._SECRET, self.state.get("secrets", {}))

    def test_a_dry_run_plans_the_binding(self) -> None:
        self.set_state(self._with_secret())

        result = self.provision("--step", "colab-secret")

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn(
            f"[dry-run] gcloud secrets add-iam-policy-binding {self._SECRET} --project {_PROJECT} "
            f"--member={_RUNTIME} --role={self._ACCESSOR}",
            result.stdout,
        )
        self.assertEqual(self.mutating_calls(), [])

    def test_a_missing_runtime_account_is_named_not_granted(self) -> None:
        self.set_state(self._with_secret(service_accounts=[]))

        result = self.provision("--apply", "--step", "colab-secret")

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("run --step runtime-sa --apply", result.stderr)
        self.assertEqual(self.mutating_calls(), [])


_COMPUTE = "843253221534-compute@developer.gserviceaccount.com"
_EDITOR = [_PROJECT, "roles/editor", f"serviceAccount:{_COMPUTE}"]


class DropEditorTests(ProvisionHarness):
    """Phase 9: roles/editor leaves the Compute Engine default service account."""

    _REMOVE = (
        f"projects remove-iam-policy-binding {_PROJECT} --member=serviceAccount:{_COMPUTE} "
        "--role=roles/editor --condition=None"
    )

    @staticmethod
    def _taken_over(**overrides) -> dict:
        """Provisioned and deployed: linebot-runtime and linebot-builder do the work."""
        runtime = f"linebot-runtime@{_PROJECT}.iam.gserviceaccount.com"
        state = fresh_project()
        state["buckets"][_STAGING] = {"location": "US-WEST1"}
        state.update(
            service_accounts=[runtime, f"linebot-builder@{_PROJECT}.iam.gserviceaccount.com"],
            bindings=[list(_EDITOR), [_PROJECT, "roles/datastore.user", _RUNTIME]],
            service_account_on_service=runtime,
            run_services=[{"name": "linebot-cloud-agent", "region": "us-west1", "account": runtime}],
            # The deploy's cron job, its token in a header.
            scheduler_jobs={"linebot-cloud-agent-cron": {"httpTarget": {"headers": "X-Cron-Token=t"}}},
            # A VM with an account of its own.
            instances=[
                {
                    "name": "own-vm",
                    "zone": "us-west1-b",
                    "status": "RUNNING",
                    "serviceAccounts": [{"email": f"own-vm@{_PROJECT}.iam.gserviceaccount.com"}],
                }
            ],
        )
        state.update(overrides)
        return state

    def test_a_dry_run_plans_the_removal_after_the_simulator_and_with_the_rollback(self) -> None:
        self.set_state(self._taken_over())

        result = self.provision("--step", "drop-editor")

        self.assertEqual(result.returncode, 0, result.stderr)
        out = self.section(result.stdout, "drop-editor")
        simulator = out.index(f"https://console.cloud.google.com/iam-admin/iam?project={_PROJECT}")
        rollback = out.index(
            f"Roll back: gcloud projects add-iam-policy-binding {_PROJECT} --member=serviceAccount:{_COMPUTE} "
            "--role=roles/editor --condition=None"
        )
        planned = out.index(f"[dry-run] gcloud {self._REMOVE}")
        self.assertLess(simulator, planned)
        self.assertLess(rollback, planned)
        self.assertIn("Test changes", out)
        self.assertEqual(self.mutating_calls(), [])
        self.assertIn(_EDITOR, self.state["bindings"])
        # Listing the jobs never copies their headers into gcloud's log.
        self.assertEqual(self.state.get("file_logged", []), [])

    def test_apply_removes_only_the_role_and_a_second_run_has_nothing_to_do(self) -> None:
        self.set_state(self._taken_over())

        result = self.provision("--apply", "--step", "drop-editor")

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(self.mutating_calls(), [self._REMOVE])
        self.assertEqual(self.state["bindings"], [[_PROJECT, "roles/datastore.user", _RUNTIME]])
        # The account stays: the step only ever describes service accounts.
        accounts = [call for call in self.calls() if call.startswith("iam service-accounts")]
        self.assertTrue(accounts)
        self.assertFalse([call for call in accounts if not call.startswith("iam service-accounts describe")])
        self.assertIn("never delete it", result.stdout)

        again = self.provision("--apply", "--step", "drop-editor")

        self.assertEqual(again.returncode, 0, again.stderr)
        self.assertIn(f"ok: {_COMPUTE} does not hold roles/editor; nothing to do", again.stdout)
        self.assertEqual(self.mutating_calls(), [])

    def test_it_refuses_until_the_replacements_are_in_place(self) -> None:
        runtime = f"linebot-runtime@{_PROJECT}.iam.gserviceaccount.com"
        builder = f"linebot-builder@{_PROJECT}.iam.gserviceaccount.com"
        buckets = fresh_project()["buckets"]
        for change, error in (
            ({"service_accounts": [builder]}, f"{runtime} does not exist: run --step runtime-sa --apply"),
            ({"service_accounts": [runtime]}, f"{builder} does not exist: run --step builder-sa --apply"),
            ({"buckets": buckets}, f"{_STAGING} is not a bucket of {_PROJECT}"),
            (
                {"service_account_on_service": _COMPUTE},
                f"Cloud Run service linebot-cloud-agent runs as {_COMPUTE}, not {runtime}",
            ),
            ({"service_missing": True}, "Cloud Run service linebot-cloud-agent does not exist in us-west1"),
        ):
            with self.subTest(error=error):
                self.set_state(self._taken_over(**change))

                result = self.provision("--apply", "--step", "drop-editor")

                self.assertEqual(result.returncode, 1)
                self.assertIn(f"Not removing roles/editor from {_COMPUTE}:", result.stderr)
                self.assertIn(error, result.stderr)
                self.assertEqual(self.mutating_calls(), [])
                self.assertIn(_EDITOR, self.state["bindings"])

    def test_it_names_and_refuses_everything_else_that_runs_as_the_account(self) -> None:
        runtime = f"linebot-runtime@{_PROJECT}.iam.gserviceaccount.com"
        state = self._taken_over()
        state["run_services"] += [
            {"name": "legacy", "region": "us-central1", "account": _COMPUTE},
            # No account named: Cloud Run uses the default one.
            {"name": "unnamed", "region": "asia-east1", "account": ""},
            {"name": "own-account", "region": "us-west1", "account": runtime},
        ]
        state["scheduler_jobs"]["nightly"] = {
            "httpTarget": {"oidcToken": {"serviceAccountEmail": _COMPUTE}}
        }
        state["instances"].append({**_OLD_VM, "serviceAccounts": [{"email": _COMPUTE}]})
        self.set_state(state)

        for args in (("--step", "drop-editor"), ("--apply", "--step", "drop-editor")):
            with self.subTest(args=args):
                result = self.provision(*args)

                self.assertEqual(result.returncode, 1)
                for user in (
                    "Cloud Run service legacy (us-central1)",
                    "Cloud Run service unnamed (asia-east1)",
                    "Cloud Scheduler job nightly",
                    "VM instance-gcp-banana (us-west1-b, TERMINATED)",
                ):
                    self.assertIn(f"- {user} runs as {_COMPUTE}", result.stderr)
                for fine in ("linebot-cloud-agent (", "own-account", "linebot-cloud-agent-cron", "own-vm"):
                    self.assertNotIn(fine, result.stderr)
                # Refused before the removal is even planned.
                self.assertNotIn("remove-iam-policy-binding", result.stdout)
                self.assertEqual(self.mutating_calls(), [])
                self.assertIn(_EDITOR, self.state["bindings"])

    def test_a_disabled_compute_api_reads_as_no_vm(self) -> None:
        """Regression (review finding 6): compute is not in REQUIRED_APIS, so
        a project that turned the Compute Engine API off after deleting its
        last VM stopped the step as a failed read."""
        compute_wording = (
            "Some requests did not succeed:\n - Compute Engine API has not been used in project "
            "843253221534 before or it is disabled. Enable it by visiting "
            "https://console.developers.google.com/apis/api/compute.googleapis.com/overview"
            "?project=843253221534 then retry."
        )
        for name, change in (
            ("SERVICE_DISABLED", {"disabled_apis": ["compute.googleapis.com"]}),
            ("compute wording", {"fail": {"compute instances list": compute_wording}}),
        ):
            with self.subTest(error=name):
                self.set_state(self._taken_over(**change))

                result = self.provision("--apply", "--step", "drop-editor")

                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertIn(
                    f"compute.googleapis.com is not enabled in project {_PROJECT}, so this reads as: none",
                    result.stdout,
                )
                self.assertEqual(self.mutating_calls(), [self._REMOVE])
                # Reading it never turns the API on.
                self.assertEqual(self.state.get("disabled_apis", []), change.get("disabled_apis", []))

    def test_another_disabled_api_still_stops_it(self) -> None:
        run_off = (
            "PERMISSION_DENIED: Cloud Run Admin API has not been used in project 843253221534 "
            "before or it is disabled.\n  reason: SERVICE_DISABLED\n  service: run.googleapis.com"
        )
        self.set_state(self._taken_over(fail={"run services list": run_off}))

        result = self.provision("--apply", "--step", "drop-editor")

        self.assertEqual(result.returncode, 1)
        self.assertIn("not with NOT_FOUND", result.stderr)
        self.assertEqual(self.mutating_calls(), [])
        self.assertIn(_EDITOR, self.state["bindings"])

    def test_a_read_error_stops_it(self) -> None:
        denied = "PERMISSION_DENIED: Permission denied."
        for prefix in ("projects get-iam-policy", "run services list", "scheduler jobs list", "compute instances list"):
            with self.subTest(prefix=prefix):
                self.set_state(self._taken_over(fail={prefix: denied}))

                result = self.provision("--apply", "--step", "drop-editor")

                self.assertEqual(result.returncode, 1)
                self.assertIn("not with NOT_FOUND", result.stderr)
                self.assertNotIn("nothing to do", result.stdout)
                self.assertEqual(self.mutating_calls(), [])
                self.assertIn(_EDITOR, self.state["bindings"])


class SafetyTests(ProvisionHarness):
    def test_purge_needs_yes_purge(self) -> None:
        result = self.provision("--apply", "--step", "purge-cloudbuild")

        self.assertEqual(result.returncode, 1)
        self.assertIn("rerun with --yes-purge", result.stderr)
        self.assertIn("PERMANENTLY deletes", result.stderr)
        self.assertEqual(self.mutating_calls(), [])
        self.assertEqual(self.state["cloudbuild_objects"], 58)

    def test_purge_with_yes_purge_clears_soft_delete_first(self) -> None:
        result = self.provision("--apply", "--step", "purge-cloudbuild", "--yes-purge")

        self.assertEqual(result.returncode, 0, result.stderr)
        mutating = self.mutating_calls()
        self.assertEqual(len(mutating), 3, mutating)
        self.assertIn("--clear-soft-delete", mutating[0])
        self.assertTrue(mutating[1].startswith(f"storage rm {_LEGACY}/source/**"))
        self.assertIn("lifecycle-delete-1d.json", mutating[2])
        bucket = self.state["buckets"][_LEGACY]
        self.assertEqual(self.state["cloudbuild_objects"], 0)
        self.assertEqual(bucket["soft_delete_policy"]["retentionDurationSeconds"], "0")
        self.assertEqual(bucket["lifecycle_config"]["rule"][0]["condition"], {"age": 1})

    def test_purge_without_the_old_bucket_does_nothing(self) -> None:
        state = fresh_project()
        del state["buckets"][_LEGACY]
        self.set_state(state)

        result = self.provision("--apply", "--step", "purge-cloudbuild")

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("nothing to purge", result.stdout)
        self.assertEqual(self.mutating_calls(), [])

    def test_ar_cleanup_is_enforced_only_after_its_dry_run(self) -> None:
        refused = self.provision("--apply", "--step", "ar-cleanup-enforce")
        self.assertEqual(refused.returncode, 1)
        self.assertIn("run --step ar-cleanup --apply first", refused.stderr)

        self.assertEqual(self.provision("--apply", "--step", "ar-cleanup").returncode, 0)
        self.assertTrue(self.state["repos"]["linebot-cloud-agent"]["cleanupPolicyDryRun"])
        enforced = self.provision("--apply", "--step", "ar-cleanup-enforce")

        self.assertEqual(enforced.returncode, 0, enforced.stderr)
        self.assertIn("--no-dry-run", enforced.stdout)
        repo = self.state["repos"]["linebot-cloud-agent"]
        self.assertFalse(repo["cleanupPolicyDryRun"])
        self.assertEqual(sorted(repo["cleanupPolicies"]), ["delete-older-than-1d", "keep-2-most-recent"])
        # Running either step again changes nothing.
        self.assertEqual(self.provision("--apply", "--step", "ar-cleanup").returncode, 0)
        self.assertEqual(self.mutating_calls(), [])

    def test_the_all_steps_run_never_enforces_ar_cleanup(self) -> None:
        result = self.provision("--apply")

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertNotIn("--no-dry-run", result.stdout)
        self.assertTrue(self.state["repos"]["linebot-cloud-agent"]["cleanupPolicyDryRun"])

    def test_ar_cleanup_warns_when_more_revisions_are_kept_than_it_keeps_images(self) -> None:
        self.env_file.write_text(_ENV + "DEPLOY_KEEP_REVISIONS=3\n", encoding="utf-8")

        result = self.provision("--step", "ar-cleanup")

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("DEPLOY_KEEP_REVISIONS / DEPLOY_KEEP_IMAGES is above 2", result.stderr)

    def test_secrets_step_refuses_values_python_dotenv_would_expand(self) -> None:
        self.env_file.write_text(
            f"GCP_PROJECT_ID={_PROJECT}\nTOKEN=abc${{HOME}}do-not-print\n", encoding="utf-8"
        )

        result = self.provision("--apply", "--step", "secrets")

        self.assertEqual(result.returncode, 1)
        self.assertIn("TOKEN contains '${'", result.stderr)
        self.assertNotIn("do-not-print", result.stdout + result.stderr)
        self.assertNotIn("secrets", self.state)

    def test_secrets_step_names_a_line_missing_its_equals_sign_by_number_only(self) -> None:
        # The base64 padding makes the whole line read as KEY=VALUE, the
        # token in the key.
        self.env_file.write_text(
            _ENV + "COLAB_TUNNEL_TOKEN eyJhIjoiZG8tbm90LXByaW50In0==\n", encoding="utf-8"
        )

        result = self.provision("--apply", "--step", "secrets")

        self.assertEqual(result.returncode, 1)
        self.assertIn("line 7: not KEY=VALUE", result.stderr)
        self.assertIn("Fix the keys and lines listed above", result.stderr)
        self.assertNotIn("eyJhIjoi", result.stdout + result.stderr)
        self.assertNotIn("secrets", self.state)

    def test_secrets_step_reads_an_env_file_that_starts_with_a_byte_order_mark(self) -> None:
        # Kept, the mark hid GCP_PROJECT_ID (provisioning fell back to
        # gcloud's default project) and made line 1 no KEY=VALUE.
        self.env_file.write_bytes(("\ufeff" + _ENV).encode("utf-8"))

        result = self.provision("--apply", "--step", "secrets")

        self.assertEqual(result.returncode, 0, result.stderr)
        creates = [call for call in self.calls() if call.startswith("secrets create")]
        self.assertEqual(len(creates), 1)
        self.assertIn(f"--project {_PROJECT}", creates[0])
        blob = base64.b64decode(self.state["secrets"]["linebot-dotenv"][0]["data"]).decode("utf-8")
        self.assertTrue(blob.startswith("GCS_BUCKET_NAME='media-bucket'\n"), blob.splitlines()[0])

    def test_a_read_error_is_never_reported_as_absent(self) -> None:
        # Only NOT_FOUND means missing: a permission or outage error stops the
        # step instead of printing "ok" over a bucket or secret that is still
        # there.
        denied = "PERMISSION_DENIED: Permission denied (or it may not exist)."
        not_found_only = "not with NOT_FOUND"
        cases = (
            (
                "purge-cloudbuild",
                {f"storage buckets describe {_LEGACY}": "HTTPError 403: no storage.buckets.get access"},
                not_found_only,
                "nothing to purge",
            ),
            (
                "purge-cloudbuild",
                {"storage ls": "HTTPError 503: Backend Error"},
                f"Could not list {_LEGACY}/source/",
                "ok: no objects",
            ),
            # The Colab queue's absence (NOT_FOUND in this project) is reported before the secrets.
            ("runtime-sa", {"secrets describe": denied}, not_found_only, "skipped: secret"),
            ("runtime-sa", {"tasks queues describe": denied}, not_found_only, "does not exist yet"),
            ("apis", {"services list": denied}, not_found_only, "gcloud services enable"),
        )
        for step, fail, error, absent_note in cases:
            with self.subTest(step=step, fail=next(iter(fail))):
                state = fresh_project()
                state["fail"] = fail
                self.set_state(state)

                result = self.provision("--apply", "--step", step, "--yes-purge")

                self.assertEqual(result.returncode, 1)
                self.assertIn(error, result.stderr)
                self.assertNotIn(absent_note, result.stdout)
                self.assertEqual(self.state["cloudbuild_objects"], 58)
                self.assertFalse([call for call in self.calls() if call.startswith("storage rm")])

    def test_a_staging_bucket_outside_the_project_is_refused(self) -> None:
        # Bucket names are global and this one is predictable: a bucket of
        # someone else's must never be adopted, granted to or staged in.
        for name, state_change in (
            ("another project's", {"buckets": {_STAGING: {"location": "US-WEST1", "project": "squatter"}}}),
            ("unreadable", {"fail": {f"storage buckets describe {_STAGING}": "HTTPError 403: no access"}}),
        ):
            with self.subTest(name):
                state = fresh_project()
                if "buckets" in state_change:
                    state["buckets"].update(state_change["buckets"])
                else:
                    state.update(state_change)
                self.set_state(state)

                staged = self.provision("--apply", "--step", "staging-bucket")
                granted = self.provision("--apply", "--step", "builder-sa")

                self.assertEqual(staged.returncode, 1)
                self.assertIn(f"{_STAGING} exists outside project {_PROJECT}", staged.stderr)
                self.assertEqual(granted.returncode, 1)
                self.assertIn(f"{_STAGING} is not a bucket of project {_PROJECT}", granted.stderr)
                self.assertFalse(
                    [b for b in self.state.get("bindings", []) if b[0] == _STAGING]
                )
                self.assertNotIn("soft_delete_policy", self.state["buckets"].get(_STAGING, {}))

    def test_the_env_file_must_parse_before_anything_reads_it(self) -> None:
        # The secrets step's own check, for every step: a key python-dotenv
        # drops would otherwise take its default without a word.
        self.env_file.write_text(_ENV + "GCS_BUCKET_NAME='other-bucket\n", encoding="utf-8")

        result = self.provision("--step", "runtime-sa")

        self.assertEqual(result.returncode, 1)
        self.assertIn("cannot parse the quoted value of GCS_BUCKET_NAME", result.stderr)
        self.assertNotIn("[step]", result.stdout)
        self.assertEqual(self.calls(), [])

    def test_the_policy_files_are_valid(self) -> None:
        policies = json.loads((_ROOT / "infra/gcp/ar-cleanup.json").read_text(encoding="utf-8"))
        by_name = {policy["name"]: policy for policy in policies}
        self.assertEqual(by_name["keep-2-most-recent"]["mostRecentVersions"], {"keepCount": 2})
        self.assertEqual(by_name["delete-older-than-1d"]["condition"]["olderThan"], "1d")
        for name, age in (("lifecycle-delete-3d.json", 3), ("lifecycle-delete-1d.json", 1)):
            rules = json.loads((_ROOT / "infra/gcp" / name).read_text(encoding="utf-8"))["rule"]
            self.assertEqual(rules, [{"action": {"type": "Delete"}, "condition": {"age": age}}])


class ProjectGuardTests(ProvisionHarness):
    """Neither provision_gcp.sh nor mint_colab_adc.sh (which runs its
    colab-secret step) ever falls back to gcloud's default project."""

    # run_script snapshots the env file, which one case makes unreadable.
    rehearse = False
    _RUNS = {"provision": (_SCRIPT, "--apply", "--step", "apis"), "mint": (_MINT,)}

    def setUp(self) -> None:
        super().setUp()
        state = fresh_project()
        state["config_project"] = "someone-elses-proj"
        self.set_state(state)

    def run_both(self, *args: str) -> dict[str, tuple[subprocess.CompletedProcess, list[str]]]:
        """Each script's result and gcloud calls."""
        runs = {}
        for name, (script, *script_args) in self._RUNS.items():
            self.log_path.unlink(missing_ok=True)
            result = subprocess.run(
                ["bash", str(script), "--env-file", str(self.env_file), *script_args, *args],
                capture_output=True,
                text=True,
                env=self.environ,
                timeout=120,
            )
            runs[name] = (result, self.calls())
        return runs

    def test_a_bad_env_file_stops_both_scripts_before_any_gcloud_call(self) -> None:
        cases = (
            ("missing", None, "Env file not found"),
            ("unreadable", b"GCP_PROJECT_ID=test-project\n", "Cannot read"),
            ("not UTF-8", b"GCP_PROJECT_ID=test-project\nBOT_NAME=caf\xe9\n", "UnicodeDecodeError"),
            ("unclosed quote", b"BOT_NAME='oops\nGCP_PROJECT_ID=test-project\n", "quoted value of BOT_NAME"),
            ("no project", b"GCP_PROJECT_ID=\nGCS_BUCKET_NAME=media-bucket\n", "GCP_PROJECT_ID is empty"),
        )
        for name, content, reason in cases:
            with self.subTest(name):
                self.env_file.unlink(missing_ok=True)
                if content is not None:
                    self.env_file.write_bytes(content)
                if name == "unreadable":
                    self.env_file.chmod(0)
                    if os.access(self.env_file, os.R_OK):
                        self.skipTest("running as root: no file is unreadable")
                try:
                    runs = self.run_both()
                finally:
                    if content is not None:
                        self.env_file.chmod(0o600)

                for script, (result, calls) in runs.items():
                    with self.subTest(script=script):
                        self.assertEqual(result.returncode, 1, result.stdout)
                        self.assertIn(reason, result.stderr)
                        self.assertNotIn("someone-elses-proj", result.stdout + result.stderr)
                        self.assertNotIn("[step]", result.stdout)
                        self.assertEqual(calls, [])
                self.assertEqual(self.state, {**fresh_project(), "config_project": "someone-elses-proj"})

    def test_an_explicit_project_must_match_the_env_file(self) -> None:
        for script, (result, calls) in self.run_both("--project-id", "other-project").items():
            with self.subTest(script=script):
                self.assertEqual(result.returncode, 1)
                self.assertIn(
                    f"--project-id other-project does not match GCP_PROJECT_ID={_PROJECT}", result.stderr
                )
                self.assertEqual(calls, [])

    def test_only_an_explicit_project_runs_without_an_env_file(self) -> None:
        self.env_file.unlink()

        runs = self.run_both("--project-id", _PROJECT)

        for script, (result, _calls) in runs.items():
            with self.subTest(script=script):
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertIn(f"project {_PROJECT}", result.stdout)
                self.assertNotIn("someone-elses-proj", result.stdout + result.stderr)
        self.assertIn("Env file not found", runs["provision"][0].stderr)


if __name__ == "__main__":
    unittest.main()
