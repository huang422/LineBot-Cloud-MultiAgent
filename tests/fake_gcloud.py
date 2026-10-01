"""A stateful stand-in for ``gcloud``, for the provisioning and deploy script tests.

The tests put a ``gcloud`` wrapper on PATH that runs this file. State (what
exists in the fake project) lives in the JSON file ``$FAKE_GCLOUD_STATE`` and
changes as the scripts create, update and delete things; every call is
appended to ``$FAKE_GCLOUD_LOG`` as one line of arguments. Only the commands
the scripts use are modelled; anything else succeeds silently. Nothing here
talks to Google.

A missing resource fails like gcloud does, with NOT_FOUND on stderr.
``state["fail"]`` maps a command prefix (its words, without flags, such as
``"secrets describe"``) to the error that command fails with instead, for
permission and outage tests. A command whose output or flags carry a secret
(a payload, a job's headers, ``run deploy --env-vars-file``) and runs without
``CLOUDSDK_CORE_DISABLE_FILE_LOGGING=true`` is recorded in
``state["file_logged"]``: real gcloud copies its output, and every parsed
flag, into ``~/.config/gcloud/logs``.

``state["disabled_apis"]`` lists APIs that are not enabled: their commands
fail as gcloud does (PERMISSION_DENIED, reason SERVICE_DISABLED) until
``services enable`` turns them on. Such a command run without
``CLOUDSDK_CORE_DISABLE_PROMPTS`` is recorded in
``state["enable_api_prompts"]``: real gcloud first asks "enable and retry?
(y/N)" on stderr and waits for an answer.

``auth application-default login`` writes ``state["adc"]`` (a fake
authorized_user credential unless a test sets another) to
``$CLOUDSDK_CONFIG/application_default_credentials.json`` and records the
config directory, scopes and flags in ``state["adc_logins"]``. Without
``CLOUDSDK_CONFIG`` it refuses: real gcloud would overwrite the person's own
ADC. Every other command run with ``CLOUDSDK_CONFIG`` set is recorded in
``state["scratch_config_calls"]``.

Every call starts a new python, so the fake needs nothing beyond the
standard library (the script tests run it with ``python -S``, skipping
site-packages) and imports base64 and hashlib only where it uses them.
"""

from __future__ import annotations

import json
import os
import sys

# Flags the scripts pass as `--flag value` (two arguments).
_VALUE_FLAGS = frozenset(
    {
        "--project",
        "--location",
        "--region",
        "--secret",
        "--service",
        "--image",
        "--config",
        "--substitutions",
        "--description",
        "--timeout",
        "--memory",
        "--cpu",
        "--min-instances",
        "--max-instances",
        "--max",
        "--env-vars-file",
        "--service-account",
        "--platform",
    }
)


def _split(argv: list[str]) -> tuple[list[str], dict[str, str], set[str]]:
    """Positional words, ``--flag=value``/``--flag value`` pairs, bare flags."""
    words: list[str] = []
    values: dict[str, str] = {}
    bare: set[str] = set()
    index = 0
    while index < len(argv):
        arg = argv[index]
        if arg in _VALUE_FLAGS and index + 1 < len(argv):
            values[arg] = argv[index + 1]
            index += 2
            continue
        if arg.startswith("--"):
            name, sep, value = arg.partition("=")
            if sep:
                values[name] = value
            else:
                bare.add(name)
        else:
            words.append(arg)
        index += 1
    return words, values, bare


def _resource(words: list[str], verb: str) -> str:
    """The resource an IAM verb applies to: the word after it."""
    index = words.index(verb)
    return words[index + 1] if index + 1 < len(words) else ""


def _policy(state: dict, resource: str) -> str:
    grouped: dict[str, list[str]] = {}
    for bound, role, member in state.get("bindings", []):
        if bound == resource:
            grouped.setdefault(role, []).append(member)
    return json.dumps(
        {"bindings": [{"role": role, "members": members} for role, members in grouped.items()]}
    )


def _secret_versions(state: dict, name: str) -> list[dict]:
    return state.setdefault("secrets", {}).setdefault(name, [])


def _live(versions: list[dict]) -> list[dict]:
    return [version for version in versions if version["state"] != "DESTROYED"]


def _version_name(secret: str, version: dict) -> str:
    return f"projects/843253221534/secrets/{secret}/versions/{version['id']}"


def _digest(image: str) -> str:
    """The digest an image tag gets when the fake build pushes it."""
    import hashlib

    return "sha256:" + hashlib.sha256(image.rsplit(":", 1)[-1].encode()).hexdigest()


# Commands whose output carries a secret (the payload, a job's headers).
_SECRET_OUTPUT = (("secrets", "versions", "access"), ("scheduler", "jobs"))

# The API behind each command group, and its title in gcloud's errors.
_GROUP_APIS = {
    "tasks": ("cloudtasks.googleapis.com", "Cloud Tasks API"),
    "secrets": ("secretmanager.googleapis.com", "Secret Manager API"),
    "compute": ("compute.googleapis.com", "Compute Engine API"),
}


def _service_disabled(command: str, api: str, title: str) -> tuple[int, str, str]:
    """What gcloud prints for a call to an API the project has not enabled."""
    number = "843253221534"
    url = f"https://console.developers.google.com/apis/api/{api}/overview?project={number}"
    message = (
        f"{title} has not been used in project {number} before or it is disabled. "
        f"Enable it by visiting {url} then retry. If you enabled this API recently, "
        "wait a few minutes for the action to propagate to our systems and retry."
    )
    return 1, "", (
        f"ERROR: (gcloud.{command}) PERMISSION_DENIED: {message} This command is authenticated "
        "as deployer@example.com which is the active account specified by the [core/account] property.\n"
        "- '@type': type.googleapis.com/google.rpc.ErrorInfo\n"
        "  domain: googleapis.com\n"
        "  metadata:\n"
        f"    activationUrl: {url}\n"
        f"    consumer: projects/{number}\n"
        f"    containerInfo: '{number}'\n"
        f"    service: {api}\n"
        f"    serviceTitle: {title}\n"
        "  reason: SERVICE_DISABLED\n"
        "- '@type': type.googleapis.com/google.rpc.LocalizedMessage\n"
        "  locale: en-US\n"
        f"  message: {message}\n"
        "- '@type': type.googleapis.com/google.rpc.Help\n"
        "  links:\n"
        "  - description: Google developers console API activation\n"
        f"    url: {url}"
    )

# What `gcloud auth application-default login` writes: an authorized_user
# credential of the Cloud SDK's OAuth client.
FAKE_ADC = {
    "account": "",
    "client_id": "764086051850-fake.apps.googleusercontent.com",
    "client_secret": "fake-client-secret",
    "refresh_token": "1//fake-refresh-token-do-not-print",
    "type": "authorized_user",
    "universe_domain": "googleapis.com",
}


def _adc_login(values: dict[str, str], bare: set[str], state: dict) -> tuple[int, str] | tuple[int, str, str]:
    config = os.environ.get("CLOUDSDK_CONFIG", "")
    state.setdefault("adc_logins", []).append(
        {"config": config, "scopes": values.get("--scopes", ""), "flags": sorted(bare)}
    )
    if not config:
        return 1, "", "fake gcloud: CLOUDSDK_CONFIG is unset; this login would replace ~/.config/gcloud's ADC"
    if state.get("adc_login_fails"):
        return 1, "", "ERROR: (gcloud.auth.application-default.login) There was a problem with web authentication."
    path = os.path.join(config, "application_default_credentials.json")
    payload = state.get("adc", FAKE_ADC)
    with open(path, "w", encoding="utf-8") as handle_:
        handle_.write(payload if isinstance(payload, str) else json.dumps(payload, indent=2))
    return 0, f"\nCredentials saved to file: [{path}]\n"

_NOT_FOUND = (1, "", "ERROR: (gcloud) NOT_FOUND: Requested entity was not found.")


def _words(argv: list[str]) -> list[str]:
    words = _split(argv)[0]
    return words[1:] if words[:1] == ["beta"] else words


def handle(argv: list[str], state: dict) -> tuple[int, str] | tuple[int, str, str]:
    words, values, bare = _split(argv)
    if words[:1] == ["beta"]:
        words = words[1:]
    project = values.get("--project", "test-project")
    fmt = values.get("--format", "")
    cmd = tuple(words[:3])

    if words[:2] == ["auth", "list"]:
        return 0, state.get("active_account", "deployer@example.com")
    if words[:3] == ["config", "get-value", "project"]:
        # gcloud's default project, which the scripts must never fall back to.
        return 0, state.get("config_project", "")
    if words[:3] == ["auth", "application-default", "login"]:
        return _adc_login(values, bare, state)
    if words[:2] == ["meta", "list-files-for-upload"]:
        return 0, "main.py"
    if words[:2] == ["services", "list"]:
        return 0, "\n".join(state.get("apis", []))
    if words[:2] == ["services", "enable"]:
        state.setdefault("apis", []).extend(words[2:])
        state["disabled_apis"] = [api for api in state.get("disabled_apis", []) if api not in words[2:]]
        return 0, ""

    if "get-iam-policy" in words:
        return 0, _policy(state, _resource(words, "get-iam-policy"))
    if "add-iam-policy-binding" in words:
        state.setdefault("bindings", []).append(
            [_resource(words, "add-iam-policy-binding"), values["--role"], values["--member"]]
        )
        return 0, ""
    if "remove-iam-policy-binding" in words:
        binding = [_resource(words, "remove-iam-policy-binding"), values["--role"], values["--member"]]
        state["bindings"] = [bound for bound in state.get("bindings", []) if bound != binding]
        return 0, ""
    if words[:2] == ["projects", "describe"]:
        return 0, state.get("project_number", "843253221534")

    if cmd == ("iam", "service-accounts", "describe"):
        email = words[3]
        return (0, email) if email in state.get("service_accounts", []) else _NOT_FOUND
    if cmd == ("iam", "service-accounts", "create"):
        state.setdefault("service_accounts", []).append(
            f"{words[3]}@{project}.iam.gserviceaccount.com"
        )
        return 0, ""

    if cmd == ("compute", "instances", "list"):
        return 0, json.dumps(state.get("instances", []))

    buckets = state.setdefault("buckets", {})
    if cmd == ("storage", "buckets", "list"):
        # Only the project's own buckets; "project" marks someone else's.
        return 0, "\n".join(
            name.removeprefix("gs://")
            for name, bucket in buckets.items()
            if bucket.get("project", project) == project
        )
    if cmd == ("storage", "buckets", "describe"):
        bucket = buckets.get(words[3])
        if bucket is None:
            return 1, "", f"ERROR: (gcloud.storage.buckets.describe) {words[3]} not found: 404."
        return 0, words[3].removeprefix("gs://") if fmt == "value(name)" else json.dumps(bucket)
    if cmd == ("storage", "buckets", "create"):
        with open(values["--lifecycle-file"], encoding="utf-8") as handle_:
            lifecycle = json.load(handle_)
        buckets[words[3]] = {
            "location": values.get("--location", "").upper(),
            "default_storage_class": values.get("--default-storage-class", "STANDARD"),
            "soft_delete_policy": {"retentionDurationSeconds": values.get("--soft-delete-duration", "604800")},
            "lifecycle_config": lifecycle,
        }
        return 0, ""
    if cmd == ("storage", "buckets", "update"):
        bucket = buckets[words[3]]
        if "--clear-soft-delete" in bare:
            bucket["soft_delete_policy"] = {"retentionDurationSeconds": "0"}
        if "--lifecycle-file" in values:
            with open(values["--lifecycle-file"], encoding="utf-8") as handle_:
                bucket["lifecycle_config"] = json.load(handle_)
        return 0, ""
    if words[:2] == ["storage", "ls"]:
        count = state.get("cloudbuild_objects", 0) if "_cloudbuild/" in words[2] else 0
        if not count:
            return 1, "", "ERROR: (gcloud.storage.ls) One or more URLs matched no objects."
        return 0, "\n".join(f"{words[2].split('**')[0]}{n}.tgz" for n in range(count))
    if words[:2] == ["storage", "rm"]:
        state["cloudbuild_objects"] = 0
        return 0, ""

    if cmd == ("tasks", "queues", "describe"):
        if words[3] not in state.get("queues", []):
            return _NOT_FOUND
        if fmt == "value(state)":
            return 0, "RUNNING"
        if "maxRetryDuration" in fmt:
            return 0, ""
        return 0, words[3]
    if cmd in {("tasks", "queues", "create"), ("tasks", "queues", "update")}:
        if words[3] not in state.setdefault("queues", []):
            state["queues"].append(words[3])
        return 0, ""

    if words[:2] == ["secrets", "describe"]:
        return (0, words[2]) if words[2] in state.get("secrets", {}) else _NOT_FOUND
    if words[:2] == ["secrets", "create"]:
        import base64

        data = sys.stdin.buffer.read()
        state.setdefault("secrets", {})[words[2]] = [
            {"id": 1, "state": "ENABLED", "data": base64.b64encode(data).decode()}
        ]
        state.setdefault("secret_creates", []).append(values.get("--replication-policy", ""))
        return 0, ""
    if words[:3] == ["secrets", "versions", "access"]:
        import base64

        live = _live(_secret_versions(state, values["--secret"]))
        if not live:
            return _NOT_FOUND
        payload = live[-1]["data"]
        if fmt == "get(payload.data)":
            return 0, base64.urlsafe_b64encode(base64.b64decode(payload)).decode()
        return 0, base64.b64decode(payload).decode()
    if words[:3] == ["secrets", "versions", "add"]:
        import base64

        versions = _secret_versions(state, words[3])
        data = sys.stdin.buffer.read()
        version = {
            "id": max((v["id"] for v in versions), default=0) + 1,
            "state": "ENABLED",
            "data": base64.b64encode(data).decode(),
        }
        versions.append(version)
        return 0, _version_name(words[3], version)
    if words[:3] == ["secrets", "versions", "describe"]:
        live = _live(_secret_versions(state, values["--secret"]))
        return (0, _version_name(values["--secret"], live[-1])) if live else _NOT_FOUND
    if words[:3] == ["secrets", "versions", "list"]:
        if words[3] not in state.get("secrets", {}):
            return _NOT_FOUND
        listed = list(reversed(_live(_secret_versions(state, words[3]))))
        if values.get("--filter") == "state=ENABLED":
            listed = [v for v in listed if v["state"] == "ENABLED"]
        if "--limit" in values:
            listed = listed[: int(values["--limit"])]
        return 0, "\n".join(_version_name(words[3], v) for v in listed)
    if words[:3] == ["secrets", "versions", "destroy"]:
        for version in _secret_versions(state, values["--secret"]):
            if str(version["id"]) == words[3]:
                version["state"] = "DESTROYED"
        return 0, ""

    repos = state.setdefault("repos", {})
    if cmd == ("artifacts", "repositories", "describe"):
        repo = repos.get(words[3])
        if repo is None:
            return _NOT_FOUND
        return 0, words[3] if fmt == "value(name)" else json.dumps(repo)
    if cmd == ("artifacts", "repositories", "create"):
        repos[words[3]] = {}
        return 0, ""
    if cmd == ("artifacts", "repositories", "set-cleanup-policies"):
        with open(values["--policy"], encoding="utf-8") as handle_:
            policies = json.load(handle_)
        repos[words[3]] = {
            "cleanupPolicies": {policy["name"]: policy for policy in policies},
            "cleanupPolicyDryRun": "--dry-run" in bare,
        }
        return 0, ""

    if cmd == ("artifacts", "docker", "images") and words[3:4] == ["list"]:
        return 0, "\n".join(state.get("images", []))
    if cmd == ("artifacts", "docker", "images") and words[3:4] == ["delete"]:
        digest = words[4].rsplit("@", 1)[-1]
        state["images"] = [image for image in state.get("images", []) if image != digest]
        return 0, ""
    if words[:2] == ["builds", "submit"]:
        substitutions = dict(
            pair.split("=", 1) for pair in values.get("--substitutions", "").split(",") if "=" in pair
        )
        state.setdefault("images", []).insert(0, _digest(substitutions.get("_IMAGE_TAG", "")))
        # [path, text]: someone edits a file (the env file) while the build runs.
        if state.get("edit_during_build"):
            path, text = state["edit_during_build"]
            with open(path, "a", encoding="utf-8") as handle_:
                handle_.write(text)
        return 0, ""

    if cmd == ("run", "services", "list"):
        # state["run_services"]: {"name", "region", "account"} per service.
        return 0, json.dumps(
            [
                {
                    "metadata": {
                        "name": service["name"],
                        "labels": {"cloud.googleapis.com/location": service["region"]},
                    },
                    "spec": {"template": {"spec": {"serviceAccountName": service.get("account", "")}}},
                }
                for service in state.get("run_services", [])
            ]
        )
    if cmd == ("run", "services", "describe"):
        if state.get("service_missing"):
            # Cloud Run's own wording, not NOT_FOUND.
            return 1, "", f"ERROR: (gcloud.run.services.describe) Cannot find service [{words[3]}]"
        if fmt == "value(status.url)":
            return 0, state.get("service_url", "")
        if fmt == "value(status.latestReadyRevisionName)":
            return 0, "linebot-cloud-agent-00002-abc"
        if fmt == "value(spec.template.spec.serviceAccountName)":
            return 0, state.get("service_account_on_service", "")
        if "volumes" in fmt:
            volumes = state.get("service_volumes", [])
            return 0, json.dumps({"spec": {"template": {"spec": {"volumes": volumes}}}})
        return 0, ""
    if words[:2] == ["run", "deploy"]:
        state.setdefault("deploys", []).append(argv)
        state.pop("service_missing", None)
        if "--env-vars-file" in values:
            with open(values["--env-vars-file"], encoding="utf-8") as handle_:
                state["env_vars_file"] = handle_.read()
        if "--service-account" in values:
            state["service_account_on_service"] = values["--service-account"]
        # The template's volumes, and the names of those its container mounts
        # (all of them unless state["service_mounts"] says otherwise).
        volumes = list(state.get("service_volumes", []))
        mounts = list(state.get("service_mounts", [volume["name"] for volume in volumes]))
        # gcloud applies these before the secret flags.
        if "--clear-volume-mounts" in bare:
            mounts = []
        if "--clear-volumes" in bare:
            volumes = []
        if "--clear-secrets" in bare or "--set-secrets" in values:
            # Like gcloud 561: the secret mounts go, and so does every secret
            # volume nothing mounted, but the ones the container mounted until
            # now stay in the template, unmounted (a live template, 2026-10-01:
            # four secret volumes, two of them mounted).
            secret_volumes = {volume["name"] for volume in volumes if volume.get("secret")}
            volumes = [v for v in volumes if not v.get("secret") or v["name"] in mounts]
            mounts = [name for name in mounts if name not in secret_volumes]
        if "--set-secrets" in values:
            # PATH=SECRET:VERSION[,PATH=SECRET:VERSION...]: one volume each.
            for number, mapping in enumerate(values["--set-secrets"].split(","), 1):
                path, _, ref = mapping.partition("=")
                secret, _, pinned = ref.partition(":")
                item = {"key": pinned, "path": path.rsplit("/", 1)[-1]}
                name = f"{secret}-d{len(state['deploys'])}-{number}"
                volumes.append({"name": name, "secret": {"secretName": secret, "items": [item]}})
                mounts.append(name)
        state["service_volumes"] = volumes
        state["service_mounts"] = mounts
        revisions = state.setdefault("revisions", [])
        image = values["--image"]
        revisions.append(
            {
                "service": words[2],
                "name": f"{words[2]}-deploy{len(state['deploys'])}",
                "image": f"{image.rsplit(':', 1)[0]}@{_digest(image)}",
                "volumes": volumes,
                "mounts": mounts,
            }
        )
        return 0, ""
    if cmd == ("run", "revisions", "list"):
        # Every service's revisions in the region, or one service's with
        # --service; a revision without "service" belongs to every service,
        # one without "region" to every region. "ready": False marks one
        # that never became Ready (a failed deploy); "mounts" names the
        # volumes its container mounts (without it, every volume).
        service = values.get("--service")
        region = values.get("--region")
        newest_first = [
            revision
            for revision in reversed(state.get("revisions", []))
            if (service is None or revision.get("service", service) == service)
            and revision.get("region", region) == region
        ]
        if fmt == "value(metadata.name)":
            return 0, "\n".join(revision["name"] for revision in newest_first)
        if fmt == "value(status.imageDigest)":
            return 0, "\n".join(revision["image"] for revision in newest_first)
        return 0, json.dumps(
            [
                {
                    "metadata": {"name": revision["name"]},
                    "spec": {
                        "volumes": revision["volumes"],
                        "containers": [
                            {
                                "volumeMounts": [
                                    {"name": name, "mountPath": f"/mnt/{name}"}
                                    for name in revision.get(
                                        "mounts", [volume["name"] for volume in revision["volumes"]]
                                    )
                                ]
                            }
                        ],
                    },
                    "status": {
                        "conditions": [
                            {"type": "Ready", "status": "True" if revision.get("ready", True) else "False"}
                        ]
                    },
                }
                for revision in newest_first
            ]
        )
    if cmd == ("run", "revisions", "delete"):
        state["revisions"] = [r for r in state.get("revisions", []) if r["name"] != words[3]]
        return 0, ""

    jobs = state.get("scheduler_jobs", {})
    if cmd == ("scheduler", "jobs", "list"):
        location = values.get("--location", "")
        return 0, json.dumps(
            [{"name": f"projects/{project}/locations/{location}/jobs/{name}", **job} for name, job in jobs.items()]
        )
    if cmd == ("scheduler", "jobs", "describe"):
        if words[3] not in jobs:
            return 1, "", "ERROR: (gcloud.scheduler.jobs.describe) NOT_FOUND: Job not found."
        # The API echoes the job's headers, the cron token included.
        return 0, words[3] if fmt == "value(name)" else json.dumps(jobs[words[3]])
    if cmd == ("scheduler", "jobs", "create") or cmd == ("scheduler", "jobs", "update"):
        headers = values.get("--headers") or values.get("--update-headers", "")
        state.setdefault("scheduler_jobs", {})[words[4]] = {"httpTarget": {"headers": headers}}
        return 0, ""
    if cmd == ("scheduler", "jobs", "delete"):
        state.get("scheduler_jobs", {}).pop(words[3], None)
        return 0, ""
    return 0, ""


def main() -> int:
    state_path = os.environ["FAKE_GCLOUD_STATE"]
    with open(state_path, encoding="utf-8") as handle_:
        state = json.load(handle_)
    argv = sys.argv[1:]
    with open(os.environ["FAKE_GCLOUD_LOG"], "a", encoding="utf-8") as log:
        log.write(" ".join(argv) + "\n")
    words = _words(argv)
    carries_secret = any(tuple(words[: len(prefix)]) == prefix for prefix in _SECRET_OUTPUT) or (
        words[:2] == ["run", "deploy"] and any(arg.startswith("--env-vars-file") for arg in argv)
    )
    if carries_secret and os.environ.get("CLOUDSDK_CORE_DISABLE_FILE_LOGGING") != "true":
        state.setdefault("file_logged", []).append(" ".join(words[:3]))
    if os.environ.get("CLOUDSDK_CONFIG") and words[:3] != ["auth", "application-default", "login"]:
        state.setdefault("scratch_config_calls", []).append(" ".join(words[:3]))
    joined = " ".join(words)
    failure = next(
        (message for prefix, message in state.get("fail", {}).items() if joined.startswith(prefix)),
        None,
    )
    api, title = _GROUP_APIS.get(words[0] if words else "", ("", ""))
    if failure is not None:
        result: tuple = (1, "", f"ERROR: (gcloud.{'.'.join(words[:3])}) {failure}")
    elif api in state.get("disabled_apis", []):
        if os.environ.get("CLOUDSDK_CORE_DISABLE_PROMPTS", "").lower() not in {"1", "true", "yes", "on"}:
            state.setdefault("enable_api_prompts", []).append(" ".join(words[:3]))
        result = _service_disabled(".".join(words[:3]), api, title)
    else:
        result = handle(argv, state)
    code, out = result[0], result[1]
    err = result[2] if len(result) > 2 else ""
    with open(state_path, "w", encoding="utf-8") as handle_:
        json.dump(state, handle_)
    if out:
        sys.stdout.write(out if out.endswith("\n") else out + "\n")
    if code:
        sys.stderr.write((err or "ERROR: (gcloud) failed") + "\n")
    return code


if __name__ == "__main__":
    raise SystemExit(main())
