"""Helpers for .env-style files, used by scripts/deploy_cloud_run.sh.

The deploy wrapper shells out to these subcommands with a bare ``python3``,
so module-level imports stay in the standard library. ``src.config`` (and
with it pydantic) is imported lazily, only by the v3 model guard, which falls
back to local copies of the shared constants when that import fails.

Only ``get``, ``dump`` and ``to-dotenv`` print env values (their stdout is
captured or piped by the wrapper, never shown); every other subcommand
reports key names, file names or a digest, never values.
"""

from __future__ import annotations

import argparse
import base64
import binascii
import codecs
import contextlib
import hashlib
import json
import os
import re
import shlex
import stat
import subprocess
import sys
import tempfile
from collections.abc import Iterable, Iterator, Mapping
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]

_PLACEHOLDER_VALUES = {
    "your_channel_secret",
    "your_access_token",
    "your_openrouter_api_key",
    "your_nvidia_api_key",
    "your_tavily_api_key",
    "your_project_id",
    "replace_me",
    "changeme",
    "todo",
}

# ── v3 model guard ────────────────────────────────────────────
# Local copies of src.config.RETIRED_MODEL_IDS / LEGACY_ENV_KEYS /
# REMOVED_ENV_KEYS for a python3 without pydantic. tests/test_envfile.py
# fails if they drift.
_FALLBACK_RETIRED_MODEL_IDS = frozenset(
    {
        "qwen/qwen3.5-397b-a17b",
        "qwen/qwen3.5-122b-a10b",
        "nvidia/nv-embedqa-e5-v5",
    }
)
_FALLBACK_LEGACY_ENV_KEYS = frozenset({"ORCHESTRATOR_MODEL", "ORCHESTRATOR_FALLBACK_MODEL"})

# Keys the app no longer reads (the model settings the role table replaced
# among them). Harmless, so check-v3 only notes them; migrate-v3 deletes them.
_FALLBACK_REMOVED_ENV_KEYS = frozenset(
    {
        "GCS_MEDIA_CLEANUP_DELAY_SECONDS",
        "THINKING_TIMEOUT_SECONDS",
        "NVIDIA_THINKING_MODEL",
        "NVIDIA_MODEL",
        "NVIDIA_VISION_MODEL",
        "NVIDIA_UTILITY_MODEL",
        "NVIDIA_EMBEDDING_MODEL",
        "ROUTER_MODEL",
        "ROUTER_FALLBACK_MODEL",
        "AGENT_FALLBACK_MODEL",
        "AGENT_FALLBACK_MODEL_2",
        "VISION_FALLBACK_MODEL",
        "VISION_FALLBACK_MODEL_2",
        "UTILITY_FALLBACK_MODEL",
        "IMAGE_GEN_PRIMARY_MODEL",
        "IMAGE_GEN_FALLBACK_MODEL",
        "IMAGE_GEN_STEPS",
        "IMAGE_GEN_CFG_SCALE",
        "NVIDIA_THINKING_ENABLED",
        "OPENROUTER_REASONING_ENABLED",
        "OPENROUTER_REASONING_EFFORT",
        "OPENROUTER_REASONING_EXCLUDE",
        "OPENROUTER_THINKING_BUDGET",
        "REQUIRE_REASONING_MODELS",
        "REQUIRE_REASONING_TOKENS",
        "WEB_SEARCH_MONTHLY_QUOTA",
        "FIRESTORE_LOCATION",
        "SEARXNG_BASE_URL",
        "SEARXNG_TIMEOUT_SECONDS",
        "SEARXNG_TIMEOUT_LIMIT_SECONDS",
        "LINE_PUSH_FALLBACK_ENABLED",
        "LINE_REPLY_FALLBACK_PUSH_ENABLED",
        "LINE_PUSH_RESERVE",
        "LINE_PUSH_MONTHLY_LIMIT",
        "COLAB_REPLY_WINDOW_SECONDS",
    }
)

# REQUIRE_REASONING_MODELS=true was the v2 switch that silently dropped
# fallback models; check-v3 still refuses it so an old env file is noticed.
# Truthy spellings match pydantic's bool parsing.
_REQUIRE_REASONING_KEY = "REQUIRE_REASONING_MODELS"
_TRUTHY_VALUES = frozenset({"1", "on", "t", "true", "y", "yes"})
# JSON {role: [{"provider", "model", ...}]}; check-v3 looks inside for
# retired ids, since the app would refuse them only at startup.
_MODEL_ROLES_KEY = "MODEL_ROLES"
# Model families the v3 defaults dropped (2026-10-01: NIM's Gemma 4 31B hung
# on every live call, OpenRouter's Gemma :free tiers answered 429 to every
# one). The ids still work, so check-v3 only notes a MODEL_ROLES naming one.
_DROPPED_MODEL_FAMILIES = ("gemma",)

# ── Cloud Build upload guard ─────────────────────────────────
# Paths (as printed by `gcloud meta list-files-for-upload`) that must never
# leave this machine: at least every secret .gcloudignore lists (a test
# checks). .env.example is a template without secrets.
_DOTENV_PATH_RE = re.compile(r"(^|/)\.env(\.[^/]*)?$")
_UPLOAD_LEAK_RES = (
    re.compile(r"(^|/)\.cloudrun-env[^/]*$"),
    re.compile(r"(^|/)settings\.local\.json$"),
    re.compile(r"(^|/)\.claude(/|$)"),
    re.compile(r"(^|/)credentials\.json$"),
    # *.bak, and migrate-v3's <env file>.bak.<YYYYmmddHHMMSS> backups.
    re.compile(r"\.bak(\.[0-9]{14})?$"),
)
_ALLOWED_DOTENV_NAME = ".env.example"

# python-dotenv breaks lines only at CR LF, LF and CR: not at the other
# breaks str.splitlines() knows (U+2028, U+2029, NEL, VT, FF, FS/GS/RS),
# which stay inside a value. It drops a leading byte order mark, and
# `export` may be followed by any blanks.
_LINE_RE = re.compile(r"[^\r\n]*(?:\r\n|\n|\r)|[^\r\n]+")
_NEWLINE_RE = re.compile(r"\r\n|\n|\r")
_BOM = "\ufeff"
_EXPORT_RE = re.compile(r"export[^\S\r\n]+")
# python-dotenv drops " #..." after an unquoted value and allows a comment
# after the closing quote of a quoted one; match both so the deployed env
# and the guard see the same values the app sees locally.
_INLINE_COMMENT_RE = re.compile(r"\s+#.*$")
# python-dotenv 1.2.3's own patterns (dotenv/parser.py): a quoted value runs,
# across lines, to the first quote no backslash escapes (a backslash always
# escapes the next character), and only blanks or a comment may follow it
# on that line. Anything else makes python-dotenv skip the statement.
_QUOTED_VALUE_RES = {
    "'": re.compile(r"'((?:\\.|[^'\\])*)'", re.DOTALL),
    '"': re.compile(r'"((?:\\.|[^"\\])*)"', re.DOTALL),
}
_AFTER_CLOSING_QUOTE_RE = re.compile(r"[^\S\r\n]*(?:#.*)?")
# The escapes python-dotenv decodes inside quoted values.
_QUOTE_ESCAPE_RES = {
    "'": re.compile(r"\\[\\']"),
    '"': re.compile(r"\\[\\'\"abfnrtv]"),
}

# ── Deploy-only keys ──────────────────────────────────────────
# Read by scripts/deploy_cloud_run.sh, scripts/provision_gcp.sh or the GCP
# SDK, never by the app on Cloud Run, so they stay out of what is deployed
# (to-dotenv, to-cloudrun-yaml). PORT is Cloud Run's own.
DEPLOY_ONLY_KEYS = frozenset(
    {
        "GOOGLE_APPLICATION_CREDENTIALS",
        "PORT",
        "HOST",
        "GCP_PROJECT_ID",
        "CLOUD_RUN_SERVICE_NAME",
        "CLOUD_RUN_REGION",
        "CLOUD_RUN_MIN_INSTANCES",
        "CLOUD_RUN_SERVICE_ACCOUNT",
        "DEPLOY_KEEP_REVISIONS",
        "DEPLOY_KEEP_IMAGES",
        "DEPLOY_ENABLE_APIS",
    }
)

# ── Secret Manager dotenv blob ────────────────────────────────
# Secret Manager's payload limit per version.
SECRET_PAYLOAD_LIMIT = 64 * 1024
_DOTENV_KEY_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
# What to-cloudrun-yaml escapes beyond JSON (see _yaml_quote).
_YAML_UNSAFE_RE = re.compile("[\x7f-\x9f\u2028\u2029\ufffe\uffff]")


def is_placeholder_value(value: str) -> bool:
    normalized = (value or "").strip().strip("'\"").lower()
    if not normalized:
        return False
    return normalized in _PLACEHOLDER_VALUES or normalized.startswith("your_")


# ── Parsing ──────────────────────────────────────────────────


@dataclass(frozen=True)
class _Assignment:
    """One ``KEY=value`` binding and the (inclusive) line range it spans.

    ``malformed``: a quoted value python-dotenv cannot parse (a quote that
    never closes, or text after the closing one). python-dotenv skips the
    statement, so ``value`` is empty and the key is not set.
    """

    key: str
    value: str
    start: int
    end: int
    malformed: bool = False


def _split_lines(text: str) -> list[str]:
    """*text*'s lines, each with its line ending, split where python-dotenv splits."""
    return _LINE_RE.findall(text)


def _strip_ending(line: str) -> str:
    return line.rstrip("\r\n")


def _consume_quoted_value(
    lines: list[str],
    start_index: int,
    value: str,
) -> tuple[str | None, int]:
    """The decoded value and the index of its last line, like python-dotenv.

    *lines* keep their line endings; *value* runs to the end of the opening
    line. The value is None when python-dotenv rejects the statement. It
    then resumes after the rest of the line where it stopped: the opening
    line for a quote that never closes, the closing line for text after the
    closing quote.
    """
    quote = value[0]
    opening = lines[start_index]
    rest = value + opening[len(_strip_ending(opening)) :] + "".join(lines[start_index + 1 :])
    match = _QUOTED_VALUE_RES[quote].match(rest)
    if match is None:
        return None, start_index
    end_index = start_index + len(_NEWLINE_RE.findall(match.group(0)))
    tail = _NEWLINE_RE.split(rest[match.end() :], 1)[0]
    if not _AFTER_CLOSING_QUOTE_RE.fullmatch(tail):
        return None, end_index
    return _decode_escapes(quote, match.group(1)), end_index


def _decode_escapes(quote: str, body: str) -> str:
    """Decode a quoted value's escapes the way python-dotenv does."""
    return _QUOTE_ESCAPE_RES[quote].sub(
        lambda match: codecs.decode(match.group(0), "unicode-escape"), body
    )


def _iter_assignments(lines: list[str]) -> Iterator[_Assignment]:
    """The assignments in *lines* (from :func:`_split_lines`, without a BOM)."""
    index = 0
    while index < len(lines):
        start = index
        # Only the left: trailing blanks on the opening line of a
        # multi-line quoted value belong to the value.
        line = _strip_ending(lines[index]).lstrip()
        if not line or line.startswith("#"):
            index += 1
            continue
        export = _EXPORT_RE.match(line)
        if export:
            line = line[export.end() :]
        if "=" not in line:
            index += 1
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.lstrip()
        if not key:
            index += 1
            continue
        malformed = False
        if value.startswith(("'", '"')):
            decoded, index = _consume_quoted_value(lines, index, value)
            malformed = decoded is None
            value = decoded or ""
        else:
            value = _INLINE_COMMENT_RE.sub("", value).rstrip()
        yield _Assignment(key=key, value=value, start=start, end=index, malformed=malformed)
        index += 1


def _assignments(text: str) -> Iterator[_Assignment]:
    """The assignments in an env file's *text*, read like python-dotenv reads it."""
    if text.startswith(_BOM):
        text = text[len(_BOM) :]
    return _iter_assignments(_split_lines(text))


def parse_env_text(text: str) -> dict[str, str]:
    """The values python-dotenv reads; statements it skips are left out."""
    return {
        assignment.key: assignment.value
        for assignment in _assignments(text)
        if not assignment.malformed
    }


def malformed_env_keys(text: str) -> tuple[str, ...]:
    """Keys whose quoted value python-dotenv cannot parse (and so skips).

    Keys that are no env keys are left to :func:`invalid_key_lines`.
    """
    return tuple(
        sorted({a.key for a in _assignments(text) if a.malformed and _DOTENV_KEY_RE.match(a.key)})
    )


def invalid_key_lines(text: str) -> tuple[int, ...]:
    """The lines (from 1) whose text before ``=`` is no env key: a line
    missing its ``=`` before a value holding one, say. Name them by number
    only: the text may be a secret."""
    return tuple(a.start + 1 for a in _assignments(text) if not _DOTENV_KEY_RE.match(a.key))


def _lines_note(numbers: Iterable[int]) -> str:
    numbers = list(numbers)
    label = "line" if len(numbers) == 1 else "lines"
    return (
        f"{label} {', '.join(map(str, numbers))}: not KEY=VALUE (the text before '=' is no env key; "
        "a missing '=' after the key?). Not shown: it may hold a secret"
    )


def parse_env_file(path: Path) -> dict[str, str]:
    return parse_env_text(path.read_text(encoding="utf-8"))


def upper_keys(values: Mapping[str, str]) -> dict[str, str]:
    """*values* keyed in upper case, as the app's settings read them
    (pydantic-settings, ``case_sensitive=False``): of two keys that differ
    only in case, the later one wins."""
    return {key.upper(): value for key, value in values.items()}


def remove_env_keys(text: str, keys: Iterable[str]) -> str:
    """Drop every assignment of ``keys`` (case-insensitive), keeping all other lines
    (and a leading byte order mark) as-is."""
    targets = {key.upper() for key in keys}
    bom = _BOM if text.startswith(_BOM) else ""
    lines = _split_lines(text[len(bom) :])
    dropped: set[int] = set()
    for assignment in _iter_assignments(lines):
        if assignment.key.upper() in targets:
            dropped.update(range(assignment.start, assignment.end + 1))
    return bom + "".join(line for index, line in enumerate(lines) if index not in dropped)


# ── v3 model guard ────────────────────────────────────────────


def _load_config_module():
    """Import src.config lazily; None when its dependencies are not installed."""
    if str(_REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(_REPO_ROOT))
    try:
        from src import config
    except ImportError:
        return None
    return config


def retired_model_ids() -> frozenset[str]:
    ids = getattr(_load_config_module(), "RETIRED_MODEL_IDS", None)
    return frozenset(ids) if ids else _FALLBACK_RETIRED_MODEL_IDS


def legacy_env_keys() -> frozenset[str]:
    keys = getattr(_load_config_module(), "LEGACY_ENV_KEYS", None)
    return frozenset(keys) if keys else _FALLBACK_LEGACY_ENV_KEYS


def removed_env_keys() -> frozenset[str]:
    keys = getattr(_load_config_module(), "REMOVED_ENV_KEYS", None)
    return frozenset(keys) if keys else _FALLBACK_REMOVED_ENV_KEYS


@dataclass(frozen=True)
class V3Report:
    """What check-v3 found in one env file: key names only, never values."""

    retired: tuple[str, ...] = ()
    legacy: tuple[str, ...] = ()
    require_reasoning: bool = False
    # Keys the app no longer reads. Informational: a stale value is
    # harmless, so it does not fail the check.
    removed: tuple[str, ...] = ()
    # MODEL_ROLES is set but is not a JSON object (the app would not start).
    invalid_model_roles: bool = False
    # MODEL_ROLES names a model of a family the defaults dropped.
    # Informational, like ``removed``.
    dropped_model_roles: bool = False
    # Quoted values python-dotenv cannot parse: the app skips these keys.
    malformed: tuple[str, ...] = ()
    # Lines that are no KEY=VALUE, by number (their text may be a secret).
    invalid_lines: tuple[int, ...] = ()

    @property
    def ok(self) -> bool:
        return not (
            self.retired
            or self.legacy
            or self.require_reasoning
            or self.invalid_model_roles
            or self.malformed
            or self.invalid_lines
        )

    @property
    def keys_to_remove(self) -> tuple[str, ...]:
        """Keys migrate-v3 deletes; MODEL_ROLES is the operator's to fix."""
        keys = (set(self.retired) - {_MODEL_ROLES_KEY}) | set(self.legacy) | set(self.removed)
        if self.require_reasoning:
            keys.add(_REQUIRE_REASONING_KEY)
        return tuple(sorted(keys))


def check_v3(
    values: Mapping[str, str],
    *,
    retired: Iterable[str] | None = None,
    legacy: Iterable[str] | None = None,
    removed: Iterable[str] | None = None,
    malformed: Iterable[str] = (),
    invalid_lines: Iterable[int] = (),
) -> V3Report:
    """Classify env keys that would keep the bot on retired v2 settings.

    Keys compare case-insensitively, like pydantic-settings does; a key
    that is no env key is never named (``invalid_lines`` numbers it).
    ``malformed`` (from :func:`malformed_env_keys`) and ``invalid_lines``
    (from :func:`invalid_key_lines`) fail the check too.
    """
    retired_ids = {model.lower() for model in (retired_model_ids() if retired is None else retired)}
    legacy_keys = {key.upper() for key in (legacy_env_keys() if legacy is None else legacy)}
    removed_keys = {key.upper() for key in (removed_env_keys() if removed is None else removed)}
    normalized = {key.upper(): value.strip() for key, value in values.items() if _DOTENV_KEY_RE.match(key)}

    retired_keys = {key for key, value in normalized.items() if value.lower() in retired_ids}
    role_models = _model_roles_ids(normalized.get(_MODEL_ROLES_KEY, ""))
    if role_models and any(model.lower() in retired_ids for model in role_models):
        retired_keys.add(_MODEL_ROLES_KEY)
    return V3Report(
        retired=tuple(sorted(retired_keys)),
        legacy=tuple(sorted(key for key in normalized if key in legacy_keys)),
        require_reasoning=normalized.get(_REQUIRE_REASONING_KEY, "").lower() in _TRUTHY_VALUES,
        removed=tuple(sorted(key for key in normalized if key in removed_keys)),
        invalid_model_roles=role_models is None,
        dropped_model_roles=any(
            family in model.lower() for model in role_models or () for family in _DROPPED_MODEL_FAMILIES
        ),
        malformed=tuple(sorted(malformed)),
        invalid_lines=tuple(invalid_lines),
    )


def _model_roles_ids(raw: str) -> list[str] | None:
    """The model ids a MODEL_ROLES value names; None when it is not a JSON object.

    Only the shape check-v3 needs: the app validates everything else at
    startup.
    """
    raw = raw.strip()
    if not raw:
        return []
    try:
        data = json.loads(raw)
    except ValueError:
        return None
    if not isinstance(data, dict):
        return None
    return [
        tier["model"].strip()
        for tiers in data.values()
        if isinstance(tiers, list)
        for tier in tiers
        if isinstance(tier, dict) and isinstance(tier.get("model"), str)
    ]


def _display_path(path: Path) -> str:
    try:
        shown = os.path.relpath(path)
    except ValueError:
        shown = str(path)
    if shown.startswith(".."):
        shown = str(path)
    return shlex.quote(shown)


def format_v3_report(report: V3Report, env_file: Path) -> list[str]:
    lines: list[str] = []
    if report.retired:
        lines.append(
            "[v3-check] retired model ids (HTTP 410) are still selected by: "
            + ", ".join(report.retired)
        )
    if report.legacy:
        lines.append(
            "[v3-check] legacy keys are no longer read and must be removed: "
            + ", ".join(report.legacy)
        )
    if report.require_reasoning:
        lines.append(
            f"[v3-check] {_REQUIRE_REASONING_KEY}=true silently drops fallback models; "
            "remove it (the v3 default is false)"
        )
    # migrate-v3 never rewrites MODEL_ROLES: these are fixed by hand.
    if report.invalid_model_roles:
        lines.append(
            f"[v3-check] {_MODEL_ROLES_KEY} is not a JSON object ({{role: [tiers]}}), so the "
            f"app would refuse to start; fix it by hand in {_display_path(env_file)} "
            "(see .env.example)"
        )
    if _MODEL_ROLES_KEY in report.retired:
        lines.append(
            f"[v3-check] {_MODEL_ROLES_KEY} names a retired model id; replace that tier's "
            f"model by hand in {_display_path(env_file)}"
        )
    if report.malformed:
        lines.append(
            "[v3-check] python-dotenv cannot parse the quoted value of: "
            + ", ".join(report.malformed)
            + " (a quote that never closes, or text after the closing quote), so the "
            f"app skips these keys; fix their quoting in {_display_path(env_file)}"
        )
    if report.invalid_lines:
        lines.append(f"[v3-check] {_lines_note(report.invalid_lines)}; fix them in {_display_path(env_file)}")
    if report.removed:
        lines.append(
            "[v3-check] note: these keys are no longer read (harmless; migrate-v3 "
            "removes them, MODEL_ROLES now overrides models): " + ", ".join(report.removed)
        )
    if report.dropped_model_roles:
        lines.append(
            f"[v3-check] note: {_MODEL_ROLES_KEY} names a Gemma model, which the defaults "
            "dropped on 2026-10-01 (it hung on NVIDIA NIM and was rate limited on OpenRouter); "
            f"replace that tier by hand in {_display_path(env_file)} unless you mean to keep it"
        )
    if report.ok:
        lines.append("[v3-check] ok: no retired model ids or legacy model keys")
    elif set(report.retired) - {_MODEL_ROLES_KEY} or report.legacy or report.require_reasoning:
        # Only when migrate-v3 fixes one of the failures above.
        lines.append(
            f"[v3-check] fix: python3 {_display_path(Path(__file__).resolve())} migrate-v3 "
            f"--file {_display_path(env_file)}"
        )
    return lines


@dataclass(frozen=True)
class MigrationResult:
    removed: tuple[str, ...]
    backup: Path | None


def _write_private_file(path: Path, data: bytes) -> None:
    fd = os.open(str(path), os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(fd, "wb") as handle:
        handle.write(data)
    os.chmod(path, 0o600)


def _replace_file(path: Path, data: bytes) -> None:
    """Rewrite ``path`` atomically, keeping its permission bits."""
    mode = stat.S_IMODE(path.stat().st_mode)
    fd, tmp_name = tempfile.mkstemp(dir=str(path.parent), prefix=f".{path.name}.", suffix=".tmp")
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(data)
        os.chmod(tmp_name, mode)
        os.replace(tmp_name, path)
    except BaseException:
        with contextlib.suppress(FileNotFoundError):
            os.unlink(tmp_name)
        raise


def migrate_v3(
    path: Path,
    *,
    now: datetime | None = None,
    dry_run: bool = False,
) -> MigrationResult:
    """Delete the keys check-v3 objects to, plus keys no longer read, after a 0600 backup."""
    path = path.resolve()
    raw = path.read_bytes()
    text = raw.decode("utf-8")
    keys = check_v3(parse_env_text(text)).keys_to_remove
    if not keys or dry_run:
        return MigrationResult(removed=keys, backup=None)

    stamp = (now or datetime.now()).strftime("%Y%m%d%H%M%S")
    backup = path.with_name(f"{path.name}.bak.{stamp}")
    _write_private_file(backup, raw)
    _replace_file(path, remove_env_keys(text, keys).encode("utf-8"))
    return MigrationResult(removed=keys, backup=backup)


# ── Secret Manager dotenv blob ────────────────────────────────


class DotenvBlobError(ValueError):
    """The env file cannot become a dotenv blob; the message names keys only."""


def deployable_values(values: Mapping[str, str], exclude: Iterable[str] = ()) -> dict[str, str]:
    """*values* without the deploy-only keys and *exclude* (case-insensitive)."""
    excluded = DEPLOY_ONLY_KEYS | {key.upper() for key in exclude}
    return {key: value for key, value in values.items() if key.upper() not in excluded}


def to_dotenv(values: Mapping[str, str], *, exclude: Iterable[str] = ()) -> str:
    """The dotenv blob stored in Secret Manager and mounted on Cloud Run.

    Every value is single-quoted, with backslashes and single quotes
    escaped, so python-dotenv (which pydantic-settings reads it with) gets
    back exactly *values*: newlines, ``#`` and double quotes included.
    Refused, by key name: a value with ``${`` (python-dotenv would expand it
    as a variable), a trailing backslash (python-dotenv before 1.2.3 reads
    it as escaping the closing quote), a carriage return (reading the file
    turns it into a newline), a key python-dotenv cannot read, and a blob
    over Secret Manager's 64 KiB limit.

    Raises:
        DotenvBlobError: Any of the above.
    """
    lines: list[str] = []
    problems: list[str] = []
    for key, value in deployable_values(values, exclude).items():
        if not _DOTENV_KEY_RE.match(key):
            # Not named: a line missing its '=' puts a secret in the key.
            problems.append("a key is not a valid env key (check-v3 names its line)")
            continue
        if "${" in value:
            problems.append(f"{key} contains '${{' (python-dotenv would expand it)")
        if value.endswith("\\"):
            problems.append(f"{key} ends with a backslash")
        if "\r" in value:
            problems.append(f"{key} contains a carriage return")
        escaped = value.replace("\\", "\\\\").replace("'", "\\'")
        lines.append(f"{key}='{escaped}'")
    if problems:
        raise DotenvBlobError("; ".join(problems))
    blob = "".join(f"{line}\n" for line in lines)
    size = len(blob.encode("utf-8"))
    if size > SECRET_PAYLOAD_LIMIT:
        raise DotenvBlobError(
            f"the blob is {size} bytes, over Secret Manager's {SECRET_PAYLOAD_LIMIT}-byte limit"
        )
    return blob


def sha256_hex(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def decode_secret_payload(text: str) -> bytes:
    """Bytes of a ``--format='get(payload.data)'`` secret payload (base64)."""
    encoded = "".join(text.split()).translate(str.maketrans("+/", "-_"))
    return base64.urlsafe_b64decode(encoded + "=" * (-len(encoded) % 4))


# ── Cloud Build upload guard ─────────────────────────────────


def find_upload_leaks(paths: Iterable[str], secrets: Iterable[str] = ()) -> list[str]:
    """Return the upload paths that are secret files and must not be uploaded.

    *secrets* are more such paths, whatever their names (the env file being
    deployed, relative to the upload root).
    """
    secret_paths = set(secrets)
    leaks: list[str] = []
    for raw_path in paths:
        path = raw_path.strip().replace("\\", "/")
        while path.startswith("./"):
            path = path[2:]
        if not path:
            continue
        is_dotenv = bool(_DOTENV_PATH_RE.search(path)) and (
            path.rsplit("/", 1)[-1] != _ALLOWED_DOTENV_NAME
        )
        if is_dotenv or path in secret_paths or any(pattern.search(path) for pattern in _UPLOAD_LEAK_RES):
            leaks.append(path)
    return leaks


def paths_under(root: Path, path: Path) -> list[str]:
    """*path* relative to *root*, as written and with symlinks resolved; [] when outside it."""
    found: list[str] = []
    for base, target in ((Path(os.path.abspath(root)), Path(os.path.abspath(path))), (root.resolve(), path.resolve())):
        with contextlib.suppress(ValueError):
            relative = target.relative_to(base).as_posix()
            if relative not in found:
                found.append(relative)
    return found


def git_ignores(root: Path, relative: str) -> bool | None:
    """Whether git ignores *relative* in the work tree *root*; None when git cannot tell."""
    try:
        result = subprocess.run(
            ["git", "-C", str(root), "check-ignore", "-q", "--", relative],
            capture_output=True,
            check=False,
        )
    except OSError:
        return None
    return {0: True, 1: False}.get(result.returncode)


# ── CLI ──────────────────────────────────────────────────────


def cmd_get(args: argparse.Namespace) -> int:
    values = upper_keys(parse_env_file(Path(args.file)))
    value = values.get(args.key.upper())
    if value is None:
        return 1
    print(value)
    return 0


def cmd_dump(args: argparse.Namespace) -> int:
    values = upper_keys(parse_env_file(Path(args.file)))
    fields: list[str] = []
    for key, value in values.items():
        fields += [key, value, "1" if is_placeholder_value(value) else "0"]
    # Written whole: a file that cannot be read prints nothing at all.
    sys.stdout.write("".join(field + "\0" for field in fields))
    return 0


def _refuse_malformed(text: str, command: str) -> bool:
    """Print the keys python-dotenv cannot parse and the numbers of lines
    that are no KEY=VALUE; True when there are any."""
    malformed = malformed_env_keys(text)
    invalid = invalid_key_lines(text)
    if malformed:
        print(
            f"[{command}] refused: python-dotenv cannot parse the quoted value of "
            + ", ".join(malformed)
            + " (a quote that never closes, or text after the closing quote)",
            file=sys.stderr,
        )
    if invalid:
        print(f"[{command}] refused: {_lines_note(invalid)}", file=sys.stderr)
    return bool(malformed or invalid)


def _yaml_quote(value: str) -> str:
    """*value* as a YAML double-quoted scalar that reads back unchanged.

    JSON escapes the C0 controls, quotes and backslashes; the characters
    YAML reads as line breaks (NEL, U+2028, U+2029, which a raw scalar
    folds) or refuses (DEL, the C1 controls) are escaped here too.
    """
    return _YAML_UNSAFE_RE.sub(
        lambda match: f"\\u{ord(match.group(0)):04x}", json.dumps(value, ensure_ascii=False)
    )


def cmd_to_cloudrun_yaml(args: argparse.Namespace) -> int:
    text = Path(args.file).read_text(encoding="utf-8")
    if _refuse_malformed(text, "to-cloudrun-yaml"):
        return 1
    values = deployable_values(parse_env_text(text), args.exclude)
    for assignment in args.set:
        key, sep, value = assignment.partition("=")
        if not sep or not _DOTENV_KEY_RE.match(key):
            # Not echoed: a missing '=' leaves the value in the key.
            print("[to-cloudrun-yaml] --set needs KEY=VALUE with an env key name", file=sys.stderr)
            return 2
        values[key] = value
    output = Path(args.output)
    lines = [f"{key}: {_yaml_quote(value)}" for key, value in values.items()]
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return 0


def cmd_to_dotenv(args: argparse.Namespace) -> int:
    env_file = Path(args.file)
    if not _require_file(env_file, "to-dotenv"):
        return 2
    text = env_file.read_text(encoding="utf-8")
    if _refuse_malformed(text, "to-dotenv"):
        return 1
    try:
        blob = to_dotenv(parse_env_text(text), exclude=args.exclude)
    except DotenvBlobError as exc:
        print(f"[to-dotenv] refused: {exc}", file=sys.stderr)
        return 1
    if args.sha256:
        print(sha256_hex(blob.encode("utf-8")))
    else:
        sys.stdout.write(blob)
    return 0


def cmd_sha256(args: argparse.Namespace) -> int:
    data = sys.stdin.buffer.read()
    if args.base64:
        try:
            data = decode_secret_payload(data.decode("ascii"))
        except (UnicodeDecodeError, binascii.Error, ValueError):
            print("[sha256] stdin is not base64", file=sys.stderr)
            return 1
    print(sha256_hex(data))
    return 0


def cmd_is_placeholder(args: argparse.Namespace) -> int:
    return 0 if is_placeholder_value(args.value) else 1


def _require_file(path: Path, command: str) -> bool:
    if path.is_file():
        return True
    print(f"[{command}] env file not found: {_display_path(path)}", file=sys.stderr)
    return False


def cmd_check_v3(args: argparse.Namespace) -> int:
    env_file = Path(args.file)
    if not _require_file(env_file, "v3-check"):
        return 2
    text = env_file.read_text(encoding="utf-8")
    report = check_v3(
        parse_env_text(text), malformed=malformed_env_keys(text), invalid_lines=invalid_key_lines(text)
    )
    for line in format_v3_report(report, env_file):
        print(line)
    return 0 if report.ok else 1


def cmd_migrate_v3(args: argparse.Namespace) -> int:
    env_file = Path(args.file)
    if not _require_file(env_file, "migrate-v3"):
        return 2
    try:
        result = migrate_v3(env_file, dry_run=args.dry_run)
    except FileExistsError as exc:
        print(f"[migrate-v3] backup already exists, retry in a second: {exc.filename}", file=sys.stderr)
        return 1
    if not result.removed:
        print("[migrate-v3] nothing to migrate")
        return 0
    if result.backup is None:
        print("[migrate-v3] would remove: " + ", ".join(result.removed))
        return 0
    print(f"[migrate-v3] backup: {_display_path(result.backup)} (mode 0600)")
    print("[migrate-v3] removed: " + ", ".join(result.removed))
    print("[migrate-v3] the model role table defaults (or MODEL_ROLES) now apply")
    return 0


def cmd_check_upload(args: argparse.Namespace) -> int:
    # The env file being deployed holds every secret: inside the repository
    # it must be one git ignores and gcloud does not upload.
    env_paths = paths_under(Path(args.root), Path(args.env_file)) if args.root and args.env_file else []
    leaks = find_upload_leaks(sys.stdin, env_paths)
    tracked = [path for path in env_paths[:1] if git_ignores(Path(args.root), path) is False]
    if not leaks and not tracked:
        print("[upload-check] ok: no secret files in the upload")
        return 0
    if leaks:
        print("[upload-check] these files would be uploaded to Cloud Build:", file=sys.stderr)
        for path in leaks:
            print(f"  {path}", file=sys.stderr)
    for path in tracked:
        print(
            f"[upload-check] the env file {path} is inside the repository and git does not ignore it: "
            "move it out of the repository, or add it to .gitignore and .gcloudignore",
            file=sys.stderr,
        )
    return 1


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Utilities for reading .env-style files.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    get_parser = subparsers.add_parser(
        "get", help="Read a single key from an env file (case-insensitive, like the app)."
    )
    get_parser.add_argument("--file", required=True, help="Path to env file.")
    get_parser.add_argument("key", help="Key to read.")
    get_parser.set_defaults(func=cmd_get)

    dump_parser = subparsers.add_parser(
        "dump",
        help=(
            "Print every key python-dotenv reads, in upper case (the app reads keys "
            "case-insensitively), each followed by its value and 1 "
            "when that value is an example placeholder (else 0), all NUL-terminated: "
            "one python3 start for a script that reads many keys."
        ),
    )
    dump_parser.add_argument("--file", required=True, help="Path to env file.")
    dump_parser.set_defaults(func=cmd_dump)

    yaml_parser = subparsers.add_parser(
        "to-cloudrun-yaml",
        help=(
            "Convert an env file into a Cloud Run env-vars YAML file (the deploy's "
            "fallback before provisioning), without the deploy-only keys."
        ),
    )
    yaml_parser.add_argument("--file", required=True, help="Path to env file.")
    yaml_parser.add_argument(
        "--exclude",
        action="append",
        default=[],
        help="Another env key to leave out. Can be passed multiple times.",
    )
    yaml_parser.add_argument(
        "--set",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="Add or override one variable. Can be passed multiple times.",
    )
    yaml_parser.add_argument("output", help="Path to output YAML file.")
    yaml_parser.set_defaults(func=cmd_to_cloudrun_yaml)

    dotenv_parser = subparsers.add_parser(
        "to-dotenv",
        help=(
            "Print the env file as the single-quoted dotenv blob stored in Secret "
            "Manager (without the deploy-only keys). Refuses values python-dotenv "
            "would not read back unchanged, naming keys only."
        ),
    )
    dotenv_parser.add_argument("--file", required=True, help="Path to env file.")
    dotenv_parser.add_argument(
        "--exclude",
        action="append",
        default=[],
        help="Another env key to leave out. Can be passed multiple times.",
    )
    dotenv_parser.add_argument(
        "--sha256",
        action="store_true",
        help="Print only the blob's SHA-256 (after the same checks).",
    )
    dotenv_parser.set_defaults(func=cmd_to_dotenv)

    sha_parser = subparsers.add_parser(
        "sha256",
        help="Print the SHA-256 of stdin (with --base64: of the base64-decoded stdin).",
    )
    sha_parser.add_argument(
        "--base64",
        action="store_true",
        help="stdin is `gcloud secrets versions access --format='get(payload.data)'` output.",
    )
    sha_parser.set_defaults(func=cmd_sha256)

    placeholder_parser = subparsers.add_parser(
        "is-placeholder",
        help="Exit 0 when the supplied value is still an example placeholder.",
    )
    placeholder_parser.add_argument("value", help="Value to inspect.")
    placeholder_parser.set_defaults(func=cmd_is_placeholder)

    check_parser = subparsers.add_parser(
        "check-v3",
        help=(
            "Exit 1 when the env file still selects retired models, sets legacy model "
            "keys or REQUIRE_REASONING_MODELS=true, or has a quoted value python-dotenv "
            "cannot parse. Prints key names only."
        ),
    )
    check_parser.add_argument("--file", required=True, help="Path to env file.")
    check_parser.set_defaults(func=cmd_check_v3)

    migrate_parser = subparsers.add_parser(
        "migrate-v3",
        help=(
            "Back up the env file (0600), then delete the keys check-v3 rejects plus "
            "the keys the app no longer reads. Prints key names only."
        ),
    )
    migrate_parser.add_argument("--file", required=True, help="Path to env file.")
    migrate_parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Only list the keys that would be removed.",
    )
    migrate_parser.set_defaults(func=cmd_migrate_v3)

    upload_parser = subparsers.add_parser(
        "check-upload",
        help=(
            "Read `gcloud meta list-files-for-upload` output on stdin; exit 1 when a "
            "secret file (.env*, .cloudrun-env*, settings.local.json, .claude/, "
            "credentials.json, *.bak, *.bak.<time>) is listed, "
            "or --env-file inside --root is listed or not ignored by git."
        ),
    )
    upload_parser.add_argument("--root", default="", help="The directory gcloud uploads.")
    upload_parser.add_argument("--env-file", default="", help="The env file being deployed.")
    upload_parser.set_defaults(func=cmd_check_upload)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
