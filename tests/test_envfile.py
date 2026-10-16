from __future__ import annotations

import base64
import contextlib
import importlib.metadata
import io
import json
import os
import re
import shutil
import stat
import subprocess
import tempfile
import tomllib
import unittest
from datetime import datetime
from pathlib import Path
from unittest.mock import patch

import yaml
from dotenv import dotenv_values

from scripts import envfile
from scripts.envfile import (
    V3Report,
    check_v3,
    find_upload_leaks,
    format_v3_report,
    is_placeholder_value,
    migrate_v3,
    parse_env_file,
    parse_env_text,
    remove_env_keys,
)
from src.config import Settings

_REPO_ROOT = Path(__file__).resolve().parents[1]

# Mirrors the fallback copies in scripts/envfile.py; passed explicitly so the
# guard tests do not depend on how far src/config.py has been migrated.
_RETIRED = {"qwen/qwen3.5-397b-a17b", "qwen/qwen3.5-122b-a10b", "nvidia/nv-embedqa-e5-v5"}
_LEGACY = {"ORCHESTRATOR_MODEL", "ORCHESTRATOR_FALLBACK_MODEL"}
# Keys the app no longer reads: the per-model settings the role table replaced.
_REMOVED = {
    "NVIDIA_MODEL",
    "NVIDIA_THINKING_MODEL",
    "NVIDIA_VISION_MODEL",
    "NVIDIA_EMBEDDING_MODEL",
    "AGENT_FALLBACK_MODEL",
    "VISION_FALLBACK_MODEL",
    "IMAGE_GEN_PRIMARY_MODEL",
    "IMAGE_GEN_FALLBACK_MODEL",
    "REQUIRE_REASONING_MODELS",
}

# A v2-era .env: retired ids, legacy router keys, the reasoning filter and
# per-model keys the role table replaced, mixed with unrelated keys that
# must survive a migration byte-for-byte.
_V2_ENV = """# ============================================================
# LINE API
LINE_CHANNEL_SECRET=secret-value-do-not-print
OPENROUTER_API_KEY=sk-or-value-do-not-print
REQUIRE_REASONING_MODELS=true
# 預設模型
NVIDIA_MODEL=qwen/qwen3.5-397b-a17b
NVIDIA_THINKING_MODEL=nvidia/nemotron-3-ultra-550b-a55b
NVIDIA_EMBEDDING_MODEL=nvidia/nv-embedqa-e5-v5
ORCHESTRATOR_MODEL=nvidia/nemotron-3-super-120b-a12b:free
ORCHESTRATOR_FALLBACK_MODEL=nvidia/nemotron-3-super-120b-a12b
ORCHESTRATOR_TEMPERATURE=0
AGENT_FALLBACK_MODEL=nvidia/nemotron-3-super-120b-a12b:free
VISION_FALLBACK_MODEL=qwen/qwen3.8-27b:free
IMAGE_GEN_PRIMARY_MODEL=black-forest-labs/flux.1-dev
IMAGE_GEN_FALLBACK_MODEL=
SCHEDULED_WEEKLY_MESSAGES='[
  {"id":"monday","day_of_week":"mon","hour":21,"minute":0,"message":"明天操一下嗎"}
]'
FIRESTORE_DATABASE="(default)"
"""


def _check(values: dict[str, str]) -> V3Report:
    return check_v3(values, retired=_RETIRED, legacy=_LEGACY, removed=_REMOVED)


def _dotenv_lock_mismatch() -> str:
    """Why the installed python-dotenv is not the reference parser ('' when it is).

    The app reads its env files with the version uv.lock pins; a parity test
    against another version proves nothing.
    """
    lock = _REPO_ROOT / "uv.lock"
    if not lock.is_file():
        return "uv.lock is not part of this checkout"
    packages = tomllib.loads(lock.read_text(encoding="utf-8")).get("package", [])
    pinned = next((p["version"] for p in packages if p.get("name") == "python-dotenv"), None)
    if pinned is None:
        return "uv.lock does not pin python-dotenv"
    installed = importlib.metadata.version("python-dotenv")
    if installed != pinned:
        return f"python-dotenv {installed} is installed, uv.lock pins {pinned}"
    return ""


# Line breaks str.splitlines() knows but python-dotenv does not (it breaks
# only at CR LF, LF and CR), the export and BOM rules, and trailing blanks
# on the opening line of a multi-line quoted value.
_DOTENV_PARITY_CASES = {
    "line separator, unquoted": "BOT_NAME=Hi\u2028there\nB=1\n",
    "paragraph separator and NEL": "A=x\u2029y\x85z\nB=1\n",
    "VT, FF and FS/GS/RS": "A=x\x0by\x0cz\x1cw\x1dv\x1eu\nB=1\n",
    "separators in quoted values": (
        "MODEL_ROLES='{\"m\": \"a\u2028b\u2029c\x85d\"}'\nB=\"x\x0cy\"\nC=3\n"
    ),
    "separator before a comment": "A=x\u2028# note\nB=y\x0c#c\n",
    "a comment holding a separator": "# note\u2028HIDDEN=1\nB=1\n",
    "leading separators": "\u2028A=1\n\x0c\nB=2\n",
    "export with a tab": "export\tA=1\nexport \t B=2\nexported=3\n",
    "trailing blanks on an opening line": "A=\"abc   \nxyz\"\nB='q \t\nr'\nC=1\n",
    "byte order mark": "\ufeffGCP_PROJECT_ID=p\nB=1\n",
    "byte order mark, then export": "\ufeffexport A=1\n",
    "CR LF": "A=1\r\nB='x\r\ny'\r\nC=2\r\n",
    "lone CR": "A=1\rB='x\ry'\rC=2\r",
}


def _run_cli_bytes(argv: list[str], stdin: bytes) -> tuple[int, str, str]:
    out, err = io.StringIO(), io.StringIO()
    fake_stdin = io.TextIOWrapper(io.BytesIO(stdin), encoding="utf-8")
    with (
        patch("sys.stdin", fake_stdin),
        contextlib.redirect_stdout(out),
        contextlib.redirect_stderr(err),
    ):
        code = envfile.main(argv)
    return code, out.getvalue(), err.getvalue()


def _run_cli(argv: list[str], stdin: str = "") -> tuple[int, str, str]:
    out, err = io.StringIO(), io.StringIO()
    with (
        patch("sys.stdin", io.StringIO(stdin)),
        contextlib.redirect_stdout(out),
        contextlib.redirect_stderr(err),
    ):
        code = envfile.main(argv)
    return code, out.getvalue(), err.getvalue()


class EnvfileParsingTests(unittest.TestCase):
    def test_parse_env_file_supports_multiline_single_quoted_values(self) -> None:
        content = """SCHEDULED_WEEKLY_MESSAGES='[
  {"id":"monday_workout_reminder","day_of_week":"mon","hour":21,"minute":0,"message":"明天操一下嗎"},
  {"id":"pineapple_workout_reminder","day_of_week":"mon","hour":21,"minute":30,"message":"啊哈！@鳳梨 還沒回覆督促一下"}
]'
"""

        with tempfile.TemporaryDirectory() as tmpdir:
            env_path = Path(tmpdir) / ".env"
            env_path.write_text(content, encoding="utf-8")

            values = parse_env_file(env_path)

        self.assertIn("SCHEDULED_WEEKLY_MESSAGES", values)
        self.assertTrue(values["SCHEDULED_WEEKLY_MESSAGES"].startswith("[\n"))
        self.assertIn("pineapple_workout_reminder", values["SCHEDULED_WEEKLY_MESSAGES"])

    def test_parse_env_file_strips_quotes_from_default_firestore_database(self) -> None:
        content = 'FIRESTORE_DATABASE="(default)"\nNVIDIA_API_KEY=\n'

        with tempfile.TemporaryDirectory() as tmpdir:
            env_path = Path(tmpdir) / ".env"
            env_path.write_text(content, encoding="utf-8")
            values = parse_env_file(env_path)

        self.assertEqual(values["FIRESTORE_DATABASE"], "(default)")
        self.assertEqual(values["NVIDIA_API_KEY"], "")

    def test_placeholder_detection_rejects_example_values(self) -> None:
        self.assertTrue(is_placeholder_value("your_openrouter_api_key"))
        self.assertTrue(is_placeholder_value("replace_me"))
        self.assertFalse(is_placeholder_value(""))
        self.assertFalse(is_placeholder_value("real-value-123"))

    def test_settings_accept_multiline_scheduler_json(self) -> None:
        content = """SCHEDULED_MESSAGES_ENABLED=true
SCHEDULED_GROUP_ID=C1234567890abcdef1234567890abcdef
SCHEDULED_WEEKLY_MESSAGES='[
  {"id":"monday_workout_reminder","day_of_week":"mon","hour":21,"minute":0,"message":"明天操一下嗎"},
  {"id":"pineapple_workout_reminder","day_of_week":"mon","hour":21,"minute":30,"message":"啊哈！@鳳梨 還沒回覆督促一下"}
]'
SCHEDULED_YEARLY_MESSAGES='[
  {"id":"birthday_reminder_0312","month":3,"day":12,"hour":9,"minute":0,"message":"啊哈！@保羅 生日快樂！"}
]'
"""

        with tempfile.TemporaryDirectory() as tmpdir:
            env_path = Path(tmpdir) / ".env"
            env_path.write_text(content, encoding="utf-8")

            settings = Settings(_env_file=env_path)

        self.assertTrue(settings.scheduled_messages_enabled)
        self.assertEqual(
            [job.id for job in settings.scheduled_weekly_messages],
            ["monday_workout_reminder", "pineapple_workout_reminder"],
        )
        self.assertEqual(
            [job.id for job in settings.scheduled_yearly_messages],
            ["birthday_reminder_0312"],
        )

    def test_inline_comments_are_stripped_like_python_dotenv(self) -> None:
        values = parse_env_text("A=plain # note\nB=with#hash\nC='quoted' # note\n")

        self.assertEqual(values, {"A": "plain", "B": "with#hash", "C": "quoted"})

    def test_quoted_value_followed_by_comment_does_not_swallow_later_keys(self) -> None:
        # The old parser only closed a quote at end of line, so the comment
        # made it consume NVIDIA_MODEL, hiding a retired id from the guard.
        values = parse_env_text(
            'FIRESTORE_DATABASE="(default)" # named DBs are not free\n'
            "NVIDIA_MODEL=qwen/qwen3.5-397b-a17b\n"
        )

        self.assertEqual(values["FIRESTORE_DATABASE"], "(default)")
        self.assertEqual(values["NVIDIA_MODEL"], "qwen/qwen3.5-397b-a17b")

    def test_parser_matches_python_dotenv_on_common_shapes(self) -> None:
        text = (
            "# comment\n"
            "A=plain\n"
            "B=plain # trailing comment\n"
            "C=with#hash\n"
            'D="quoted value" # comment\n'
            "E='single' # comment\n"
            "F=\n"
            'G=""\n'
            "export H=exported\n"
            "I = spaced\n"
            "J='[\n"
            '  {"id":"x","message":"Hi #1"}\n'
            "]'\n"
            "K=' # not a comment '\n"
            "L='it\\'s' # escaped quote\n"
            'M="say \\"hi\\"\\tnow\\nnext"\n'
            "N='C:\\\\temp\\\\x'\n"
            "O='keep \\n literal'\n"
            "P='it\\' # still inside'\n"
        )

        parsed = parse_env_text(text)

        self.assertEqual(parsed, dict(dotenv_values(stream=io.StringIO(text))))
        # The escapes are decoded like the app decodes them.
        self.assertEqual(parsed["L"], "it's")
        self.assertEqual(parsed["M"], 'say "hi"\tnow\nnext')
        self.assertEqual(parsed["N"], "C:\\temp\\x")
        self.assertEqual(parsed["O"], "keep \\n literal")
        self.assertEqual(parsed["P"], "it' # still inside")

    def test_lines_break_only_where_python_dotenv_breaks_them(self) -> None:
        # str.splitlines() also broke at U+2028/2029, NEL, VT, FF and FS/GS/RS:
        # an unquoted value lost its tail, and a quoted JSON value got a raw
        # newline inside a string, which the deployed Settings() refused.
        values = parse_env_text(
            "BOT_NAME=Hi\u2028there\n"
            "A=x\x85y\x0bz\x0cw\x1cv\x1du\x1et\u2029s\n"
            "MODEL_ROLES='{\"m\": \"a\u2028b\"}'\n"
            "export\tEXPORTED=1\n"
            'OPENING="trailing   \nnext"\n'
            "# a comment\u2028HIDDEN=1\n"
        )

        self.assertEqual(
            values,
            {
                "BOT_NAME": "Hi\u2028there",
                "A": "x\x85y\x0bz\x0cw\x1cv\x1du\x1et\u2029s",
                "MODEL_ROLES": '{"m": "a\u2028b"}',
                "EXPORTED": "1",
                "OPENING": "trailing   \nnext",
            },
        )
        self.assertEqual(json.loads(values["MODEL_ROLES"]), {"m": "a\u2028b"})
        self.assertEqual(envfile.invalid_key_lines("A=1\u2028B=2\nC D=3\n"), (2,))

    @unittest.skipIf(_dotenv_lock_mismatch(), _dotenv_lock_mismatch())
    def test_parser_matches_the_locked_python_dotenv_on_tricky_input(self) -> None:
        for name, text in _DOTENV_PARITY_CASES.items():
            with self.subTest(name):
                self.assertEqual(
                    parse_env_text(text), dict(dotenv_values(stream=io.StringIO(text)))
                )
                # Files: what the app reads (open() turns CR LF and CR into LF).
                with tempfile.TemporaryDirectory() as tmpdir:
                    env_path = Path(tmpdir) / ".env"
                    env_path.write_bytes(text.encode("utf-8"))
                    self.assertEqual(
                        parse_env_file(env_path), dict(dotenv_values(env_path, encoding="utf-8"))
                    )

    def test_a_leading_byte_order_mark_is_dropped_like_python_dotenv(self) -> None:
        # Kept, it hid GCP_PROJECT_ID (the scripts fell back to gcloud's
        # default project) and made line 1 no KEY=VALUE (deploy refused).
        with tempfile.TemporaryDirectory() as tmpdir:
            env_path = Path(tmpdir) / ".env"
            env_path.write_bytes("\ufeffGCP_PROJECT_ID=proj\nBOT_NAME=Bot\n".encode())
            results = {
                "get": _run_cli(["get", "--file", str(env_path), "GCP_PROJECT_ID"]),
                "dump": _run_cli(["dump", "--file", str(env_path)]),
                "check-v3": _run_cli(["check-v3", "--file", str(env_path)]),
                "to-dotenv": _run_cli(["to-dotenv", "--file", str(env_path)]),
            }

        self.assertEqual(results["get"], (0, "proj\n", ""))
        dump_fields = ["GCP_PROJECT_ID", "proj", "0", "BOT_NAME", "Bot", "0"]
        self.assertEqual(results["dump"], (0, "".join(f"{field}\x00" for field in dump_fields), ""))
        self.assertEqual(results["check-v3"][0], 0)
        self.assertEqual(results["to-dotenv"], (0, "BOT_NAME='Bot'\n", ""))
        # migrate-v3 keeps the mark, whichever line it drops.
        self.assertEqual(remove_env_keys("\ufeffA=1\nB=2\n", ["a"]), "\ufeffB=2\n")
        self.assertEqual(remove_env_keys("\ufeffA=1\nB=2\n", ["B"]), "\ufeffA=1\n")

    def test_get_and_dump_read_keys_case_insensitively_like_the_app(self) -> None:
        # The app's settings (pydantic-settings, case_sensitive=False) read
        # internal_cron_secret as INTERNAL_CRON_SECRET, and of two spellings
        # the later one. Case-sensitive, the deploy skipped the cron job.
        with tempfile.TemporaryDirectory() as tmpdir:
            env_path = Path(tmpdir) / ".env"
            env_path.write_text("internal_cron_secret=lower\nBot_Name=first\nBOT_NAME=second\nbot_name=third\n")
            results = {
                "get": _run_cli(["get", "--file", str(env_path), "INTERNAL_CRON_SECRET"]),
                "get any case": _run_cli(["get", "--file", str(env_path), "Bot_Name"]),
                "dump": _run_cli(["dump", "--file", str(env_path)]),
            }

        self.assertEqual(results["get"], (0, "lower\n", ""))
        self.assertEqual(results["get any case"], (0, "third\n", ""))
        dump_fields = ["INTERNAL_CRON_SECRET", "lower", "0", "BOT_NAME", "third", "0"]
        self.assertEqual(results["dump"], (0, "".join(f"{field}\x00" for field in dump_fields), ""))

    def test_remove_env_keys_splits_lines_like_python_dotenv(self) -> None:
        text = "A=x\u2028B=y\r\nC='1\r\n2'\rD=4\n"

        self.assertEqual(remove_env_keys(text, ["B"]), text)
        self.assertEqual(remove_env_keys(text, ["C"]), "A=x\u2028B=y\r\nD=4\n")

    def test_statements_python_dotenv_skips_are_skipped_and_named(self) -> None:
        # A broken quote used to swallow every later line into its value, so
        # a key the app reads (INTERNAL_CRON_SECRET here) vanished from the
        # deploy while python-dotenv still read it locally.
        cases = {
            "unterminated": "K='never closed\nINTERNAL_CRON_SECRET=s\nC=t\n",
            "unterminated, a later quote": "K='never closed\nB=s\nC='t'\nD=u\n",
            "text after the quote": "K='x' y\nINTERNAL_CRON_SECRET=s\nC='z'\n",
            "a second quote": "K='a'b'\nB=1\n",
            "double quotes": 'K="x"y\nB=1\n',
            "multi-line, then text": "K='x\ny' tail\nB=1\n",
            "fine, for contrast": "K='x' # note\nB='a\\\\'\nC=\"it\\\"s\"#c\n",
        }
        for name, text in cases.items():
            with self.subTest(name):
                parsed = parse_env_text(text)

                self.assertEqual(parsed, dict(dotenv_values(stream=io.StringIO(text))))
                expected = () if name.startswith("fine") else ("K",)
                self.assertEqual(envfile.malformed_env_keys(text), expected)
        self.assertEqual(
            parse_env_text(cases["unterminated"]), {"INTERNAL_CRON_SECRET": "s", "C": "t"}
        )

    def test_exporters_refuse_a_broken_quote_by_key_name(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            env_path = Path(tmpdir) / ".env"
            env_path.write_text("A=1\nTOKEN='do-not-print\nB=2\n", encoding="utf-8")

            blob_code, blob_out, blob_err = _run_cli(["to-dotenv", "--file", str(env_path)])
            sha_code, sha_out, _ = _run_cli(["to-dotenv", "--file", str(env_path), "--sha256"])
            yaml_code, _, yaml_err = _run_cli(
                ["to-cloudrun-yaml", "--file", str(env_path), str(Path(tmpdir) / "env.yaml")]
            )
            wrote_yaml = (Path(tmpdir) / "env.yaml").exists()

        self.assertEqual((blob_code, blob_out), (1, ""))
        self.assertEqual((sha_code, sha_out), (1, ""))
        self.assertEqual(yaml_code, 1)
        self.assertFalse(wrote_yaml)
        for err in (blob_err, yaml_err):
            self.assertIn("cannot parse the quoted value of TOKEN", err)
            self.assertNotIn("do-not-print", err)

    def test_a_line_that_is_not_key_value_is_refused_by_its_number_never_shown(self) -> None:
        # A key missing its '=' before a value that holds one (base64 padding,
        # a query string) reads as a key with the secret in it.
        text = (
            "A=1\n"
            "OPENROUTER_API_KEY sk-or-v1-do-not-print==\n"
            "c2VjcmV0/LWRvLW5vdC1wcmludA==\n"
            "NOTE do-not-print='never closed\n"
            "B=2\n"
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            env_path = Path(tmpdir) / ".env"
            env_path.write_text(text, encoding="utf-8")
            yaml_path = Path(tmpdir) / "env.yaml"
            results = {
                "to-dotenv": _run_cli(["to-dotenv", "--file", str(env_path), "--sha256"]),
                "to-cloudrun-yaml": _run_cli(["to-cloudrun-yaml", "--file", str(env_path), str(yaml_path)]),
                "check-v3": _run_cli(["check-v3", "--file", str(env_path)]),
            }
            wrote_yaml = yaml_path.exists()

        self.assertEqual(envfile.invalid_key_lines(text), (2, 3, 4))
        self.assertFalse(wrote_yaml)
        for command, (code, out, err) in results.items():
            with self.subTest(command=command):
                self.assertEqual(code, 1)
                self.assertIn("lines 2, 3, 4", out + err)
                for secret in ("do-not-print", "c2VjcmV0", "sk-or-v1"):
                    self.assertNotIn(secret, out + err)

    def test_to_cloudrun_yaml_quotes_values_and_honours_excludes(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            env_path = Path(tmpdir) / "app.env"
            output = Path(tmpdir) / "env.yaml"
            env_path.write_text(
                "LINE_CHANNEL_SECRET=abc:def\nGCP_PROJECT_ID=proj\nBOT_NAME=Bot\n"
                "APP_VERSION=stale\n"
                "SCHEDULED_WEEKLY_MESSAGES='[\n  {\"id\":\"x\"}\n]'\n",
                encoding="utf-8",
            )

            code, _, _ = _run_cli(
                [
                    "to-cloudrun-yaml",
                    "--file",
                    str(env_path),
                    "--exclude",
                    "bot_name",
                    "--set",
                    "APP_VERSION=abc123-20260930",
                    str(output),
                ]
            )
            yaml_lines = output.read_text(encoding="utf-8").splitlines()

        self.assertEqual(code, 0)
        # GCP_PROJECT_ID is deploy-only; --set overrides the file's value.
        self.assertEqual(
            yaml_lines,
            [
                'LINE_CHANNEL_SECRET: "abc:def"',
                'APP_VERSION: "abc123-20260930"',
                'SCHEDULED_WEEKLY_MESSAGES: "[\\n  {\\"id\\":\\"x\\"}\\n]"',
            ],
        )

    def test_to_cloudrun_yaml_rejects_a_bad_set(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            env_path = Path(tmpdir) / "app.env"
            env_path.write_text("A=1\n", encoding="utf-8")

            results = {
                bad: _run_cli(
                    ["to-cloudrun-yaml", "--file", str(env_path), "--set", bad, str(Path(tmpdir) / "o")]
                )
                for bad in ("NOVALUE", "=1", "NOT A KEY=1", "BAD-KEY=1")
            }
            wrote_yaml = (Path(tmpdir) / "o").exists()

        self.assertFalse(wrote_yaml)
        for bad, (code, _, err) in results.items():
            with self.subTest(bad=bad):
                self.assertEqual(code, 2)
                self.assertIn("--set needs KEY=VALUE", err)
                self.assertNotIn(bad, err)

    def test_to_cloudrun_yaml_carries_line_separators_and_controls_unchanged(self) -> None:
        # YAML reads NEL, U+2028 and U+2029 as line breaks (a raw NEL came
        # back as a space) and refuses DEL and the C1 controls.
        values = {
            "BOT_NAME": "Hi\u2028there\u2029and\x85more",
            "CONTROLS": "a\x0bb\x0cc\x1cd\x7fe\x9ff",
            "TEXT": "明天 😀 é \"q\" \\ #1",
        }
        with tempfile.TemporaryDirectory() as tmpdir:
            env_path = Path(tmpdir) / "app.env"
            env_path.write_bytes(
                "BOT_NAME=Hi\u2028there\u2029and\x85more\n"
                "CONTROLS='a\x0bb\x0cc\x1cd\x7fe\x9ff'\n"
                "TEXT=\"明天 😀 é \\\"q\\\" \\\\ #1\"\n".encode()
            )
            output = Path(tmpdir) / "env.yaml"

            code, _, _ = _run_cli(["to-cloudrun-yaml", "--file", str(env_path), str(output)])
            loaded = yaml.safe_load(output.read_text(encoding="utf-8"))

        self.assertEqual(code, 0)
        self.assertEqual(loaded, values)


# Every kind of value the app reads, written the ways people write them.
_ROUND_TRIP_ENV = """# LINE
LINE_CHANNEL_SECRET=abc#def
LINE_CHANNEL_ACCESS_TOKEN="tok/en+with=chars"  # trailing comment
OPENROUTER_API_KEY='sk-or-v1-it\\'s'
NVIDIA_API_KEY=
BOT_NAME="Bot \\"Quoted\\" O'Neil \\\\ #1"
SCHEDULED_MESSAGES_ENABLED=true
SCHEDULED_GROUP_ID=C1234567890abcdef1234567890abcdef
SCHEDULED_WEEKLY_MESSAGES='[
  {"id":"monday","day_of_week":"mon","hour":21,"minute":0,"message":"明天 #1 操一下嗎 it\\'s \\"ok\\""}
]'
SCHEDULED_YEARLY_MESSAGES='[{"id":"bday","month":2,"day":29,"hour":9,"minute":0,"message":"生日 C:\\\\\\\\temp 快樂"}]'
MODEL_ROLES={"utility": [{"provider": "nvidia", "model": "nvidia/nemotron-3-super-120b-a12b"}]}
RATE_LIMIT_MAX_REQUESTS=12
CHAT_TEMPERATURE=0.35
REPLY_WINDOW_SECONDS=75
TTS_ENABLED=false
FIRESTORE_DATABASE="(default)" # named DBs are not free
TAVILY_API_KEY=''
export LOG_LEVEL=warning
GCP_PROJECT_ID=deploy-only
CLOUD_RUN_REGION=us-west1
PORT=9999
"""


class DotenvBlobTests(unittest.TestCase):
    def test_env_file_to_blob_to_settings_round_trip(self) -> None:
        values = parse_env_text(_ROUND_TRIP_ENV)

        blob = envfile.to_dotenv(values)

        # python-dotenv (what pydantic-settings uses) reads back every value,
        # minus the deploy-only keys.
        self.assertEqual(dict(dotenv_values(stream=io.StringIO(blob))), envfile.deployable_values(values))
        for key in ("GCP_PROJECT_ID", "CLOUD_RUN_REGION", "PORT"):
            self.assertNotIn(f"{key}=", blob)
        self.assertTrue(blob.startswith("LINE_CHANNEL_SECRET='abc#def'\n"))

        with tempfile.TemporaryDirectory() as tmpdir:
            env_path = Path(tmpdir) / ".env"
            blob_path = Path(tmpdir) / "app.env"
            env_path.write_text(_ROUND_TRIP_ENV, encoding="utf-8")
            blob_path.write_text(blob, encoding="utf-8")
            with patch.dict(os.environ, {}, clear=True):
                local = Settings(_env_file=env_path).model_dump()
                deployed = Settings(_env_file=blob_path).model_dump()

        # PORT is Cloud Run's own, so it is the one value left behind.
        self.assertEqual(local.pop("port"), 9999)
        self.assertEqual(deployed.pop("port"), Settings.model_fields["port"].default)
        self.assertEqual(deployed, local)
        self.assertEqual(deployed["line_channel_secret"], "abc#def")
        self.assertEqual(deployed["openrouter_api_key"], "sk-or-v1-it's")
        self.assertEqual(deployed["bot_name"], "Bot \"Quoted\" O'Neil \\ #1")
        self.assertEqual(
            deployed["scheduled_weekly_messages"][0]["message"], '明天 #1 操一下嗎 it\'s "ok"'
        )
        self.assertEqual(deployed["scheduled_yearly_messages"][0]["message"], "生日 C:\\temp 快樂")
        self.assertIn('"provider": "nvidia"', deployed["model_roles"])
        self.assertEqual(
            (deployed["reply_window_seconds"], deployed["tts_enabled"]), (75, False)
        )
        self.assertEqual((deployed["log_level"], deployed["firestore_database"]), ("WARNING", "(default)"))

    def test_json_holding_a_line_separator_reaches_the_deployed_settings_intact(self) -> None:
        # The old parser split the quoted value at U+2028 and rejoined it with
        # a raw newline, which is invalid inside a JSON string.
        text = (
            "SCHEDULED_MESSAGES_ENABLED=true\n"
            "SCHEDULED_GROUP_ID=C1234567890abcdef1234567890abcdef\n"
            "SCHEDULED_WEEKLY_MESSAGES='[{\"id\":\"m\",\"day_of_week\":\"mon\",\"hour\":21,"
            "\"minute\":0,\"message\":\"line one\u2028line two\"}]'\n"
            "BOT_NAME=Hi\u2028there\n"
        )
        blob = envfile.to_dotenv(parse_env_text(text))

        with tempfile.TemporaryDirectory() as tmpdir:
            blob_path = Path(tmpdir) / "app.env"
            blob_path.write_bytes(blob.encode("utf-8"))
            with patch.dict(os.environ, {}, clear=True):
                deployed = Settings(_env_file=blob_path)

        self.assertEqual(deployed.scheduled_weekly_messages[0].message, "line one\u2028line two")
        self.assertEqual(deployed.bot_name, "Hi\u2028there")

    def test_the_blob_is_deterministic_and_single_quoted(self) -> None:
        values = {"A": "x", "B": "it's", "C": "a\\b", "D": "line1\nline2", "E": ""}

        blob = envfile.to_dotenv(values)

        self.assertEqual(blob, envfile.to_dotenv(dict(values)))
        self.assertEqual(
            blob, "A='x'\nB='it\\'s'\nC='a\\\\b'\nD='line1\nline2'\nE=''\n"
        )

    def test_values_python_dotenv_would_change_are_refused_by_key_name(self) -> None:
        cases = {
            "INTERPOLATED": "prefix-${HOME}-secret-value",
            "TRAILING": "secret-value\\",
            "CARRIAGE": "secret\rvalue",
        }
        for key, value in cases.items():
            with self.subTest(key=key):
                with self.assertRaises(envfile.DotenvBlobError) as caught:
                    envfile.to_dotenv({"OK": "fine", key: value})
                self.assertIn(key, str(caught.exception))
                self.assertNotIn("secret", str(caught.exception))
        # A key that is no env key may be a secret itself (a line missing its '='): not named.
        with self.assertRaises(envfile.DotenvBlobError) as caught:
            envfile.to_dotenv({"OK": "fine", "API_KEY sk-secret": "="})
        self.assertIn("not a valid env key", str(caught.exception))
        self.assertNotIn("secret", str(caught.exception))

    def test_a_blob_over_the_secret_manager_limit_is_refused(self) -> None:
        with self.assertRaises(envfile.DotenvBlobError) as caught:
            envfile.to_dotenv({"BIG": "x" * envfile.SECRET_PAYLOAD_LIMIT})

        self.assertIn("65536-byte limit", str(caught.exception))

    def test_excludes_are_case_insensitive_and_add_to_the_deploy_only_keys(self) -> None:
        blob = envfile.to_dotenv(
            {"GCP_PROJECT_ID": "p", "gcp_project_id": "p", "EXTRA": "x", "KEEP": "k"},
            exclude=["extra"],
        )

        self.assertEqual(blob, "KEEP='k'\n")

    def test_cli_prints_the_blob_or_its_digest_and_names_keys_on_refusal(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            env_path = Path(tmpdir) / ".env"
            env_path.write_text("A=1\nGCP_PROJECT_ID=p\nB='two words'\n", encoding="utf-8")

            code, out, _ = _run_cli(["to-dotenv", "--file", str(env_path)])
            digest_code, digest, _ = _run_cli(["to-dotenv", "--file", str(env_path), "--sha256"])

            env_path.write_text("TOKEN=abc${X}do-not-print\n", encoding="utf-8")
            bad_code, bad_out, bad_err = _run_cli(["to-dotenv", "--file", str(env_path)])
            missing_code, _, missing_err = _run_cli(
                ["to-dotenv", "--file", str(Path(tmpdir) / "nope.env")]
            )

        self.assertEqual((code, out), (0, "A='1'\nB='two words'\n"))
        self.assertEqual(digest_code, 0)
        self.assertEqual(digest.strip(), envfile.sha256_hex(out.encode("utf-8")))
        self.assertEqual((bad_code, bad_out), (1, ""))
        self.assertIn("TOKEN contains '${'", bad_err)
        self.assertNotIn("do-not-print", bad_err)
        self.assertEqual(missing_code, 2)
        self.assertIn("env file not found", missing_err)

    def test_sha256_of_a_secret_payload_matches_the_local_blob(self) -> None:
        blob = envfile.to_dotenv({"A": "é~?>", "B": "x"}).encode("utf-8")
        # gcloud prints payload.data as URL-safe base64.
        payload = base64.urlsafe_b64encode(blob).decode("ascii").rstrip("=") + "\n"

        code, out, _ = _run_cli_bytes(["sha256", "--base64"], payload.encode("ascii"))
        raw_code, raw_out, _ = _run_cli_bytes(["sha256"], blob)
        bad_code, _, bad_err = _run_cli_bytes(["sha256", "--base64"], b"%%%not base64")

        self.assertEqual((code, out.strip()), (0, envfile.sha256_hex(blob)))
        self.assertEqual((raw_code, raw_out.strip()), (0, envfile.sha256_hex(blob)))
        self.assertEqual(bad_code, 1)
        self.assertIn("not base64", bad_err)


class CheckV3Tests(unittest.TestCase):
    def test_flags_retired_ids_legacy_keys_and_reasoning_filter(self) -> None:
        report = _check(parse_env_text(_V2_ENV))

        self.assertFalse(report.ok)
        self.assertEqual(report.retired, ("NVIDIA_EMBEDDING_MODEL", "NVIDIA_MODEL"))
        self.assertEqual(report.legacy, ("ORCHESTRATOR_FALLBACK_MODEL", "ORCHESTRATOR_MODEL"))
        self.assertTrue(report.require_reasoning)

    def test_keys_no_longer_read_are_reported_but_do_not_fail(self) -> None:
        report = _check(
            {
                "NVIDIA_MODEL": "meta/muse-glimmer-30b",
                "IMAGE_GEN_FALLBACK_MODEL": "",
                "REQUIRE_REASONING_MODELS": "false",
                "BOT_NAME": "Assistant",
            }
        )

        self.assertTrue(report.ok)
        self.assertEqual(
            report.removed, ("IMAGE_GEN_FALLBACK_MODEL", "NVIDIA_MODEL", "REQUIRE_REASONING_MODELS")
        )
        # migrate-v3 still deletes them.
        self.assertEqual(report.keys_to_remove, report.removed)

    def test_clean_v3_env_passes(self) -> None:
        values = {
            "MODEL_ROLES": '{"utility": [{"provider": "nvidia", "model": "x"}]}',
            "MODEL_PATH_PREFERENCE": "api",
            "LINE_CHANNEL_SECRET": "x",
        }

        report = _check(values)

        self.assertTrue(report.ok)
        self.assertEqual(report.keys_to_remove, ())

    def test_a_retired_id_inside_model_roles_fails_by_key_name(self) -> None:
        values = {
            "model_roles": json.dumps(
                {
                    "chat": [{"provider": "nvidia", "model": "nvidia/nemotron-3-ultra-550b-a55b"}],
                    "embed": [{"provider": "nvidia", "model": " NVIDIA/nv-embedqa-e5-v5 "}],
                }
            )
        }

        report = _check(values)

        self.assertFalse(report.ok)
        self.assertEqual(report.retired, ("MODEL_ROLES",))
        # migrate-v3 never rewrites MODEL_ROLES: the operator edits the tier.
        self.assertEqual(report.keys_to_remove, ())
        lines = "\n".join(format_v3_report(report, Path(".env")))
        self.assertIn("MODEL_ROLES names a retired model id", lines)
        self.assertIn("by hand in .env", lines)
        self.assertNotIn("nv-embedqa", lines)
        # migrate-v3 would fix nothing here, so it is not suggested.
        self.assertNotIn("migrate-v3 --file", lines)

    def test_a_gemma_id_in_model_roles_is_noted_but_does_not_fail(self) -> None:
        # The defaults dropped Gemma (2026-10-01); a deployed MODEL_ROLES may still name it.
        values = {
            "MODEL_ROLES": json.dumps(
                {"vision": [{"provider": "openrouter", "model": "google/gemma-4-26b-a4b-it:free"}]}
            )
        }

        report = _check(values)

        self.assertTrue(report.ok)
        self.assertTrue(report.dropped_model_roles)
        lines = "\n".join(format_v3_report(report, Path(".env")))
        self.assertIn("MODEL_ROLES names a Gemma model, which the defaults dropped", lines)
        self.assertIn("by hand in .env", lines)
        self.assertNotIn("gemma-4", lines)
        for clean in ({}, {"MODEL_ROLES": '{"chat": [{"provider": "nvidia", "model": "x"}]}'}, {"MODEL_ROLES": "[]"}):
            with self.subTest(values=clean):
                self.assertFalse(_check(clean).dropped_model_roles)

    def test_an_old_colab_router_gate_is_noted_but_does_not_fail(self) -> None:
        # The default became 0.0 on 2026-10-16; an env file copied earlier sets 0.75 or 0.90.
        for value in ("0.75", "0.9", "0.90"):
            with self.subTest(value=value):
                report = _check({"colab_router_confidence": value})
                self.assertTrue(report.ok)
                self.assertTrue(report.old_router_gate)
                lines = "\n".join(format_v3_report(report, Path(".env")))
                self.assertIn("COLAB_ROUTER_CONFIDENCE keeps an old default", lines)
                self.assertIn("in .env or delete it", lines)
        for value in ("0.0", "0.5", "x", ""):
            with self.subTest(value=value):
                self.assertFalse(_check({"COLAB_ROUTER_CONFIDENCE": value}).old_router_gate)
        self.assertFalse(_check({}).old_router_gate)

    def test_model_roles_that_is_not_a_json_object_fails(self) -> None:
        for raw in ("{not json", '["chat"]'):
            with self.subTest(raw=raw):
                report = _check({"MODEL_ROLES": raw})
                self.assertFalse(report.ok)
                self.assertTrue(report.invalid_model_roles)
                lines = "\n".join(format_v3_report(report, Path(".env")))
                self.assertIn("MODEL_ROLES is not a JSON object", lines)
                self.assertIn("by hand in .env", lines)
                self.assertNotIn("migrate-v3 --file", lines)
                self.assertNotIn(raw, lines)
        self.assertTrue(_check({"MODEL_ROLES": "  "}).ok)

    def test_the_migrate_hint_appears_only_when_migrate_v3_fixes_a_failure(self) -> None:
        roles = json.dumps({"chat": [{"provider": "nvidia", "model": "qwen/qwen3.5-122b-a10b"}]})
        cases = {
            "MODEL_ROLES and a legacy key": ({"MODEL_ROLES": roles, "ORCHESTRATOR_MODEL": "x"}, True),
            "invalid MODEL_ROLES and a retired key": (
                {"MODEL_ROLES": "[]", "NVIDIA_MODEL": "qwen/qwen3.5-122b-a10b"},
                True,
            ),
            "the reasoning filter": ({"REQUIRE_REASONING_MODELS": "true"}, True),
            "MODEL_ROLES and a key no longer read": ({"MODEL_ROLES": roles, "NVIDIA_MODEL": "m"}, False),
        }
        for name, (values, hinted) in cases.items():
            with self.subTest(name):
                report = _check(values)
                lines = "\n".join(format_v3_report(report, Path(".env")))
                self.assertFalse(report.ok)
                self.assertEqual("migrate-v3 --file .env" in lines, hinted)

    def test_keys_and_retired_values_match_case_insensitively(self) -> None:
        report = _check(
            {
                "orchestrator_model": "x",
                "nvidia_vision_model": "QWEN/qwen3.5-122b-a10b",
                "require_reasoning_models": "Yes",
            }
        )

        self.assertEqual(report.legacy, ("ORCHESTRATOR_MODEL",))
        self.assertEqual(report.retired, ("NVIDIA_VISION_MODEL",))
        self.assertEqual(report.removed, ("NVIDIA_VISION_MODEL", "REQUIRE_REASONING_MODELS"))
        self.assertTrue(report.require_reasoning)

    def test_cli_prints_key_names_and_fix_command_but_never_values(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            env_path = Path(tmpdir) / ".env"
            env_path.write_text(_V2_ENV, encoding="utf-8")

            with (
                patch.object(envfile, "retired_model_ids", return_value=frozenset(_RETIRED)),
                patch.object(envfile, "legacy_env_keys", return_value=frozenset(_LEGACY)),
                patch.object(envfile, "removed_env_keys", return_value=frozenset(_REMOVED)),
            ):
                code, out, _ = _run_cli(["check-v3", "--file", str(env_path)])

        self.assertEqual(code, 1)
        for key in ("NVIDIA_MODEL", "ORCHESTRATOR_MODEL", "REQUIRE_REASONING_MODELS", "NVIDIA_THINKING_MODEL"):
            self.assertIn(key, out)
        self.assertIn("no longer read", out)
        self.assertIn("migrate-v3 --file", out)
        for value in (
            "qwen/qwen3.5-397b-a17b",
            "nemotron-3-super-120b-a12b",
            "secret-value-do-not-print",
            "sk-or-value-do-not-print",
        ):
            self.assertNotIn(value, out)

    def test_a_broken_quote_fails_by_key_name_without_the_migrate_hint(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            env_path = Path(tmpdir) / ".env"
            env_path.write_text(
                "LINE_CHANNEL_SECRET='secret-value-do-not-print' extra\n"
                'OPENROUTER_API_KEY="sk-or-value-do-not-print\n'
                "BOT_NAME=Bot\n",
                encoding="utf-8",
            )

            code, out, _ = _run_cli(["check-v3", "--file", str(env_path)])

        self.assertEqual(code, 1)
        self.assertIn(
            "cannot parse the quoted value of: LINE_CHANNEL_SECRET, OPENROUTER_API_KEY", out
        )
        # migrate-v3 cannot fix quoting, so it is not suggested.
        self.assertNotIn("migrate-v3", out)
        self.assertNotIn("do-not-print", out)
        self.assertTrue(_check({"A": "1"}).ok)
        self.assertFalse(check_v3({}, retired=(), legacy=(), removed=(), malformed=["K"]).ok)

    def test_cli_reports_missing_file(self) -> None:
        code, _, err = _run_cli(["check-v3", "--file", "/nonexistent/.env"])

        self.assertEqual(code, 2)
        self.assertIn("env file not found", err)

    def test_env_example_passes_the_v3_guard(self) -> None:
        example = _REPO_ROOT / ".env.example"
        if not example.is_file():  # .gcloudignore keeps it out of Cloud Build uploads
            self.skipTest(".env.example is not part of this checkout")

        report = check_v3(parse_env_file(example))

        self.assertTrue(report.ok, report)
        self.assertEqual(report.removed, ())

    def test_env_example_keys_exist_in_settings(self) -> None:
        example = _REPO_ROOT / ".env.example"
        if not example.is_file():
            self.skipTest(".env.example is not part of this checkout")
        # Read by the deploy / provisioning scripts or the GCP SDK, not by Settings.
        deploy_only = envfile.DEPLOY_ONLY_KEYS - {"PORT", "HOST"}

        values = parse_env_file(example)

        unknown = sorted(
            key for key in values if key.lower() not in Settings.model_fields and key not in deploy_only
        )
        self.assertEqual(unknown, [])
        # Models live in the role table now, not in per-model keys.
        self.assertEqual([key for key in values if key.endswith(("_MODEL", "_MODEL_2"))], [])

    def test_env_example_role_table_switches_match_the_code_defaults(self) -> None:
        example = _REPO_ROOT / ".env.example"
        if not example.is_file():
            self.skipTest(".env.example is not part of this checkout")

        values = parse_env_file(example)

        for key in (
            "MODEL_PATH_PREFERENCE",
            "COLAB_CHAT_THINK",
            "COLAB_DEEP_THINK",
            "NVIDIA_THINKING_BUDGET",
            "OPENROUTER_FREE_DAILY_LIMIT",
            "TURN_DEADLINE_SECONDS",
        ):
            with self.subTest(key=key):
                self.assertEqual(values[key], str(Settings.model_fields[key.lower()].default))
        # FORCE_TIER is test-only: present, and off.
        self.assertEqual(values["FORCE_TIER"], "")


class DeployScriptSecretTests(unittest.TestCase):
    def test_cron_token_gcloud_calls_disable_file_logging(self) -> None:
        # gcloud writes every parsed flag to ~/.config/gcloud/logs at DEBUG
        # level, so a flag carrying INTERNAL_CRON_SECRET must not be logged.
        script = (_REPO_ROOT / "scripts" / "deploy_cloud_run.sh").read_text(encoding="utf-8")
        commands = [
            line.strip()
            for line in script.replace("\\\n", " ").splitlines()
            if "X-Cron-Token=" in line
        ]

        self.assertEqual(len(commands), 2)
        for command in commands:
            with self.subTest(command=command[:60]):
                self.assertRegex(
                    command,
                    r"^CLOUDSDK_CORE_DISABLE_FILE_LOGGING=true\s+gcloud scheduler jobs "
                    r"(update|create) http ",
                )


class SharedConstantsTests(unittest.TestCase):
    """The fallback copies must match src.config, the source of truth."""

    def test_fallback_retired_ids_and_legacy_keys_match_src_config(self) -> None:
        from src import config

        self.assertEqual(frozenset(config.RETIRED_MODEL_IDS), envfile._FALLBACK_RETIRED_MODEL_IDS)
        self.assertEqual(frozenset(config.LEGACY_ENV_KEYS), envfile._FALLBACK_LEGACY_ENV_KEYS)
        self.assertEqual(envfile._FALLBACK_RETIRED_MODEL_IDS, frozenset(_RETIRED))

    def test_fallback_removed_keys_match_src_config(self) -> None:
        from src import config

        self.assertEqual(frozenset(config.REMOVED_ENV_KEYS), envfile._FALLBACK_REMOVED_ENV_KEYS)
        self.assertEqual(envfile.removed_env_keys(), envfile._FALLBACK_REMOVED_ENV_KEYS)
        self.assertLessEqual(_REMOVED, envfile._FALLBACK_REMOVED_ENV_KEYS)
        # None of them is read by Settings any more.
        self.assertEqual(
            [key for key in envfile._FALLBACK_REMOVED_ENV_KEYS if key.lower() in Settings.model_fields],
            [],
        )

    def test_guard_falls_back_to_local_copies_without_src_config(self) -> None:
        with patch.object(envfile, "_load_config_module", return_value=None):
            self.assertEqual(envfile.retired_model_ids(), envfile._FALLBACK_RETIRED_MODEL_IDS)
            self.assertEqual(envfile.legacy_env_keys(), envfile._FALLBACK_LEGACY_ENV_KEYS)
            self.assertEqual(envfile.removed_env_keys(), envfile._FALLBACK_REMOVED_ENV_KEYS)


class MigrateV3Tests(unittest.TestCase):
    def setUp(self) -> None:
        patchers = (
            patch.object(envfile, "retired_model_ids", return_value=frozenset(_RETIRED)),
            patch.object(envfile, "legacy_env_keys", return_value=frozenset(_LEGACY)),
            patch.object(envfile, "removed_env_keys", return_value=frozenset(_REMOVED)),
        )
        for patcher in patchers:
            patcher.start()
            self.addCleanup(patcher.stop)
        tmpdir = tempfile.TemporaryDirectory()
        self.addCleanup(tmpdir.cleanup)
        self.env_path = Path(tmpdir.name) / ".env"

    def test_removes_offending_and_unread_keys_after_private_backup(self) -> None:
        self.env_path.write_text(_V2_ENV, encoding="utf-8")
        os.chmod(self.env_path, 0o644)

        result = migrate_v3(self.env_path, now=datetime(2026, 9, 29, 18, 30, 5))

        self.assertEqual(
            result.removed,
            (
                "AGENT_FALLBACK_MODEL",
                "IMAGE_GEN_FALLBACK_MODEL",
                "IMAGE_GEN_PRIMARY_MODEL",
                "NVIDIA_EMBEDDING_MODEL",
                "NVIDIA_MODEL",
                "NVIDIA_THINKING_MODEL",
                "ORCHESTRATOR_FALLBACK_MODEL",
                "ORCHESTRATOR_MODEL",
                "REQUIRE_REASONING_MODELS",
                "VISION_FALLBACK_MODEL",
            ),
        )
        backup = self.env_path.with_name(".env.bak.20260929183005")
        self.assertEqual(result.backup, backup.resolve())
        self.assertEqual(backup.read_text(encoding="utf-8"), _V2_ENV)
        self.assertEqual(stat.S_IMODE(backup.stat().st_mode), 0o600)
        self.assertEqual(stat.S_IMODE(self.env_path.stat().st_mode), 0o644)

        migrated = self.env_path.read_text(encoding="utf-8")
        values = parse_env_text(migrated)
        self.assertTrue(_check(values).ok)
        self.assertEqual(_check(values).keys_to_remove, ())
        for key in result.removed:
            self.assertNotIn(key, values)
        # Everything else survives untouched, including comments and multi-line JSON.
        self.assertIn("# 預設模型\n", migrated)
        self.assertIn("ORCHESTRATOR_TEMPERATURE=0\n", migrated)
        self.assertIn('FIRESTORE_DATABASE="(default)"\n', migrated)
        self.assertIn('"message":"明天操一下嗎"}\n]\'\n', migrated)
        self.assertEqual(values["LINE_CHANNEL_SECRET"], "secret-value-do-not-print")

    def test_clean_file_is_left_alone_without_backup(self) -> None:
        clean = "MODEL_PATH_PREFERENCE=api\nBOT_NAME=Assistant\n"
        self.env_path.write_text(clean, encoding="utf-8")

        result = migrate_v3(self.env_path)

        self.assertEqual(result.removed, ())
        self.assertIsNone(result.backup)
        self.assertEqual(self.env_path.read_text(encoding="utf-8"), clean)
        self.assertEqual(sorted(p.name for p in self.env_path.parent.iterdir()), [".env"])

    def test_dry_run_lists_keys_without_writing(self) -> None:
        self.env_path.write_text(_V2_ENV, encoding="utf-8")

        code, out, _ = _run_cli(["migrate-v3", "--file", str(self.env_path), "--dry-run"])

        self.assertEqual(code, 0)
        self.assertIn("would remove: AGENT_FALLBACK_MODEL", out)
        self.assertEqual(self.env_path.read_text(encoding="utf-8"), _V2_ENV)
        self.assertEqual(sorted(p.name for p in self.env_path.parent.iterdir()), [".env"])

    def test_cli_prints_only_key_and_file_names(self) -> None:
        self.env_path.write_text(_V2_ENV, encoding="utf-8")

        code, out, _ = _run_cli(["migrate-v3", "--file", str(self.env_path)])

        self.assertEqual(code, 0)
        self.assertIn("removed: AGENT_FALLBACK_MODEL", out)
        self.assertIn(".env.bak.", out)
        for value in (
            "qwen/qwen3.5-397b-a17b",
            "nemotron-3-super-120b-a12b",
            "secret-value-do-not-print",
            "sk-or-value-do-not-print",
        ):
            self.assertNotIn(value, out)

    def test_remove_env_keys_handles_export_prefix_multiline_values_and_crlf(self) -> None:
        text = (
            "KEEP=1\r\n"
            "export DROP=x\r\n"
            "drop_json='[\r\n"
            '  {"id":"a"}\r\n'
            "]'\r\n"
            "# DROP=commented-out stays\r\n"
            "TAIL=2\r\n"
        )

        result = remove_env_keys(text, ["DROP", "DROP_JSON"])

        self.assertEqual(result, "KEEP=1\r\n# DROP=commented-out stays\r\nTAIL=2\r\n")


class UploadLeakGuardTests(unittest.TestCase):
    def test_flags_secret_files_anywhere_in_the_upload(self) -> None:
        leaking = [
            ".env",
            "sub/dir/.env",
            ".env.local",
            ".env.bak.20260929183005",
            ".cloudrun-env.generated.abc123.yaml",
            "nested/.cloudrun-env.generated.x.yaml",
            ".claude/settings.local.json",
            ".claude/commands/deploy.md",
            "settings.local.json",
            "./.env",
            "credentials.json",
            "infra/credentials.json",
            "notes.bak",
            "config/.env.bak",
            # migrate-v3's backup of an env file named otherwise.
            "config/prod.env.bak.20261001120000",
        ]
        safe = [
            ".env.example",
            "docs/.env.example",
            "main.py",
            "src/config.py",
            "scripts/envfile.py",
            "tests/test_envfile.py",
            "prod.env",
            "src/credentials.py",
            "docs/credentials.json.md",
            "docs/backup.md",
            "notes.bak.md",
        ]

        leaks = find_upload_leaks([*leaking, *safe])

        self.assertEqual(leaks, [path.removeprefix("./") for path in leaking])

    def test_every_secret_pattern_in_gcloudignore_is_guarded(self) -> None:
        # .gcloudignore says the deploy refuses whatever its "Secrets" block
        # names; one path per pattern must trip the guard.
        gcloudignore = _REPO_ROOT / ".gcloudignore"
        if not gcloudignore.is_file():  # it ignores itself, so Cloud Build has no copy
            self.skipTest(".gcloudignore is not part of this checkout")
        text = gcloudignore.read_text(encoding="utf-8")
        block = text.split("\n# Secrets.", 1)[1].split("\n", 1)[1].split("\n\n", 1)[0]
        patterns = [line for line in block.splitlines() if line and not line.startswith("#")]
        self.assertIn("credentials.json", patterns)

        for pattern in patterns:
            with self.subTest(pattern=pattern):
                sample = re.sub(r"\[(.)[^]]*\]", r"\1", pattern).replace("*", "x")
                sample += "x" if pattern.endswith("/") else ""
                expected = [sample, f"sub/{sample}"]
                self.assertEqual(find_upload_leaks(expected), expected)

    def test_env_example_inside_claude_dir_is_still_a_leak(self) -> None:
        self.assertEqual(find_upload_leaks([".claude/.env.example"]), [".claude/.env.example"])

    def test_the_deployed_env_file_inside_the_repository_is_a_leak_whatever_its_name(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "repo"
            (root / "config").mkdir(parents=True)
            env_file = root / "config" / "prod.env"
            env_file.write_text("LINE_CHANNEL_SECRET=do-not-print\n", encoding="utf-8")
            outside = Path(tmp) / "prod.env"
            outside.write_text("LINE_CHANNEL_SECRET=do-not-print\n", encoding="utf-8")

            for spelled in (env_file, root / "config" / ".." / "config" / "prod.env"):
                with self.subTest(env_file=str(spelled)):
                    argv = ["check-upload", "--root", str(root), "--env-file", str(spelled)]
                    code, _, err = _run_cli(argv, stdin="main.py\nconfig/prod.env\n")
                    self.assertEqual(code, 1)
                    self.assertIn("  config/prod.env\n", err)
                    self.assertNotIn("do-not-print", err)
                    # Left out of the upload (.gcloudignore), it may stay.
                    self.assertEqual(_run_cli(argv, stdin="main.py\n")[0], 0)
            # An env file outside the repository is never uploaded.
            argv = ["check-upload", "--root", str(root), "--env-file", str(outside)]
            self.assertEqual(_run_cli(argv, stdin="main.py\nprod.env\n")[0], 0)

    @unittest.skipUnless(shutil.which("git"), "needs git")
    def test_the_deployed_env_file_inside_a_git_repository_must_be_ignored_by_git(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "repo"
            # The user's own git config (a global excludes file) must not count.
            isolated = {"GIT_CONFIG_GLOBAL": os.devnull, "GIT_CONFIG_NOSYSTEM": "1", "XDG_CONFIG_HOME": tmp}
            with patch.dict(os.environ, isolated):
                subprocess.run(["git", "init", "-q", str(root)], check=True)
                env_file = root / "prod.env"
                env_file.write_text("LINE_CHANNEL_SECRET=do-not-print\n", encoding="utf-8")
                argv = ["check-upload", "--root", str(root), "--env-file", str(env_file)]

                code, _, err = _run_cli(argv, stdin="main.py\n")
                self.assertEqual(code, 1)
                self.assertIn("prod.env", err)
                self.assertIn(".gitignore", err)
                self.assertNotIn("do-not-print", err)

                (root / ".gitignore").write_text("prod.env\n", encoding="utf-8")
                self.assertEqual(_run_cli(argv, stdin="main.py\n")[0], 0)

    def test_check_upload_cli_exit_codes(self) -> None:
        ok_code, ok_out, _ = _run_cli(["check-upload"], stdin="main.py\nsrc/config.py\n")
        bad_code, _, bad_err = _run_cli(
            ["check-upload"], stdin="main.py\n.claude/settings.local.json\n.env\n"
        )

        self.assertEqual(ok_code, 0)
        self.assertIn("ok", ok_out)
        self.assertEqual(bad_code, 1)
        self.assertIn(".claude/settings.local.json", bad_err)
        self.assertIn("  .env\n", bad_err)
