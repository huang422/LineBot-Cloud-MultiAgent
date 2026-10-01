"""The image installs a complete, hash-pinned lock on a digest-pinned base.

The Dockerfile runs ``pip install --require-hashes --no-deps -r
requirements.lock``, so a runtime import whose distribution is missing from
the lock (or pinned without a hash) only fails when the image is built or,
worse, when the code path first runs on Cloud Run. These tests catch that
from the source tree.
"""

from __future__ import annotations

import ast
import re
import sys
import unittest
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]

# Top-level import (dotted prefix) -> distribution that provides it.
_IMPORT_TO_DIST = {
    "fastapi": "fastapi",
    "uvicorn": "uvicorn",
    "httpx": "httpx",
    "httpcore": "httpcore",
    "pydantic": "pydantic",
    "pydantic_settings": "pydantic-settings",
    # Comes with pydantic, which pins its exact version.
    "pydantic_core": "pydantic-core",
    "dotenv": "python-dotenv",
    "PIL": "pillow",
    "edge_tts": "edge-tts",
    "mutagen": "mutagen",
    # Imported lazily only (see LazyImportTests).
    "trafilatura": "trafilatura",
    "charset_normalizer": "charset-normalizer",
    "opencc": "opencc-python-reimplemented",
    "google.cloud.storage": "google-cloud-storage",
    "google.cloud.firestore": "google-cloud-firestore",
    "google.cloud.firestore_v1": "google-cloud-firestore",
    "google.auth": "google-auth",
    "google.oauth2": "google-auth",
    "google.api_core": "google-api-core",
    # zoneinfo falls back to this package if the image's system zoneinfo is missing.
    "tzdata": "tzdata",
    # The Colab control plane, lazily (see ColabLibraryTests).
    "colab_cli": "google-colab-cli",
    "requests": "requests",
}

_REQUIRED_RUNTIME_IMPORTS = (
    "fastapi",
    "uvicorn",
    "httpx",
    "pydantic",
    "pydantic_settings",
    "dotenv",
    "PIL",
    "edge_tts",
    "mutagen",
    "trafilatura",
    "charset_normalizer",
    "opencc",
    "google.cloud.storage",
    "google.cloud.firestore",
    "tzdata",
)

# colab/ is the Colab VM package: ours, shipped in the image (the launcher
# uploads it), standard library only.
_FIRST_PARTY = frozenset({"src", "main", "colab"})
_DIGEST_PINNED_BASE = re.compile(r"python:3\.12-slim@sha256:[0-9a-f]{64}")
_HASH = re.compile(r"--hash=sha256:([0-9a-f]{64})")


def _normalize(name: str) -> str:
    return re.sub(r"[-_.]+", "-", name).lower()


def _parse_lock(path: Path) -> dict[str, tuple[str, list[str]]]:
    """Return ``{normalized name: (pinned version, sha256 hashes)}``."""
    text = path.read_text(encoding="utf-8").replace("\\\n", " ")
    pins: dict[str, tuple[str, list[str]]] = {}
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        match = re.match(r"([A-Za-z0-9][A-Za-z0-9._-]*)(?:\[[^\]]*\])?==([^\s;]+)", line)
        if match is None:
            raise AssertionError(f"{path.name}: not an exact pin: {line[:80]}")
        pins[_normalize(match.group(1))] = (match.group(2), _HASH.findall(line))
    return pins


def _third_party_imports() -> dict[str, set[str]]:
    """Map each absolute non-stdlib import in main.py and src/ to its files."""
    found: dict[str, set[str]] = {}
    for path in [_REPO_ROOT / "main.py", *sorted((_REPO_ROOT / "src").rglob("*.py"))]:
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                modules = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
                # `from google.cloud import storage` imports google.cloud.storage.
                modules = [f"{node.module}.{alias.name}" for alias in node.names]
            else:
                continue
            for module in modules:
                top = module.split(".", 1)[0]
                if top in sys.stdlib_module_names or top in _FIRST_PARTY:
                    continue
                found.setdefault(module, set()).add(str(path.relative_to(_REPO_ROOT)))
    return found


def _dist_for(module: str) -> str | None:
    for prefix in sorted(_IMPORT_TO_DIST, key=len, reverse=True):
        if module == prefix or module.startswith(prefix + "."):
            return _IMPORT_TO_DIST[prefix]
    return None


def _non_comment_lines(path: Path) -> list[str]:
    lines = (line.strip() for line in path.read_text(encoding="utf-8").splitlines())
    return [line for line in lines if line and not line.startswith("#")]


class RequirementsLockTests(unittest.TestCase):
    def setUp(self) -> None:
        self.lock = _parse_lock(_REPO_ROOT / "requirements.lock")

    def test_runtime_imports_are_pinned_with_hashes(self) -> None:
        for module in _REQUIRED_RUNTIME_IMPORTS:
            with self.subTest(module=module):
                dist = _normalize(_IMPORT_TO_DIST[module])
                self.assertIn(dist, self.lock, f"{dist} (import {module}) is not locked")
                _, hashes = self.lock[dist]
                self.assertTrue(hashes, f"{dist} is pinned without a hash")

    def test_every_third_party_import_in_src_is_locked(self) -> None:
        for module, files in sorted(_third_party_imports().items()):
            with self.subTest(module=module):
                dist = _dist_for(module)
                self.assertIsNotNone(
                    dist,
                    f"import {module} ({', '.join(sorted(files))}) has no entry in "
                    "_IMPORT_TO_DIST; add its package to requirements.in and re-lock",
                )
                self.assertIn(_normalize(dist), self.lock, f"{dist} is not in requirements.lock")

    def test_every_lock_entry_has_a_hash(self) -> None:
        # --require-hashes rejects the whole install if any single entry lacks one.
        for name in ("requirements.lock", "requirements-dev.lock"):
            for dist, (_, hashes) in _parse_lock(_REPO_ROOT / name).items():
                with self.subTest(lock=name, dist=dist):
                    self.assertTrue(hashes)

    def test_dev_lock_pins_runtime_packages_like_the_image(self) -> None:
        dev = _parse_lock(_REPO_ROOT / "requirements-dev.lock")
        for dist, (version, _) in self.lock.items():
            with self.subTest(dist=dist):
                self.assertIn(dist, dev)
                self.assertEqual(dev[dist][0], version)

    def test_requirements_txt_files_only_include_the_locks(self) -> None:
        self.assertEqual(
            _non_comment_lines(_REPO_ROOT / "requirements.txt"), ["-r requirements.lock"]
        )
        self.assertEqual(
            _non_comment_lines(_REPO_ROOT / "requirements-dev.txt"), ["-r requirements-dev.lock"]
        )

    def test_requirements_in_allows_no_pydantic_settings_without_no_decode(self) -> None:
        # src/config.py imports NoDecode, new in pydantic-settings 2.7.
        text = (_REPO_ROOT / "requirements.in").read_text(encoding="utf-8")
        floor = re.search(r"^pydantic-settings>=([0-9.]+)$", text, re.MULTILINE)
        self.assertIsNotNone(floor, "requirements.in has no pydantic-settings floor")
        self.assertGreaterEqual(tuple(int(part) for part in floor.group(1).split(".")), (2, 7))

    def test_requirements_in_allows_no_pydantic_without_use_default(self) -> None:
        # src/config.py raises pydantic_core's PydanticUseDefault from a
        # before-validator; checked on pydantic 2.10.3 (pydantic-core 2.27.1).
        text = (_REPO_ROOT / "requirements.in").read_text(encoding="utf-8")
        floor = re.search(r"^pydantic>=([0-9.]+)$", text, re.MULTILINE)
        self.assertIsNotNone(floor, "requirements.in has no pydantic floor")
        floor_version = tuple(int(part) for part in floor.group(1).split("."))
        self.assertGreaterEqual(floor_version, (2, 10, 3))
        locked = tuple(int(part) for part in self.lock["pydantic"][0].split("."))
        self.assertGreaterEqual(locked, floor_version, "requirements.lock is below the pydantic floor")


# Imports too heavy for a cold start: only ever inside a function body, so
# they load on first use (trafilatura: about a second and ~50 MB; the Colab
# libraries ~16 MB, and only when COLAB_ENABLED), and only in these files.
_LAZY_ONLY = {
    "trafilatura": ["src/services/search/extract.py"],
    "colab_cli": ["src/services/colab/control_plane.py"],
}


def _module_level_imports(path: Path) -> set[str]:
    """Top-level names imported outside any function body."""
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    found: set[str] = set()

    def visit(node: ast.AST) -> None:
        for child in ast.iter_child_nodes(node):
            if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda)):
                continue
            if isinstance(child, ast.Import):
                found.update(alias.name.split(".", 1)[0] for alias in child.names)
            elif isinstance(child, ast.ImportFrom) and child.level == 0 and child.module:
                found.add(child.module.split(".", 1)[0])
            visit(child)

    visit(tree)
    return found


class LazyImportTests(unittest.TestCase):
    def test_heavy_packages_are_imported_inside_functions_only(self) -> None:
        for path in [_REPO_ROOT / "main.py", *sorted((_REPO_ROOT / "src").rglob("*.py"))]:
            eager = _module_level_imports(path) & set(_LAZY_ONLY)
            with self.subTest(path=str(path.relative_to(_REPO_ROOT))):
                self.assertEqual(eager, set())
        for package, expected in _LAZY_ONLY.items():
            users = [
                str(path.relative_to(_REPO_ROOT))
                for path in sorted((_REPO_ROOT / "src").rglob("*.py"))
                if package in {module.split(".", 1)[0] for module in _imports_in(path)}
            ]
            with self.subTest(package=package):
                self.assertEqual(users, expected)

    def test_starting_the_app_does_not_load_them(self) -> None:
        import subprocess

        code = (
            "import sys, main; "
            "from src.runtime.composition import build_container; "
            "from src.config import Settings; "
            "build_container(Settings(_env_file=None)); "
            f"print(sorted(m for m in {[*_LAZY_ONLY, 'jupyter_kernel_client']!r} if m in sys.modules))"
        )
        result = subprocess.run(
            [sys.executable, "-c", code], cwd=_REPO_ROOT, capture_output=True, text=True, timeout=120
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(result.stdout.strip().splitlines()[-1], "[]")


def _imports_in(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            names.add(node.module)
    return names


# What colab_cli's CLI and optional paths need but colab_cli.client, .runtime
# and .contents never import: left out of both locks with --no-emit-package
# (DEPLOY.md), pyarrow above all (~157 MB, through jupyter-mimetypes).
_NOT_SHIPPED = (
    "fastjsonschema",
    "google-auth-oauthlib",
    "jsonschema",
    "jsonschema-specifications",
    "nbformat",
    "oauthlib",
    "prompt-toolkit",
    "pyarrow",
    "referencing",
    "requests-oauthlib",
    "rpds-py",
    "shellingham",
    "typer",
    "wcwidth",
)
# Top-level modules of those, blocked when importing the colab_cli modules.
_NOT_SHIPPED_MODULES = (
    "fastjsonschema",
    "google_auth_oauthlib",
    "jsonschema",
    "jsonschema_specifications",
    "nbformat",
    "oauthlib",
    "prompt_toolkit",
    "pyarrow",
    "referencing",
    "requests_oauthlib",
    "rpds",
    "shellingham",
    "typer",
    "wcwidth",
)
# The parts of colab_cli that keep global state, write under $HOME, log
# tokens or call exit(): the server never imports them.
_FORBIDDEN_COLAB_MODULES = ("colab_cli.common", "colab_cli.auth", "colab_cli.commands", "colab_cli.cli")


class ColabLibraryTests(unittest.TestCase):
    def test_the_locks_leave_out_what_the_used_modules_never_import(self) -> None:
        for name in ("requirements.lock", "requirements-dev.lock"):
            lock = _parse_lock(_REPO_ROOT / name)
            for dist in _NOT_SHIPPED:
                with self.subTest(lock=name, dist=dist):
                    self.assertNotIn(dist, lock)

    def test_the_colab_libraries_are_pinned_where_it_matters(self) -> None:
        lock = _parse_lock(_REPO_ROOT / "requirements.lock")
        # 0.7.4's metadata pins jupyter-kernel-client==0.9.0 (the PyPI package,
        # not the googlecolab fork); its runtime.py handles both.
        self.assertEqual(lock["google-colab-cli"][0], "0.7.4")
        self.assertEqual(lock["jupyter-kernel-client"][0], "0.9.0")
        self.assertEqual(lock["jupyter-mimetypes"][0], "0.2.0")
        # google-colab-cli#81: an older pyzmq crashes the kernel client.
        self.assertGreaterEqual(int(lock["pyzmq"][0].split(".")[0]), 26)
        websocket = tuple(int(part) for part in lock["websocket-client"][0].split(".")[:2])
        self.assertGreaterEqual(websocket, (1, 6))

    def test_src_never_imports_the_cli_parts_of_colab_cli(self) -> None:
        offenders = sorted(
            f"{path.relative_to(_REPO_ROOT)}: {module}"
            for path in [_REPO_ROOT / "main.py", *sorted((_REPO_ROOT / "src").rglob("*.py"))]
            for module in _imports_in(path)
            if any(
                module == forbidden or module.startswith(forbidden + ".")
                for forbidden in _FORBIDDEN_COLAB_MODULES
            )
        )
        self.assertEqual(offenders, [])

    def test_the_used_modules_import_without_the_left_out_packages(self) -> None:
        import importlib.util
        import subprocess

        if importlib.util.find_spec("colab_cli") is None:
            self.skipTest("google-colab-cli is not installed (see requirements.lock)")
        code = (
            "import importlib.abc, sys\n"
            "class Block(importlib.abc.MetaPathFinder):\n"
            "    def find_spec(self, name, path=None, target=None):\n"
            f"        if name.split('.')[0] in {_NOT_SHIPPED_MODULES!r}:\n"
            "            raise ImportError(f'{name} is not shipped')\n"
            "sys.meta_path.insert(0, Block())\n"
            "import colab_cli.client, colab_cli.runtime, colab_cli.contents\n"
            "import google.oauth2.credentials, google.auth.transport.requests\n"
            f"print(sorted(m for m in sys.modules if m in {_FORBIDDEN_COLAB_MODULES!r}))\n"
        )
        result = subprocess.run(
            [sys.executable, "-c", code], cwd=_REPO_ROOT, capture_output=True, text=True, timeout=120
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(result.stdout.strip().splitlines()[-1], "[]")


class ImagePinningTests(unittest.TestCase):
    def setUp(self) -> None:
        self.dockerfile = (_REPO_ROOT / "Dockerfile").read_text(encoding="utf-8")

    def test_base_image_is_pinned_by_digest(self) -> None:
        from_lines = [
            line.split(None, 1)[1]
            for line in self.dockerfile.splitlines()
            if line.upper().startswith("FROM ")
        ]
        self.assertTrue(from_lines)
        for image in from_lines:
            self.assertRegex(image.strip(), _DIGEST_PINNED_BASE)

    def test_dependencies_install_with_require_hashes(self) -> None:
        installs = [
            line for line in self.dockerfile.splitlines() if re.search(r"\bpip install\b", line)
        ]
        self.assertEqual(len(installs), 1, installs)
        for flag in ("--require-hashes", "--no-deps", "-r requirements.lock"):
            self.assertIn(flag, installs[0])

    def test_cloud_build_tests_on_the_image_base(self) -> None:
        base = _DIGEST_PINNED_BASE.search(self.dockerfile)
        self.assertIsNotNone(base)
        cloudbuild = (_REPO_ROOT / "cloudbuild.yaml").read_text(encoding="utf-8")
        self.assertIn(f"name: '{base.group(0)}'", cloudbuild)
        self.assertIn("--require-hashes --no-deps -r requirements-dev.lock", cloudbuild)


if __name__ == "__main__":
    unittest.main()
