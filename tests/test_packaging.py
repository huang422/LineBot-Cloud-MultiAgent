"""The image installs a complete, hash-pinned lock on a digest-pinned base.

pyproject.toml lists the dependencies, uv.lock pins them with hashes, and the
Dockerfile installs ``uv export --frozen --no-dev`` of it with ``pip install
--require-hashes --no-deps``. A runtime import whose distribution is missing
from the lock (or pinned without a hash) only fails when the image is built
or, worse, when the code path first runs on Cloud Run. These tests catch that
from the source tree.
"""

from __future__ import annotations

import ast
import functools
import re
import shutil
import sys
import tomllib
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
_DIGEST_PINNED_UV = re.compile(r"ghcr\.io/astral-sh/uv:[0-9.]+@sha256:[0-9a-f]{64}")
_PROJECT = "linebot-cloud-agent"


def _normalize(name: str) -> str:
    return re.sub(r"[-_.]+", "-", name).lower()


@functools.cache
def _pyproject() -> dict:
    return tomllib.loads((_REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8"))


@functools.cache
def _uv_lock() -> dict:
    return tomllib.loads((_REPO_ROOT / "uv.lock").read_text(encoding="utf-8"))


def _locked() -> dict[str, tuple[str, list[str]]]:
    """``{name: (pinned version, sha256 hashes)}`` for every package in uv.lock."""
    pins: dict[str, tuple[str, list[str]]] = {}
    for package in _uv_lock()["package"]:
        if package["name"] == _PROJECT:
            continue
        files = [*package.get("wheels", []), *([package["sdist"]] if "sdist" in package else [])]
        pins[package["name"]] = (package["version"], [f["hash"] for f in files if "hash" in f])
    return pins


def _closure(*, dev: bool) -> set[str]:
    """Names uv export installs: the project's dependencies (and the dev group)."""
    packages = {package["name"]: package for package in _uv_lock()["package"]}
    root = packages[_PROJECT]
    todo = [dep["name"] for dep in root.get("dependencies", [])]
    if dev:
        todo += [dep["name"] for deps in root.get("dev-dependencies", {}).values() for dep in deps]
    seen: set[str] = set()
    while todo:
        name = todo.pop()
        if name in seen:
            continue
        seen.add(name)
        package = packages[name]
        todo += [dep["name"] for dep in package.get("dependencies", [])]
        for extra in package.get("optional-dependencies", {}).values():
            todo += [dep["name"] for dep in extra]
    return seen


def _floor(name: str) -> tuple[int, ...]:
    for spec in _pyproject()["project"]["dependencies"]:
        match = re.fullmatch(rf"{re.escape(name)}>=([0-9.]+)", spec)
        if match:
            return tuple(int(part) for part in match.group(1).split("."))
    raise AssertionError(f"pyproject.toml has no {name} floor")


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


class RequirementsLockTests(unittest.TestCase):
    def setUp(self) -> None:
        self.lock = _locked()
        self.runtime = _closure(dev=False)

    def test_runtime_imports_are_pinned_with_hashes(self) -> None:
        for module in _REQUIRED_RUNTIME_IMPORTS:
            with self.subTest(module=module):
                dist = _normalize(_IMPORT_TO_DIST[module])
                self.assertIn(dist, self.runtime, f"{dist} (import {module}) is not a runtime dependency")
                _, hashes = self.lock[dist]
                self.assertTrue(hashes, f"{dist} is pinned without a hash")

    def test_every_third_party_import_in_src_is_locked(self) -> None:
        for module, files in sorted(_third_party_imports().items()):
            with self.subTest(module=module):
                dist = _dist_for(module)
                self.assertIsNotNone(
                    dist,
                    f"import {module} ({', '.join(sorted(files))}) has no entry in "
                    "_IMPORT_TO_DIST; add its package to pyproject.toml and re-lock",
                )
                self.assertIn(_normalize(dist), self.runtime, f"{dist} is not a runtime dependency in uv.lock")

    def test_every_lock_entry_has_a_hash(self) -> None:
        # --require-hashes rejects the whole install if any single entry lacks one.
        for dist, (_, hashes) in self.lock.items():
            with self.subTest(dist=dist):
                self.assertTrue(hashes)

    def test_dev_tools_stay_out_of_the_image(self) -> None:
        dev_only = _closure(dev=True) - self.runtime
        self.assertLessEqual({"pytest", "pytest-asyncio", "ruff"}, dev_only)

    def test_the_lock_matches_pyproject(self) -> None:
        # Cloud Build and the Dockerfile export with --frozen: a dependency
        # edited in pyproject.toml without `uv lock` must fail here instead.
        from packaging.requirements import Requirement
        from packaging.specifiers import SpecifierSet

        def key(spec: str | dict) -> tuple[str, tuple[str, ...], str]:
            if isinstance(spec, dict):
                specifier = str(SpecifierSet(spec.get("specifier", "")))
                return (spec["name"], tuple(sorted(spec.get("extras", []))), specifier)
            req = Requirement(spec)
            return (_normalize(req.name), tuple(sorted(req.extras)), str(req.specifier))

        project = _pyproject()
        self.assertEqual(project["project"]["requires-python"], _uv_lock()["requires-python"])
        root = next(p for p in _uv_lock()["package"] if p["name"] == _PROJECT)
        metadata = root["metadata"]
        self.assertEqual(
            sorted(map(key, project["project"]["dependencies"])),
            sorted(map(key, metadata["requires-dist"])),
            "pyproject.toml changed since uv.lock was made: run `uv lock`",
        )
        self.assertEqual(
            {group: sorted(map(key, specs)) for group, specs in project["dependency-groups"].items()},
            {group: sorted(map(key, specs)) for group, specs in metadata.get("requires-dev", {}).items()},
            "pyproject.toml changed since uv.lock was made: run `uv lock`",
        )
        self.assertEqual(
            sorted(project["tool"]["uv"]["exclude-dependencies"]),
            sorted(_uv_lock()["manifest"]["excludes"]),
            "pyproject.toml changed since uv.lock was made: run `uv lock`",
        )

    def test_pyproject_allows_no_pydantic_settings_without_no_decode(self) -> None:
        # src/config.py imports NoDecode, new in pydantic-settings 2.7.
        self.assertGreaterEqual(_floor("pydantic-settings"), (2, 7))

    def test_pyproject_allows_no_pydantic_without_use_default(self) -> None:
        # src/config.py raises pydantic_core's PydanticUseDefault from a
        # before-validator; checked on pydantic 2.10.3 (pydantic-core 2.27.1).
        floor = _floor("pydantic")
        self.assertGreaterEqual(floor, (2, 10, 3))
        locked = tuple(int(part) for part in self.lock["pydantic"][0].split("."))
        self.assertGreaterEqual(locked, floor, "uv.lock is below the pydantic floor")


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
# and .contents never import: left out of the resolution by pyproject.toml's
# [tool.uv] exclude-dependencies, pyarrow above all (~157 MB, through
# jupyter-mimetypes). Only the first five are listed there; the rest are
# reached only through them.
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
    def test_the_lock_leaves_out_what_the_used_modules_never_import(self) -> None:
        lock = _locked()
        for dist in _NOT_SHIPPED:
            with self.subTest(dist=dist):
                self.assertNotIn(dist, lock)
        excluded = set(_pyproject()["tool"]["uv"]["exclude-dependencies"])
        self.assertEqual(
            excluded,
            {"google-auth-oauthlib", "nbformat", "prompt-toolkit", "pyarrow", "typer"},
        )
        self.assertLessEqual(excluded, set(_NOT_SHIPPED))

    def test_the_colab_libraries_are_pinned_where_it_matters(self) -> None:
        lock = _locked()
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
            self.skipTest("google-colab-cli is not installed (see DEPLOY.md, 本機測試)")
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
        self.assertEqual(len(from_lines), 2, from_lines)
        for image in from_lines:
            self.assertRegex(image.strip(), _DIGEST_PINNED_BASE)
        # Both stages on the same base: the export and the install agree on
        # the interpreter.
        self.assertEqual(len(set(_DIGEST_PINNED_BASE.findall(self.dockerfile))), 1)

    def test_dependencies_install_with_require_hashes(self) -> None:
        installs = [
            line for line in self.dockerfile.splitlines() if re.search(r"\bpip install\b", line)
        ]
        self.assertEqual(len(installs), 1, installs)
        for flag in ("--require-hashes", "--no-deps", "-r /tmp/requirements.txt"):
            self.assertIn(flag, installs[0])
        # The exported file does not stay in the image.
        self.assertIn("&& rm /tmp/requirements.txt", self.dockerfile)
        exports = [line for line in self.dockerfile.splitlines() if "uv export" in line]
        self.assertEqual(len(exports), 1, exports)
        for flag in ("--frozen", "--no-dev", "--no-emit-project", "-o /requirements.txt"):
            self.assertIn(flag, exports[0])

    def test_uv_is_pinned_by_digest_and_only_in_the_build_stage(self) -> None:
        stages = re.split(r"(?m)^FROM ", self.dockerfile)[1:]
        self.assertEqual(len(stages), 2)
        self.assertRegex(stages[0], _DIGEST_PINNED_UV)
        self.assertNotIn("astral-sh/uv", stages[1])
        # Nothing in the image stage runs or copies uv (comments may name uv.lock).
        instructions = [line for line in stages[1].splitlines() if not line.lstrip().startswith("#")]
        self.assertEqual([line for line in instructions if re.search(r"\buv\b(?!\.lock)", line)], [])
        # The pinned uv is one pyproject.toml's required-version accepts
        # (exclude-dependencies needs it).
        from packaging.specifiers import SpecifierSet

        version = re.search(r"astral-sh/uv:([0-9.]+)@", stages[0]).group(1)
        self.assertIn(version, SpecifierSet(_pyproject()["tool"]["uv"]["required-version"]))

    def test_cloud_build_tests_on_the_image_base(self) -> None:
        base = _DIGEST_PINNED_BASE.search(self.dockerfile)
        uv = _DIGEST_PINNED_UV.search(self.dockerfile)
        self.assertIsNotNone(base)
        self.assertIsNotNone(uv)
        cloudbuild = (_REPO_ROOT / "cloudbuild.yaml").read_text(encoding="utf-8")
        self.assertIn(f"name: '{base.group(0)}'", cloudbuild)
        self.assertIn(f"name: '{uv.group(0)}'", cloudbuild)
        # One uv and one base everywhere: bumping one without the other fails.
        self.assertEqual(set(_DIGEST_PINNED_UV.findall(cloudbuild)), {uv.group(0)})
        self.assertEqual(set(_DIGEST_PINNED_BASE.findall(cloudbuild)), {base.group(0)})
        # The export step keeps the dev group (pytest, ruff); the image's has --no-dev.
        export = re.search(r"args: \[export, [^\]]*\]", cloudbuild)
        self.assertIsNotNone(export)
        self.assertIn("--frozen", export.group(0))
        self.assertNotIn("--no-dev", export.group(0))
        self.assertIn("--require-hashes --no-deps -r /deps/dev.txt", cloudbuild)

    def test_the_lock_reaches_the_build(self) -> None:
        # The image's first stage and Cloud Build's export step read both.
        from tests.test_ignore_files import _docker_excludes, _git_ignored

        kept = ("pyproject.toml", "uv.lock")
        self.assertEqual([path for path in kept if _docker_excludes(path)], [])
        gcloudignore = _REPO_ROOT / ".gcloudignore"
        if not gcloudignore.is_file():  # it ignores itself, so Cloud Build has no copy
            self.skipTest(".gcloudignore is not part of this checkout")
        if shutil.which("git") is None:
            self.skipTest("needs git")
        self.assertEqual(_git_ignored(gcloudignore, kept), [])


if __name__ == "__main__":
    unittest.main()
