"""Secret files stay out of git, the Cloud Build upload and the image.

.gcloudignore uses .gitignore syntax, so git itself checks both, in an empty
repository of their own. Docker anchors .dockerignore patterns at the build
context root, which _docker_excludes follows.
"""

from __future__ import annotations

import os
import re
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]

# scripts/envfile.py migrate-v3 backs up <env file> as <env file>.bak.<YYYYmmddHHMMSS>.
_BACKUP = "prod.env.bak.20261001120000"
_SECRETS = (
    _BACKUP,
    f"config/{_BACKUP}",
    ".env.bak.20261001120000",
    "old.bak",
    "sub/old.bak",
    "credentials.json",
    "src/credentials.json",
)
_KEPT = ("main.py", "src/credentials.py", "notes.bak.md", "backup.20261001120000")


def _git_ignored(ignore_file: Path, paths: tuple[str, ...]) -> list[str]:
    """The *paths* git ignores with *ignore_file* as a repository's only .gitignore."""
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp) / "repo"
        # The user's own git config (a global excludes file) must not count.
        env = {**os.environ, "GIT_CONFIG_GLOBAL": os.devnull, "GIT_CONFIG_NOSYSTEM": "1", "XDG_CONFIG_HOME": tmp}
        subprocess.run(["git", "init", "-q", str(root)], check=True, env=env)
        shutil.copyfile(ignore_file, root / ".gitignore")
        result = subprocess.run(
            ["git", "-C", str(root), "check-ignore", "--no-index", "--stdin"],
            input="".join(f"{path}\n" for path in paths),
            capture_output=True,
            text=True,
            check=False,
            env=env,
        )
        return result.stdout.splitlines()


def _docker_pattern(pattern: str) -> re.Pattern[str]:
    # **/ is any number of directories; * and ? never cross a /.
    parts = re.split(r"(\*\*/|\*|\?|\[[^\]]*\])", pattern.strip("/"))
    swaps = {"**/": "(?:.*/)?", "*": "[^/]*", "?": "[^/]"}
    return re.compile(
        "".join(swaps.get(part, part if part.startswith("[") else re.escape(part)) for part in parts)
    )


def _docker_excludes(path: str) -> bool:
    lines = (_REPO_ROOT / ".dockerignore").read_text(encoding="utf-8").splitlines()
    patterns = [line.strip() for line in lines if line.strip() and not line.startswith("#")]
    excluded = False
    parents = path.split("/")
    for pattern in patterns:  # the last matching pattern wins
        negated = pattern.startswith("!")
        regex = _docker_pattern(pattern.removeprefix("!"))
        # A pattern that matches a directory excludes what is inside it.
        if any(regex.fullmatch("/".join(parents[:depth])) for depth in range(1, len(parents) + 1)):
            excluded = not negated
    return excluded


@unittest.skipUnless(shutil.which("git"), "needs git")
class GitAndUploadIgnoreTests(unittest.TestCase):
    def _check(self, name: str) -> None:
        ignore_file = _REPO_ROOT / name
        if not ignore_file.is_file():  # .gcloudignore leaves both out of the Cloud Build upload
            self.skipTest(f"{name} is not part of this checkout")
        ignored = _git_ignored(ignore_file, (*_SECRETS, *_KEPT))
        self.assertEqual([path for path in _SECRETS if path not in ignored], [])
        self.assertEqual([path for path in _KEPT if path in ignored], [])

    def test_git_ignores_secret_files(self) -> None:
        self._check(".gitignore")

    def test_cloud_build_upload_leaves_out_secret_files(self) -> None:
        self._check(".gcloudignore")


class DockerIgnoreTests(unittest.TestCase):
    def test_the_image_leaves_out_secret_files_at_any_depth(self) -> None:
        self.assertEqual([path for path in _SECRETS if not _docker_excludes(path)], [])
        self.assertEqual([path for path in ("main.py", "src/credentials.py") if _docker_excludes(path)], [])


if __name__ == "__main__":
    unittest.main()
