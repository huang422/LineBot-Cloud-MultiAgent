"""Script-run rehearsals for the shell-script tests.

tests/test_deploy_script.py and tests/test_provision_gcp.py run their
scripts against the fake gcloud (tests/fake_gcloud.py) through
:class:`RehearsedScripts`; each module's setUpModule calls
:func:`start_rehearsals` and its tearDownModule :func:`stop_rehearsals`.
"""

from __future__ import annotations

import contextlib
import os
import subprocess
import tempfile
import unittest
from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path
from types import ModuleType


class RehearsedScripts:
    """Script tests whose script runs happen ahead of time, concurrently.

    A script run against the fake gcloud takes a second or more, and no
    test's runs depend on another test's. So setUpModule rehearses the
    module's tests on a thread pool (start_rehearsals): each runs once with
    its assertions ignored, recording every script run it makes with its
    inputs (the command, the environment, the env file and the fake
    project's state), its result, and the state and gcloud log it leaves.
    The tests then run for real in the same work directories: a run whose
    inputs equal the next recorded run's is replayed from it, and any other
    run is live, as is every run after it. A class with ``rehearse = False``
    runs live.

    The harness keeps its files in ``work_dir()``, sets ``env_file``,
    ``state_path``, ``log_path`` and ``environ`` (from ``base_environ``) in
    setUp, and runs its script with ``run_script``.
    tests/test_provision_gcp.py uses it too.
    """

    rehearse = True
    _rehearsing = False

    @classmethod
    def _rehearse(cls, name: str) -> None:
        test = cls(name)
        test._rehearsing = True
        # Only the real run's assertions count.
        with contextlib.suppress(Exception):
            test.setUp()
            getattr(test, name)()
        test.doCleanups()

    def setUp(self) -> None:
        super().setUp()
        rehearsal = _REHEARSALS.futures.get((type(self), self._testMethodName))
        if rehearsal is not None and not self._rehearsing:
            rehearsal.result()

    @property
    def base_environ(self) -> dict[str, str]:
        return _REHEARSALS.environ

    def work_dir(self) -> Path:
        """This test's directory, the same in its rehearsal and its real run."""
        return _REHEARSALS.work_dirs[type(self), self._testMethodName]

    def run_script(self, argv: list[str], timeout: int) -> subprocess.CompletedProcess:
        inputs = (argv, sorted(self.environ.items()), _snapshot(self.env_file), _snapshot(self.state_path))
        recorded = _REHEARSALS.runs[type(self), self._testMethodName]
        if recorded and not self._rehearsing:
            recorded_inputs, result, state, log = recorded.pop(0)
            if recorded_inputs == inputs:
                _restore(self.state_path, state)
                _restore(self.log_path, log)
                return result
            recorded.clear()
        result = subprocess.run(argv, capture_output=True, text=True, env=self.environ, timeout=timeout)
        if self._rehearsing:
            recorded.append((inputs, result, _snapshot(self.state_path), _snapshot(self.log_path)))
        return result


class _Rehearsals:
    """One test module's rehearsals, keyed by (test class, test name)."""

    def __init__(self) -> None:
        self.environ: dict[str, str] = {}
        self.work_dirs: dict[tuple[type, str], Path] = {}
        self.runs: dict[tuple[type, str], list] = {}
        self.futures: dict[tuple[type, str], Future] = {}
        self.cleanups = contextlib.ExitStack()


_REHEARSALS = _Rehearsals()


def start_rehearsals(module: ModuleType) -> None:
    """For setUpModule: a work directory for each RehearsedScripts test, and the rehearsals."""
    # os.environ, read once: pytest keeps changing it (PYTEST_CURRENT_TEST)
    # while the rehearsals run on their threads.
    _REHEARSALS.environ = {
        key: value for key, value in os.environ.items() if key != "PYTEST_CURRENT_TEST"
    }
    root = _REHEARSALS.cleanups.enter_context(tempfile.TemporaryDirectory())
    pool = ThreadPoolExecutor(max_workers=min(8, os.cpu_count() or 1))
    _REHEARSALS.cleanups.callback(pool.shutdown, cancel_futures=True)
    loader = unittest.TestLoader()
    for cls in vars(module).values():
        if not (isinstance(cls, type) and issubclass(cls, RehearsedScripts)):
            continue
        for name in loader.getTestCaseNames(cls):
            _REHEARSALS.work_dirs[cls, name] = Path(tempfile.mkdtemp(dir=root))
            _REHEARSALS.runs[cls, name] = []
            if cls.rehearse:
                _REHEARSALS.futures[cls, name] = pool.submit(cls._rehearse, name)


def stop_rehearsals() -> None:
    """For tearDownModule: waits for the rehearsals, then removes the work directories."""
    _REHEARSALS.cleanups.close()
    for table in (_REHEARSALS.work_dirs, _REHEARSALS.runs, _REHEARSALS.futures):
        table.clear()


def _snapshot(path: Path) -> bytes | None:
    return path.read_bytes() if path.exists() else None


def _restore(path: Path, data: bytes | None) -> None:
    if data is None:
        path.unlink(missing_ok=True)
    else:
        path.write_bytes(data)
