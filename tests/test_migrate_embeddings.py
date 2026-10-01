"""Offline tests for scripts/migrate_embeddings.py.

Firestore and the embedding API are replaced with small fakes; the script's
live path (real client, real .env) only ever runs when a human starts it.
"""

from __future__ import annotations

import unittest
from typing import Any

from scripts import migrate_embeddings as migrate
from scripts.migrate_embeddings import (
    DELETE,
    KEEP,
    REEMBED,
    Throttle,
    apply_plan,
    chunked,
    classify_episode,
    plan_migration,
    run,
)

_MODEL = "nvidia/nemotron-3-embed-1b"
_LEGACY = "nvidia/nv-embedqa-e5-v5"


class _FakeDocRef:
    def __init__(self, doc_id: str) -> None:
        self.id = doc_id
        self.updates: list[dict[str, Any]] = []
        self.deleted = False

    def update(self, data: dict[str, Any]) -> None:
        self.updates.append(data)

    def delete(self) -> None:
        self.deleted = True


class _FakeSnapshot:
    def __init__(self, ref: _FakeDocRef, data: dict[str, Any]) -> None:
        self.reference = ref
        self._data = data

    def to_dict(self) -> dict[str, Any]:
        return dict(self._data)


class _FakeParent:
    """An episodes/{chat_key} document that exists only as a path."""

    def __init__(self, chat_key: str, snapshots: list[_FakeSnapshot]) -> None:
        self.id = chat_key
        self._snapshots = snapshots

    def collection(self, name: str):
        assert name == "items", name
        return self

    def stream(self):
        return iter(self._snapshots)


class _FakeCollection:
    def __init__(self, parents: list[_FakeParent]) -> None:
        self._parents = parents

    def list_documents(self):
        return iter(self._parents)


class _FakeClient:
    def __init__(self, collections: dict[str, _FakeCollection]) -> None:
        self._collections = collections
        self.requested: list[str] = []

    def collection(self, name: str) -> _FakeCollection:
        self.requested.append(name)
        return self._collections[name]


class _FakeEmbedder:
    model = _MODEL
    dimensions = 4

    def __init__(self, *, configured: bool = True, fail: bool = False) -> None:
        self.is_configured = configured
        self.fail = fail
        self.calls: list[tuple[list[str], str]] = []

    async def embed_batch(self, texts, *, input_type: str = "passage"):
        texts = list(texts)
        self.calls.append((texts, input_type))
        if self.fail:
            return []
        return [[0.5, 0.5, 0.5, 0.5] for _ in texts]


class _NoWaitThrottle(Throttle):
    def __init__(self) -> None:
        super().__init__(40)

    async def wait(self) -> None:
        return None


def _episode(summary: str | None, **fields: Any) -> dict[str, Any]:
    data: dict[str, Any] = {"ts": 1.0, **fields}
    if summary is not None:
        data["summary"] = summary
    return data


def _fixture() -> tuple[_FakeClient, dict[str, _FakeDocRef]]:
    refs = {name: _FakeDocRef(name) for name in ("current", "legacy", "other", "empty")}
    parents = [
        _FakeParent(
            "user::U1",
            [
                _FakeSnapshot(
                    refs["current"],
                    _episode(
                        "已是新向量",
                        embedding=[0.1] * 4,
                        embedding_model=_MODEL,
                        embedding_dim=4,
                    ),
                ),
                _FakeSnapshot(refs["legacy"], _episode("使用者喜歡巧克力", embedding=[0.2] * 4)),
            ],
        ),
        _FakeParent(
            "group::G1",
            [
                _FakeSnapshot(
                    refs["other"],
                    _episode("群組討論爬山", embedding=[0.3] * 4, embedding_model=_LEGACY),
                ),
                _FakeSnapshot(refs["empty"], _episode("   ", embedding=[0.4] * 4)),
            ],
        ),
    ]
    client = _FakeClient({"linebot_episodes": _FakeCollection(parents)})
    return client, refs


class ClassifyEpisodeTests(unittest.TestCase):
    def test_classification(self) -> None:
        cases = [
            ("current", _episode("x", embedding=[1.0], embedding_model=_MODEL, embedding_dim=4), KEEP),
            ("untagged legacy", _episode("x", embedding=[1.0]), REEMBED),
            ("other model", _episode("x", embedding=[1.0], embedding_model=_LEGACY), REEMBED),
            (
                "other dimension",
                _episode("x", embedding=[1.0], embedding_model=_MODEL, embedding_dim=2048),
                REEMBED,
            ),
            ("tagged but no vector", _episode("x", embedding_model=_MODEL, embedding_dim=4), REEMBED),
            ("no summary", _episode(None, embedding=[1.0]), DELETE),
            ("blank summary", _episode("  \n", embedding=[1.0]), DELETE),
            ("non-text summary", {"summary": 123, "embedding": [1.0]}, DELETE),
        ]
        for label, data, expected in cases:
            with self.subTest(label):
                self.assertEqual(classify_episode(data, model=_MODEL, dimensions=4), expected)


class PlanMigrationTests(unittest.TestCase):
    def test_plan_counts_actions_and_chats(self) -> None:
        client, refs = _fixture()

        plan = plan_migration(
            migrate.iter_episode_documents(client, "linebot_episodes"),
            model=_MODEL,
            dimensions=4,
        )

        self.assertEqual(client.requested, ["linebot_episodes"])
        self.assertEqual(plan.chats, 2)
        self.assertEqual(plan.scanned, 4)
        self.assertEqual(plan.keep, 1)
        self.assertEqual([item.ref for item in plan.reembed], [refs["legacy"], refs["other"]])
        # Episodes with only a summary are embedded from the summary.
        self.assertEqual([item.text for item in plan.reembed], ["使用者喜歡巧克力", "群組討論爬山"])
        self.assertEqual(plan.delete, [refs["empty"]])
        self.assertEqual(
            plan.stale_reasons, {"untagged (legacy)": 2, "other model": 1}
        )

    def test_reembed_text_is_what_the_app_embeds_a_new_episode_from(self) -> None:
        from src.services.memory_service import episode_passage_text

        snippet_episode = _episode("使用者在學鋼琴", snippet="User: 我開始學鋼琴\nAssistant: 很棒", embedding=[0.2])
        legacy_episode = _episode(
            "聊到爬山",
            embedding=[0.2],
            messages=[{"role": "user", "content": "週末去爬山"}, {"role": "assistant", "content": "好"}],
        )
        documents = [("user::U1", "snippet", snippet_episode), ("user::U1", "legacy", legacy_episode)]

        plan = plan_migration(documents, model=_MODEL, dimensions=4)

        self.assertEqual(
            [item.text for item in plan.reembed],
            [
                "使用者在學鋼琴\n\nUser: 我開始學鋼琴\nAssistant: 很棒",
                "聊到爬山\n\nUser: 週末去爬山\nAssistant: 好",
            ],
        )
        self.assertEqual(plan.reembed[0].text, episode_passage_text(snippet_episode))

    def test_chunked_splits_into_batches(self) -> None:
        self.assertEqual([len(c) for c in chunked(list(range(35)), 16)], [16, 16, 3])
        self.assertEqual(list(chunked([], 16)), [])
        with self.assertRaises(ValueError):
            list(chunked([1], 0))


class ThrottleTests(unittest.IsolatedAsyncioTestCase):
    async def test_throttle_spaces_requests_for_max_rpm(self) -> None:
        now = [0.0]
        slept: list[float] = []

        async def fake_sleep(seconds: float) -> None:
            slept.append(seconds)
            now[0] += seconds

        throttle = Throttle(40, clock=lambda: now[0], sleep=fake_sleep)
        for _ in range(3):
            await throttle.wait()

        # 40 requests/minute -> at least 1.5 s between requests; no wait first.
        self.assertEqual(slept, [1.5, 1.5])

    async def test_throttle_skips_sleep_when_enough_time_passed(self) -> None:
        now = [0.0]
        slept: list[float] = []

        async def fake_sleep(seconds: float) -> None:
            slept.append(seconds)

        throttle = Throttle(40, clock=lambda: now[0], sleep=fake_sleep)
        await throttle.wait()
        now[0] = 10.0
        await throttle.wait()

        self.assertEqual(slept, [])

    def test_throttle_rejects_non_positive_rate(self) -> None:
        with self.assertRaises(ValueError):
            Throttle(0)


class ApplyPlanTests(unittest.IsolatedAsyncioTestCase):
    async def test_apply_reembeds_in_batches_and_deletes_textless(self) -> None:
        refs = [_FakeDocRef(f"d{i}") for i in range(20)]
        plan = plan_migration(
            [("user::U1", ref, _episode(f"摘要 {i}")) for i, ref in enumerate(refs)]
            + [("user::U1", _FakeDocRef("gone"), _episode(None))],
            model=_MODEL,
            dimensions=4,
        )
        embedder = _FakeEmbedder()

        result = await apply_plan(
            plan,
            embedder,
            vector_factory=lambda values: ("VECTOR", tuple(values)),
            batch_size=16,
            throttle=_NoWaitThrottle(),
        )

        self.assertEqual([len(texts) for texts, _ in embedder.calls], [16, 4])
        self.assertTrue(all(input_type == "passage" for _, input_type in embedder.calls))
        self.assertEqual(embedder.calls[0][0][0], "摘要 0")
        self.assertEqual(result.reembedded, 20)
        self.assertEqual(result.deleted, 1)
        self.assertEqual(result.failed, 0)
        self.assertEqual(result.requests, 2)
        self.assertTrue(plan.delete[0].deleted)
        self.assertEqual(
            refs[0].updates,
            [
                {
                    "embedding": ("VECTOR", (0.5, 0.5, 0.5, 0.5)),
                    "embedding_model": _MODEL,
                    "embedding_dim": 4,
                }
            ],
        )

    async def test_failed_batch_is_counted_and_left_untouched(self) -> None:
        refs = [_FakeDocRef(f"d{i}") for i in range(3)]
        plan = plan_migration(
            [("user::U1", ref, _episode("摘要")) for ref in refs],
            model=_MODEL,
            dimensions=4,
        )

        result = await apply_plan(
            plan,
            _FakeEmbedder(fail=True),
            vector_factory=list,
            throttle=_NoWaitThrottle(),
        )

        self.assertEqual(result.failed, 3)
        self.assertEqual(result.reembedded, 0)
        self.assertTrue(all(not ref.updates for ref in refs))


class RunTests(unittest.IsolatedAsyncioTestCase):
    async def _run(self, **kwargs) -> tuple[int, str, dict[str, _FakeDocRef], _FakeEmbedder]:
        client, refs = _fixture()
        embedder = kwargs.pop("embedder", None) or _FakeEmbedder()
        lines: list[str] = []
        code = await run(
            client=client,
            collection="linebot_episodes",
            embedder=embedder,
            vector_factory=list,
            throttle=_NoWaitThrottle(),
            out=lines.append,
            **kwargs,
        )
        return code, "\n".join(lines), refs, embedder

    def _assert_no_private_text(self, output: str) -> None:
        for private in ("巧克力", "爬山", "已是新向量", "user::U1", "group::G1"):
            self.assertNotIn(private, output)

    async def test_dry_run_writes_nothing_and_prints_counts_only(self) -> None:
        code, output, refs, embedder = await self._run(apply=False)

        self.assertEqual(code, 0)
        self.assertEqual(embedder.calls, [])
        self.assertTrue(all(not ref.updates and not ref.deleted for ref in refs.values()))
        self.assertIn("re-embed:               2", output)
        self.assertIn("delete (no summary):    1", output)
        self.assertIn("Dry run", output)
        self._assert_no_private_text(output)

    async def test_apply_migrates_and_reports_counts(self) -> None:
        code, output, refs, embedder = await self._run(apply=True)

        self.assertEqual(code, 0)
        self.assertEqual(len(embedder.calls), 1)
        self.assertEqual(refs["legacy"].updates[0]["embedding_model"], _MODEL)
        self.assertEqual(refs["other"].updates[0]["embedding_dim"], 4)
        self.assertTrue(refs["empty"].deleted)
        self.assertEqual(refs["current"].updates, [])
        self.assertIn("re-embedded 2, deleted 1, failed 0", output)
        self._assert_no_private_text(output)

    async def test_apply_without_nvidia_key_refuses_before_writing(self) -> None:
        code, output, refs, embedder = await self._run(
            apply=True, embedder=_FakeEmbedder(configured=False)
        )

        self.assertEqual(code, 2)
        self.assertEqual(embedder.calls, [])
        self.assertTrue(all(not ref.updates and not ref.deleted for ref in refs.values()))
        self.assertIn("NVIDIA_API_KEY", output)

    async def test_retired_model_stops_before_reading_or_writing(self) -> None:
        # An embedder built around the role table (e.g. by hand) that still
        # names nv-embedqa-e5-v5: re-embedding "to" it would 410 on every
        # batch while the deletions still ran.
        retired = _FakeEmbedder()
        retired.model = _LEGACY
        client, refs = _fixture()
        client.collection = lambda name: self.fail("Firestore was read")
        lines: list[str] = []

        code = await run(
            client=client,
            collection="linebot_episodes",
            embedder=retired,
            apply=True,
            vector_factory=list,
            throttle=_NoWaitThrottle(),
            out=lines.append,
        )

        self.assertEqual(code, 2)
        self.assertEqual(retired.calls, [])
        self.assertTrue(all(not ref.updates and not ref.deleted for ref in refs.values()))
        self.assertIn("Fix the embed tier in MODEL_ROLES", "\n".join(lines))

    async def test_apply_with_failed_batch_exits_non_zero(self) -> None:
        code, output, _, _ = await self._run(apply=True, embedder=_FakeEmbedder(fail=True))

        self.assertEqual(code, 1)
        self.assertIn("failed 2", output)


class ParserTests(unittest.TestCase):
    def test_defaults_are_dry_run_with_free_tier_limits(self) -> None:
        args = migrate.build_parser().parse_args([])

        self.assertFalse(args.apply)
        self.assertEqual(args.batch_size, 16)
        self.assertEqual(args.max_rpm, 40)
        self.assertEqual(args.env_file.name, ".env")


if __name__ == "__main__":
    unittest.main()
