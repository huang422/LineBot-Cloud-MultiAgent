"""PassiveMessageBuffer: threshold flag, due keys, and no timers of its own."""

from __future__ import annotations

import asyncio
import unittest
from unittest.mock import patch

from src.config import Settings
from src.services.memory_backends import InMemoryBackend, MemoryBackendError
from src.services.memory_service import MemoryService
from src.services.passive_buffer import PassiveMessageBuffer


class _FakeClock:
    def __init__(self, now: float = 1_000.0) -> None:
        self.now = now

    def __call__(self) -> float:
        return self.now


class PassiveMessageBufferTests(unittest.IsolatedAsyncioTestCase):
    def _buffer(
        self,
        *,
        threshold: int = 3,
        delay: int = 30,
        enabled: bool = True,
        flush=None,
        max_items: int | None = None,
    ):
        self.flushed: list[tuple[str, dict, list]] = []
        self.clock = _FakeClock()

        async def record(key, context, items):
            self.flushed.append((key, context, list(items)))
            return "written"

        return PassiveMessageBuffer(
            enabled=enabled,
            threshold=threshold,
            delay=delay,
            flush=flush or record,
            clock=self.clock,
            max_items=max_items,
        )

    async def test_enqueue_never_creates_tasks(self) -> None:
        buffer = self._buffer(threshold=2)
        with patch.object(asyncio, "create_task") as create_task, patch.object(
            asyncio.get_running_loop(), "create_task"
        ) as loop_create_task:
            for i in range(5):
                buffer.enqueue("group::G1", i, context={"chat_id": "G1"})

        create_task.assert_not_called()
        loop_create_task.assert_not_called()
        self.assertEqual(self.flushed, [])
        self.assertEqual(buffer.pending_count("group::G1"), 5)

    async def test_enqueue_reports_only_the_crossing_of_the_threshold(self) -> None:
        buffer = self._buffer(threshold=3)

        flags = [buffer.enqueue("group::G1", i) for i in range(5)]

        self.assertEqual(flags, [False, False, True, False, False])
        await buffer.flush("group::G1")
        # A fresh buffer crosses the threshold again.
        self.assertEqual([buffer.enqueue("group::G1", i) for i in range(3)], [False, False, True])

    async def test_a_lost_flush_request_is_repeated_after_the_delay(self) -> None:
        # The passive_flush task for the threshold crossing never runs.
        buffer = self._buffer(threshold=3, delay=30)
        self.assertEqual([buffer.enqueue("group::G1", i) for i in range(3)], [False, False, True])

        # Within the delay the first request still covers new items.
        self.clock.now += 29
        self.assertFalse(buffer.enqueue("group::G1", 3))
        # After it, the next item asks again, and so on once per delay.
        self.clock.now += 1
        self.assertTrue(buffer.enqueue("group::G1", 4))
        self.assertFalse(buffer.enqueue("group::G1", 5))
        self.clock.now += 30
        self.assertTrue(buffer.enqueue("group::G1", 6))
        self.assertEqual(buffer.pending_count("group::G1"), 7)

    async def test_an_old_buffer_asks_for_a_flush_on_its_next_item(self) -> None:
        buffer = self._buffer(threshold=8, delay=30)
        self.assertFalse(buffer.enqueue("group::G1", "a"))
        self.clock.now += 29
        self.assertFalse(buffer.enqueue("group::G1", "b"))
        self.clock.now += 1
        # Below the threshold, but the oldest item has waited the delay.
        self.assertTrue(buffer.enqueue("group::G1", "c"))

    async def test_each_buffer_is_capped_oldest_first(self) -> None:
        buffer = self._buffer(threshold=3, max_items=5)
        with patch("src.services.passive_buffer.logger.warning") as log_warning:
            for i in range(8):
                buffer.enqueue("group::G1", i)

        self.assertEqual(buffer.pending_count("group::G1"), 5)
        self.assertEqual(log_warning.call_count, 3)
        await buffer.flush("group::G1")
        self.assertEqual(self.flushed[0][2], [3, 4, 5, 6, 7])
        # The cap never goes below the threshold, and defaults to 4x it.
        self.assertEqual(self._buffer(threshold=3, max_items=1).max_items, 3)
        self.assertEqual(self._buffer(threshold=3).max_items, 12)

    async def test_flush_returns_the_callback_result(self) -> None:
        buffer = self._buffer()
        self.assertIsNone(await buffer.flush("group::G1"))  # nothing pending
        buffer.enqueue("group::G1", "a")
        self.assertEqual(await buffer.flush("group::G1"), "written")

    async def test_a_failed_flush_puts_its_items_back_in_order(self) -> None:
        calls: list[list] = []

        async def fails_once(key, context, items):
            calls.append(list(items))
            if len(calls) == 1:
                # A message arrives while the write is in progress.
                buffer.enqueue(key, "c")
                raise RuntimeError("firestore down")

        buffer = self._buffer(threshold=3, flush=fails_once)
        buffer.enqueue("group::G1", "a", context={"chat_id": "G1"})
        self.assertFalse(buffer.enqueue("group::G1", "b"))
        first_at = self.clock.now
        self.clock.now += 5

        with self.assertRaises(RuntimeError):
            await buffer.flush("group::G1")

        self.assertEqual(calls, [["a", "b"]])
        self.assertEqual(buffer.pending_count("group::G1"), 3)
        self.assertEqual(buffer.context("group::G1"), {"chat_id": "G1"})
        # The restored buffer keeps its age, so the cron sweep still sees it.
        self.assertEqual(buffer.due_keys(now=first_at + 30), ["group::G1"])
        # And it may ask for a flush again right away.
        self.assertTrue(buffer.enqueue("group::G1", "d"))
        await buffer.flush("group::G1")
        self.assertEqual(calls[-1], ["a", "b", "c", "d"])
        self.assertEqual(buffer.pending_keys(), [])

    async def test_a_cancelled_flush_puts_its_items_back(self) -> None:
        started = asyncio.Event()

        async def hanging(key, context, items):
            started.set()
            await asyncio.sleep(60)

        buffer = self._buffer(flush=hanging)
        buffer.enqueue("group::G1", "a")
        task = asyncio.create_task(buffer.flush("group::G1"))
        await started.wait()
        self.assertEqual(buffer.pending_count("group::G1"), 0)

        task.cancel()
        with self.assertRaises(asyncio.CancelledError):
            await task

        self.assertEqual(buffer.pending_count("group::G1"), 1)

    async def test_a_discard_during_a_failed_flush_is_not_undone(self) -> None:
        async def failing(key, context, items):
            buffer.discard(key)  # !new while the write is in progress
            raise RuntimeError("firestore down")

        buffer = self._buffer(flush=failing)
        buffer.enqueue("group::G1", "a")
        with self.assertRaises(RuntimeError):
            await buffer.flush("group::G1")

        self.assertEqual(buffer.pending_keys(), [])

    async def test_threshold_of_one_flags_every_first_item(self) -> None:
        buffer = self._buffer(threshold=1)
        self.assertTrue(buffer.enqueue("group::G1", "a"))

    async def test_disabled_or_keyless_enqueue_is_ignored(self) -> None:
        disabled = self._buffer(enabled=False)
        self.assertFalse(disabled.enqueue("group::G1", "a"))
        self.assertEqual(disabled.pending_keys(), [])

        enabled = self._buffer()
        self.assertFalse(enabled.enqueue("", "a"))
        self.assertEqual(enabled.pending_keys(), [])

    async def test_due_keys_use_the_first_item_timestamp(self) -> None:
        buffer = self._buffer(delay=30)
        buffer.enqueue("group::OLD", "a")
        self.clock.now += 20
        buffer.enqueue("group::NEW", "b")
        # A later item does not push the old buffer's deadline back.
        buffer.enqueue("group::OLD", "c")

        self.assertEqual(buffer.due_keys(), [])
        self.clock.now += 10
        self.assertEqual(buffer.due_keys(), ["group::OLD"])
        self.assertEqual(buffer.due_keys(now=self.clock.now + 20), ["group::OLD", "group::NEW"])

    async def test_flush_hands_items_and_context_to_the_callback_once(self) -> None:
        buffer = self._buffer()
        buffer.enqueue("group::G1", "a", context={"chat_id": "G1"})
        buffer.enqueue("group::G1", "b", context={"ignored": True})

        await buffer.flush("group::G1")
        await buffer.flush("group::G1")  # nothing left: no second write

        self.assertEqual(self.flushed, [("group::G1", {"chat_id": "G1"}, ["a", "b"])])
        self.assertEqual(buffer.pending_keys(), [])
        self.assertEqual(buffer.due_keys(now=self.clock.now + 999), [])

    async def test_discard_drops_items_without_flushing(self) -> None:
        buffer = self._buffer()
        buffer.enqueue("group::G1", "a")

        buffer.discard("group::G1")
        await buffer.flush("group::G1")

        self.assertEqual(self.flushed, [])
        self.assertEqual(buffer.pending_count("group::G1"), 0)

    async def test_close_flushes_everything(self) -> None:
        buffer = self._buffer()
        buffer.enqueue("group::G1", "a")
        buffer.enqueue("room::R1", "b")

        await buffer.close()

        self.assertEqual(sorted(key for key, _, _ in self.flushed), ["group::G1", "room::R1"])
        self.assertEqual(buffer.pending_keys(), [])

    async def test_close_keeps_going_past_a_failed_flush(self) -> None:
        written: list[str] = []

        async def flaky(key, context, items):
            if key == "group::G1":
                raise RuntimeError("firestore down")
            written.append(key)

        buffer = self._buffer(flush=flaky)
        buffer.enqueue("group::G1", "a")
        buffer.enqueue("room::R1", "b")
        with patch("src.services.passive_buffer.logger.warning") as log_warning:
            await buffer.close()

        self.assertEqual(written, ["room::R1"])
        log_warning.assert_called_once()


class MemoryServicePassiveTests(unittest.IsolatedAsyncioTestCase):
    def _service(self, **overrides) -> MemoryService:
        values = {
            "passive_group_recording_enabled": True,
            "passive_group_flush_threshold": 3,
            "passive_group_flush_delay_seconds": 30,
            **overrides,
        }
        settings = Settings(_env_file=None, **values)
        return MemoryService(settings, backend=InMemoryBackend())

    def _enqueue(self, service: MemoryService, text: str, *, chat_id: str = "G1", source_type="group"):
        return service.enqueue_passive_message(
            source_type=source_type,
            chat_scope="multi",
            chat_id=chat_id,
            user_id="U1",
            text=text,
        )

    async def test_enqueue_returns_the_threshold_flag(self) -> None:
        service = self._service()
        self.assertEqual([self._enqueue(service, f"m{i}") for i in range(3)], [False, False, True])
        # DMs, blank text and disabled recording never ask for a flush.
        self.assertFalse(self._enqueue(service, "dm", source_type="user", chat_id="U1"))
        self.assertFalse(self._enqueue(service, "   "))
        disabled = self._service(passive_group_recording_enabled=False)
        self.assertFalse(self._enqueue(disabled, "x"))

    async def test_flush_passive_persists_the_batch_in_one_write(self) -> None:
        service = self._service()
        for i in range(2):
            self._enqueue(service, f"m{i}")

        await service.flush_passive("group::G1")

        stored = service.backend_instance.store["group::G1"]
        self.assertEqual([m["content"] for m in stored["recent_messages"]], ["m0", "m1"])

    async def test_flush_due_passive_flushes_only_old_buffers(self) -> None:
        service = self._service()
        clock = _FakeClock()
        service._passive._clock = clock
        self._enqueue(service, "old", chat_id="G_OLD")
        clock.now += 25
        self._enqueue(service, "new", chat_id="G_NEW")

        flushed = await service.flush_due_passive(now=clock.now + 5)

        self.assertEqual(flushed, ["group::G_OLD"])
        self.assertIn("group::G_OLD", service.backend_instance.store)
        self.assertNotIn("group::G_NEW", service.backend_instance.store)
        self.assertEqual(service.get_stats()["passive_buffered_chats"], 1)

    async def test_flush_due_passive_is_bounded(self) -> None:
        service = self._service()
        for i in range(4):
            self._enqueue(service, "x", chat_id=f"G{i}")

        flushed = await service.flush_due_passive(now=10**12, limit=3)

        self.assertEqual(len(flushed), 3)
        self.assertEqual(len(service._passive.pending_keys()), 1)

    async def test_flush_due_passive_starts_nothing_past_its_deadline(self) -> None:
        service = self._service()
        for i in range(2):
            self._enqueue(service, "x", chat_id=f"G{i}")

        with patch("src.services.memory_service.logger.warning") as log_warning:
            flushed = await service.flush_due_passive(now=10**12, deadline=0.0)

        self.assertEqual(flushed, [])
        self.assertEqual(len(service._passive.pending_keys()), 2)
        self.assertIn("2 chat(s) wait for the next tick", log_warning.call_args.args[0])

    async def test_flush_due_passive_reports_only_chats_it_wrote(self) -> None:
        service = self._service()
        for chat_id in ("G_OK", "G_BAD"):
            self._enqueue(service, "x", chat_id=chat_id)
        real_save = service.backend_instance.save_chat

        async def save(key, document):
            if key == "group::G_BAD":
                raise MemoryBackendError("quota exceeded")
            await real_save(key, document)

        with patch.object(service.backend_instance, "save_chat", side_effect=save):
            flushed = await service.flush_due_passive(now=10**12)

        self.assertEqual(flushed, ["group::G_OK"])
        # The failed chat keeps its batch for the next sweep.
        self.assertEqual(service._passive.pending_keys(), ["group::G_BAD"])

    async def test_buffer_cap_matches_the_recent_window_hard_cap(self) -> None:
        service = self._service()
        self.assertEqual(service._passive.max_items, service._recent_hard_cap)


if __name__ == "__main__":
    unittest.main()
