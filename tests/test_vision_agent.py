"""An image prompt stays near 4k tokens (IMAGE_CONTEXT_CHARS).

The 2026-10-01 incident: a quoted photo in a busy group sent muse-glimmer a
~6.2k-token prompt (long assistant turns, the summary, profile facts), which
took about 40 s; at ~4k tokens it answered in 8.5-19 s. Every agent that
answers sends an image request to the vision tiers (BaseAgent.targets_for),
so each keeps to the budget.
"""

from __future__ import annotations

import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock

from src.agents.base_agent import IMAGE_CONTEXT_CHARS as CONTEXT_CHARS
from src.agents.base_agent import SUMMARY_HEADER, within_context_budget
from src.agents.chat_agent import ChatAgent
from src.agents.vision_agent import VisionAgent
from src.agents.web_search_agent import WebSearchAgent
from src.config import Settings
from src.models.agent_request import AgentRequest

_IMAGE = "data:image/jpeg;base64,/9j/AAAA"
_SUMMARY = "- 這個群組是大學同學的聊天群，常駐成員有四位，常一起規劃旅行與聚餐。\n" * 15
_FACTS = ["住在新北市板橋", "對花生過敏"]


def _turn(role: str, index: int, chars: int) -> dict:
    head = f"第{index}輪："
    return {"role": role, "content": head + "台" * (chars - len(head))}


def _history(sizes: list[int]) -> list[dict]:
    return [_turn("user" if index % 2 == 0 else "assistant", index, size) for index, size in enumerate(sizes)]


def _request(**overrides) -> AgentRequest:
    values = {
        "user_id": "U0f3e9c2d8b7a6e5f4d3c2b1a0e9fa1b2",
        "group_id": "C1",
        "source_type": "group",
        "chat_scope": "multi",
        "text": "解釋這張圖片",
        "image_base64": _IMAGE,
        "quoted_message_id": "589012345678901234",
        "quoted_image_base64": _IMAGE,
        "memory_summary": _SUMMARY,
        "user_profile_facts": list(_FACTS),
        "disable_thinking": True,
        **overrides,
    }
    return AgentRequest(**values)


def _context_chars(request: AgentRequest) -> int:
    turns = sum(len(turn["content"]) for turn in request.conversation_history or ())
    return turns + len(request.memory_summary) + sum(len(fact) for fact in request.user_profile_facts)


class ContextBudgetTests(unittest.TestCase):
    def test_the_incident_sized_context_is_cut_to_the_budget_newest_turns_first(self) -> None:
        # Six turns like the incident's: short questions, long answers.
        request = _request(conversation_history=_history([150, 1300, 200, 1400, 180, 1300]))
        self.assertGreater(_context_chars(request), 5000)

        trimmed = within_context_budget(request)

        self.assertEqual(_context_chars(trimmed), CONTEXT_CHARS)
        # Whole turns, newest first: the last three fit, the older ones go,
        # and the summary keeps what is left of the budget.
        self.assertEqual(trimmed.conversation_history, request.conversation_history[3:])
        self.assertEqual(len(trimmed.memory_summary), CONTEXT_CHARS - 12 - (1400 + 180 + 1300))
        self.assertTrue(trimmed.memory_summary.endswith("…"))
        self.assertEqual(trimmed.user_profile_facts, _FACTS)
        # The message, the quote and the image are untouched, and so is the request.
        for field in ("text", "image_base64", "quoted_image_base64", "quoted_message_id", "request_id"):
            self.assertEqual(getattr(trimmed, field), getattr(request, field))
        self.assertEqual(len(request.conversation_history), 6)
        self.assertEqual(request.memory_summary, _SUMMARY)

    def test_a_small_context_is_kept_whole(self) -> None:
        request = _request(conversation_history=_history([100, 400, 120, 500]))

        trimmed = within_context_budget(request)

        self.assertEqual(trimmed.conversation_history, request.conversation_history)
        self.assertEqual(trimmed.memory_summary, _SUMMARY)

    def test_the_summary_keeps_what_the_turns_leave(self) -> None:
        request = _request(conversation_history=_history([200, 2500]))
        left = CONTEXT_CHARS - sum(len(fact) for fact in _FACTS) - 2700

        trimmed = within_context_budget(request)

        self.assertEqual(len(trimmed.conversation_history), 2)
        self.assertEqual(len(trimmed.memory_summary), left)
        self.assertTrue(trimmed.memory_summary.endswith("…"))
        self.assertTrue(_SUMMARY.startswith(trimmed.memory_summary[:-1]))

    def test_a_newest_turn_longer_than_the_budget_drops_the_history(self) -> None:
        trimmed = within_context_budget(_request(conversation_history=_history([100, CONTEXT_CHARS + 1])))

        self.assertEqual(trimmed.conversation_history, [])
        self.assertEqual(trimmed.memory_summary, _SUMMARY)

    def test_no_room_left_drops_the_summary(self) -> None:
        request = _request(conversation_history=_history([CONTEXT_CHARS - 12]))

        trimmed = within_context_budget(request)

        self.assertEqual(len(trimmed.conversation_history), 1)
        self.assertEqual(trimmed.memory_summary, "")


class VisionMessagesTests(unittest.IsolatedAsyncioTestCase):
    def _agent(self, chain=None) -> VisionAgent:
        return VisionAgent(Settings(_env_file=None, bot_name="小幫手"), chain or SimpleNamespace(), targets=[])

    def test_the_messages_carry_the_bounded_context_and_everything_else_whole(self) -> None:
        agent = self._agent()
        request = _request(conversation_history=_history([150, 1300, 200, 1400, 180, 1300]))

        messages = agent._build_messages(request)

        text = "\n".join(m["content"] for m in messages if isinstance(m["content"], str))
        self.assertNotIn("第0輪", text)
        self.assertNotIn("第2輪", text)
        for kept in ("第3輪", "第4輪", "第5輪"):
            self.assertIn(kept, text)
        self.assertIn(SUMMARY_HEADER, text)
        self.assertNotIn(_SUMMARY, text)
        self.assertIn("對花生過敏", text)
        # The system prompt and the user's message with its image, whole.
        self.assertEqual(messages[0]["role"], "system")
        self.assertIn(agent.system_prompt.split("{current_time}")[0], messages[0]["content"])
        last = messages[-1]["content"]
        self.assertIn("解釋這張圖片", last[0]["text"])
        self.assertEqual(last[1], {"type": "image_url", "image_url": {"url": _IMAGE}})

    async def test_process_sends_the_bounded_messages(self) -> None:
        chain = SimpleNamespace(generate=AsyncMock(return_value=SimpleNamespace(text="紅色的圓形", model="m")))
        agent = self._agent(chain)
        request = _request(conversation_history=_history([150, 1300, 200, 1400, 180, 1300]))

        await agent.process(request)

        sent = chain.generate.await_args.kwargs["messages"]
        self.assertEqual(sent, agent._build_messages(request))
        self.assertFalse(any("第0輪" in str(message["content"]) for message in sent))


def _text(messages: list[dict]) -> str:
    return "\n".join(str(message["content"]) for message in messages)


class OtherAgentsImagePromptTests(unittest.TestCase):
    def setUp(self) -> None:
        self.settings = Settings(_env_file=None, bot_name="小幫手")
        self.history = _history([150, 1300, 200, 1400, 180, 1300])

    def test_chat_and_web_search_bound_the_context_of_an_image_request(self) -> None:
        for agent in (
            ChatAgent(self.settings, SimpleNamespace(), targets=[]),
            WebSearchAgent(self.settings, SimpleNamespace(), targets=[]),
        ):
            with self.subTest(agent=agent.name):
                text = _text(agent._build_messages(_request(conversation_history=self.history)))

                self.assertNotIn("第0輪", text)
                self.assertIn("第5輪", text)
                self.assertNotIn(_SUMMARY, text)

    def test_a_text_request_and_the_routers_text_only_view_keep_the_whole_context(self) -> None:
        agent = ChatAgent(self.settings, SimpleNamespace(), targets=[])
        text_only = _request(
            conversation_history=self.history, image_base64="", quoted_image_base64="", quoted_message_id=""
        )

        for messages in (
            agent._build_messages(text_only),
            agent._build_messages(_request(conversation_history=self.history), sees_images=False),
        ):
            text = _text(messages)
            self.assertIn("第0輪", text)
            self.assertIn(_SUMMARY.strip(), text)


if __name__ == "__main__":
    unittest.main()
