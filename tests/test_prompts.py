"""The prompt files keep what the 2026-09-30 prompt audit settled.

Replies go out as LINE plain text and thinking is set per request, not by
prose, so the reply prompts carry no Markdown emphasis to copy and no
"think carefully first" steering; the router prompt names no model.
"""

from __future__ import annotations

import re
import unittest
from pathlib import Path

_PROMPTS = Path(__file__).resolve().parent.parent / "prompts"
_REPLY_PROMPTS = ("chat.md", "vision.md", "web_search.md")
_BOLD_SPAN = re.compile(r"\*\*[^*\n]+\*\*")


def _prompt(name: str) -> str:
    return (_PROMPTS / name).read_text(encoding="utf-8")


class PromptFileTests(unittest.TestCase):
    def test_reply_prompts_have_no_bold_for_the_model_to_mirror(self) -> None:
        for name in _REPLY_PROMPTS:
            with self.subTest(prompt=name):
                self.assertEqual(_BOLD_SPAN.findall(_prompt(name)), [])

    def test_no_prompt_steers_hidden_thinking(self) -> None:
        for path in sorted(_PROMPTS.glob("*.md")):
            with self.subTest(prompt=path.name):
                text = path.read_text(encoding="utf-8")
                for phrase in ("先認真思考", "在心中完成", "內部判斷（禁止顯示）", "必須在內部完成"):
                    self.assertNotIn(phrase, text)

    def test_reply_prompts_rank_the_profile_block_as_background(self) -> None:
        # The agents add a [使用者 profile，僅供參考] block (src/agents/base_agent.py).
        for name in _REPLY_PROMPTS:
            with self.subTest(prompt=name):
                self.assertIn("歷史對話、長期記憶摘要與使用者 profile", _prompt(name))

    def test_reply_prompts_keep_the_profile_out_of_group_replies(self) -> None:
        # Facts given in a 1:1 chat reach the user's group replies too; the
        # agents add the same rule to a group reply's system prompt.
        for name in _REPLY_PROMPTS:
            with self.subTest(prompt=name):
                self.assertIn(
                    "群組中，除非使用者在當前訊息自己提到或要求，不要主動提及或透露使用者 profile 的內容",
                    _prompt(name),
                )

    def test_reply_prompts_explain_the_speaker_line_marker(self) -> None:
        # Group history, the message and its quote carry it
        # (src/utils/speaker_lines.py); review finding 4: only the summary
        # prompt said what it means, so a reply could copy it into code.
        from src.utils.speaker_lines import SPEAKER_LINE_MARKER

        for name in _REPLY_PROMPTS:
            with self.subTest(prompt=name):
                text = _prompt(name)
                self.assertIn(f"行首的「{SPEAKER_LINE_MARKER}」是系統加上的標記", text)
                self.assertIn("不要照抄這個標記", text)

    def test_reply_prompts_say_replies_are_plain_line_text(self) -> None:
        for name in _REPLY_PROMPTS:
            with self.subTest(prompt=name):
                text = _prompt(name)
                self.assertIn("LINE 不", text)
                self.assertIn("Markdown", text)
                # No Markdown blockquote marker for the model to copy.
                self.assertNotRegex(text, r"(?m)^>")
        self.assertNotIn("markdown code block", _prompt("chat.md"))

    def test_reply_prompts_ask_for_code_in_fences(self) -> None:
        # to_line_text keeps a fenced block verbatim and drops only its fence
        # lines; unfenced code goes through the Markdown/LaTeX rules
        # (code-review round 3: "__init__" reached the user as "init").
        for name in _REPLY_PROMPTS:
            with self.subTest(prompt=name):
                text = _prompt(name)
                self.assertIn("程式碼放在 ``` 圍欄內", text)
                self.assertNotIn("不加 ``` 圍欄", text)

    def test_search_prompt_keeps_numbers_to_the_results_only_when_there_are_results(self) -> None:
        # Without results the agent answers from the model's own knowledge
        # (rule 5); rule 6 must not forbid that.
        text = _prompt("web_search.md")
        self.assertIn("有搜尋結果時，數字、日期、名稱、價格等具體資訊必須直接取自搜尋結果", text)
        self.assertIn("沒有搜尋結果時，依既有知識回答", text)

    def test_search_prompt_explains_the_labels_the_search_context_carries(self) -> None:
        from src.services.search.types import AI_SUMMARY_LABEL, PUBLISHED_LABEL

        text = _prompt("web_search.md")
        for label in (AI_SUMMARY_LABEL, f"{PUBLISHED_LABEL}:"):
            self.assertIn(f"「{label}」", text)
        self.assertNotIn("搜尋結果不附發布日期", text)

    def test_the_router_prompt_has_no_example_it_never_sees(self) -> None:
        # A message with an image (sent or quoted) never reaches the router:
        # the fast rules route it, so an "（引用圖片）" example is dead text.
        self.assertNotIn("（引用圖片）→", _prompt("orchestrator.md"))

    def test_the_router_prompt_names_no_model_and_defaults_to_no_thinking(self) -> None:
        text = _prompt("orchestrator.md")
        for stale in ("Nemotron", "OpenRouter", "專用思考模型", "單行 JSON"):
            self.assertNotIn(stale, text)
        self.assertIn('"needs_thinking":false', text)

    def test_the_tool_contracts_live_in_the_tool_descriptions(self) -> None:
        # The chat prompt no longer restates (and contradicts) the schemas.
        text = _prompt("chat.md")
        for stale in ("update_user_profile(facts: object)", "recall_memory(query", "連續三輪內"):
            self.assertNotIn(stale, text)

    def test_diary_mode_saves_only_what_the_sender_states_about_themself(self) -> None:
        # Code-review round 4 #14: diary mode told the model to store
        # inferred family structure and relationships, against the
        # update_user_profile rule; profile facts reach every chat, groups too.
        text = _prompt("chat.md")
        for stale in ("家庭結構", "重要人際關係", "就把穩定事實"):
            self.assertNotIn(stale, text)
        self.assertIn("使用者在訊息中明確自述", text)
        self.assertIn("不要寫進個人資料", text)

    def test_search_prompt_forbids_summarising_a_page_it_could_not_open(self) -> None:
        # Code-review round 4 #11: the agent names the unread link in a
        # system note; the prompt backs it.
        self.assertIn("系統註明某個網址無法開啟時，你沒有讀到那個網頁：不可猜測、摘要或評論它的內容", _prompt("web_search.md"))

    def test_image_prompt_does_not_promise_edits_of_a_reference_image(self) -> None:
        text = _prompt("image_gen.md")
        self.assertNotIn("以原圖為基準做定向調整", text)
        self.assertIn("不是在原圖上直接編輯", text)


if __name__ == "__main__":
    unittest.main()
