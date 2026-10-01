"""normalize_messages: the provider-safe shape both chat providers send."""

from __future__ import annotations

import copy
import unittest

from src.providers.messages import has_image_parts, normalize_messages, strip_image_parts


class NormalizeMessagesTests(unittest.TestCase):
    def test_consecutive_user_strings_join_with_blank_line(self) -> None:
        messages = [
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "memory summary"},
            {"role": "user", "content": "quoted text"},
            {"role": "user", "content": "question"},
        ]

        self.assertEqual(
            normalize_messages(messages),
            [
                {"role": "system", "content": "sys"},
                {"role": "user", "content": "memory summary\n\nquoted text\n\nquestion"},
            ],
        )

    def test_user_part_list_merge_converts_strings_to_text_parts_in_order(self) -> None:
        image = {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,abc"}}
        messages = [
            {"role": "user", "content": "context"},
            {"role": "user", "content": [{"type": "text", "text": "what is this?"}, image]},
            {"role": "user", "content": "trailing note"},
        ]

        self.assertEqual(
            normalize_messages(messages),
            [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "context"},
                        {"type": "text", "text": "what is this?"},
                        image,
                        {"type": "text", "text": "trailing note"},
                    ],
                }
            ],
        )

    def test_leading_system_messages_merge_into_one_string(self) -> None:
        messages = [
            {"role": "system", "content": "persona"},
            {"role": "system", "content": [{"type": "text", "text": "rules"}]},
            {"role": "user", "content": "hi"},
        ]

        self.assertEqual(
            normalize_messages(messages),
            [
                {"role": "system", "content": "persona\n\nrules"},
                {"role": "user", "content": "hi"},
            ],
        )

    def test_later_system_message_is_not_merged_into_the_leading_one(self) -> None:
        messages = [
            {"role": "system", "content": "persona"},
            {"role": "user", "content": "hi"},
            {"role": "system", "content": "late note"},
            {"role": "system", "content": "another"},
        ]

        self.assertEqual(normalize_messages(messages), messages)

    def test_assistant_and_tool_turns_are_never_merged(self) -> None:
        tool_call = {
            "id": "call_1",
            "type": "function",
            "function": {"name": "recall_memory", "arguments": "{}"},
        }
        messages = [
            {"role": "user", "content": "q"},
            {"role": "assistant", "content": None, "tool_calls": [tool_call]},
            {"role": "assistant", "content": "second assistant"},
            {"role": "tool", "tool_call_id": "call_1", "name": "recall_memory", "content": "a"},
            {"role": "tool", "tool_call_id": "call_2", "name": "recall_memory", "content": "b"},
        ]

        self.assertEqual(normalize_messages(messages), messages)

    def test_assistant_part_list_collapses_to_text(self) -> None:
        messages = [
            {"role": "user", "content": "q"},
            {
                "role": "assistant",
                "content": [
                    {"type": "thinking", "thinking": "private"},
                    {"type": "text", "text": "part one, "},
                    {"type": "text", "text": "part two"},
                ],
                "tool_calls": [],
            },
        ]

        normalized = normalize_messages(messages)

        self.assertEqual(normalized[1]["content"], "part one, part two")
        self.assertEqual(normalized[1]["tool_calls"], [])

    def test_input_list_and_dicts_are_not_mutated(self) -> None:
        messages = [
            {"role": "system", "content": "a"},
            {"role": "system", "content": "b"},
            {"role": "user", "content": [{"type": "text", "text": "x"}]},
            {"role": "user", "content": "y"},
            {"role": "assistant", "content": [{"type": "text", "text": "z"}]},
        ]
        snapshot = copy.deepcopy(messages)

        normalized = normalize_messages(messages)

        self.assertEqual(messages, snapshot)
        self.assertIsNot(normalized, messages)
        self.assertIsNot(normalized[0], messages[0])

    def test_empty_user_strings_do_not_leave_stray_separators(self) -> None:
        messages = [
            {"role": "user", "content": ""},
            {"role": "user", "content": "real question"},
        ]

        self.assertEqual(
            normalize_messages(messages),
            [{"role": "user", "content": "real question"}],
        )

    def test_empty_list_stays_empty(self) -> None:
        self.assertEqual(normalize_messages([]), [])


class HasImagePartsTests(unittest.TestCase):
    def test_detects_image_url_parts_in_any_message(self) -> None:
        image = {"type": "image_url", "image_url": {"url": "data:image/png;base64,AAAA"}}

        self.assertTrue(
            has_image_parts(
                [
                    {"role": "system", "content": "sys"},
                    {"role": "user", "content": [{"type": "text", "text": "hi"}, image]},
                ]
            )
        )
        self.assertFalse(has_image_parts([{"role": "user", "content": "hi"}]))
        self.assertFalse(
            has_image_parts([{"role": "user", "content": [{"type": "text", "text": "hi"}]}])
        )
        self.assertFalse(has_image_parts([]))


if __name__ == "__main__":
    unittest.main()


class StripImagePartsTests(unittest.TestCase):
    def test_image_parts_go_and_text_only_turns_become_strings(self) -> None:
        image = {"type": "image_url", "image_url": {"url": "data:image/png;base64,AAAA"}}
        messages = [
            {"role": "system", "content": "sys"},
            {"role": "user", "content": [{"type": "text", "text": "引用的圖片"}, image]},
            {"role": "user", "content": [image]},
            {"role": "user", "content": [{"type": "text", "text": "畫成水彩"}, image, {"type": "text", "text": "謝謝"}]},
        ]
        original = copy.deepcopy(messages)

        stripped = strip_image_parts(messages)

        self.assertEqual(
            stripped,
            [
                {"role": "system", "content": "sys"},
                {"role": "user", "content": "引用的圖片"},
                {"role": "user", "content": "畫成水彩\n\n謝謝"},
            ],
        )
        self.assertFalse(has_image_parts(stripped))
        self.assertEqual(messages, original)

    def test_other_part_types_are_kept(self) -> None:
        audio = {"type": "input_audio", "input_audio": {"data": "x"}}
        messages = [{"role": "user", "content": [{"type": "text", "text": "hi"}, audio]}]

        self.assertEqual(strip_image_parts(messages), messages)
