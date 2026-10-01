"""Tolerant JSON reading: valid JSON comes back unchanged (audit A8)."""

from __future__ import annotations

import unittest

from src.utils.jsonish import iter_json_candidates, load_jsonish_dict


class LoadJsonishDictTests(unittest.TestCase):
    def test_valid_json_with_curly_quotes_and_full_width_commas_is_kept_as_is(self) -> None:
        raw = '{"task_description": "查詢“台北”天氣，然後回覆", "reasoning": "需要：即時資訊"}'

        self.assertEqual(
            load_jsonish_dict(raw),
            {"task_description": "查詢“台北”天氣，然後回覆", "reasoning": "需要：即時資訊"},
        )

    def test_full_width_punctuation_is_normalized_only_when_parsing_fails(self) -> None:
        self.assertEqual(load_jsonish_dict("{“agent”：“chat”，“ok”：true}"), {"agent": "chat", "ok": True})

    def test_python_literals_are_accepted(self) -> None:
        self.assertEqual(
            load_jsonish_dict("{'agent': 'chat', 'needs_thinking': True, 'reasoning': None}"),
            {"agent": "chat", "needs_thinking": True, "reasoning": None},
        )
        self.assertEqual(
            load_jsonish_dict('{"agent": "chat", "needs_thinking": TRUE, "x": null}'),
            {"agent": "chat", "needs_thinking": True, "x": None},
        )

    def test_anything_but_a_dict_is_none(self) -> None:
        for raw in ("[1, 2]", '"text"', "not json", "{[1]: 2}", ""):
            with self.subTest(raw=raw):
                self.assertIsNone(load_jsonish_dict(raw))


class IterJsonCandidatesTests(unittest.TestCase):
    def test_fenced_bare_and_embedded_objects(self) -> None:
        text = 'Sure:\n```json\n{"a": 1}\n```\nalso {"b": "}"} end'

        self.assertEqual(list(iter_json_candidates(text)), ['{"a": 1}', '{"b": "}"}'])


if __name__ == "__main__":
    unittest.main()
