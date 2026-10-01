"""to_line_text: model Markdown/LaTeX becomes plain text for LINE.

The samples are shaped like real Nemotron Ultra replies (live gate
2026-09-30): bold and LaTeX even in one-line answers, headings, lists,
tables and code in longer ones.
"""

from __future__ import annotations

import time
import unittest
from types import SimpleNamespace

from src.pipeline.event_pipeline import build_assistant_memory_text
from src.utils.line_text import to_line_text


class UltraStyleReplyTests(unittest.TestCase):
    def test_one_line_answer_with_bold(self) -> None:
        reply = "**台北 101** 是位於台灣台北市信義區的超高層摩天大樓，高 **508 公尺**，曾是世界最高建築。"

        self.assertEqual(
            to_line_text(reply),
            "台北 101 是位於台灣台北市信義區的超高層摩天大樓，高 508 公尺，曾是世界最高建築。",
        )

    def test_inline_latex_math(self) -> None:
        reply = (
            "兩管同時打開時，每小時可注入水池的 \\( \\frac{1}{6} + \\frac{1}{3} = \\frac{1}{2} \\)，"
            "所以需要 **2 小時** 注滿。"
        )

        self.assertEqual(
            to_line_text(reply), "兩管同時打開時，每小時可注入水池的 1/6 + 1/3 = 1/2，所以需要 2 小時 注滿。"
        )

    def test_structured_answer_with_headings_list_table_and_display_math(self) -> None:
        reply = (
            "### 解題步驟\n"
            "\n"
            "1. **設未知數**：小明現在 \\(x\\) 歲，小華 \\(x - 3\\) 歲。\n"
            "2. __列式__：\n"
            "\\[\n"
            "(x + 5) + (x - 3 + 5) = 31\n"
            "\\]\n"
            "3. 解得 $$x = 12$$\n"
            "\n"
            "| 人物 | 現在 | 五年後 |\n"
            "|------|:----:|------:|\n"
            "| 小明 | 12 | 17 |\n"
            "| 小華 | 9 | 14 |\n"
            "\n"
            "## 答案 ##\n"
            "小明現在 **12 歲** \\(\\times\\) 不是 \\(\\text{13 歲}\\)。"
        )

        self.assertEqual(
            to_line_text(reply),
            "解題步驟\n"
            "\n"
            "1. 設未知數：小明現在 x 歲，小華 x - 3 歲。\n"
            "2. 列式：\n"
            "(x + 5) + (x - 3 + 5) = 31\n"
            "3. 解得 x = 12\n"
            "\n"
            "| 人物 | 現在 | 五年後 |\n"
            "| 小明 | 12 | 17 |\n"
            "| 小華 | 9 | 14 |\n"
            "\n"
            "答案\n"
            "小明現在 12 歲 × 不是 13 歲。",
        )

    def test_code_keeps_its_lines_without_fences(self) -> None:
        reply = (
            "可以這樣寫：\n"
            "```python\n"
            "class Cart:\n"
            "    def __init__(self, **items):\n"
            "        self.total = sum(items.values())  # **不是粗體**\n"
            "```\n"
            "然後用 `Cart(**prices)` 建立，或執行 ``pip install `x` ``。"
        )

        self.assertEqual(
            to_line_text(reply),
            "可以這樣寫：\n"
            "class Cart:\n"
            "    def __init__(self, **items):\n"
            "        self.total = sum(items.values())  # **不是粗體**\n"
            "然後用 Cart(**prices) 建立，或執行 pip install `x`。",
        )

    def test_a_reply_cut_off_inside_a_code_block_keeps_the_code(self) -> None:
        reply = "範例：\n~~~js\nconst total = a ** b;\n**還沒結束**"

        self.assertEqual(to_line_text(reply), "範例：\nconst total = a ** b;\n**還沒結束**")

    def test_a_fence_under_a_list_item_is_still_a_fence(self) -> None:
        # 5a review: a fence indented 4+ spaces used to leave "`python".
        numbered = "1. 步驟一\n\n    ```python\n    print('x')\n    ```\n\n2. **步驟二**"
        nested = "- 父項\n  - 子項：\n    ```bash\n    ls -la **x**\n    ```"

        self.assertEqual(to_line_text(numbered), "1. 步驟一\n\n    print('x')\n\n2. 步驟二")
        self.assertEqual(to_line_text(nested), "- 父項\n  - 子項：\n    ls -la **x**")

    def test_a_line_that_starts_with_triple_backtick_inline_code_is_not_a_fence(self) -> None:
        # A backtick fence's info string cannot hold a backtick; this used
        # to swallow the first line and leave the rest unsanitized.
        self.assertEqual(
            to_line_text("```echo hi``` 然後 **粗體**\n第二行 **粗體**"), "echo hi 然後 粗體\n第二行 粗體"
        )

    def test_links_become_text_and_url_and_urls_stay_whole(self) -> None:
        reply = (
            "請看 [中央氣象署](https://www.cwa.gov.tw/V8/C/) 的預報，"
            "或 [https://example.com/a](https://example.com/a)。"
            "原始連結 https://example.com/__init__/**x?q=a__b 😀 保持不變。"
        )

        self.assertEqual(
            to_line_text(reply),
            "請看 中央氣象署 https://www.cwa.gov.tw/V8/C/ 的預報，"
            "或 https://example.com/a。"
            "原始連結 https://example.com/__init__/**x?q=a__b 😀 保持不變。",
        )

    def test_bold_around_a_url_goes_but_the_url_stays(self) -> None:
        self.assertEqual(
            to_line_text("官網：**https://www.taipei-101.com.tw/tw/**"),
            "官網：https://www.taipei-101.com.tw/tw/",
        )

    def test_a_url_ends_at_chinese_text_so_code_and_markup_after_it_still_go(self) -> None:
        # Review: the URL ran on to the next space and swallowed the held
        # inline code, which reached the user as a raw private-use marker.
        cases = {
            "參考 https://example.com，使用`pip install foo`安裝": "參考 https://example.com，使用pip install foo安裝",
            "https://example.com/a使用`x`與**重點**": "https://example.com/a使用x與重點",
            "見 https://a.example/`b` **c**": "見 https://a.example/b c",
        }
        for reply, expected in cases.items():
            with self.subTest(reply=reply):
                self.assertEqual(to_line_text(reply), expected)

    def test_private_use_characters_in_the_reply_cannot_pose_as_held_text(self) -> None:
        # One used to raise IndexError in the restore (the whole reply
        # became an error message) or bring back the wrong held text.
        self.assertEqual(to_line_text("a 0 b"), "a 0 b")
        self.assertEqual(to_line_text("`x` 與 0 與 **y**"), "x 與 0 與 y")


class ReasoningMathTests(unittest.TestCase):
    """LaTeX commands reasoning models put in final answers (5a review)."""

    def test_boxed_percent_and_square_roots(self) -> None:
        self.assertEqual(to_line_text("答案是 \\[ \\boxed{42} \\]"), "答案是 42")
        self.assertEqual(to_line_text("年增率約 \\( 3.2\\% \\)"), "年增率約 3.2%")
        self.assertEqual(to_line_text("\\( \\boxed{\\frac{1}{2}} \\)"), "1/2")
        self.assertEqual(to_line_text("\\( \\sqrt{2} + \\sqrt{x+1} \\)"), "√2 + √(x+1)")

    def test_a_compound_numerator_or_denominator_keeps_its_grouping(self) -> None:
        # Review: \frac{a+b}{c+d} came out as "a+b/c+d", which reads a + b/c + d.
        cases = {
            "\\( \\frac{a+b}{c+d} \\)": "(a+b)/(c+d)",
            "\\( \\frac{a+b}{2} \\)": "(a+b)/2",
            "\\( x = \\frac{-b \\pm \\sqrt{b^2-4ac}}{2a} \\)": "x = (-b ± √(b^2-4ac))/(2a)",
            "\\( P = \\frac{3}{4+6} \\)": "P = 3/(4+6)",
            "\\( \\frac{\\frac{1}{2}}{3} \\)": "(1/2)/3",
            "\\( \\frac{ 3.5 }{ x } = \\frac{\\text{總距離}}{\\text{時間}} = \\frac{\\pi}{2} \\)": "3.5/x = 總距離/時間 = \\pi/2",
        }
        for reply, expected in cases.items():
            with self.subTest(reply=reply):
                self.assertEqual(to_line_text(reply), expected)

    def test_left_right_and_aligned_rows(self) -> None:
        self.assertEqual(to_line_text("\\( \\left( a+b \\right)^2 \\)"), "( a+b )^2")
        aligned = "\\[\n\\begin{aligned}\nx &= 1 + 2 \\\\\n  &= 3\n\\end{aligned}\n\\]"
        self.assertEqual(to_line_text(aligned), "x = 1 + 2\n= 3")
        self.assertEqual(to_line_text("\\( a \\& b \\)"), "a & b")

    def test_a_reply_of_only_markup_is_blank(self) -> None:
        # send_response replaces a blank result with its default text.
        self.assertEqual(to_line_text("```\n```").strip(), "")


class StoredReplyTests(unittest.TestCase):
    def test_memory_keeps_the_reply_as_the_user_saw_it(self) -> None:
        # The next turn's history must not show the model its own markup.
        reply = SimpleNamespace(text="**2 小時**，因為 \\( \\frac{1}{2} \\) 池／時", output_format="text")
        self.assertEqual(build_assistant_memory_text(reply), "2 小時，因為 1/2 池／時")

        blank = SimpleNamespace(text="```\n```", output_format="image")
        self.assertEqual(build_assistant_memory_text(blank), "[已傳送圖片]")


_CODE = (
    "class Cart:\n"
    "    def __init__(self, items):\n"
    "        self.items = items\n"
    "\n"
    "# 依類別分派\n"
    "result = handlers[kind](event)\n"
    'nums = re.findall(r"\\((\\d+)\\)", "(1) [2]")\n'
    'tags = re.match(r"\\[(\\w+)\\]", line)\n'
    'if __name__ == "__main__":\n'
    "  # 啟動\n"
    "\tmain(**kwargs)\n"
)


class CodeTests(unittest.TestCase):
    """Code reaches the user runnable (code-review round 3: CODE-MANGLE)."""

    def test_a_fenced_block_comes_back_verbatim_without_its_fences(self) -> None:
        # Nothing inside a fence is touched, not even what is markup outside.
        body = (
            _CODE + "## 不是標題\n**x** `y` [a](https://a.example) \\( z \\) $$w$$\n|---|---|\n  \n"
        )
        for fence in ("```", "```python", "~~~"):
            with self.subTest(fence=fence):
                reply = f"可以這樣寫：\n{fence}\n{body}{fence[:3]}\n然後執行。"
                self.assertEqual(to_line_text(reply), f"可以這樣寫：\n{body}然後執行。")

    def test_unfenced_code_is_left_as_it_is(self) -> None:
        # Markdown heuristics used to give "def init", "handlerskind event",
        # r"(\d+)", 'if name == "main"', and drop the comments' "#".
        self.assertEqual(to_line_text(_CODE), _CODE)
        self.assertEqual(to_line_text("可以這樣寫：\n" + _CODE), "可以這樣寫：\n" + _CODE)

    def test_code_constructs_that_look_like_markup_stay(self) -> None:
        for line in (
            "super().__init__(**kwargs)",
            "from __future__ import annotations",
            "vars = self.__dict__",
            'path.join(__dirname, "__tests__")',
            'printf("%s:%d", __FILE__, __LINE__);',
            'fn = registry["save"](doc)',
            '"include": ["src/**/*.ts", "tests/**/*.ts"]',
            "sed 's/\\(.*\\)=\\(.*\\)/\\2=\\1/' f",
            "find . \\( -name '*.py' -o -name '*.md' \\) -print",
            "const re = /\\(\\d{3}\\) \\d{3}-\\d{4}/;",
            'LINK = re.compile(r"\\[(.+?)\\]\\((.+?)\\)")',
            'm = re.search(r"(\\(\\d+\\))", s)',
            "a\\[0\\] = 1",
            '在 Python 裡 __init__ 是建構子，__name__ 在直接執行時是 "__main__"。',
        ):
            with self.subTest(line=line):
                self.assertEqual(to_line_text(line), line)

    def test_escaped_parens_far_apart_do_not_become_one_math_span(self) -> None:
        code = 'OPEN = re.compile(r"\\(")\n\ndef check(a, b):\n    return a & b\n\nCLOSE = re.compile(r"\\)")'
        bash = 'echo "pid $$"\nif [ -f lock ] && true; then\n    kill $$\nfi'
        for text in (code, bash):
            with self.subTest(text=text):
                self.assertEqual(to_line_text(text), text)

    def test_markup_in_prose_still_goes(self) -> None:
        cases = {
            "### 重點\n__注意__：見 [官網](https://a.example/x)。": "重點\n注意：見 官網 https://a.example/x。",
            "所以 \\( f'(x) = 2x \\)，x > 0 時遞增": "所以 f'(x) = 2x，x > 0 時遞增",
        }
        for reply, expected in cases.items():
            with self.subTest(reply=reply):
                self.assertEqual(to_line_text(reply), expected)

    def test_a_long_backtick_run_is_fast(self) -> None:
        # (`+)(.+?)\1 backtracked cubically: 6000 backticks took ~5 s.
        for text in (
            "x" + "`" * 10_000,
            "答案：" + "`" * 10_000,
            "`" * 10_000 + " 結束 `",
            "` `` " * 2_000,
        ):
            with self.subTest(text=text[:8]):
                started = time.perf_counter()
                to_line_text(text)
                self.assertLess(time.perf_counter() - started, 0.2)


class PlainTextTests(unittest.TestCase):
    def test_plain_text_and_emoji_are_unchanged(self) -> None:
        for text in (
            "今天台北多雲，午後可能有雷陣雨 🌦️",
            "#台北美食 推薦這家，5 * 3 = 15，C# 也可以。",
            "- 第一點\n- 第二點\n* 第三點\n\n---\n結尾",
            "價格是 $100 到 $200，snake_case_name 與 a__b__c 不變。",
            "",
        ):
            with self.subTest(text=text):
                self.assertEqual(to_line_text(text), text)

    def test_bold_italic_and_underscores(self) -> None:
        self.assertEqual(to_line_text("***非常重要***：__先備份__再升級"), "非常重要：先備份再升級")

    def test_exponents_and_kwargs_are_not_bold(self) -> None:
        # Review: "2**3 + 4**2" used to lose its operators as if bolded.
        for text in (
            "計算 2**3 + 4**2 = 24",
            "x**2 + y**2 = r**2",
            "a**b**c 與 2***3 + 4***2",
            "呼叫 f(*args, **kwargs) 或 g(**opts)，再算 n**2",
        ):
            with self.subTest(text=text):
                self.assertEqual(to_line_text(text), text)

    def test_bold_next_to_cjk_english_and_punctuation_still_goes(self) -> None:
        cases = {
            "這是**重點**。": "這是重點。",
            "This is **important**, really.": "This is important, really.",
            "(**注意**)、「**提醒**」與 ***小心***！": "(注意)、「提醒」與 小心！",
            "**答案**：2**3 = 8，**不是 2**3 = 6**。": "答案：2**3 = 8，不是 2**3 = 6。",
        }
        for text, expected in cases.items():
            with self.subTest(text=text):
                self.assertEqual(to_line_text(text), expected)

    def test_only_table_separator_rows_are_dropped(self) -> None:
        table = "項目 | 價格\n--- | ---\n咖啡 | 60\n|---|\n| 茶 | 40 |"

        self.assertEqual(to_line_text(table), "項目 | 價格\n咖啡 | 60\n| 茶 | 40 |")


if __name__ == "__main__":
    unittest.main()
