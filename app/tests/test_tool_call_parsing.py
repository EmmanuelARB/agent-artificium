"""Tolerant reading of textual tool requests.

The cases below are taken from a real Qwen life-loop whose rejected requests
were diagnosed one by one: most carried a complete, unambiguous request that
the old tag-to-tag regular expression refused (missing or repeated tags, raw
newlines or lone backslashes inside JSON strings). Requests with a genuine
ambiguity must still be rejected.
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from artificium.life_loop import parse_life_loop_output, render_normalized_life_loop_output


def calls(content: str, **kwargs) -> list[tuple[str, dict, str | None]]:
    parsed = parse_life_loop_output(content, **kwargs)
    return [(t.name, t.arguments, t.parse_error) for t in parsed.tools]


class MissingClosingTag(unittest.TestCase):
    def test_reply_ending_right_after_the_object_is_accepted(self) -> None:
        content = (
            '<tool_call>{"tool":"run_shell","command":"rm -rf x; python3 -c \\"import mpmath\\"",'
            '"timeout_seconds":300}'
        )
        self.assertEqual(
            calls(content, finish_reason="stop"),
            [("run_shell", {"command": 'rm -rf x; python3 -c "import mpmath"', "timeout_seconds": 300}, None)],
        )

    def test_truncated_reply_keeps_an_unclosed_request_invalid(self) -> None:
        content = '<tool_call>{"tool":"read_file","path":"a"}'
        [(name, arguments, error)] = calls(content, finish_reason="length")
        self.assertEqual(name, "read_file")
        self.assertEqual(arguments, {})
        self.assertIn("cut off", error)

    def test_truncated_reply_with_closed_request_is_fine(self) -> None:
        content = '<tool_call>{"tool":"read_file","path":"a"}</tool_call> and then'
        self.assertEqual(calls(content, finish_reason="length"), [("read_file", {"path": "a"}, None)])

    def test_two_requests_without_closing_tag_between_them_are_both_kept(self) -> None:
        # Previously "Extra data": both requests were merged and lost.
        content = (
            '<tool_call>{"tool":"run_shell","command":"cat log","cwd":"code"}\n'
            '<tool_call>{"tool":"read_file","path":"mind/memory/projects/riemann/index.txt"}\n'
            "</tool_call>"
        )
        self.assertEqual(
            calls(content),
            [
                ("run_shell", {"command": "cat log", "cwd": "code"}, None),
                ("read_file", {"path": "mind/memory/projects/riemann/index.txt"}, None),
            ],
        )

    def test_stray_closing_tag_is_not_visible_text(self) -> None:
        parsed = parse_life_loop_output('<tool_call>{"tool":"list_interactions"}\n\n</tool_call>\nDone.')
        self.assertEqual(parsed.visible, "Done.")


class RepeatedAndEmptyTags(unittest.TestCase):
    def test_repeated_opening_tags_are_one_request(self) -> None:
        # Previously "Expecting value: line 1 column 1 (char 0)".
        content = (
            '<tool_call>{"tool":"save_memory","path":"a.txt","content":"x","retrieve_when":"y"}\n'
            "</tool_call>\n<tool_call>\n<tool_call>\n<tool_call>\n"
            '{"tool":"run_shell","command":"mkdir -p mind/space/riemann && echo created","timeout_seconds":15}\n'
            "</tool_call>"
        )
        self.assertEqual([c[0] for c in calls(content)], ["save_memory", "run_shell"])
        self.assertTrue(all(error is None for _, _, error in calls(content)))

    def test_empty_request_pair_requests_nothing(self) -> None:
        content = '<tool_call></tool_call><tool_call>{"tool":"list_interactions"}</tool_call>'
        self.assertEqual(calls(content), [("list_interactions", {}, None)])

    def test_two_objects_in_one_tag_are_two_requests(self) -> None:
        content = '<tool_call>\n{"tool":"read_file","path":"a"}\n{"tool":"read_file","path":"b"}\n</tool_call>'
        self.assertEqual(
            calls(content), [("read_file", {"path": "a"}, None), ("read_file", {"path": "b"}, None)]
        )


class StringContents(unittest.TestCase):
    def test_raw_newlines_inside_strings_are_accepted(self) -> None:
        # Previously "Invalid control character".
        content = (
            '<tool_call>{"tool":"write_file","content":"import numpy as np\n\n'
            'def f(u):\n    return u\n","path":"code/dbn.py","mode":"create"}</tool_call>'
        )
        [(name, arguments, error)] = calls(content)
        self.assertIsNone(error)
        self.assertEqual(arguments["content"], "import numpy as np\n\ndef f(u):\n    return u\n")

    def test_lone_backslashes_are_literal(self) -> None:
        # Previously "Invalid \\escape".
        content = (
            '<tool_call>{"tool":"run_shell","command":"grep -n -E \'dps|prec|mp\\.mp\' z.py; '
            "grep -nE 'def |t \\+=' r.py\"}</tool_call>"
        )
        [(name, arguments, error)] = calls(content)
        self.assertIsNone(error)
        self.assertEqual(arguments["command"], "grep -n -E 'dps|prec|mp\\.mp' z.py; grep -nE 'def |t \\+=' r.py")

    def test_valid_escapes_next_to_a_lone_backslash_keep_their_meaning(self) -> None:
        content = '<tool_call>{"tool":"run_shell","command":"printf \\"a\\\\n\\" | grep \\d"}</tool_call>'
        [(_, arguments, error)] = calls(content)
        self.assertIsNone(error)
        self.assertEqual(arguments["command"], 'printf "a\\n" | grep \\d')

    def test_tags_quoted_inside_a_string_are_data(self) -> None:
        content = (
            '<tool_call>{"tool":"save_memory","path":"harness/p.txt",'
            '"content":"Call tools as <tool_call>{\\"tool\\":\\"x\\"}</tool_call>; think in </think>.",'
            '"retrieve_when":"always"}</tool_call>'
        )
        [(name, arguments, error)] = calls(content)
        self.assertIsNone(error)
        self.assertIn("</tool_call>; think in </think>.", arguments["content"])


class UnescapedQuotes(unittest.TestCase):
    def test_quotes_inside_code_are_literal(self) -> None:
        # Previously "Expecting ',' delimiter".
        content = (
            '<tool_call>{"tool":"run_shell","command":"cd code && cat out.txt; '
            'echo "---procs---"; for f in *.log; do printf \'%s: \' "$f"; done"}</tool_call>'
        )
        [(_, arguments, error)] = calls(content)
        self.assertIsNone(error)
        self.assertEqual(
            arguments["command"],
            'cd code && cat out.txt; echo "---procs---"; for f in *.log; do printf \'%s: \' "$f"; done',
        )

    def test_python_print_in_a_written_file_is_kept_whole(self) -> None:
        content = (
            '<tool_call>{"tool":"write_file","path":"a.py","content":"x = 1\n'
            'print("m   Phi", x)\n","mode":"create"}</tool_call>'
        )
        [(_, arguments, error)] = calls(content)
        self.assertIsNone(error)
        self.assertEqual(arguments, {"path": "a.py", "content": 'x = 1\nprint("m   Phi", x)\n', "mode": "create"})

    def test_json_looking_code_never_cuts_a_value_short(self) -> None:
        # Repairing `{"a": "b", "c": 1}` inside the string would end the object
        # early and invent a `c` argument; the request must be rejected instead.
        content = (
            '<tool_call>{"tool":"write_file","path":"a.py",'
            '"content":"d = {"a": "b", "c": 1}\nprint(d)\n"}</tool_call>'
        )
        [(_, arguments, error)] = calls(content)
        self.assertIn("invalid tool-call JSON", error)
        self.assertEqual(arguments, {})

    def test_doubled_quote_is_rejected(self) -> None:
        content = '<tool_call>{"tool":"run_shell","command":"echo \'(none)\'"","timeout_seconds":30}</tool_call>'
        [(_, _, error)] = calls(content)
        self.assertIn("invalid tool-call JSON", error)


class GenuineErrorsStayErrors(unittest.TestCase):
    def test_ambiguous_escaped_backtick_is_rejected(self) -> None:
        # `\\\`` could mean `\`` or `\\`` for the shell: different commands.
        content = '<tool_call>{"tool":"run_shell","command":"python3 -c \\"a=\'\\\\\\`x\\\\\\`\'\\""}</tool_call>'
        [(_, _, error)] = calls(content)
        self.assertIn("Invalid \\escape", error)

    def test_key_missing_its_opening_quote_is_recovered_not_swallowed_into_content(self) -> None:
        # The dangling key already carries its closing quote, which is the
        # only unambiguous reading: a new key lost its opening quote. It must
        # be restored as its own argument, never merged into the preceding
        # string (which would silently rewrite `content`).
        content = (
            '<tool_call>{"tool":"write_file","path":"a.py","content":"print(\\"x\\")\\n",'
            'mode":"create"}</tool_call>'
        )
        [(_, arguments, error)] = calls(content)
        self.assertIsNone(error)
        self.assertEqual(arguments, {"path": "a.py", "content": 'print("x")\n', "mode": "create"})

    def test_missing_key_quote_is_recovered_and_the_next_request_survives(self) -> None:
        content = (
            '<tool_call>{"tool":"run_shell","command":"ls" ,cwd":"."}</tool_call>'
            '<tool_call>{"tool":"read_file","path":"a"}</tool_call>'
        )
        result = calls(content)
        self.assertEqual(result[0], ("run_shell", {"command": "ls", "cwd": "."}, None))
        self.assertEqual(result[1], ("read_file", {"path": "a"}, None))

    def test_unclosed_undecodable_request_is_rejected(self) -> None:
        [(name, _, error)] = calls('<tool_call>{"tool":"read_file","path":"a"')
        self.assertIn("no closing", error)
        self.assertEqual(name, "read_file")

    def test_non_json_body_with_closing_tag_is_rejected(self) -> None:
        [(_, _, error)] = calls("<tool_call>\nplease list the files\n</tool_call>")
        self.assertIn("invalid tool-call JSON", error)


class OtherRequestForms(unittest.TestCase):
    def test_qwen_native_function_form(self) -> None:
        content = (
            "<tool_call>\n<function=run_shell>\n<parameter=command>\n"
            'cd code && echo "=== tail ===" && tail -n 15 a.log\n</parameter>\n'
            "<parameter=timeout_seconds>\n30\n</parameter>\n</function>\n</tool_call>\n"
            "<tool_call>\n<function=list_directory>\n<parameter=path>\nmind/memory/context\n</parameter>\n"
            "<parameter=include_hidden>\nfalse\n</parameter>\n</function>\n</tool_call>"
        )
        self.assertEqual(
            calls(content),
            [
                ("run_shell", {"command": 'cd code && echo "=== tail ===" && tail -n 15 a.log', "timeout_seconds": 30}, None),
                ("list_directory", {"path": "mind/memory/context", "include_hidden": False}, None),
            ],
        )

    def test_mangled_object_starts(self) -> None:
        for body in ('<tool":"read_file","path":"a"}', '<tool="read_file","path":"a"}'):
            with self.subTest(body=body):
                self.assertEqual(calls(f"<tool_call>\n{body}\n</tool_call>"), [("read_file", {"path": "a"}, None)])

    def test_mangled_start_with_a_redirect_is_not_a_repeated_tag(self) -> None:
        # `<tool=...2>` once matched as a repeated opening tag up to the `>`.
        content = '<tool_call>\n<tool="run_shell","command":"ls 2>/dev/null; echo done"}</tool_call>'
        self.assertEqual(calls(content), [("run_shell", {"command": "ls 2>/dev/null; echo done"}, None)])

    def test_stray_text_before_the_object(self) -> None:
        for prefix in ("tool:", "<function=tool>", "tool\n<parameter>\n"):
            with self.subTest(prefix=prefix):
                content = f'<tool_call>{prefix}{{"tool":"read_file","path":"a"}}</tool_call>'
                self.assertEqual(calls(content), [("read_file", {"path": "a"}, None)])

    def test_key_value_lines(self) -> None:
        content = "<tool_call>\ntool: read_file\npath: mind/memory/a.txt\n</tool_call>"
        self.assertEqual(calls(content), [("read_file", {"path": "mind/memory/a.txt"}, None)])

    def test_dangling_opening_tag_requests_nothing(self) -> None:
        content = '<tool_call>{"tool":"read_file","path":"a"}\n<tool_call>'
        self.assertEqual(calls(content), [("read_file", {"path": "a"}, None)])


class ThoughtsAndProse(unittest.TestCase):
    def test_unopened_leading_thought_hides_its_draft_request(self) -> None:
        # The template opened <think> in the prompt; the draft inside the
        # thought must not run next to the real request.
        content = (
            'draft: <tool_call>{"tool":"run_shell","command":"ps -p 1 && cat a.log"}\n'
            "</invoke>\n\n</think>\n\n"
            '<tool_call>{"tool":"run_shell","command":"ps -p 1; cat a.log"}\n</tool_call>'
        )
        parsed = parse_life_loop_output(content)
        self.assertEqual([t.arguments["command"] for t in parsed.tools], ["ps -p 1; cat a.log"])
        self.assertEqual(len(parsed.thoughts), 1)
        self.assertEqual(parsed.visible, "")

    def test_prose_mention_of_the_tag_is_text(self) -> None:
        content = 'I will now emit a <tool_call> to read it.\n<tool_call>{"tool":"read_file","path":"x"}</tool_call>'
        parsed = parse_life_loop_output(content)
        self.assertEqual([(t.name, t.parse_error) for t in parsed.tools], [("read_file", None)])

    def test_backticked_mention_is_text(self) -> None:
        parsed = parse_life_loop_output("Close each `<tool_call>` block.")
        self.assertEqual(parsed.tools, [])

    def test_request_drafted_inside_a_closed_thought_is_ignored(self) -> None:
        content = (
            '<think>maybe <tool_call>{"tool":"read_file","path":"a"}</tool_call></think>'
            '<tool_call>{"tool":"read_file","path":"b"}</tool_call>'
        )
        self.assertEqual(calls(content), [("read_file", {"path": "b"}, None)])

    def test_thought_left_open_is_thinking_not_a_malformed_request(self) -> None:
        content = '<think>Next I will emit <tool_call>{"tool":"run_shell","command":'
        parsed = parse_life_loop_output(content, finish_reason="stop")
        self.assertEqual(parsed.tools, [])
        self.assertEqual(len(parsed.thoughts), 1)
        self.assertEqual(parsed.visible, "")

    def test_valid_request_after_a_forgotten_close_still_runs(self) -> None:
        content = '<think>check it <tool_call>{"tool":"read_file","path":"a"}</tool_call>'
        parsed = parse_life_loop_output(content)
        self.assertEqual([(t.name, t.parse_error) for t in parsed.tools], [("read_file", None)])
        self.assertEqual(parsed.thoughts, ["check it"])

    def test_normalized_context_uses_the_canonical_form(self) -> None:
        parsed = parse_life_loop_output('<think>t</think><tool_call>\n<tool_call>{"tool":"list_interactions"}')
        self.assertEqual(
            render_normalized_life_loop_output(parsed),
            '<think>t</think>\n<tool_call>{"tool":"list_interactions"}</tool_call>',
        )


if __name__ == "__main__":
    unittest.main()
