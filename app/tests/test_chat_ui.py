"""Terminal chat: prompt-free startup, multiline input, decisions."""

from __future__ import annotations

import io
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest import mock

import artificium.cli  # noqa: F401  (cli and cli_chat import each other)
from artificium.cli_chat import (
    _chat_entity,
    _choose_interaction,
    _edit_message,
    _decision_panel,
    _panel_rows,
    _print_event,
    _read_chat_message,
    _route_message,
)
from artificium.filesystem import Paths
from artificium.interactions import ArtificiumClient


class ChatStartupCase(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.paths = Paths(Path(self.temporary.name))
        self.client = ArtificiumClient(self.paths.install)

    def test_entity_defaults_then_remembers_an_explicit_choice(self) -> None:
        self.assertEqual(_chat_entity(self.paths, None), "user_1")
        self.assertEqual(_chat_entity(self.paths, "alice"), "alice")
        self.assertEqual(_chat_entity(self.paths, None), "alice")

    def test_a_corrupt_entity_store_falls_back_to_the_default(self) -> None:
        store = self.paths.runtime / "chat-client.json"
        store.parent.mkdir(parents=True, exist_ok=True)
        store.write_text("{not json", encoding="utf-8")
        self.assertEqual(_chat_entity(self.paths, None), "user_1")
        store.write_text('{"entity": "../x"}', encoding="utf-8")
        self.assertEqual(_chat_entity(self.paths, None), "user_1")

    def test_startup_never_prompts_and_resumes_the_latest_thread(self) -> None:
        with mock.patch("builtins.input", side_effect=AssertionError("prompted")):
            first = _choose_interaction(self.client, "user_1", None, None)
            self.assertEqual(_choose_interaction(self.client, "user_1", None, None), first)
            fresh = _choose_interaction(self.client, "user_1", None, None, new=True)
            named = _choose_interaction(self.client, "user_1", None, "Release plan")
        self.assertNotEqual(fresh, first)
        self.assertTrue(named.startswith("release-plan-"))
        self.assertEqual(
            _choose_interaction(self.client, "user_1", "given-thread", None), "given-thread"
        )


def _scripted(*answers):
    queue = list(answers)

    def read(prompt: str) -> str:
        shown.append(prompt)
        item = queue.pop(0)
        if isinstance(item, BaseException):
            raise item
        return item

    shown: list[str] = []
    return read, shown


class MultilineInputCase(unittest.TestCase):
    def read_message(self, *answers, edit=lambda: "edited"):
        read, shown = _scripted(*answers)
        prompts: list[str] = []
        message = _read_chat_message("me> ", prompts.append, read=read, edit=edit)
        return message, shown, prompts

    def test_a_single_line_is_unchanged(self) -> None:
        message, shown, prompts = self.read_message("  hello  ")
        self.assertEqual(message, "hello")
        self.assertEqual(shown, ["me> "])
        self.assertEqual(prompts[-1], "me> ")

    def test_trailing_backslash_continues_with_a_continuation_prompt(self) -> None:
        message, shown, prompts = self.read_message("first\\", "second\\", "third")
        self.assertEqual(message, "first\nsecond\nthird")
        self.assertEqual(shown, ["me> ", "... ", "... "])
        self.assertIn("... ", prompts)
        self.assertEqual(prompts[-1], "me> ")

    def test_paste_reads_until_a_lone_dot_and_keeps_slashes_literal(self) -> None:
        message, _, _ = self.read_message("/paste", "/attach x", "line two", "  .  ")
        self.assertEqual(message, "/attach x\nline two")

    def test_paste_ends_on_end_of_input(self) -> None:
        message, _, _ = self.read_message("/paste", "only", EOFError())
        self.assertEqual(message, "only")

    def test_edit_returns_the_editor_text(self) -> None:
        message, _, _ = self.read_message("/edit", edit=lambda: "long\ntext")
        self.assertEqual(message, "long\ntext")

    def test_edit_and_paste_are_plain_text_inside_a_draft(self) -> None:
        message, _, _ = self.read_message("start\\", "/edit")
        self.assertEqual(message, "start\n/edit")

    def test_interrupt_discards_a_draft_but_still_quits_on_an_empty_prompt(self) -> None:
        message, _, prompts = self.read_message("draft\\", KeyboardInterrupt())
        self.assertEqual(message, "")
        self.assertEqual(prompts[-1], "me> ")
        with self.assertRaises(KeyboardInterrupt):
            self.read_message(KeyboardInterrupt())


class EditorCase(unittest.TestCase):
    def run_editor(self, editor: str) -> str:
        with mock.patch.dict("os.environ", {"EDITOR": editor, "VISUAL": ""}):
            return _edit_message()

    def test_editor_text_is_returned_trimmed(self) -> None:
        text = self.run_editor("""sh -c 'printf "line one\\nline two\\n\\n" > "$0"'""")
        self.assertEqual(text, "line one\nline two")

    def test_a_failing_editor_sends_nothing(self) -> None:
        self.assertEqual(self.run_editor("false"), "")
        self.assertEqual(self.run_editor("definitely-not-an-editor-xyz"), "")


def _decision(number: int, text: str, options=None, stamp="2026-10-06T08:00:00Z") -> dict:
    event = {"id": f"event_{number}", "direction": "outbound", "kind": "decision",
             "sender": "artificium", "content": text, "created_at": stamp}
    if options:
        event["options"] = options
    return event


class DecisionPanelCase(unittest.TestCase):
    def test_no_decisions_means_no_panel(self) -> None:
        self.assertEqual(_decision_panel([]), [])

    def test_panel_numbers_decisions_and_lists_options(self) -> None:
        lines = _decision_panel(
            [_decision(1, "Which toolchain?\nIt matters.", ["clang", "gcc"]),
             _decision(2, "Delete the cache?")], columns=60)
        text = "\n".join(lines)
        self.assertIn("2 decisions waiting for you", lines[0])
        self.assertIn("[1]", text)
        self.assertIn("[2]", text)
        self.assertIn("1) clang", text)
        self.assertIn("It matters.", text)
        self.assertIn("/reply N TEXT", lines[-1])
        self.assertNotIn("just type", lines[-1])

    def test_a_single_decision_says_plain_typing_answers_it(self) -> None:
        lines = _decision_panel([_decision(1, "Proceed?")])
        self.assertIn("1 decision waiting", lines[0])
        self.assertIn("just type your answer", lines[-1])

    def test_a_long_decision_is_trimmed_in_the_panel_only(self) -> None:
        body = "\n".join(f"line {n}" for n in range(30))
        text = "\n".join(_decision_panel([_decision(1, body)]))
        self.assertIn("line 9", text)
        self.assertNotIn("line 10", text)
        self.assertIn("20 more lines", text)

    def test_color_is_off_unless_requested(self) -> None:
        self.assertNotIn("\x1b", "\n".join(_decision_panel([_decision(1, "Q?")])))
        self.assertIn("\x1b", "\n".join(_decision_panel([_decision(1, "Q?")], color=True)))

    def test_row_count_ignores_color_codes_and_counts_wrapping(self) -> None:
        self.assertEqual(_panel_rows(["\x1b[1mabc\x1b[0m", ""], columns=10), 2)
        self.assertEqual(_panel_rows(["x" * 25], columns=10), 3)

    def test_history_marks_decisions_and_dims_only_agent_updates(self) -> None:
        def show(event, color=False):
            buffer = io.StringIO()
            with redirect_stdout(buffer):
                _print_event(event, local_entity="user_1", color=color)
            return buffer.getvalue()

        decision = show(_decision(1, "Pick?", ["a", "b"]), color=True)
        self.assertIn("· decision", decision)
        self.assertIn("1) a", decision)
        self.assertNotIn("\x1b[2m", decision)
        update = {"direction": "outbound", "sender": "artificium", "content": "Done.",
                  "created_at": "2026-10-06T08:00:00Z"}
        self.assertIn("\x1b[2mDone.", show(update, color=True))
        self.assertNotIn("\x1b", show(update))


class ReplyRoutingCase(unittest.TestCase):
    def test_one_open_decision_is_answered_by_a_plain_message(self) -> None:
        content, target, note = _route_message("go ahead", [_decision(1, "Proceed?")])
        self.assertEqual((content, target), ("go ahead", "event_1"))
        self.assertIn("answering", note)

    def test_a_bare_option_number_sends_that_option(self) -> None:
        decisions = [_decision(1, "Which?", ["clang", "gcc"])]
        self.assertEqual(_route_message("2", decisions)[:2], ("gcc", "event_1"))
        self.assertEqual(_route_message("3", decisions)[:2], ("3", "event_1"))

    def test_several_open_decisions_are_never_guessed(self) -> None:
        decisions = [_decision(1, "A?"), _decision(2, "B?", ["x", "y"])]
        content, target, note = _route_message("sure", decisions)
        self.assertEqual((content, target), ("sure", None))
        self.assertIn("/reply N", note)
        self.assertEqual(_route_message("/reply 2 2", decisions)[:2], ("y", "event_2"))
        self.assertEqual(_route_message("/reply 1 yes, do it", decisions)[:2],
                         ("yes, do it", "event_1"))

    def test_bad_reply_commands_are_explained_not_sent(self) -> None:
        decisions = [_decision(1, "A?")]
        for text in ("/reply", "/reply 1", "/reply 9 hi", "/reply x hi"):
            content, target, note = _route_message(text, decisions)
            self.assertIsNone(target, text)
            self.assertIn("Usage", note)
        self.assertIn("No decision", _route_message("/reply 1 hi", [])[2])

    def test_with_nothing_open_a_message_is_just_a_message(self) -> None:
        self.assertEqual(_route_message("hello", []), ("hello", None, None))
        self.assertEqual(_route_message("/replying now", []), ("/replying now", None, None))


if __name__ == "__main__":
    unittest.main()
