"""Terminal chat: prompt-free startup, multiline input, decisions."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest import mock

import artificium.cli  # noqa: F401  (cli and cli_chat import each other)
from artificium.cli_chat import (
    _chat_entity,
    _choose_interaction,
    _edit_message,
    _read_chat_message,
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


if __name__ == "__main__":
    unittest.main()
