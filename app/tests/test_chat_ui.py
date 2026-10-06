"""Terminal chat: prompt-free startup, multiline input, decisions."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest import mock

import artificium.cli  # noqa: F401  (cli and cli_chat import each other)
from artificium.cli_chat import _chat_entity, _choose_interaction
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


if __name__ == "__main__":
    unittest.main()
