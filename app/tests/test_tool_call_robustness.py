"""Tool-call robustness against real Qwen life-loop failure shapes.

The cases below are shortened/anonymized reproductions of raw model output
captured in production runs (``tool_request_rejected`` events over ~44 hours
across two long-running instances). The dominant real failure was not
malformed escaping but the model closing its JSON object one field early
and then continuing to emit more "fields" as bare text, e.g.::

    {"tool":"write_file","content":"...huge script..."}
    path":"mind/space/world/scripts/check.py","mode":"create"}
    </tool_call>

That premature ``}`` makes ``json.loads`` succeed early with only `content`,
silently dropping `path`/`mode` — previously surfacing as "missing a required
argument: 'path'" even though the model's intent was completely unambiguous.
This file also covers the batch-withholding change (a valid prefix before an
invalid call now executes) and the truncated-response chunking guidance.
"""

from __future__ import annotations

import dataclasses
import json
import os
import shutil
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from artificium.config import Config, ConfigStore
from artificium.filesystem import Paths
from artificium.life_loop import parse_life_loop_output
from artificium.records import Console
from artificium.runtime import Artificium

from test_final_release import ROOT, ScriptedEngine, call


def calls(content: str, **kwargs) -> list[tuple[str, dict, str | None]]:
    parsed = parse_life_loop_output(content, **kwargs)
    return [(t.name, t.arguments, t.parse_error) for t in parsed.tools]


class PrematureObjectClose(unittest.TestCase):
    """`{...}key":"value"...}`: the model closed the object a field early."""

    def test_single_dangling_field_after_a_premature_close(self) -> None:
        # Anonymized from a real rejection: a large write_file whose `path`
        # and `mode` landed after the content string's closing `}` instead
        # of before it.
        content = (
            '<tool_call>{"tool":"write_file","content":"def main():\\n    pass\\n"}\n'
            'path":"mind/space/world/scripts/check.py","mode":"create"}\n</tool_call>'
        )
        [(name, arguments, error)] = calls(content)
        self.assertEqual(name, "write_file")
        self.assertIsNone(error)
        self.assertEqual(
            arguments,
            {
                "content": "def main():\n    pass\n",
                "path": "mind/space/world/scripts/check.py",
                "mode": "create",
            },
        )

    def test_chained_dangling_fields_after_two_premature_closes(self) -> None:
        # A rarer variant: the model closed early more than once in the same
        # request. Each dangling field still carries its own closing quote,
        # so the repair is still unambiguous and can chain.
        content = (
            '<tool_call>{"tool":"write_file","content":"x = 1\\n"}\n'
            'mode":"overwrite"}\n'
            'path":"mind/space/world/scripts/_nearcheck.py"}\n</tool_call>'
        )
        [(name, arguments, error)] = calls(content)
        self.assertEqual(name, "write_file")
        self.assertIsNone(error)
        self.assertEqual(
            arguments,
            {
                "content": "x = 1\n",
                "mode": "overwrite",
                "path": "mind/space/world/scripts/_nearcheck.py",
            },
        )

    def test_two_genuinely_separate_calls_are_not_merged(self) -> None:
        # A complete object followed by another complete `{...}` object (no
        # dangling bare key) is two intended requests, not a merge target.
        content = (
            '<tool_call>{"tool":"write_file","path":"a.py","content":"real"}\n'
            '{"tool":"write_file","path":"b.py","content":"placeholder"}</tool_call>'
        )
        self.assertEqual(
            calls(content),
            [
                ("write_file", {"path": "a.py", "content": "real"}, None),
                ("write_file", {"path": "b.py", "content": "placeholder"}, None),
            ],
        )

    def test_dangling_text_that_is_not_a_key_stays_rejected(self) -> None:
        # Genuinely ambiguous: nothing shaped like `ident":` follows the
        # premature close, so there is no unambiguous key to restore.
        content = '<tool_call>{"tool":"write_file","content":"done"}\nand then what?</tool_call>'
        [(_, arguments, error)] = calls(content)
        self.assertIsNone(error)
        # Nothing to merge: this simply is a complete, if incomplete-looking,
        # request; the trailing prose is left as visible text, not swallowed.
        self.assertEqual(arguments, {"content": "done"})


class MissingOpeningKeyQuote(unittest.TestCase):
    """`,key":"value"`: the key already carries its closing quote."""

    def test_dangling_key_missing_its_opening_quote_is_restored(self) -> None:
        # Anonymized from a real rejection (`run_shell` gained a `cwd`
        # argument whose key lost its opening quote after a stray space).
        content = '<tool_call>{"tool":"run_shell","command":"ls -la" ,cwd":"mind/space"}</tool_call>'
        [(name, arguments, error)] = calls(content)
        self.assertEqual(name, "run_shell")
        self.assertIsNone(error)
        self.assertEqual(arguments, {"command": "ls -la", "cwd": "mind/space"})

    def test_still_rejects_a_doubled_quote_with_no_key_shape(self) -> None:
        # The text after the stray quote does not look like `ident":`, so
        # there is no unambiguous repair; this must stay rejected.
        content = '<tool_call>{"tool":"run_shell","command":"echo hi""}</tool_call>'
        [(_, _, error)] = calls(content)
        self.assertIn("invalid tool-call JSON", error)


class TruncatedResponseGuidance(unittest.TestCase):
    def test_cut_off_write_file_suggests_chunking_with_append(self) -> None:
        # A complete, well-formed object that simply never got its closing
        # `</tool_call>` because the output limit hit right after it.
        content = (
            '<tool_call>{"tool":"write_file","path":"mind/space/big.txt","mode":"create",'
            '"content":"first part of a very long file"}'
        )
        [(name, arguments, error)] = calls(content, finish_reason="length")
        self.assertEqual(name, "write_file")
        self.assertEqual(arguments, {})
        self.assertIn("cut off", error)
        self.assertIn('mode "append"', error)

    def test_cut_off_mid_string_also_suggests_chunking(self) -> None:
        # A truncated response can also cut off mid-string with no closing
        # brace or tag at all; that path raises a decode failure rather than
        # returning a clean-but-incomplete object, and should carry the same
        # chunking hint.
        content = '<tool_call>{"tool":"write_file","path":"a.txt","content":"unterminated'
        [(_, _, error)] = calls(content, finish_reason="length")
        self.assertIn("no closing", error)
        self.assertIn('mode "append"', error)


class SequentialPrefixExecutes(unittest.TestCase):
    """The all-or-nothing batch withholding is now a prefix/suffix split."""

    def setUp(self) -> None:
        # Mirrors the minimal fixture in test_final_release.FinalReleaseCase
        # without inheriting its own test methods.
        check = mock.patch(
            "artificium.setup.verify_connection",
            side_effect=lambda paths, config, key, **kw: (config, {}),
        )
        check.start()
        self.addCleanup(check.stop)
        temporary = tempfile.TemporaryDirectory(dir=ROOT.parent)
        self.addCleanup(temporary.cleanup)
        self.paths = Paths(Path(temporary.name))
        shutil.copytree(ROOT / "app/prompts", self.paths.prompts)
        shutil.copytree(ROOT / "app/seed", self.paths.seed)
        self.paths.ensure_layout()
        self.config = Config(
            provider="custom", model="test", base_url="http://example.invalid/v1",
            context_window_tokens=50000, max_life_loop_rounds=1,
        )
        ConfigStore(self.paths).save(self.config)
        environment = mock.patch.dict(os.environ, {}, clear=True)
        environment.start()
        self.addCleanup(environment.stop)
        network = mock.patch("urllib.request.urlopen", side_effect=OSError("offline fixture"))
        network.start()
        self.addCleanup(network.stop)

    def agent(self, engine=None, **settings):
        self.config = dataclasses.replace(self.config, **settings)
        ConfigStore(self.paths).save(self.config)
        agent = Artificium(self.paths, engine=engine or ScriptedEngine(), console=Console(quiet=True))
        agent.initialization.finish("Self, meta-memory, tools, environment, and pending events inspected.")
        return agent

    def fill_context(self, agent):
        agent.working.append({"role": "assistant", "content": "source detail " * 8500}, origin="test")

    def test_valid_calls_before_the_first_invalid_one_run(self) -> None:
        engine = ScriptedEngine(
            call("write_file", path="mind/space/one.txt", content="first")
            + call("write_file", path="mind/space/two.txt", content="second")
            + call("not_a_real_tool")
        )
        agent = self.agent(engine, max_life_loop_rounds=1)
        agent.run_turn()
        self.assertEqual((self.paths.space / "one.txt").read_text(), "first")
        self.assertEqual((self.paths.space / "two.txt").read_text(), "second")
        repair_context = self.paths.working_context.read_text()
        self.assertIn("already executed", repair_context)
        self.assertIn("not_a_real_tool", repair_context)

    def test_mandatory_offload_still_withholds_the_whole_batch(self) -> None:
        # A call that mandatory offloading would allow on its own (save_memory)
        # must still wait behind a later blocked call in the same response:
        # this all-or-nothing gate is deliberately unchanged by the prefix fix.
        engine = ScriptedEngine(
            call("save_memory", path="test/note", content="a lesson", retrieve_when="later")
            + call("write_file", path="mind/space/blocked.txt", content="blocked")
        )
        agent = self.agent(engine, mandatory_offload=True)
        self.fill_context(agent)
        agent.run_turn()
        self.assertFalse((self.paths.space / "blocked.txt").exists())
        self.assertFalse((self.paths.memory / "test/note.txt").exists())


if __name__ == "__main__":
    unittest.main()
