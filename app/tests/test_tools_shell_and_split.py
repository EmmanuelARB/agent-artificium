"""Coverage for B4 (run_shell robustness), B8 (head/tail previews on bounded
tool output), tool-execution timing, the public control()/save_control()
accessors, and the S7 split of ToolRegistry into domain mixins.

These tests exercise the tool layer directly (no engine/model involved), the
same way tests/test_layout_and_overrides.py does, so a fresh workspace is
built per test in a temporary directory.
"""
from __future__ import annotations

import os
import shlex
import shutil
import subprocess
import sys
import tempfile
import time
import unittest
from pathlib import Path

from artificium.config import Config
from artificium.filesystem import Paths, json_dumps, read_json
from artificium.initialization import Initialization
from artificium.interactions import InteractionStore, NotificationStore
from artificium.life_loop import ToolIntent
from artificium.memory import InfiniteAttention, LongTermMemory, TokenEstimator, WorkingMemory
from artificium.prompts import PromptPack
from artificium.records import Console, Records
from artificium.tool_attention import AttentionToolsMixin
from artificium.tool_files import FileToolsMixin
from artificium.tool_interactions import InteractionToolsMixin
from artificium.tool_loader import load_mind_tool
from artificium.tool_memory import MemoryToolsMixin
from artificium.tools import ToolExecution, ToolRegistry, SleepRequest
from artificium.vision import VisualContext


ROOT = Path(__file__).resolve().parents[1]
PROMPTS_SRC = ROOT / "prompts"
SEED_SRC = ROOT / "seed"


def _install(root: Path) -> Paths:
    paths = Paths(root)
    shutil.copytree(PROMPTS_SRC, paths.prompts)
    shutil.copytree(SEED_SRC, paths.seed)
    paths.ensure_layout()
    return paths


def _tool_registry(paths: Paths, **config_overrides) -> ToolRegistry:
    config = Config(
        provider="custom",
        model="test-model",
        base_url="http://example.invalid/v1",
        context_window_tokens=20_000,
        **config_overrides,
    )
    records = Records(paths)
    notifications = NotificationStore(paths, records)
    interactions = InteractionStore(paths, notifications, records)
    estimator = TokenEstimator(config.chars_per_token)
    working = WorkingMemory(paths, config, estimator, records)
    visual = VisualContext(paths, config, records)
    memory = LongTermMemory(paths, records)
    attention = InfiniteAttention(paths, config, estimator, records)
    initialization = Initialization(paths, records)
    scheduler_type = load_mind_tool(paths, "scheduler.py", "Scheduler")
    scheduler = scheduler_type(paths, interactions, records)
    return ToolRegistry(
        paths=paths, config=config, prompts=PromptPack(paths), records=records,
        console=Console(quiet=True), interactions=interactions, memory=memory,
        working=working, streams=attention, visual=visual,
        initialization=initialization, scheduler=scheduler,
    ), records


PY = shlex.quote(sys.executable)


class ToolsBaseCase(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.install = Path(self.temporary.name) / "install"
        self.paths = _install(self.install)

    def _tools(self, **config_overrides):
        tools, records = _tool_registry(self.paths, **config_overrides)
        return tools, records


class ShellTimeoutKillsProcessGroupCase(ToolsBaseCase):
    def test_timeout_kills_a_background_grandchild_quickly(self) -> None:
        tools, _ = self._tools()
        pidfile = self.paths.root / "bg.pid"
        # One job runs in the background (the "grandchild" this test checks
        # is really gone), one runs in the foreground so the shell itself
        # blocks past the timeout.
        command = f"sleep 30 & echo $! > {shlex.quote(str(pidfile))}; sleep 30"
        started = time.monotonic()
        with self.assertRaises(subprocess.TimeoutExpired):
            tools.run_shell(command, timeout_seconds=1)
        elapsed = time.monotonic() - started
        self.assertLess(elapsed, 5)
        self.assertTrue(pidfile.is_file())
        background_pid = int(pidfile.read_text().strip())
        # A killed orphan can linger briefly as a zombie until init reaps it.
        deadline = time.monotonic() + 3
        while time.monotonic() < deadline:
            try:
                state = Path(f"/proc/{background_pid}/stat").read_text().rsplit(")", 1)[1].split()[0]
            except (FileNotFoundError, ProcessLookupError):
                break
            if state == "Z":
                break
            time.sleep(0.05)
        else:
            self.fail(f"background process {background_pid} survived the timeout")

    def test_execute_still_surfaces_shell_timeout_guidance(self) -> None:
        # execute() must keep catching subprocess.TimeoutExpired the same
        # way it always has (the shell_timeout + tool_execution_error
        # notifications), now with partial output attached when available.
        tools, records = self._tools()
        command = f"{PY} -c \"import sys,time; sys.stdout.write('partial'); sys.stdout.flush(); time.sleep(10)\""
        intent = ToolIntent(id="evt_1", name="run_shell",
                             arguments={"command": command, "timeout_seconds": 0.3})
        execution = tools.execute(intent)
        self.assertEqual(execution.result["status"], "error")
        self.assertIn("TimeoutExpired", execution.result["summary"])
        self.assertIn("SHELL TIMEOUT", execution.result["guidance"])
        self.assertTrue(
            any("SHELL TIMEOUT" in n or "TOOL EXECUTION ERROR" in n for n in execution.notifications)
        )
        # Partial stdout produced before the kill should have made it through.
        if "stdout" in execution.result:
            self.assertIsInstance(execution.result["stdout"], str)


class ShellStdinAndBinaryCase(ToolsBaseCase):
    def test_stdin_is_closed_so_a_blocking_read_returns_immediately(self) -> None:
        tools, _ = self._tools()
        started = time.monotonic()
        result = tools.run_shell('read x; echo "got:$x"')
        elapsed = time.monotonic() - started
        self.assertLess(elapsed, 5)
        self.assertEqual(result["status"], "ok")
        self.assertEqual(result["stdout"].strip(), "got:")

    def test_binary_output_does_not_raise(self) -> None:
        tools, _ = self._tools()
        command = (
            f"{PY} -c \"import sys; "
            "sys.stdout.buffer.write(bytes([0xff, 0xfe, 0xfd, 0x00, 65, 66]))\""
        )
        result = tools.run_shell(command)
        self.assertEqual(result["status"], "ok")
        self.assertIsInstance(result["stdout"], str)
        self.assertIn("AB", result["stdout"])

    def test_command_runs_in_its_own_session(self) -> None:
        tools, _ = self._tools()
        own_pgid = os.getpgrp()
        result = tools.run_shell(f"{PY} -c \"import os; print(os.getpgrp())\"")
        child_pgid = int(result["stdout"].strip())
        self.assertNotEqual(child_pgid, own_pgid)


class HeadTailPreviewCase(ToolsBaseCase):
    def test_bound_result_splits_stdout_and_stderr_into_head_and_tail(self) -> None:
        tools, _ = self._tools(max_tool_output_chars=3_000)
        stdout_text = "A" * 20_000
        stderr_text = "B" * 10_000
        intent = ToolIntent(id="evt_stdout", name="run_shell", arguments={"command": "true"})
        result = {
            "status": "ok",
            "summary": "shell exited 0",
            "command": "true",
            "cwd": str(self.paths.root),
            "returncode": 0,
            "stdout": stdout_text,
            "stderr": stderr_text,
        }
        bounded = tools._bound_result(intent, result)
        self.assertEqual(bounded["status"], "output_saved")
        self.assertEqual(bounded["returncode"], 0)
        self.assertTrue(Path(bounded["full_output_path"]).is_file())
        saved = read_json(Path(bounded["full_output_path"]), None)
        self.assertEqual(saved["stdout"], stdout_text)
        self.assertEqual(saved["stderr"], stderr_text)
        self.assertIn("stdout_head", bounded)
        self.assertIn("stdout_tail", bounded)
        self.assertIn("stderr_head", bounded)
        self.assertIn("stderr_tail", bounded)
        self.assertTrue(stdout_text.startswith(bounded["stdout_head"]))
        self.assertTrue(stdout_text.endswith(bounded["stdout_tail"]))
        self.assertTrue(stderr_text.startswith(bounded["stderr_head"]))
        self.assertTrue(stderr_text.endswith(bounded["stderr_tail"]))
        # The whole bounded envelope must stay well under the configured
        # limit even though the original stdout/stderr were nowhere close.
        self.assertLess(len(json_dumps(bounded, pretty=True)), tools.config.max_tool_output_chars)

    def test_bound_result_falls_back_to_output_head_tail_without_stdout(self) -> None:
        tools, _ = self._tools(max_tool_output_chars=2_000)
        intent = ToolIntent(id="evt_list", name="list_directory", arguments={"path": "mind"})
        big_entries = [{"path": f"mind/file-{i}.txt", "type": "file", "size_bytes": i} for i in range(2_000)]
        result = {"status": "ok", "summary": "listed many entries", "entries": big_entries}
        bounded = tools._bound_result(intent, result)
        self.assertEqual(bounded["status"], "output_saved")
        self.assertNotIn("stdout_head", bounded)
        self.assertIn("output_head", bounded)
        self.assertIn("output_tail", bounded)
        self.assertLess(len(json_dumps(bounded, pretty=True)), tools.config.max_tool_output_chars)

    def test_end_to_end_large_shell_output_is_bounded_with_previews(self) -> None:
        tools, _ = self._tools(max_tool_output_chars=4_000)
        command = f"{PY} -c \"import sys; sys.stdout.write('Z' * 50000)\""
        intent = ToolIntent(id="evt_big", name="run_shell", arguments={"command": command})
        execution = tools.execute(intent)
        result = execution.result
        self.assertEqual(result["status"], "output_saved")
        self.assertIn("stdout_head", result)
        self.assertIn("stdout_tail", result)
        self.assertLess(len(json_dumps(result, pretty=True)), tools.config.max_tool_output_chars)
        full = Path(result["full_output_path"]).read_text()
        self.assertIn("Z" * 100, full)


class ToolExecutionDurationCase(ToolsBaseCase):
    def test_duration_seconds_is_recorded_on_emit_and_life_records(self) -> None:
        tools, records = self._tools()
        intent = ToolIntent(id="evt_dir", name="list_directory", arguments={"path": "mind"})
        execution = tools.execute(intent)
        self.assertIsInstance(execution, ToolExecution)
        operational = records.recent_operational(50)
        executed = [r for r in operational if r.get("kind") == "tool_executed" and r.get("tool_id") == "evt_dir"]
        self.assertEqual(len(executed), 1)
        self.assertIn("duration_seconds", executed[0])
        self.assertIsInstance(executed[0]["duration_seconds"], float)
        self.assertGreaterEqual(executed[0]["duration_seconds"], 0.0)

        life = records.recent_life(50)
        results = [r for r in life if r.get("kind") == "tool_result" and r.get("tool_id") == "evt_dir"]
        self.assertEqual(len(results), 1)
        self.assertIn("duration_seconds", results[0])
        self.assertIsInstance(results[0]["duration_seconds"], float)

    def test_duration_is_rounded_to_three_decimals(self) -> None:
        tools, records = self._tools()
        intent = ToolIntent(id="evt_dir2", name="list_directory", arguments={"path": "mind"})
        tools.execute(intent)
        executed = [
            r for r in records.recent_operational(50)
            if r.get("kind") == "tool_executed" and r.get("tool_id") == "evt_dir2"
        ][0]
        value = executed["duration_seconds"]
        self.assertEqual(round(value, 3), value)


class ControlAccessorCase(ToolsBaseCase):
    def test_public_and_private_control_accessors_are_equivalent(self) -> None:
        tools, _ = self._tools()
        self.assertEqual(tools.control(), tools._control())
        tools.save_control({"working_memory_offload_pending": {"proposed_path": "x"}})
        self.assertEqual(tools.control(), {"working_memory_offload_pending": {"proposed_path": "x"}})
        self.assertEqual(tools._control(), tools.control())
        tools._save_control({"self_revision_pending": {"reason": "y"}})
        self.assertEqual(tools.control(), {"self_revision_pending": {"reason": "y"}})
        self.assertEqual(tools.save_control, tools._save_control)
        self.assertEqual(tools.control, tools._control)


class SplitModulesCase(ToolsBaseCase):
    def test_tool_registry_inherits_the_domain_mixins(self) -> None:
        self.assertTrue(issubclass(ToolRegistry, FileToolsMixin))
        self.assertTrue(issubclass(ToolRegistry, MemoryToolsMixin))
        self.assertTrue(issubclass(ToolRegistry, InteractionToolsMixin))
        self.assertTrue(issubclass(ToolRegistry, AttentionToolsMixin))

    def test_mixins_do_not_import_the_tools_module(self) -> None:
        import artificium.tool_attention as attention_mod
        import artificium.tool_files as files_mod
        import artificium.tool_interactions as interactions_mod
        import artificium.tool_memory as memory_mod

        for module in (attention_mod, files_mod, interactions_mod, memory_mod):
            self.assertNotIn("tools", getattr(module, "__dict__", {}))
            source = Path(module.__file__).read_text()
            self.assertNotIn("from .tools import", source)
            self.assertNotIn("import artificium.tools", source)

    def test_every_tool_name_still_resolves_to_a_bound_method(self) -> None:
        tools, _ = self._tools()
        for name in tools._functions:
            self.assertTrue(callable(getattr(tools, name)))

    def test_legacy_public_imports_from_tools_module_still_work(self) -> None:
        from artificium.tools import ToolRegistry as ImportedRegistry
        from artificium.tools import ToolExecution as ImportedExecution
        from artificium.tools import SleepRequest as ImportedSleep

        self.assertIs(ImportedRegistry, ToolRegistry)
        request = ImportedSleep(mode="timed", seconds=5.0)
        self.assertEqual((request.mode, request.seconds), ("timed", 5.0))
        execution = ImportedExecution(
            name="sleep", tool_id="t1", result={"status": "ok"},
            started_at="a", finished_at="b",
        )
        self.assertEqual(execution.notifications, [])


if __name__ == "__main__":
    unittest.main()
