"""Coverage for the write-mode/clobber-protection safety work:

A. `write_file` and `save_memory` accept the same `create`/`overwrite`/
   `append` modes, and every mode error names the existing file's size and
   says exactly which mode to use instead.
B. Overwriting an existing file is backed up (rotated, outside `mind/`) and
   refused if it would shrink a file bigger than 4KB to under half its size,
   unless `allow_shrink=true` is passed; the refusal names a partial prior
   `read_file` of that path when there was one.
C. A not-found read names `mind/...` as a hint when the path was given
   without that prefix (or vice versa); writes never auto-resolve.

This exercises the tool layer directly (no engine/model involved), the same
way tests/test_tools_shell_and_split.py does, so a fresh workspace is built
per test in a temporary directory.
"""
from __future__ import annotations

import shutil
import tempfile
import unittest
from pathlib import Path

from artificium.config import Config
from artificium.filesystem import Paths
from artificium.initialization import Initialization
from artificium.interactions import InteractionStore, NotificationStore
from artificium.memory import InfiniteAttention, LongTermMemory, TokenEstimator, WorkingMemory
from artificium.prompts import PromptPack
from artificium.records import Console, Records
from artificium.tool_loader import load_mind_tool
from artificium.tools import ToolRegistry
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
    )


class FileSafetyBaseCase(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.install = Path(self.temporary.name) / "install"
        self.paths = _install(self.install)
        self.tools = _tool_registry(self.paths)


# --- A: consistent modes with actionable errors -----------------------------


class WriteFileModeCase(FileSafetyBaseCase):
    def test_create_on_existing_file_names_size_and_next_mode(self) -> None:
        target = self.paths.space / "note.txt"
        self.tools.write_file(str(target), "hello world", mode="create")
        with self.assertRaises(FileExistsError) as caught:
            self.tools.write_file(str(target), "new", mode="create")
        message = str(caught.exception)
        self.assertIn(str(len("hello world")), message)
        self.assertIn("overwrite", message)
        self.assertIn("append", message)
        self.assertIn("Read it fully first", message)

    def test_invalid_mode_lists_all_three_modes(self) -> None:
        with self.assertRaisesRegex(ValueError, "create, overwrite, or append"):
            self.tools.write_file("mind/space/x.txt", "a", mode="bogus")

    def test_append_and_overwrite_still_work(self) -> None:
        target = self.paths.space / "log.txt"
        self.tools.write_file(str(target), "first", mode="create")
        self.tools.write_file(str(target), " second", mode="append")
        self.assertEqual(target.read_text(), "first second")
        self.tools.write_file(str(target), "replaced", mode="overwrite")
        self.assertEqual(target.read_text(), "replaced")


class SaveMemoryModeCase(FileSafetyBaseCase):
    def test_create_mode_is_now_accepted(self) -> None:
        result = self.tools.save_memory(
            path="projects/widget/plan",
            content="Plan content.",
            retrieve_when="Retrieve while planning the widget project.",
            mode="create",
        )
        self.assertEqual(result["status"], "remembered")

    def test_create_on_existing_memory_names_size_and_next_mode(self) -> None:
        self.tools.save_memory(
            path="projects/widget/plan",
            content="Plan content.",
            retrieve_when="Retrieve while planning the widget project.",
            mode="create",
        )
        with self.assertRaises(FileExistsError) as caught:
            self.tools.save_memory(
                path="projects/widget/plan",
                content="New plan.",
                retrieve_when="Retrieve while planning the widget project.",
                mode="create",
            )
        message = str(caught.exception)
        self.assertIn("overwrite", message)
        self.assertIn("append", message)
        self.assertIn("bytes", message)

    def test_invalid_mode_lists_all_three_modes(self) -> None:
        with self.assertRaisesRegex(ValueError, "create, overwrite, or append"):
            self.tools.save_memory(
                path="projects/widget/plan",
                content="x",
                retrieve_when="Retrieve while planning the widget project.",
                mode="bogus",
            )

    def test_default_overwrite_mode_unchanged_for_a_new_path(self) -> None:
        result = self.tools.save_memory(
            path="projects/widget/fresh-note",
            content="Fresh content.",
            retrieve_when="Retrieve while planning the widget project.",
        )
        self.assertEqual(result["status"], "remembered")


# --- B: backup-before-overwrite and the shrink guard -------------------------


class BackupBeforeOverwriteCase(FileSafetyBaseCase):
    def test_write_file_overwrite_backs_up_prior_content_outside_mind(self) -> None:
        target = self.paths.mind / "space" / "big.txt"
        self.tools.write_file(str(target), "x" * 5000, mode="create")
        result = self.tools.write_file(str(target), "y" * 4900, mode="overwrite")
        self.assertIn("backup_path", result)
        backup = Path(result["backup_path"])
        self.assertTrue(backup.is_file())
        self.assertEqual(backup.read_text(), "x" * 5000)
        # The backup must live outside mind/, so reorganizing or deleting
        # memory cannot also delete the safety net.
        self.assertNotIn(str(self.paths.mind), str(backup))
        self.assertIn(str(self.paths.logs), str(backup))

    def test_save_memory_overwrite_backs_up_prior_content(self) -> None:
        self.tools.save_memory(
            path="projects/widget/plan",
            content="x" * 5000,
            retrieve_when="Retrieve while planning the widget project.",
        )
        result = self.tools.save_memory(
            path="projects/widget/plan",
            content="y" * 4900,
            retrieve_when="Retrieve while planning the widget project.",
        )
        self.assertIn("backup_path", result)
        backup = Path(result["backup_path"])
        self.assertTrue(backup.is_file())
        self.assertEqual(backup.read_text(), "x" * 5000 + "\n")

    def test_versions_are_rotated_and_bounded_in_count(self) -> None:
        target = self.paths.space / "rotating.txt"
        self.tools.write_file(str(target), "seed", mode="create")
        backups: list[Path] = []
        for i in range(25):
            result = self.tools.write_file(str(target), f"content-{i}", mode="overwrite")
            backups.append(Path(result["backup_path"]))
        directory = backups[-1].parent
        remaining = sorted(directory.glob("*.bak"))
        self.assertLessEqual(len(remaining), 20)
        # The most recent backups must have survived rotation.
        self.assertIn(backups[-1], remaining)
        self.assertNotIn(backups[0], remaining)

    def test_no_backup_when_file_does_not_yet_exist(self) -> None:
        target = self.paths.space / "brand-new.txt"
        result = self.tools.write_file(str(target), "content", mode="create")
        self.assertNotIn("backup_path", result)


class ShrinkGuardCase(FileSafetyBaseCase):
    def test_large_shrink_is_refused_by_default(self) -> None:
        target = self.paths.space / "project-state.txt"
        self.tools.write_file(str(target), "A" * 41_800, mode="create")
        with self.assertRaises(ValueError) as caught:
            self.tools.write_file(str(target), "B" * 8_500, mode="overwrite")
        message = str(caught.exception)
        self.assertIn("41800", message)
        self.assertIn("allow_shrink", message)
        self.assertEqual(target.read_text(), "A" * 41_800)

    def test_allow_shrink_true_permits_the_overwrite(self) -> None:
        target = self.paths.space / "project-state.txt"
        self.tools.write_file(str(target), "A" * 41_800, mode="create")
        result = self.tools.write_file(
            str(target), "B" * 8_500, mode="overwrite", allow_shrink=True
        )
        self.assertEqual(result["status"], "written")
        self.assertEqual(target.read_text(), "B" * 8_500)

    def test_small_shrink_is_not_blocked(self) -> None:
        target = self.paths.space / "medium.txt"
        self.tools.write_file(str(target), "A" * 5000, mode="create")
        result = self.tools.write_file(str(target), "B" * 3000, mode="overwrite")
        self.assertEqual(result["status"], "written")

    def test_small_existing_file_is_never_blocked(self) -> None:
        target = self.paths.space / "tiny.txt"
        self.tools.write_file(str(target), "A" * 100, mode="create")
        result = self.tools.write_file(str(target), "B", mode="overwrite")
        self.assertEqual(result["status"], "written")

    def test_partial_read_is_named_in_the_refusal(self) -> None:
        target = self.paths.mind / "memory" / "projects" / "riemann" / "state.txt"
        target.parent.mkdir(parents=True, exist_ok=True)
        self.tools.write_file(str(target), "A" * 41_800, mode="create")
        read = self.tools.read_file(str(target), start=44, max_characters=1000)
        self.assertFalse(read["complete"])
        with self.assertRaises(ValueError) as caught:
            self.tools.write_file(str(target), "B" * 8_500, mode="overwrite")
        message = str(caught.exception)
        self.assertIn("partial", message)

    def test_full_read_clears_the_partial_marker(self) -> None:
        target = self.paths.space / "state.txt"
        self.tools.write_file(str(target), "small content", mode="create")
        read = self.tools.read_file(str(target))
        self.assertTrue(read["complete"])
        # Grow the file past the shrink-guard floor, then confirm a full
        # subsequent read is not flagged as partial in a later refusal.
        self.tools.write_file(str(target), "A" * 41_800, mode="overwrite")
        full = self.tools.read_file(str(target))
        self.assertTrue(full["complete"])
        with self.assertRaises(ValueError) as caught:
            self.tools.write_file(str(target), "B" * 8_500, mode="overwrite")
        self.assertNotIn("partial", str(caught.exception))

    def test_save_memory_shrink_guard_matches_write_file(self) -> None:
        self.tools.save_memory(
            path="projects/widget/plan",
            content="A" * 41_800,
            retrieve_when="Retrieve while planning the widget project.",
        )
        with self.assertRaises(ValueError) as caught:
            self.tools.save_memory(
                path="projects/widget/plan",
                content="B" * 8_500,
                retrieve_when="Retrieve while planning the widget project.",
            )
        self.assertIn("allow_shrink", str(caught.exception))
        result = self.tools.save_memory(
            path="projects/widget/plan",
            content="B" * 8_500,
            retrieve_when="Retrieve while planning the widget project.",
            allow_shrink=True,
        )
        self.assertEqual(result["status"], "remembered")

    def test_self_and_meta_memory_are_exempt_from_the_shrink_guard(self) -> None:
        # self.txt and mind/meta_memory.md are agent-authored maps/identity
        # files expected to be rewritten (including shrinking) as ordinary
        # upkeep; only their shrink guard is exempt, not their backup.
        meta_before_size = self.paths.meta_memory.stat().st_size
        self.assertGreater(meta_before_size, 4096)
        result = self.tools.write_file(
            str(self.paths.meta_memory), "Updated memory routes", mode="overwrite"
        )
        self.assertEqual(result["status"], "written")
        self.assertIn("backup_path", result)
        self.assertEqual(self.paths.meta_memory.read_text(), "Updated memory routes")


# --- C: "did you mean mind/...?" hints on not-found reads --------------------


class PathRootHintCase(FileSafetyBaseCase):
    def test_read_file_missing_mind_prefix_gets_a_hint(self) -> None:
        target = self.paths.mind / "memory" / "projects" / "riemann-hypothesis" / "notes.txt"
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text("notes")
        with self.assertRaises(FileNotFoundError) as caught:
            self.tools.read_file("memory/projects/riemann-hypothesis/notes.txt")
        message = str(caught.exception)
        self.assertIn("Did you mean", message)
        self.assertIn("mind/memory/projects/riemann-hypothesis/notes.txt", message)

    def test_list_directory_missing_mind_prefix_gets_a_hint(self) -> None:
        (self.paths.mind / "memory" / "projects").mkdir(parents=True, exist_ok=True)
        with self.assertRaises(NotADirectoryError) as caught:
            self.tools.list_directory("memory/projects")
        self.assertIn("Did you mean mind/memory/projects", str(caught.exception))

    def test_no_hint_when_no_alternate_path_exists(self) -> None:
        with self.assertRaises(FileNotFoundError) as caught:
            self.tools.read_file("mind/memory/does/not/exist.txt")
        self.assertNotIn("Did you mean", str(caught.exception))

    def test_write_file_never_auto_resolves_a_missing_mind_prefix(self) -> None:
        # Writes must not silently "fix" the destination: the file lands
        # exactly where the (possibly wrong) path says, without a hint.
        result = self.tools.write_file(
            "memory/projects/example/plan.txt", "content", mode="create"
        )
        written = Path(result["path"])
        self.assertEqual(written, (self.paths.root / "memory/projects/example/plan.txt").resolve())
        self.assertFalse((self.paths.mind / "memory/projects/example/plan.txt").exists())


if __name__ == "__main__":
    unittest.main()
