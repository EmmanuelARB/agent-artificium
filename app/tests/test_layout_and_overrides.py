"""Coverage for the install/app/workspace restructuring: anchors, the seed/
gap-fill mechanism the reset story relies on, the prompt and code overlays,
the file tools' read-only application boundary, and the CLI surface for all
of it.
"""
from __future__ import annotations

import contextlib
import io
import json
import os
import shutil
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from artificium import _bootstrap
from artificium.cli import main
from artificium.config import Config, ConfigStore, SecretsStore, configured_paths
from artificium.filesystem import Paths
from artificium.initialization import Initialization
from artificium.interactions import InteractionStore, NotificationStore
from artificium.memory import InfiniteAttention, LongTermMemory, TokenEstimator, WorkingMemory
from artificium.prompts import PromptPack
from artificium.records import Console, Records
from artificium.tool_loader import load_mind_tool
from artificium.tools import ToolRegistry
from artificium.vision import VisualContext


PROMPTS_SRC = Path(__file__).resolve().parents[1] / "prompts"
SEED_SRC = Path(__file__).resolve().parents[1] / "seed"


def _install(root: Path) -> Paths:
    """Build a fresh, fully seeded fake installation at ``root``."""

    paths = Paths(root)
    shutil.copytree(PROMPTS_SRC, paths.prompts)
    shutil.copytree(SEED_SRC, paths.seed)
    paths.ensure_layout()
    return paths


def _tool_registry(paths: Paths) -> ToolRegistry:
    config = Config(provider="custom", model="test-model", base_url="http://example.invalid/v1",
                     context_window_tokens=20_000)
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


# --- 1. Anchors -------------------------------------------------------------

class AnchorsCase(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.install = Path(self.temp.name) / "install"

    def test_config_and_secrets_sit_at_the_install_root_not_in_the_code_tree(self) -> None:
        paths = Paths(self.install)
        self.assertEqual(paths.config, self.install / "config.json")
        self.assertEqual(paths.secrets, self.install / ".secrets.json")
        self.assertEqual(paths.config.parent, self.install)
        self.assertEqual(paths.secrets.parent, self.install)

    def test_root_app_code_prompts_seed_and_overrides_are_positioned_correctly(self) -> None:
        paths = Paths(self.install)
        self.assertEqual(paths.app, self.install / "app")
        self.assertEqual(paths.root, self.install / "workspace")
        self.assertEqual(paths.code, paths.app)
        self.assertEqual(paths.launcher, self.install / "artificium.py")
        self.assertEqual(paths.prompts, paths.app / "prompts")
        self.assertEqual(paths.seed, paths.app / "seed")
        self.assertEqual(paths.overrides, paths.root / "overrides")
        self.assertEqual(paths.prompt_overrides, paths.overrides / "prompts")
        self.assertEqual(paths.code_overrides, paths.overrides / "code")
        self.assertEqual(paths.quarantined_overrides, paths.root / "overrides.quarantined")
        self.assertEqual(paths.mind, paths.root / "mind")
        self.assertEqual(paths.logs, paths.root / "logs")

    def test_explicit_app_dir_is_used_without_disturbing_root_or_settings(self) -> None:
        elsewhere = Path(self.temp.name) / "elsewhere-app"
        paths = Paths(self.install, elsewhere)
        self.assertEqual(paths.app, elsewhere)
        self.assertEqual(paths.code, elsewhere)
        self.assertEqual(paths.root, self.install / "workspace")
        self.assertEqual(paths.config, self.install / "config.json")


# --- 2. from_code_file / for_app --------------------------------------------

class FromCodeFileAndForAppCase(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)

    def test_a_clone_of_any_name_is_its_own_root(self) -> None:
        for name in ("agent-artificium", "my-agent", "app"):
            with self.subTest(name=name):
                clone = Path(self.temp.name) / name
                paths = Paths.from_code_file(clone / "app" / "artificium" / "config.py")
                self.assertEqual(paths.install, clone.resolve())
                self.assertEqual(paths.app, (clone / "app").resolve())
                self.assertEqual(paths.root, (clone / "workspace").resolve())

    def test_for_app_addresses_another_application_tree(self) -> None:
        tree = Path(self.temp.name) / "some-clone" / "app"
        paths = Paths.for_app(tree)
        self.assertEqual(paths.app, tree.resolve())
        self.assertEqual(paths.install, tree.parent.resolve())
        self.assertEqual(paths.code, tree.resolve())


# --- 3. configured_paths precedence -----------------------------------------

class ConfiguredPathsPrecedenceCase(unittest.TestCase):
    def test_explicit_argument_beats_environment_beats_derived_location(self) -> None:
        derived = Paths(Path("/derived/install"))
        with mock.patch("artificium.config.Paths.from_code_file", return_value=derived) as from_code:
            with mock.patch.dict(os.environ, {}, clear=True):
                self.assertIs(configured_paths(None), derived)
            from_code.assert_called_once()

            with mock.patch.dict(os.environ, {"ARTIFICIUM_ROOT": "/env/install"}):
                by_env = configured_paths(None)
                self.assertEqual(by_env.install, Path("/env/install").expanduser().resolve())

                explicit = configured_paths("/explicit/install")
                self.assertEqual(explicit.install, Path("/explicit/install").expanduser().resolve())


# --- 4. fill_from_seed --------------------------------------------------------

class FillFromSeedCase(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.paths = Paths(Path(self.temp.name) / "install")

    def test_copies_every_missing_seed_file_and_returns_their_relative_paths(self) -> None:
        shutil.copytree(SEED_SRC, self.paths.seed)
        created = self.paths.fill_from_seed()
        expected = sorted(
            p.relative_to(SEED_SRC).as_posix() for p in SEED_SRC.rglob("*")
            if p.is_file() and "__pycache__" not in p.parts and p.suffix != ".pyc"
        )
        self.assertEqual(sorted(created), expected)
        self.assertEqual((self.paths.mind / "self.txt").read_text(), (SEED_SRC / "self.txt").read_text())

    def test_second_call_is_idempotent_and_never_overwrites_a_learned_file(self) -> None:
        shutil.copytree(SEED_SRC, self.paths.seed)
        self.paths.fill_from_seed()
        (self.paths.mind / "self.txt").write_text("Learned and changed through experience.\n")
        again = self.paths.fill_from_seed()
        self.assertEqual(again, [])
        self.assertEqual((self.paths.mind / "self.txt").read_text(), "Learned and changed through experience.\n")

    def test_refreshes_an_unedited_delivered_file_but_not_an_edited_one(self) -> None:
        shutil.copytree(SEED_SRC, self.paths.seed)
        self.paths.fill_from_seed()
        tool = self.paths.seed / "tools" / "search.py"
        tool.write_text(tool.read_text() + "\n# newer release\n")
        self.assertEqual(self.paths.fill_from_seed(), ["tools/search.py"])
        self.assertTrue((self.paths.mind / "tools" / "search.py").read_text().endswith("# newer release\n"))
        (self.paths.mind / "tools" / "search.py").write_text("# the agent's own version\n")
        tool.write_text(tool.read_text() + "# even newer\n")
        self.assertEqual(self.paths.fill_from_seed(), [])
        self.assertEqual((self.paths.mind / "tools" / "search.py").read_text(), "# the agent's own version\n")

    def test_legacy_name_only_record_refreshes_a_known_earlier_release(self) -> None:
        from artificium.filesystem import _EARLIER_SEED_RELEASES, atomic_write_json, sha256_file
        shutil.copytree(SEED_SRC, self.paths.seed)
        self.paths.fill_from_seed()
        names = sorted(p.relative_to(self.paths.seed).as_posix() for p in self.paths.seed.rglob("*") if p.is_file())
        atomic_write_json(self.paths.seed_deliveries, names)
        target = self.paths.mind / "tools" / "search.py"
        target.write_text("old shipped search\n")
        with mock.patch.dict(_EARLIER_SEED_RELEASES, {"tools/search.py": (sha256_file(target),)}):
            self.assertEqual(self.paths.fill_from_seed(), ["tools/search.py"])
        self.assertEqual(target.read_text(), (self.paths.seed / "tools" / "search.py").read_text())

    def test_never_delivers_bytecode_caches(self) -> None:
        shutil.copytree(SEED_SRC, self.paths.seed)
        cache = self.paths.seed / "tools" / "__pycache__"
        cache.mkdir(exist_ok=True)
        (cache / "search.cpython-314.pyc").write_bytes(b"\0")
        created = self.paths.fill_from_seed()
        self.assertFalse(any("__pycache__" in name for name in created))
        self.assertFalse((self.paths.mind / "tools" / "__pycache__").exists())

    def test_silent_no_op_without_a_seed_directory(self) -> None:
        created = self.paths.fill_from_seed()
        self.assertEqual(created, [])
        self.assertFalse(self.paths.mind.exists())


# --- 5. The reset story, end to end -----------------------------------------

class ResetStoryCase(unittest.TestCase):
    def test_deleting_the_workspace_and_reseeding_restores_a_factory_mind(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            paths = Paths(Path(temp) / "install")
            shutil.copytree(SEED_SRC, paths.seed)
            paths.ensure_layout()

            original_self = paths.self_file.read_text()
            # The agent lives here: it revises its Self and invents a memory.
            paths.self_file.write_text("Mutated through lived experience.\n")
            invented = paths.memory / "invented/private-note.txt"
            invented.parent.mkdir(parents=True, exist_ok=True)
            invented.write_text("A private invented memory that was never shipped.\n")
            # The operator's settings sit outside the workspace entirely.
            paths.config.write_text('{"model": {"model": "test-model"}}\n')
            paths.secrets.write_text('{"api_key": "very-secret"}\n')

            shutil.rmtree(paths.root)
            paths.ensure_layout()

            self.assertEqual(paths.self_file.read_text(), original_self)
            self.assertFalse(invented.exists())
            self.assertEqual(
                (paths.memory / "harness/infinite-attention.txt").read_text(),
                (SEED_SRC / "memory/harness/infinite-attention.txt").read_text(),
            )
            self.assertEqual(paths.config.read_text(), '{"model": {"model": "test-model"}}\n')
            self.assertEqual(paths.secrets.read_text(), '{"api_key": "very-secret"}\n')


# --- 6. Prompt overlay --------------------------------------------------------

class PromptOverlayCase(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.paths = _install(Path(self.temp.name) / "install")

    def test_no_overlay_reports_nothing_overridden(self) -> None:
        pack = PromptPack(self.paths)
        self.assertEqual(pack.overridden(), [])

    def test_overlay_file_shadows_the_shipped_one_and_is_reported(self) -> None:
        override = self.paths.prompt_overrides / "always/core.md"
        override.parent.mkdir(parents=True, exist_ok=True)
        override.write_text("OVERRIDDEN CORE CONTENT\n")
        pack = PromptPack(self.paths)
        self.assertEqual(pack.overridden(), ["always/core.md"])
        self.assertIn("OVERRIDDEN CORE CONTENT", pack.always())

    def test_manifest_relative_path_cannot_escape_either_tree(self) -> None:
        pack = PromptPack(self.paths)
        with self.assertRaises(ValueError):
            pack._path("../outside.md")


# --- 7. Code overlay (_bootstrap) --------------------------------------------

class CodeOverlayCase(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.install = Path(self.temp.name) / "install"
        self.code_file = self.install / "app" / "artificium" / "__init__.py"
        self.workspace = self.install / "workspace"
        self.overlay = self.workspace / "overrides" / "code"
        self.attempts = self.workspace / "overrides" / ".attempts.json"

    def test_returns_none_when_the_overlay_is_absent_or_empty(self) -> None:
        self.assertIsNone(_bootstrap.code_overlay(self.code_file, argv=[]))
        self.overlay.mkdir(parents=True)
        self.assertIsNone(_bootstrap.code_overlay(self.code_file, argv=[]))

    def test_returns_the_overlay_directory_for_valid_python(self) -> None:
        self.overlay.mkdir(parents=True)
        (self.overlay / "runtime.py").write_text("VALUE = 1\n")
        result = _bootstrap.code_overlay(self.code_file, argv=[])
        self.assertEqual(result, self.overlay)

    def test_returns_none_when_any_file_fails_to_parse(self) -> None:
        self.overlay.mkdir(parents=True)
        (self.overlay / "runtime.py").write_text("VALUE = 1\n")
        (self.overlay / "broken.py").write_text("def broken(\n")
        self.assertIsNone(_bootstrap.code_overlay(self.code_file, argv=[]))

    def test_no_overrides_flag_and_environment_variable_disable_it(self) -> None:
        self.overlay.mkdir(parents=True)
        (self.overlay / "runtime.py").write_text("VALUE = 1\n")
        self.assertIsNone(_bootstrap.code_overlay(self.code_file, argv=["--no-overrides"]))
        with mock.patch.dict(os.environ, {"ARTIFICIUM_NO_OVERRIDES": "1"}):
            self.assertIsNone(_bootstrap.code_overlay(self.code_file, argv=[]))
        self.assertFalse(self.attempts.exists())

    def test_empty_or_absent_directory_never_raises_the_counter(self) -> None:
        self.assertIsNone(_bootstrap.code_overlay(self.code_file, argv=[]))
        self.assertFalse(self.attempts.exists())

    def test_activating_the_overlay_raises_the_failed_start_counter(self) -> None:
        self.overlay.mkdir(parents=True)
        (self.overlay / "runtime.py").write_text("VALUE = 1\n")
        _bootstrap.code_overlay(self.code_file, argv=[])
        self.assertEqual(json.loads(self.attempts.read_text())["failed_starts"], 1)
        _bootstrap.code_overlay(self.code_file, argv=[])
        self.assertEqual(json.loads(self.attempts.read_text())["failed_starts"], 2)

    def test_quarantines_the_whole_overrides_directory_after_three_activations(self) -> None:
        self.overlay.mkdir(parents=True)
        (self.overlay / "runtime.py").write_text("VALUE = 1\n")
        prompt_override = self.workspace / "overrides" / "prompts" / "always/core.md"
        prompt_override.parent.mkdir(parents=True)
        prompt_override.write_text("Custom.\n")
        for _ in range(3):
            self.assertEqual(_bootstrap.code_overlay(self.code_file, argv=[]), self.overlay)
        quarantined = self.workspace / "overrides.quarantined"
        self.assertFalse(quarantined.exists())

        result = _bootstrap.code_overlay(self.code_file, argv=[])

        self.assertIsNone(result)
        self.assertTrue(quarantined.is_dir())
        self.assertTrue((quarantined / "code" / "runtime.py").is_file())
        self.assertTrue((quarantined / "prompts" / "always/core.md").is_file())
        self.assertFalse((self.workspace / "overrides").exists())

    def test_clear_attempts_removes_the_counter(self) -> None:
        self.overlay.mkdir(parents=True)
        (self.overlay / "runtime.py").write_text("VALUE = 1\n")
        _bootstrap.code_overlay(self.code_file, argv=[])
        self.assertTrue(self.attempts.exists())
        _bootstrap.clear_attempts(self.workspace)
        self.assertFalse(self.attempts.exists())

    def test_clear_attempts_is_a_silent_no_op_without_an_overrides_directory(self) -> None:
        _bootstrap.clear_attempts(self.workspace)  # must not raise


# --- 8. The application is read-only to the file tools -----------------------

class ReadOnlyApplicationCase(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.paths = _install(Path(self.temp.name) / "install")
        self.registry = _tool_registry(self.paths)

    def test_writing_a_shipped_prompt_is_refused_and_points_at_the_overlay(self) -> None:
        target = self.paths.prompts / "always/core.md"
        with self.assertRaises(PermissionError) as caught:
            self.registry._resolve_write(str(target))
        self.assertIn("overrides/prompts", str(caught.exception))

    def test_writing_a_package_module_is_refused_and_points_at_the_overlay(self) -> None:
        target = self.paths.code / "artificium" / "runtime.py"
        with self.assertRaises(PermissionError) as caught:
            self.registry._resolve_write(str(target))
        self.assertIn("overrides/code", str(caught.exception))

    def test_the_launcher_readme_and_git_are_refused_too(self) -> None:
        for name in ("artificium.py", "README.md", ".git/config", ".gitignore"):
            with self.subTest(name=name):
                with self.assertRaisesRegex(PermissionError, "read-only outside workspace/"):
                    self.registry._resolve_write(str(self.paths.install / name))

    def test_settings_and_paths_outside_the_project_stay_writable(self) -> None:
        for target in (self.paths.config, self.paths.secrets, Path(self.temp.name).parent / "elsewhere.txt"):
            with self.subTest(target=target):
                self.assertEqual(self.registry._resolve_write(str(target)), target.resolve())

    def test_an_ordinary_workspace_path_resolves_normally(self) -> None:
        resolved = self.registry._resolve_write("mind/space/example.txt")
        self.assertEqual(resolved, (self.paths.root / "mind/space/example.txt").resolve())


# --- 9. CLI: overrides and reset ---------------------------------------------

class CLIOverridesAndResetCase(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.install = Path(self.temp.name) / "install"
        self.paths = _install(self.install)
        ConfigStore(self.paths).save(Config(provider="custom", model="test-model",
                                            base_url="http://example.invalid/v1",
                                            context_window_tokens=20_000))
        SecretsStore(self.paths).save_api_key("secret-key")
        not_running = mock.patch("artificium.cli.process_state", return_value={"alive": False})
        not_running.start()
        self.addCleanup(not_running.stop)

    def run_cli(self, *args: str):
        out, err = io.StringIO(), io.StringIO()
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
            code = main(["--root", str(self.install), *args])
        return code, out.getvalue(), err.getvalue()

    def test_overrides_list_reports_none_then_the_added_override(self) -> None:
        code, out, _ = self.run_cli("overrides")
        self.assertEqual(code, 0)
        self.assertIn("No overrides", out)

        override = self.paths.prompt_overrides / "always/core.md"
        override.parent.mkdir(parents=True, exist_ok=True)
        override.write_text("Custom core content.\n")
        code, out, _ = self.run_cli("overrides")
        self.assertEqual(code, 0)
        self.assertIn("prompts/always/core.md", out)
        self.assertIn("shadows the shipped file", out)

    def test_overrides_diff_shows_the_change(self) -> None:
        override = self.paths.prompt_overrides / "always/core.md"
        override.parent.mkdir(parents=True, exist_ok=True)
        override.write_text("Custom core content.\n")
        code, out, _ = self.run_cli("overrides", "diff")
        self.assertEqual(code, 0)
        self.assertIn("Custom core content.", out)
        self.assertIn("shipped/prompts/always/core.md", out)

    def test_overrides_clear_removes_the_file(self) -> None:
        override = self.paths.prompt_overrides / "always/core.md"
        override.parent.mkdir(parents=True, exist_ok=True)
        override.write_text("Custom core content.\n")
        code, out, _ = self.run_cli("overrides", "clear", "--yes")
        self.assertEqual(code, 0)
        self.assertFalse(override.exists())
        self.assertIn("Removed", out)

    def test_reset_deletes_the_workspace_and_grows_a_fresh_one(self) -> None:
        marker = self.paths.memory / "harness/infinite-attention.txt"
        marker.write_text("MUTATED\n")
        learned = self.paths.memory / "project/learned.txt"
        learned.parent.mkdir(parents=True, exist_ok=True)
        learned.write_text("learned\n")
        code, _, _ = self.run_cli("reset", "--yes")
        self.assertEqual(code, 0)
        self.assertNotEqual(marker.read_text(), "MUTATED\n")
        self.assertFalse(learned.exists())
        self.assertFalse((self.install / "archives").exists())
        self.assertTrue(self.paths.self_file.is_file())
        self.assertTrue(self.paths.config.is_file())
        self.assertTrue(self.paths.secrets.is_file())

    def test_reset_refuses_while_the_agent_is_running(self) -> None:
        with mock.patch("artificium.cli.process_state", return_value={"alive": True}):
            code, _, err = self.run_cli("reset", "--yes")
        self.assertEqual(code, 2)
        self.assertIn("Stop Artificium", err)


if __name__ == "__main__":
    unittest.main()
