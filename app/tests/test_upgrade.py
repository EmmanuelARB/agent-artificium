from __future__ import annotations

import contextlib
import io
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from artificium.cli import main
from artificium.filesystem import Paths
from artificium.initialization import initialize_mind
from artificium.records import Records
from artificium.upgrade import _repository_key, upgrade


ROOT = Path(__file__).resolve().parents[2]


@unittest.skipUnless(shutil.which("git"), "Git is required for upgrade integration tests")
class UpgradeCase(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.directory = Path(temporary.name)
        self.source = self.directory / "upstream"
        self.source.mkdir()
        # The upstream repository holds what the project tracks.
        shutil.copytree(ROOT / "app", self.source / "app", ignore=shutil.ignore_patterns("__pycache__"))
        for name in ("artificium.py", "README.md", ".gitignore"):
            shutil.copy2(ROOT / name, self.source / name)
        (self.source / "app/artificium/obsolete.py").write_text("OLD = True\n")
        self.git(self.source, "init", "-q", "-b", "main")
        self.commit()
        # The instance is a clone of it that has been set up and has lived.
        clone = self.directory / "instance"
        subprocess.run(["git", "clone", "-q", str(self.source), str(clone)], check=True, capture_output=True)
        self.paths = Paths(clone)
        self.paths.ensure_layout()
        self.protected = {
            self.paths.self_file: b"My learned purpose.\n",
            self.paths.meta_memory: b"My learned memory map.\n",
            self.paths.memory / "harness/tool-building-and-workspace.txt": b"My learned operating strategy.\n",
            self.paths.memory / "project/new-discovery.txt": b"A new result.\n",
            self.paths.created_tools / "custom.py": b"# agent-built tool\n",
            self.paths.space / "result.txt": b"result\n",
            self.paths.interactions / "chat/events/event.json": b"{}\n",
            self.paths.config: b'{"temperature":0.3}\n',
            self.paths.secrets: b'{"api_key":"private"}\n',
            # An untracked file the operator left in the application.
            self.paths.app / "my-contract.json": b'{"private":"contract"}\n',
        }
        for path, content in self.protected.items():
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(content)
        (self.source / "app/artificium/new_feature.py").write_text("NEW = True\n")
        (self.source / "app/artificium/obsolete.py").unlink()
        (self.source / "README.md").write_text("Updated documentation.\n")
        self.commit()

    def git(self, directory, *args):
        return subprocess.check_output(["git", "-C", str(directory), *args], stderr=subprocess.PIPE, text=True)

    def commit(self, directory=None):
        directory = directory or self.source
        self.git(directory, "add", "--all")
        self.git(directory, "-c", "user.name=Test", "-c", "user.email=test@example.invalid",
                 "commit", "-qm", "Fixture revision")

    def run_upgrade(self, **options):
        with mock.patch("artificium.upgrade.process_state", return_value={"alive": False}):
            return upgrade(self.paths, report=lambda _: None, **options)

    def assert_preserved(self):
        for path, content in self.protected.items():
            self.assertEqual(path.read_bytes(), content, path)

    def assert_not_upgraded(self):
        self.assertFalse((self.paths.app / "artificium/new_feature.py").exists())
        self.assertTrue((self.paths.app / "artificium/obsolete.py").exists())

    def test_upgrade_fast_forwards_code_and_preserves_workspace_and_settings(self):
        result = self.run_upgrade()
        self.assertEqual(result["changed_files"], 3)
        self.assertTrue((self.paths.app / "artificium/new_feature.py").is_file())
        self.assertFalse((self.paths.app / "artificium/obsolete.py").exists())
        self.assertEqual((self.paths.install / "README.md").read_text(), "Updated documentation.\n")
        self.assert_preserved()

    def test_check_is_read_only_even_while_the_agent_is_running(self):
        with mock.patch("artificium.upgrade.process_state", return_value={"alive": True}):
            result = upgrade(self.paths, check=True, report=lambda _: None)
        self.assertEqual(result["changed_files"], 3)
        self.assert_not_upgraded()
        self.assert_preserved()

    def moved_home(self):
        """A new home that carries one more commit than the original project."""
        home = self.directory / "fork"
        subprocess.run(["git", "clone", "-q", str(self.source), str(home)], check=True, capture_output=True)
        (home / "app/artificium/fork_only.py").write_text("FORK = True\n")
        self.commit(home)
        return home

    def remote_url(self):
        return self.git(self.paths.install, "remote", "get-url", "origin").strip()

    def test_a_clone_of_the_original_project_moves_to_the_new_home(self):
        home = self.moved_home()
        with mock.patch("artificium.upgrade.ORIGINAL_REPOSITORY", str(self.source) + ".git/"), \
                mock.patch("artificium.upgrade.REPOSITORY_URL", str(home)):
            result = self.run_upgrade()
        self.assertEqual(self.remote_url(), str(home))
        self.assertEqual(result["changed_files"], 4)
        self.assertTrue((self.paths.app / "artificium/fork_only.py").is_file())
        self.assert_preserved()

    def test_check_previews_the_new_home_without_moving_the_remote(self):
        home = self.moved_home()
        with mock.patch("artificium.upgrade.ORIGINAL_REPOSITORY", str(self.source)), \
                mock.patch("artificium.upgrade.REPOSITORY_URL", str(home)), \
                mock.patch("artificium.upgrade.process_state", return_value={"alive": True}):
            result = upgrade(self.paths, check=True, report=lambda _: None)
        self.assertEqual(result["changed_files"], 4)
        self.assertEqual(self.remote_url(), str(self.source))
        self.assert_not_upgraded()

    def test_a_clone_of_another_repository_is_left_where_it_is(self):
        with mock.patch("artificium.upgrade.REPOSITORY_URL", str(self.directory / "elsewhere")):
            self.run_upgrade()
        self.assertEqual(self.remote_url(), str(self.source))

    def test_repository_key_matches_every_spelling_of_one_repository(self):
        spellings = (
            "https://github.com/officialgr/agent-artificium",
            "https://github.com/officialgr/agent-artificium/",
            "https://github.com/OfficialGR/agent-artificium.git",
            "git@github.com:officialgr/agent-artificium.git",
            "ssh://git@github.com/officialgr/agent-artificium.git",
        )
        for url in spellings:
            self.assertEqual(_repository_key(url), "github.com/officialgr/agent-artificium", url)
        self.assertNotEqual(_repository_key("https://github.com/someone/agent-artificium"),
                            "github.com/officialgr/agent-artificium")

    def test_apply_refuses_a_running_agent(self):
        with mock.patch("artificium.upgrade.process_state", return_value={"alive": True}):
            with self.assertRaisesRegex(RuntimeError, "Stop Artificium"):
                upgrade(self.paths, report=lambda _: None)
        self.assert_not_upgraded()

    def test_local_edits_to_the_application_are_never_overwritten(self):
        edited = self.paths.app / "prompts/always/core.md"
        edited.write_text("Edited in place.\n")
        with self.assertRaisesRegex(RuntimeError, "local changes"):
            self.run_upgrade()
        self.assertEqual(edited.read_text(), "Edited in place.\n")
        self.assert_not_upgraded()

    def test_local_commits_block_the_fast_forward_without_changes(self):
        (self.paths.app / "artificium/local.py").write_text("LOCAL = True\n")
        self.commit(self.paths.install)
        (self.source / "app/artificium/local.py").write_text("UPSTREAM = True\n")
        self.commit()
        with self.assertRaisesRegex(RuntimeError, "fast-forward"):
            self.run_upgrade()
        self.assertEqual((self.paths.app / "artificium/local.py").read_text(), "LOCAL = True\n")
        self.assert_preserved()

    def test_a_copy_that_is_not_a_git_clone_is_refused(self):
        shutil.rmtree(self.paths.install / ".git")
        with self.assertRaisesRegex(RuntimeError, "not a Git clone"):
            self.run_upgrade()

    def test_overrides_of_changed_files_are_reported(self):
        self.paths.code_overrides.mkdir(parents=True, exist_ok=True)
        (self.paths.code_overrides / "new_feature.py").write_text("MINE = True\n")
        (self.paths.code_overrides / "unrelated.py").write_text("MINE = True\n")
        result = self.run_upgrade(check=True)
        self.assertEqual(result["shadowed_overrides"], ["code/new_feature.py"])

    def test_a_discarded_memory_stays_discarded_across_upgrade_and_startup(self):
        # The agent curates its own mind. A seed file is delivered once; after
        # that, deleting it is a decision, not a gap to fill.
        deleted = self.paths.memory / "harness/infinite-attention.txt"
        deleted.unlink()
        self.run_upgrade()
        self.assertFalse(deleted.exists())
        for _ in range(3):
            initialize_mind(self.paths, Records(self.paths))
            self.assertFalse(deleted.exists())
        self.assert_preserved()

    def test_a_memory_a_newer_application_ships_is_delivered_at_startup(self):
        added = "memory/harness/newly-shipped-capability.txt"
        (self.source / "app/seed" / added).write_text(
            "operating guidance for a capability this version adds\n", encoding="utf-8"
        )
        self.commit()
        self.run_upgrade()
        self.assertFalse((self.paths.mind / added).exists())
        initialize_mind(self.paths, Records(self.paths))
        self.assertTrue((self.paths.mind / added).exists())
        self.assert_preserved()

    def test_a_structural_file_is_repaired_rather_than_left_missing(self):
        # self.txt and meta_memory.md are not curatable: the harness cannot
        # start without them, so their absence is damage, not a choice.
        self.paths.self_file.unlink()
        initialize_mind(self.paths, Records(self.paths))
        self.assertTrue(self.paths.self_file.is_file())

    def test_missing_seed_file_is_reported_without_inventing_a_replacement(self):
        self.paths.meta_memory.unlink()
        (self.paths.seed / "meta_memory.md").unlink()
        with self.assertRaisesRegex(FileNotFoundError, "did not produce it"):
            initialize_mind(self.paths, Records(self.paths))
        self.assertFalse(self.paths.meta_memory.exists())

    def test_cli_upgrades_without_configuration(self):
        self.paths.config.unlink()
        del self.protected[self.paths.config]
        with contextlib.redirect_stdout(io.StringIO()), \
                mock.patch("artificium.upgrade.process_state", return_value={"alive": False}):
            result = main(["--root", str(self.paths.install), "upgrade"])
        self.assertEqual(result, 0)
        self.assertFalse(self.paths.config.exists())
        self.assertTrue((self.paths.app / "artificium/new_feature.py").is_file())
        self.assert_preserved()


if __name__ == "__main__":
    unittest.main()
