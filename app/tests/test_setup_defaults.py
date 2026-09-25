"""Fresh setups write the recommended defaults; saved configurations keep theirs."""
from __future__ import annotations

import contextlib
import io
import os
from pathlib import Path
import shutil
import sys
import tempfile
import unittest
from unittest import mock

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "app"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from artificium.config import NEW_SETUP_DEFAULTS, Config, ConfigStore
from artificium.filesystem import Paths
from artificium.setup import SetupOptions, SetupWizard
from test_connection_flow import TestServer


class SetupDefaultsCase(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(dir=ROOT.parent)
        self.addCleanup(self.temp.cleanup)
        self.paths = Paths(Path(self.temp.name) / "instance")
        shutil.copytree(ROOT / "app/prompts", self.paths.prompts)
        shutil.copytree(ROOT / "app/seed", self.paths.seed)
        self.paths.ensure_layout()
        self.server = TestServer()
        self.addCleanup(self.server.close)
        env = mock.patch.dict(os.environ, {}, clear=True)
        env.start()
        self.addCleanup(env.stop)
        quiet = contextlib.redirect_stdout(io.StringIO())
        quiet.__enter__()
        self.addCleanup(quiet.__exit__, None, None, None)

    def setup(self, **values):
        SetupWizard(self.paths).run(SetupOptions(provider="llamacpp", api_url=self.server.url, **values))
        return ConfigStore(self.paths).load()

    def test_fresh_setup_writes_recommended_defaults(self):
        config = self.setup()
        for name, value in NEW_SETUP_DEFAULTS.items():
            self.assertEqual(getattr(config, name), value, name)
        self.assertEqual(
            config.working_memory_limit, int(config.context_window_tokens * 0.6)
        )

    def test_explicit_choices_win_over_new_defaults(self):
        config = self.setup(
            working_memory_tokens="same", mandatory_offload=False, auto_repair=False,
            request_timeout_seconds="off",
        )
        self.assertIsNone(config.working_memory_fraction)
        self.assertEqual(config.working_memory_limit, config.context_window_tokens)
        self.assertFalse(config.mandatory_offload)
        self.assertFalse(config.auto_repair)
        self.assertIsNone(config.request_timeout_seconds)

    def test_reconnect_keeps_saved_harness_settings(self):
        self.setup(working_memory_tokens="same", auto_repair=False)
        updated = SetupWizard(self.paths).reconfigure(SetupOptions(scope="model"))
        self.assertIsNone(updated.working_memory_fraction)
        self.assertFalse(updated.auto_repair)

    def test_saved_legacy_config_keeps_its_meaning(self):
        legacy = Config(provider="custom", model="m", base_url="http://localhost:1")
        self.assertIsNone(legacy.request_timeout_seconds)
        self.assertFalse(legacy.mandatory_offload)
        self.assertFalse(legacy.auto_repair)
        self.assertEqual(legacy.working_memory_limit, legacy.context_window_tokens)


if __name__ == "__main__":
    unittest.main()
