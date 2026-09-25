from pathlib import Path
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "app"))

from artificium.filesystem import Paths
from artificium.upgrade import shadowed_overrides


class ShadowedOverridesCase(unittest.TestCase):
    def test_reports_only_overrides_of_changed_files(self):
        with tempfile.TemporaryDirectory() as temp:
            paths = Paths(Path(temp) / "instance")
            (paths.code_overrides).mkdir(parents=True)
            (paths.code_overrides / "records.py").write_text("x = 1\n")
            (paths.code_overrides / "helper_of_my_own.py").write_text("y = 1\n")
            (paths.prompt_overrides / "always").mkdir(parents=True)
            (paths.prompt_overrides / "always" / "core.md").write_text("core\n")
            changes = ["app/artificium/records.py", "app/artificium/cli.py",
                       "app/prompts/events/wake.md"]
            self.assertEqual(shadowed_overrides(paths, changes), ["code/records.py"])
            changes.append("app/prompts/always/core.md")
            self.assertEqual(shadowed_overrides(paths, changes),
                             ["code/records.py", "prompts/always/core.md"])


if __name__ == "__main__":
    unittest.main()
