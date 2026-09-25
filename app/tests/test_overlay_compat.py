"""Workspace code-overlay compatibility for the S5 module split.

Artificium lets a workspace shadow a shipped module with
``workspace/overrides/code/<module>.py`` (see ``artificium/_bootstrap.py`` and
``artificium/__init__.py``: the overlay directory is inserted first in the
package ``__path__``). An overlay written against an older version of a
module should keep working after an upgrade as far as reasonably possible.

This test takes the *pre-split* ("baseline") copy of each module this
refactor touched -- recorded once, at authoring time, as a plain fixture file
under ``tests/fixtures/overlay_baseline/`` (so the test itself never depends
on git at runtime) -- and drops it into a fresh install's
``workspace/overrides/code/<module>.py``, one module at a time. It then runs a
*separate interpreter* (so the overlay is picked up from a cold import, the
way it would be for a real process) that imports ``artificium``, builds an
``Artificium`` instance with a scripted in-process ``Engine`` (no network),
runs one bounded turn, and also exercises ``artificium.cli.main(["status"])``.

Modules where this is expected to fail (and why) are recorded in
``EXPECTED_FAILURES`` instead of silently skipped, so a regression in an
*unexpected* module still fails the suite.
"""

from __future__ import annotations

import subprocess
import sys
import tempfile
import textwrap
import unittest
from pathlib import Path

CODE_DIR = Path(__file__).resolve().parents[1]
FIXTURES = Path(__file__).resolve().parent / "fixtures" / "overlay_baseline"

# Every module this refactor changed that also exists in the pre-refactor
# ("staged baseline") tree. Each fixture is that baseline file's content,
# verbatim, captured with `git show :app/artificium/<name>.py`.
BASELINE_MODULES = [
    "records",
    "tools",
    "memory",
    "engine",
    "cli",
    "config",
    "setup",
    "runtime",
    "context_budget",
    "connection",
    "upgrade",
]

# module -> one-line reason a *pre-split* overlay of it cannot be expected to
# work against the current harness. Anything not listed here is expected to
# import and run a full bounded turn cleanly.
#
# Every module in BASELINE_MODULES is, in fact, expected to work: an old
# module overlaid alone is a strict *subset* of the current one (it just
# doesn't use whatever feature was added since), and every place current code
# added a new cross-module dependency (a new field, function, or class) was
# fixed to degrade gracefully -- a `getattr` fallback, a `TYPE_CHECKING`-only
# import, or moving the new symbol to a small module of its own that the
# unrelated module doesn't shadow -- rather than left as a hard import/attr
# error. This dict stays here, empty, as the place a genuinely irreducible
# case (like the task description's example of an old runtime.py that simply
# cannot know about new Config fields) would be recorded if one is found.
EXPECTED_FAILURES: dict[str, str] = {}


def _script(module: str, install: Path) -> str:
    """A short program run in a fresh interpreter under `install`'s overlay."""

    return textwrap.dedent(f"""
        import os
        import sys
        sys.path.insert(0, {str(CODE_DIR)!r})

        from pathlib import Path
        install = Path({str(install)!r})
        # Must be set before the first `import artificium`: __init__.py picks
        # the code overlay (if any) exactly once, at that first import.
        os.environ["ARTIFICIUM_ROOT"] = str(install)

        from artificium.config import Config, ConfigStore, SecretsStore
        from artificium.filesystem import Paths
        from artificium.records import Console

        paths = Paths(install)
        ConfigStore(paths).save(Config(
            provider="custom", model="test-model",
            base_url="http://example.invalid/v1",
            context_window_tokens=20_000,
        ))
        SecretsStore(paths).save_api_key("test-key")

        from artificium.engine import Engine, EngineReply
        from artificium.runtime import Artificium

        class ScriptedEngine(Engine):
            def __init__(self):
                self.calls = 0

            def complete(self, messages):
                self.calls += 1
                return EngineReply("<think>No action.</think>", usage={{"input_tokens": 5, "output_tokens": 3}})

        agent = Artificium(paths, engine=ScriptedEngine(), console=Console(quiet=True))
        agent.run_once(trigger="overlay-compat-test")
        assert agent.engine.calls >= 1, "the scripted engine was never called"

        from artificium.cli import main as cli_main
        status_code = cli_main(["--root", str(install), "status"])
        assert status_code == 0, f"cli status exited {{status_code}}"

        print("OVERLAY_COMPAT_OK")
    """)


def _seed(paths_root: Path) -> None:
    """Give a fresh install its shipped prompts/seed, the way a real one gets them."""

    import shutil

    from artificium.filesystem import Paths

    paths = Paths(paths_root)
    shutil.copytree(CODE_DIR / "prompts", paths.prompts)
    shutil.copytree(CODE_DIR / "seed", paths.seed)
    paths.ensure_layout()


def _run_with_overlay(module: str) -> subprocess.CompletedProcess:
    fixture = FIXTURES / f"{module}.py"
    with tempfile.TemporaryDirectory() as temp:
        install = Path(temp) / "install"
        _seed(install)
        from artificium.filesystem import Paths

        paths = Paths(install)
        paths.code_overrides.mkdir(parents=True, exist_ok=True)
        (paths.code_overrides / f"{module}.py").write_text(
            fixture.read_text(encoding="utf-8"), encoding="utf-8"
        )
        script = paths.root / "overlay_compat_script.py"
        script.write_text(_script(module, install), encoding="utf-8")
        return subprocess.run(
            [sys.executable, str(script)],
            capture_output=True,
            text=True,
            timeout=60,
        )


class OverlayCompatCase(unittest.TestCase):
    """One test per baseline module, so failures are reported individually."""


def _make_test(module: str):
    def test(self: unittest.TestCase) -> None:
        self.assertTrue(
            (FIXTURES / f"{module}.py").is_file(),
            f"missing fixture for {module}; regenerate tests/fixtures/overlay_baseline/{module}.py",
        )
        result = _run_with_overlay(module)
        expected_reason = EXPECTED_FAILURES.get(module)
        if expected_reason is None:
            self.assertEqual(
                result.returncode, 0,
                f"a baseline `{module}.py` overlay should still run; "
                f"stdout={result.stdout!r} stderr={result.stderr[-4000:]!r}",
            )
            self.assertIn("OVERLAY_COMPAT_OK", result.stdout)
        else:
            # Recorded as an expected failure -- assert it still fails so a
            # fix nobody documented here doesn't silently regress unnoticed.
            self.assertNotEqual(
                result.returncode, 0,
                f"`{module}.py` is listed as an expected overlay-compat "
                f"failure ({expected_reason!r}) but a baseline overlay now "
                f"runs cleanly; remove it from EXPECTED_FAILURES.",
            )

    test.__name__ = f"test_{module}_baseline_overlay"
    return test


for _module in BASELINE_MODULES:
    setattr(OverlayCompatCase, f"test_{_module}_baseline_overlay", _make_test(_module))
del _module


if __name__ == "__main__":
    unittest.main()
