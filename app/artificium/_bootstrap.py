"""Activate the workspace's code overlay before the package imports itself.

This module runs first and must stay independent: it imports nothing from
Artificium, because the modules it decides about include the ones it would
otherwise rely on.  It answers one question — which directory, if any, should
shadow ``artificium/`` — and it refuses to let that answer stop a start.
"""

from __future__ import annotations

import ast
import json
import os
import shutil
import sys
from pathlib import Path


WORKSPACE_DIRECTORY = "workspace"
QUARANTINE_AFTER = 3


def _argument(argv: list[str], name: str) -> str | None:
    """Read ``--name value`` or ``--name=value`` before argparse exists."""

    for index, item in enumerate(argv):
        if item == name and index + 1 < len(argv):
            return argv[index + 1]
        if item.startswith(f"{name}="):
            return item.split("=", 1)[1]
    return None


def install_root(code_file: str | Path, argv: list[str] | None = None) -> Path:
    """Mirror ``configured_paths`` without importing it."""

    argv = sys.argv if argv is None else argv
    supplied = _argument(argv, "--root") or os.getenv("ARTIFICIUM_ROOT")
    if supplied:
        return Path(supplied).expanduser().resolve()
    # <root>/app/artificium/__init__.py
    return Path(code_file).resolve().parents[2]


def disabled(argv: list[str] | None = None) -> bool:
    argv = sys.argv if argv is None else argv
    return "--no-overrides" in argv or os.getenv("ARTIFICIUM_NO_OVERRIDES", "") not in ("", "0")


def _complain(message: str) -> None:
    print(f"artificium: {message}", file=sys.stderr)


def _readable(directory: Path) -> bool:
    """Reject the whole overlay when any module in it cannot be parsed.

    A partial overlay is worse than none: half of a changed mechanism is a
    failure mode nobody can reason about.
    """

    for module in sorted(directory.glob("*.py")):
        try:
            ast.parse(module.read_bytes(), filename=str(module))
        except (SyntaxError, ValueError, OSError) as error:
            _complain(f"ignoring code overrides: {module.name}: {error}")
            return False
    return True


def _attempts(state: Path) -> int:
    try:
        value = json.loads(state.read_text(encoding="utf-8"))
        return int(value.get("failed_starts", 0))
    except (OSError, ValueError, AttributeError):
        return 0


def _record_attempt(state: Path, count: int) -> None:
    try:
        state.parent.mkdir(parents=True, exist_ok=True)
        state.write_text(json.dumps({"failed_starts": count}) + "\n", encoding="utf-8")
    except OSError:
        pass


def _quarantine(overrides: Path, workspace: Path) -> None:
    destination = workspace / "overrides.quarantined"
    try:
        shutil.rmtree(destination, ignore_errors=True)
        overrides.rename(destination)
        _complain(
            f"overrides failed {QUARANTINE_AFTER} starts in a row and were moved "
            f"to {destination}. Starting on the shipped harness."
        )
    except OSError as error:
        _complain(f"could not quarantine {overrides}: {error}")


def clear_attempts(workspace: Path) -> None:
    """Called once a start has proven itself; see ``Artificium.__init__``."""

    try:
        (workspace / "overrides" / ".attempts.json").unlink(missing_ok=True)
    except OSError:
        pass


def code_overlay(code_file: str | Path, argv: list[str] | None = None) -> Path | None:
    """Return the directory to prepend to ``artificium.__path__``, if any."""

    if disabled(argv):
        return None
    root = install_root(code_file, argv)
    workspace = root / WORKSPACE_DIRECTORY
    overrides = workspace / "overrides"
    overlay = overrides / "code"
    if not overlay.is_dir() or not any(overlay.glob("*.py")):
        return None
    state = overrides / ".attempts.json"
    failed = _attempts(state)
    if failed >= QUARANTINE_AFTER:
        _quarantine(overrides, workspace)
        return None
    if not _readable(overlay):
        return None
    # The counter is raised before the overlay loads and cleared once the
    # runtime is alive, so a module that crashes on import cannot loop forever.
    _record_attempt(state, failed + 1)
    return overlay
