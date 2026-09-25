"""Upgrade the application tree; the workspace and settings live outside it."""
from __future__ import annotations

import ast
from contextlib import ExitStack, suppress
import os
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Callable

from .filesystem import Paths, atomic_write_json, file_lock, sortable_id
from .operator import process_state
from .prompts import PromptPack
from .runtime import ProcessLock


DEFAULT_REPOSITORY = "https://github.com/officialgr/agent-artificium.git"
MANAGED_DIRECTORIES = (
    "artificium-code/artificium", "artificium-code/prompts", "artificium-code/seed",
    "artificium-code/tests", "artificium-code/examples", "scripts",
)
MANAGED_FILES = (
    "artificium.py", "README.md", ".gitignore", "artificium-code/artificium.py",
    "artificium-code/pyproject.toml", "artificium-code/REFERENCE.md",
)
# Nothing mutable lives in app/ any more, so only build artifacts are skipped.
PRESERVED_NAMES = {"__pycache__", ".pytest_cache", ".git"}


def _preserved(path: Path) -> bool:
    return (any(part in PRESERVED_NAMES or part.startswith(".env") for part in path.parts)
            or path.suffix in {".pyc", ".pyo"})


def _regular(root: Path, path: Path) -> None:
    for part in (path, *path.parents):
        if part == root:
            break
        if part.is_symlink():
            raise ValueError(f"Cannot upgrade through a symlink: {part}")


def _files(root: Path) -> dict[str, Path]:
    result = {}
    candidates = [root / name for name in MANAGED_FILES]
    for name in MANAGED_DIRECTORIES:
        directory = root / name
        _regular(root, directory)
        candidates.extend(directory.rglob("*"))
    for path in candidates:
        relative = path.relative_to(root)
        if _preserved(relative):
            continue
        _regular(root, path)
        if path.is_file():
            result[relative.as_posix()] = path
    return result


def _require_stopped(paths: Paths) -> None:
    if process_state(paths)["alive"]:
        raise RuntimeError("Stop Artificium before upgrading: python3 artificium.py stop")


def _replace(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(prefix=".upgrade-", dir=destination.parent)
    os.close(descriptor)
    temporary = Path(name)
    try:
        shutil.copy2(source, temporary)
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)


def _apply(paths: Paths, incoming: dict[str, Path], changes: list[str], revision: str) -> Path:
    backup = paths.logs / "upgrades" / sortable_id("upgrade_")
    backup.mkdir(parents=True, mode=0o700)
    created = []
    for name in changes:
        destination = paths.app / name
        _regular(paths.app, destination)
        if destination.exists():
            saved = backup / "files" / name
            saved.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(destination, saved)
        else:
            created.append(name)
    manifest = {"revision": revision, "changed": changes, "created": created, "status": "applying"}
    atomic_write_json(backup / "upgrade.json", manifest)
    applied = []
    try:
        for name in changes:
            applied.append(name)
            destination = paths.app / name
            if name in incoming:
                _replace(incoming[name], destination)
            else:
                destination.unlink()
        atomic_write_json(backup / "upgrade.json", {**manifest, "status": "completed"})
    except BaseException:
        try:
            for name in reversed(applied):
                saved = backup / "files" / name
                if saved.is_file():
                    _replace(saved, paths.app / name)
                else:
                    (paths.app / name).unlink(missing_ok=True)
        except OSError as error:
            raise RuntimeError(f"Upgrade and restoration failed. Keep the agent stopped; code backup: {backup}") from error
        with suppress(OSError):
            atomic_write_json(backup / "upgrade.json", {**manifest, "status": "rolled_back"})
        raise RuntimeError(f"Upgrade failed; replaced code was restored. Backup: {backup}")
    # Remove source directories the new version no longer ships.
    for name in changes:
        parent = (paths.app / name).parent
        while parent != paths.app:
            try:
                parent.rmdir()
            except OSError:
                break
            parent = parent.parent
    return backup


def upgrade(paths: Paths, *, repository: str = DEFAULT_REPOSITORY, ref: str = "main",
            check: bool = False, report: Callable[[str], None] = print) -> dict:
    """Fetch a branch/tag into staging; never pull/reset the live checkout."""
    if not check:
        _require_stopped(paths)
    if not shutil.which("git"):
        raise RuntimeError("Install Git to use the upgrade command.")
    if not repository or repository.startswith("-") or not ref or ref.startswith("-"):
        raise ValueError("Provide a repository and a branch/tag name, not Git options.")
    report("Downloading the selected Artificium version…")
    with ExitStack() as stack:
        if not check:
            stack.enter_context(file_lock(paths.runtime / "upgrade.lock"))
            _require_stopped(paths)
            stack.enter_context(ProcessLock(paths.process_lock))
        directory = stack.enter_context(tempfile.TemporaryDirectory(prefix="artificium-upgrade-", dir=paths.install))
        source = Path(directory) / "source"
        try:
            subprocess.run(["git", "clone", "--quiet", "--depth", "1", "--single-branch",
                            "--branch", ref, "--", repository, str(source)],
                           check=True, capture_output=True, timeout=180)
            revision = subprocess.check_output(["git", "-C", str(source), "rev-parse", "HEAD"], text=True).strip()
        except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as error:
            raise RuntimeError("Could not download Artificium. Check the repository, branch/tag, network and Git access; no instance files were replaced.") from error
        for name in ("artificium.py", "artificium-code/artificium/cli.py", "artificium-code/pyproject.toml"):
            if not (source / name).is_file():
                raise ValueError(f"Incomplete upgrade source: missing {name}")
        incoming = _files(source)
        PromptPack(Paths.for_app(source)).validate()
        for name, path in incoming.items():
            if path.suffix == ".py":
                try:
                    ast.parse(path.read_bytes(), filename=name)
                except SyntaxError as error:
                    raise ValueError(f"Invalid Python in upgrade source: {name}: {error.msg}") from error
        current = _files(paths.app)
        changes = sorted(name for name in current.keys() | incoming.keys()
                         if name not in current or name not in incoming
                         or current[name].read_bytes() != incoming[name].read_bytes()
                         or (current[name].stat().st_mode & 0o777) != (incoming[name].stat().st_mode & 0o777))
        result = {"revision": revision, "changed_files": len(changes), "backup": None}
        if check or not changes:
            report(f"{revision[:12]}: {len(changes)} managed files would change." if check else "Managed code is already up to date.")
            return result
        backup = _apply(paths, incoming, changes, revision)
        result["backup"] = str(backup)
        report(f"Upgraded {len(changes)} managed files to {revision[:12]}. Code backup: {backup}")
        report("The workspace, settings and existing logs were untouched. Start: python3 artificium.py start")
        return result
