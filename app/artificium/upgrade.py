"""Upgrade the project with a fast-forward `git pull`.

The workspace, config.json, and .secrets.json are ignored by Git, so pulling
never touches what the agent has learned or how it connects to its model.
"""
from __future__ import annotations

from contextlib import ExitStack
import shutil
import subprocess
from typing import Callable

from .filesystem import Paths, file_lock
from .operator import process_state
from .runtime import ProcessLock

# This fork's home. A clone that still tracks the original project is moved
# here on upgrade, since the original no longer carries this fork's releases.
REPOSITORY_URL = "https://github.com/EmmanuelARB/agent-artificium.git"
ORIGINAL_REPOSITORY = "github.com/officialgr/agent-artificium"


def _git(paths: Paths, *args: str, timeout: float = 60) -> str:
    try:
        completed = subprocess.run(
            ["git", "-C", str(paths.install), *args],
            check=True, capture_output=True, text=True, timeout=timeout,
        )
    except subprocess.CalledProcessError as error:
        detail = (error.stderr or error.stdout or "").strip()
        raise RuntimeError(f"git {' '.join(args)} failed: {detail}") from error
    except subprocess.TimeoutExpired as error:
        raise RuntimeError(f"git {' '.join(args)} timed out; check the network.") from error
    return completed.stdout


def shadowed_overrides(paths: Paths, changes: list[str]) -> list[str]:
    """Workspace overrides that hide a shipped file this upgrade changes.

    An override written against the old file keeps winning after the upgrade,
    so it may pin old behaviour or fail to import the new modules around it.
    ``changes`` are paths relative to the project root, as Git reports them.
    """
    changed = set(changes)
    shadowed = []
    for overlay, shipped in (
        (paths.code_overrides, "app/artificium"),
        (paths.prompt_overrides, "app/prompts"),
    ):
        if not overlay.is_dir():
            continue
        for path in sorted(overlay.rglob("*")):
            if not path.is_file() or "__pycache__" in path.parts:
                continue
            relative = path.relative_to(overlay).as_posix()
            if f"{shipped}/{relative}" in changed:
                shadowed.append(f"{overlay.name}/{relative}")
    return shadowed


def _repository_key(url: str) -> str:
    """``host/owner/name`` for any spelling of a Git URL (https, ssh, scp-like)."""
    key = url.strip().lower()
    for prefix in ("https://", "http://", "ssh://", "git://"):
        if key.startswith(prefix):
            key = key[len(prefix):]
            break
    key = key.split("@", 1)[-1] if "@" in key.split("/", 1)[0] else key
    if ":" in key.split("/", 1)[0]:
        key = key.replace(":", "/", 1)
    key = key.rstrip("/")
    return key[:-4] if key.endswith(".git") else key


def _moved_remote(paths: Paths) -> tuple[str, str, str] | None:
    """(remote, old URL, branch) when the tracked remote is the original project."""
    try:
        branch = _git(paths, "symbolic-ref", "--short", "HEAD").strip()
        remote = _git(paths, "config", "--get", f"branch.{branch}.remote").strip()
        merge = _git(paths, "config", "--get", f"branch.{branch}.merge").strip()
        url = _git(paths, "remote", "get-url", remote).strip()
    except RuntimeError:
        return None
    if _repository_key(url) != _repository_key(ORIGINAL_REPOSITORY):
        return None
    return remote, url, merge.removeprefix("refs/heads/")


def _require_stopped(paths: Paths) -> None:
    if process_state(paths)["alive"]:
        raise RuntimeError("Stop Artificium before upgrading: python3 artificium.py stop")


def upgrade(paths: Paths, *, check: bool = False, report: Callable[[str], None] = print) -> dict:
    """Fetch the tracked branch and fast-forward to it; never merge or discard."""
    if not shutil.which("git"):
        raise RuntimeError("Install Git to use the upgrade command.")
    if not (paths.install / ".git").exists():
        raise RuntimeError(
            "This copy of Artificium is not a Git clone. Clone the repository, then move "
            "config.json, .secrets.json, and workspace/ into the new clone."
        )
    if not check:
        _require_stopped(paths)
    if _git(paths, "status", "--porcelain", "--untracked-files=no").strip():
        raise RuntimeError(
            "The application has local changes (git status). Commit or discard them first; "
            "to change prompts or code without editing the application, use workspace overrides."
        )
    try:
        upstream = _git(paths, "rev-parse", "--abbrev-ref", "--symbolic-full-name", "@{upstream}").strip()
    except RuntimeError as error:
        raise RuntimeError("The current branch tracks no remote branch; run git pull yourself.") from error
    moved = _moved_remote(paths)
    if moved and check:
        remote, url, branch = moved
        report(f"{remote} points at the original project ({url}); upgrade will move it to "
               f"{REPOSITORY_URL}. Previewing that repository…")
        _git(paths, "fetch", "--quiet", REPOSITORY_URL, branch, timeout=180)
        upstream = "FETCH_HEAD"
    else:
        if moved:
            remote, url, _ = moved
            _git(paths, "remote", "set-url", remote, REPOSITORY_URL)
            report(f"Moved {remote} from the original project ({url}) to {REPOSITORY_URL}.")
        report(f"Fetching {upstream}…")
        _git(paths, "fetch", "--quiet", timeout=180)
    changes = [line for line in _git(paths, "diff", "--name-only", "HEAD", upstream).splitlines() if line]
    revision = _git(paths, "rev-parse", "--short=12", upstream).strip()
    shadowed = shadowed_overrides(paths, changes)
    result = {"revision": revision, "changed_files": len(changes), "shadowed_overrides": shadowed}
    if shadowed:
        report("Workspace overrides hide files this upgrade changes; they keep the old "
               "version until reviewed (python3 artificium.py overrides diff NAME):")
        for name in shadowed:
            report(f"  {name}")
    if check or not changes:
        report(f"{revision}: {len(changes)} files would change." if check else "Already up to date.")
        return result
    with ExitStack() as stack:
        stack.enter_context(file_lock(paths.runtime / "upgrade.lock"))
        _require_stopped(paths)
        stack.enter_context(ProcessLock(paths.process_lock))
        try:
            _git(paths, "merge", "--ff-only", "--quiet", upstream)
        except RuntimeError as error:
            raise RuntimeError(
                f"Git refused to fast-forward, so nothing was changed. Reconcile the clone "
                f"with git yourself. {error}"
            ) from error
    report(f"Upgraded {len(changes)} files to {revision}. The workspace and settings were untouched.")
    report("Start: python3 artificium.py start")
    return result
