"""Background life-loop process control and the terminal launcher.

`_pid_state` calls `process_state` (and callers here call `_pid_state`,
`_choose_interaction`, `_chat_repl`, `_watch_life_loop`) through
`artificium.cli` at call time rather than importing them directly: tests
patch these at "artificium.cli.<name>", and only a lookup through that module
object at call time observes such a patch.
"""

from __future__ import annotations

import os
import shlex
import shutil
import signal
import subprocess
import sys
import time
from pathlib import Path

from . import cli as _cli
from .filesystem import Paths, safe_identifier
from .interactions import ArtificiumClient


def _pid_state(paths: Paths) -> tuple[int | None, bool]:
    state = _cli.process_state(paths)
    return state["pid"], bool(state["alive"] and state["owned"] is not False)


def _print_detach_status(paths: Paths, *, client: str) -> None:
    """Make detaching from a client impossible to confuse with stopping the agent."""

    pid, alive = _cli._pid_state(paths)
    label = client.upper()
    if alive and pid:
        print(f"\nWARNING: {label} CLOSED — ARTIFICIUM IS STILL RUNNING")
        print(f"Life-loop PID: {pid}")
        print("TO STOP ARTIFICIUM: python3 artificium.py stop")
    else:
        print(f"\n{label} CLOSED — ARTIFICIUM IS NOT RUNNING")


def _start_background(paths: Paths) -> int:
    pid, alive = _cli._pid_state(paths)
    if alive and pid:
        return pid
    # The daemon is spawned with the workspace as its working directory and
    # writes its own logs there, so it has to exist before the process does.
    # After `rm -rf workspace/` this start is what rebuilds it from the seed.
    paths.ensure_layout()
    launcher = paths.launcher
    output = (paths.logs / "daemon.stdout.log").open("a", encoding="utf-8")
    error = (paths.logs / "daemon.stderr.log").open("a", encoding="utf-8")
    process = subprocess.Popen(
        [sys.executable, str(launcher), "--root", str(paths.install), "run", "--quiet"],
        cwd=paths.root,
        stdin=subprocess.DEVNULL,
        stdout=output,
        stderr=error,
        start_new_session=True,
    )
    output.close()
    error.close()
    deadline = time.monotonic() + 5
    while time.monotonic() < deadline:
        found, running = _cli._pid_state(paths)
        if running and found:
            return found
        if process.poll() is not None:
            raise RuntimeError(
                f"background life-loop exited; inspect {paths.logs / 'daemon.stderr.log'}"
            )
        time.sleep(0.1)
    raise RuntimeError(
        f"background life-loop did not become ready; inspect {paths.logs / 'daemon.stderr.log'}"
    )


def _stop_background(paths: Paths) -> bool:
    state = _cli.process_state(paths)
    if state["alive"] and state["owned"] is None:
        raise RuntimeError("Cannot verify the lock PID belongs to Artificium; inspect it before stopping manually.")
    pid, alive = state["pid"], state["alive"] and state["owned"] is True
    if not alive or not pid:
        paths.process_lock.unlink(missing_ok=True)
        return False
    try:
        os.kill(pid, signal.SIGTERM)
    except ProcessLookupError:
        paths.process_lock.unlink(missing_ok=True)
        return False
    deadline = time.monotonic() + 5
    while time.monotonic() < deadline:
        _, running = _cli._pid_state(paths)
        if not running:
            return True
        time.sleep(0.1)
    try:
        os.kill(pid, signal.SIGKILL)
    except ProcessLookupError:
        pass
    deadline = time.monotonic() + 2
    while time.monotonic() < deadline:
        _, running = _cli._pid_state(paths)
        if not running:
            paths.process_lock.unlink(missing_ok=True)
            return True
        time.sleep(0.05)
    raise RuntimeError(
        f"could not stop PID {pid}; inspect {paths.process_lock} and terminate it explicitly"
    )


def _terminal_command(paths: Paths, chat_args: list[str]) -> list[str] | None:
    base = [
        sys.executable,
        str(paths.launcher),
        "--root",
        str(paths.install),
        *chat_args,
    ]
    configured = os.getenv("TERMINAL")
    candidates: list[list[str]] = []
    if configured:
        candidates.append([*shlex.split(configured), "-e", *base])
    candidates.extend(
        [
            ["gnome-terminal", "--", *base],
            ["konsole", "-e", *base],
            ["kitty", *base],
            ["wezterm", "start", "--", *base],
            ["xterm", "-e", *base],
        ]
    )
    for command in candidates:
        if shutil.which(command[0]):
            return command
    return None


def _launcher(paths: Paths) -> None:
    print("\nWhat would you like to do?\n")
    print("1. Start a chat")
    print("2. Watch the life-loop")
    choice = input("\nChoose [1]: ").strip() or "1"
    if choice == "1":
        entity = safe_identifier(
            input("Your entity name/ID [user_1]: ").strip() or "user_1",
            label="entity id",
        )
        client = ArtificiumClient(paths.install)
        interaction_id = _cli._choose_interaction(client, entity, None, None)
        pid = _start_background(paths)
        print(f"Life-loop running as PID {pid}.")
        args = ["chat", "--entity", entity, "--interaction", interaction_id]
        command = _terminal_command(paths, args)
        if command:
            subprocess.Popen(command, cwd=paths.root, start_new_session=True)
            print("Chat opened in a new terminal.")
        else:
            print("No supported terminal launcher was detected; opening chat here.")
            _cli._chat_repl(paths, entity, interaction_id, None)
        return
    if choice == "2":
        _, alive = _cli._pid_state(paths)
        if not alive:
            pid = _start_background(paths)
            print(f"Life-loop started as PID {pid}.")
        _cli._watch_life_loop(paths)
        return
    raise ValueError("choose 1 or 2")

