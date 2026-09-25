"""Read-only operator views. Inspecting a release never initializes its mind."""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from .config import ConfigStore
from .filesystem import Paths, read_json, read_jsonl
from .memory import TokenEstimator


def process_state(paths: Paths) -> dict[str, Any]:
    state = read_json(paths.process_lock, {})
    state = state if isinstance(state, dict) else {}
    try:
        pid = int(state.get("pid") or 0)
    except (TypeError, ValueError):
        pid = 0
    alive = False
    owned: bool | None = None
    if pid > 0:
        try:
            os.kill(pid, 0)
            alive = True
        except PermissionError:
            alive = True
        except ProcessLookupError:
            pass
        if alive and Path("/proc").is_dir():
            try:
                command = (Path("/proc") / str(pid) / "cmdline").read_bytes().split(b"\0")
                args = [part.decode(errors="replace") for part in command if part]
                # Resolve relative launchers against the actual process cwd.
                cwd = (Path("/proc") / str(pid) / "cwd").resolve(strict=True)
                candidates = {paths.launcher}
                owned = any(
                    (Path(arg) if Path(arg).is_absolute() else cwd / arg).resolve() in candidates
                    for arg in args[1:] if arg.endswith("artificium.py")
                ) and "run" in args
                if "-m" in args and "artificium.cli" in args and "run" in args:
                    supplied_root = args[args.index("--root") + 1] if "--root" in args else str(cwd)
                    owned = Path(supplied_root).resolve() == paths.install
                stat = (Path("/proc") / str(pid) / "stat").read_text()
                if stat.rsplit(")", 1)[1].strip().startswith("Z"):
                    alive = False
            except (OSError, IndexError):
                owned = None
    return {"pid": pid or None, "alive": alive, "owned": owned,
            "started_at": state.get("started_at"), "lock_path": str(paths.process_lock)}


def status_snapshot(paths: Paths) -> dict[str, Any]:
    config = ConfigStore(paths).load() if paths.config.is_file() else None
    process = process_state(paths)
    pending = []
    for path in sorted(paths.receipts.glob("event_*.json")):
        receipt = read_json(path, {})
        if isinstance(receipt, dict) and receipt.get("direction") == "inbound" and not receipt.get("handled_at"):
            pending.append(receipt)
    tasks = [read_json(path, {}) for path in paths.scheduler_tasks.glob("task_*.json")]
    streams = [read_json(path, {}) for path in paths.streams.glob("*/state.json")]
    estimator = TokenEstimator(config.chars_per_token if config else 4.0)
    meta = paths.meta_memory.read_text(encoding="utf-8") if paths.meta_memory.is_file() else ""
    visual = read_json(paths.visual_context, {})
    return {
        "configured": config is not None, "root": str(paths.root), "process": process,
        "provider": config.provider if config else None, "model": config.model if config else None,
        "context_window_tokens": config.context_window_tokens if config else None,
        "working_memory_tokens": config.working_memory_limit if config else None,
        "auto_repair": config.auto_repair if config else False,
        "vision_preference": config.vision_preference if config else None,
        "effective_vision": config.vision if config else None,
        "mandatory_offload": config.mandatory_offload if config else False,
        "offload_threshold_percent": config.offload_threshold_percent if config else 80,
        "offload_pending": bool(config and config.mandatory_offload and read_json(paths.runtime / "life-loop-control.json", {}).get("mandatory_offload_pending")),
        "working_context_tokens": estimator.messages(read_jsonl(paths.working_context)),
        "meta_memory_tokens": estimator.text(meta),
        "pending_notifications": len(list(paths.notifications_new.glob("*.json"))),
        "unhandled_interactions": pending,
        "scheduled_tasks": [t for t in tasks if isinstance(t, dict) and t.get("status") == "pending"],
        "attention_streams": [s for s in streams if isinstance(s, dict)],
        "sleep": read_json(paths.sleep_state, {}), "runtime": read_json(paths.runtime_state, {}),
        "initialization": read_json(paths.initialization_state, {"status": "not_started"}),
        "active_images": visual.get("images", []) if isinstance(visual, dict) else [],
    }


def format_status(state: dict[str, Any]) -> str:
    process = state["process"]
    running = process["alive"] and process["owned"] is not False
    label = "running" if running else "stopped"
    runtime = state.get("runtime") or {}
    if running and runtime.get("status") == "blocked":
        label = "running; model requests paused"
    if not state["configured"]:
        label = "not configured"
    lines = [f"Artificium: {label}", f"Workspace: {state['root']}"]
    if running and runtime.get("status") == "blocked":
        lines.append("Error: " + str(runtime.get("error", "unknown")))
        lines.append("Correct the problem, then run: python3 artificium.py restart")
    if process["alive"]:
        lines.append(f"Process: {process['pid']}" + (" (lock belongs to another process)" if process["owned"] is False else ""))
    if state["configured"]:
        lines.extend([
            f"Engine: {state['provider']} / {state['model']}",
            f"Working context: ~{state['working_context_tokens']:,} tokens; configured capacity: {state['context_window_tokens']:,}",
            f"Working-memory target: {state['working_memory_tokens']:,}; automatic repair: {'on' if state['auto_repair'] else 'off'}",
            f"Pending events: {len(state['unhandled_interactions'])}; scheduled tasks: {len(state['scheduled_tasks'])}",
            f"Vision: {state['vision_preference']} (effective: {state['effective_vision']})",
            "Mandatory offloading: " + (f"{state['offload_threshold_percent']:g}%" if state['mandatory_offload'] else "off") + ("; pending" if state['offload_pending'] else ""),
        ])
        sleep = state["sleep"]
        if sleep.get("active"):
            lines.append(f"Sleep: {sleep.get('mode', 'unknown')}" + (" (saved state; process stopped)" if not running else ""))
    lines.append("Next: python3 artificium.py setup" if not state["configured"] else
                 "Stop: python3 artificium.py stop" if running else "Start: python3 artificium.py start")
    return "\n".join(lines)
