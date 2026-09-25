from __future__ import annotations

import contextlib
import datetime as dt
import hashlib
import json
import os
import re
import shutil
import tempfile
import threading
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator


def utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat(timespec="microseconds").replace(
        "+00:00", "Z"
    )


def sortable_id(prefix: str = "") -> str:
    stamp = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    return f"{prefix}{stamp}_{uuid.uuid4().hex[:12]}"


def json_dumps(value: Any, *, pretty: bool = False) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        indent=2 if pretty else None,
        separators=None if pretty else (",", ":"),
        sort_keys=pretty,
    )


def atomic_write_text(path: Path, content: str, *, mode: int | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary_path = Path(temporary)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        if mode is not None:
            os.chmod(temporary_path, mode)
        os.replace(temporary_path, path)
    finally:
        temporary_path.unlink(missing_ok=True)


def atomic_write_json(path: Path, value: Any, *, mode: int | None = None) -> None:
    atomic_write_text(path, json_dumps(value, pretty=True) + "\n", mode=mode)


_append_locks: dict[str, threading.Lock] = {}
_append_locks_guard = threading.Lock()


def append_jsonl(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    key = str(path.resolve())
    with _append_locks_guard:
        lock = _append_locks.setdefault(key, threading.Lock())
    with lock:
        with path.open("a", encoding="utf-8") as handle:
            handle.write(json_dumps(value) + "\n")
            handle.flush()


def read_json(path: Path, default: Any = None) -> Any:
    if not path.is_file():
        return default
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return default


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return list(iter_jsonl(path))


def iter_jsonl(path: Path) -> Iterator[dict[str, Any]]:
    if not path.is_file():
        return
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            try:
                value = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(value, dict):
                yield value


def sha256_text(content: str) -> str:
    return hashlib.sha256(content.encode("utf-8")).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


# Hashes of seed files as shipped by earlier releases, for workspaces whose
# delivery record predates per-file hashes.  A mind file still equal to one of
# these was never edited by the agent and may be refreshed to the new seed.
_EARLIER_SEED_RELEASES: dict[str, tuple[str, ...]] = {
    "tools/search.py": ("b61589fe401f92788fbbdb8dce31db8cd4b15b52ab30080b9fb9d9690304d24d",),
    "memory/harness/learning-self-improvement-and-adaptation.txt": (
        "a88676a962a2f8bf4f827814f9380cc9fe2a1d8a3f92f647ca59ae66c3792e36",
    ),
    "memory/harness/self-initiative-and-sleep.txt": (
        "1a6bed8f1085c059cdf65e11b5cd3a96c23a781ae89be723a50aa454eacfeb25",
    ),
    "memory/harness/tool-building-and-workspace.txt": (
        "811750b3ffca518c228b3847fabd01dee3f9409cdce6708243d3b500988807e8",
    ),
}


def safe_identifier(value: str, *, label: str = "identifier") -> str:
    value = value.strip()
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,127}", value):
        raise ValueError(
            f"Invalid {label}: use 1-128 letters, numbers, dots, dashes, or underscores"
        )
    return value


def descriptive_slug(value: str, *, label: str = "name") -> str:
    original = value.strip()
    slug = re.sub(r"[^a-z0-9]+", "-", original.lower()).strip("-")
    slug = re.sub(r"-{2,}", "-", slug)[:120].rstrip("-")
    if len(slug) < 8 or slug in {
        "context",
        "memory",
        "summary",
        "checkpoint",
        "offload",
        "notes",
        "misc",
        "temp",
    }:
        raise ValueError(
            f"{label} must be descriptive enough to reveal what should be loaded"
        )
    return slug


def _sanitize_path_component(name: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_.-]+", "_", name).strip("._")
    return cleaned[:80] or "file"


def _file_version_directory(paths: "Paths", target: Path) -> Path:
    key = hashlib.sha256(str(target).encode("utf-8")).hexdigest()[:16]
    return paths.file_versions / f"{key}-{_sanitize_path_component(target.name)}"


def backup_before_overwrite(
    paths: "Paths",
    target: Path,
    *,
    keep_versions: int = 20,
    max_total_bytes: int = 20_000_000,
) -> Path | None:
    """Copy ``target``'s current content into a rotated backup area before an
    overwrite destroys it.

    Real incident: the model read a 41.8KB file through a partial
    ``read_file`` window, mistook it for the whole file, and overwrote it
    with 8.5KB with no backup anywhere. Backups live under
    ``logs/runtime/file-versions/`` -- outside ``mind/``, so an agent that
    reorganizes or deletes memory cannot also delete its own safety net --
    keyed by a hash of the absolute path so unrelated files never collide and
    a renamed/moved file does not inherit stale history. Only the last
    ``keep_versions`` copies (and at most ``max_total_bytes`` total) are kept
    per file; older ones are pruned first.

    Returns the new backup's path, or ``None`` if ``target`` does not exist
    (nothing to back up).
    """

    if not target.is_file():
        return None
    directory = _file_version_directory(paths, target)
    directory.mkdir(parents=True, exist_ok=True)
    origin_marker = directory / "origin.txt"
    if not origin_marker.exists():
        try:
            origin_marker.write_text(str(target) + "\n", encoding="utf-8")
        except OSError:
            pass
    backup_path = directory / f"{sortable_id()}.bak"
    shutil.copy2(target, backup_path)
    _rotate_file_versions(directory, keep_versions=keep_versions, max_total_bytes=max_total_bytes)
    return backup_path


def _rotate_file_versions(directory: Path, *, keep_versions: int, max_total_bytes: int) -> None:
    versions = sorted(p for p in directory.glob("*.bak") if p.is_file())
    while len(versions) > max(1, keep_versions):
        versions.pop(0).unlink(missing_ok=True)
    total = 0
    sizes: list[tuple[Path, int]] = []
    for version in versions:
        try:
            size = version.stat().st_size
        except OSError:
            continue
        sizes.append((version, size))
        total += size
    while total > max_total_bytes and len(sizes) > 1:
        oldest, size = sizes.pop(0)
        try:
            oldest.unlink()
            total -= size
        except OSError:
            break


def record_partial_read(paths: "Paths", target: Path, info: dict[str, Any]) -> None:
    """Remember that the latest ``read_file`` of ``target`` was incomplete."""

    state = read_json(paths.partial_reads_state, {})
    if not isinstance(state, dict):
        state = {}
    state[str(target)] = info
    atomic_write_json(paths.partial_reads_state, state)


def clear_partial_read(paths: "Paths", target: Path) -> None:
    """Forget any partial-read marker for ``target`` (its latest read was whole)."""

    state = read_json(paths.partial_reads_state, {})
    if not isinstance(state, dict) or str(target) not in state:
        return
    state.pop(str(target), None)
    atomic_write_json(paths.partial_reads_state, state)


def get_partial_read(paths: "Paths", target: Path) -> dict[str, Any] | None:
    state = read_json(paths.partial_reads_state, {})
    if not isinstance(state, dict):
        return None
    value = state.get(str(target))
    return value if isinstance(value, dict) else None


SHRINK_GUARD_MIN_EXISTING_BYTES = 4096
SHRINK_GUARD_RATIO = 0.5


def check_shrink_guard(
    *,
    paths: "Paths",
    target: Path,
    existing_size: int,
    new_size: int,
    allow_shrink: bool,
) -> None:
    """Refuse a large, unconfirmed shrink of ``target`` on overwrite.

    Overwriting a file bigger than ``SHRINK_GUARD_MIN_EXISTING_BYTES`` with
    less than ``SHRINK_GUARD_RATIO`` of its size is refused unless the caller
    passes ``allow_shrink=True`` -- confirming the loss is intentional rather
    than the result of mistaking a partial read for the whole file.
    """

    if allow_shrink or existing_size <= SHRINK_GUARD_MIN_EXISTING_BYTES:
        return
    if new_size >= existing_size * SHRINK_GUARD_RATIO:
        return
    partial = get_partial_read(paths, target)
    hint = ""
    if partial:
        hint = (
            " The last read_file of this path in this runtime was partial "
            f"(bytes {partial.get('start_byte')}-{partial.get('end_byte')} of "
            f"{partial.get('size_bytes')} total), so this overwrite may be based "
            "on an incomplete view of the file."
        )
    percent = round((1 - (new_size / existing_size)) * 100)
    raise ValueError(
        f"Refusing to overwrite {target} ({existing_size} bytes) with only "
        f"{new_size} bytes (a {percent}% shrink).{hint} Read the file fully first "
        "to confirm this is intentional, then retry with allow_shrink=true."
    )


def mind_prefix_hint(paths: "Paths", supplied: str) -> str:
    """A short \"Did you mean mind/...?\" hint for a not-found read path.

    Real incidents: the model used ``memory/projects/...`` instead of
    ``mind/memory/projects/...`` (every tool path is relative to the
    workspace root, and durable memory lives under ``mind/memory``). This
    never changes what is read or written -- it only helps the model correct
    itself on its next call. Reads get a hint on a genuine not-found; a write
    never auto-resolves, since silently "fixing" its destination would hide a
    wrong path instead of surfacing it.
    """

    clean = supplied.strip().replace("\\", "/").lstrip("/")
    if not clean:
        return ""
    candidates: list[tuple[str, Path]] = []
    head = clean.split("/", 1)[0]
    if head != "mind":
        candidates.append((f"mind/{clean}", paths.mind / clean))
    elif clean.startswith("mind/") and len(clean) > len("mind/"):
        rest = clean[len("mind/") :]
        candidates.append((rest, paths.root / rest))
    for label, candidate in candidates:
        try:
            exists = candidate.exists()
        except OSError:
            exists = False
        if exists:
            return f" Did you mean {label}?"
    return ""


def resolve_inside(root: Path, supplied: str) -> Path:
    relative = Path(supplied)
    if relative.is_absolute():
        candidate = relative.expanduser().resolve()
    else:
        candidate = (root / relative).resolve()
    try:
        candidate.relative_to(root.resolve())
    except ValueError as exc:
        raise ValueError(f"Path must remain inside {root}: {supplied}") from exc
    return candidate


@contextlib.contextmanager
def file_lock(path: Path) -> Iterator[None]:
    """Small cross-process advisory lock on Unix; thread-safe fallback elsewhere."""

    path.parent.mkdir(parents=True, exist_ok=True)
    handle = path.open("a+")
    try:
        try:
            import fcntl

            fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        except (ImportError, OSError):
            pass
        yield
    finally:
        try:
            import fcntl

            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
        except (ImportError, OSError):
            pass
        handle.close()


APP_DIRECTORY = "app"
WORKSPACE_DIRECTORY = "workspace"


@dataclass(frozen=True)
class Paths:
    """Everything is located from the project root, the directory of the clone.

    ``app`` holds the shipped application and never changes at runtime, while
    ``workspace`` holds everything the agent writes. The settings sit at the
    root too, outside the workspace, so deleting the workspace restores a
    factory agent without losing the model connection.

        <root>/
        |-- artificium.py   (launcher)
        |-- README.md
        |-- app/            (code, prompts, seed, tests)
        |-- config.json     (ignored by Git)
        |-- .secrets.json   (ignored by Git)
        `-- workspace/      (ignored by Git: mind, logs, overrides)
    """

    install: Path
    app_dir: Path | None = None

    @classmethod
    def from_code_file(cls, file: str | Path) -> "Paths":
        # <root>/app/artificium/<module>.py
        return cls(Path(file).resolve().parents[2])

    @classmethod
    def for_app(cls, app: str | Path) -> "Paths":
        """Address an application tree other than this project's, such as a clone."""

        directory = Path(app).resolve()
        return cls(directory.parent, directory)

    @property
    def app(self) -> Path:
        return self.app_dir if self.app_dir is not None else self.install / APP_DIRECTORY

    @property
    def root(self) -> Path:
        """The agent's world: every path it reads or writes is relative to this."""

        return self.install / WORKSPACE_DIRECTORY

    @property
    def launcher(self) -> Path:
        return self.app.parent / "artificium.py"

    @property
    def code(self) -> Path:
        """The directory to put on ``sys.path`` to import the ``artificium`` package."""

        return self.app

    @property
    def prompts(self) -> Path:
        return self.code / "prompts"

    @property
    def prompt_manifest(self) -> Path:
        return self.prompts / "manifest.toml"

    @property
    def seed(self) -> Path:
        """The shipped starting mind, copied into the workspace on first start."""

        return self.code / "seed"

    @property
    def config(self) -> Path:
        return self.install / "config.json"

    @property
    def secrets(self) -> Path:
        return self.install / ".secrets.json"

    @property
    def overrides(self) -> Path:
        return self.root / "overrides"

    @property
    def prompt_overrides(self) -> Path:
        return self.overrides / "prompts"

    @property
    def code_overrides(self) -> Path:
        return self.overrides / "code"

    @property
    def seed_deliveries(self) -> Path:
        """Seed files this workspace has already been given, once each."""

        return self.runtime / "seed-deliveries.json"

    @property
    def quarantined_overrides(self) -> Path:
        return self.root / "overrides.quarantined"

    @property
    def mind(self) -> Path:
        return self.root / "mind"

    @property
    def self_file(self) -> Path:
        return self.mind / "self.txt"

    @property
    def meta_memory(self) -> Path:
        return self.mind / "meta_memory.md"

    @property
    def memory(self) -> Path:
        return self.mind / "memory"

    @property
    def working_context(self) -> Path:
        return self.runtime / "context.jsonl"

    @property
    def streams(self) -> Path:
        return self.runtime / "attention"

    @property
    def interactions(self) -> Path:
        return self.mind / "interactions"

    @property
    def created_tools(self) -> Path:
        return self.mind / "tools"

    @property
    def space(self) -> Path:
        """Agent-owned workspace for projects and artifacts that are not memories."""

        return self.mind / "space"

    @property
    def logs(self) -> Path:
        return self.root / "logs"

    @property
    def lifetime_log(self) -> Path:
        return self.logs / "lifetime.jsonl"

    @property
    def life_loop_log(self) -> Path:
        return self.logs / "life-loop.jsonl"

    @property
    def model_log(self) -> Path:
        return self.logs / "model"

    @property
    def outputs(self) -> Path:
        return self.logs / "outputs"

    @property
    def context_archive(self) -> Path:
        return self.logs / "context"

    @property
    def feature_logs(self) -> Path:
        return self.logs / "features"

    @property
    def feature_summary(self) -> Path:
        return self.feature_logs / "summary.json"

    @property
    def records_lock(self) -> Path:
        return self.runtime / "records.lock"

    def feature_log(self, feature: str) -> Path:
        safe = safe_identifier(feature, label="feature log name")
        return self.feature_logs / f"{safe}.jsonl"

    @property
    def runtime(self) -> Path:
        return self.logs / "runtime"

    @property
    def runtime_state(self) -> Path:
        return self.runtime / "state.json"

    @property
    def initialization_state(self) -> Path:
        return self.runtime / "initialization.json"

    @property
    def pressure_state(self) -> Path:
        return self.runtime / "context_pressure.json"

    @property
    def visual_context(self) -> Path:
        return self.runtime / "visual-context.json"

    @property
    def sleep_state(self) -> Path:
        return self.runtime / "sleep.json"

    @property
    def self_history(self) -> Path:
        return self.runtime / "self-history"

    @property
    def process_lock(self) -> Path:
        return self.runtime / "artificium.lock"

    @property
    def interaction_lock(self) -> Path:
        return self.runtime / "interactions.lock"

    @property
    def scheduler_root(self) -> Path:
        return self.runtime / "scheduler"

    @property
    def scheduler_tasks(self) -> Path:
        return self.scheduler_root / "tasks"

    @property
    def scheduler_lock(self) -> Path:
        return self.scheduler_root / "scheduler.lock"

    @property
    def receipts(self) -> Path:
        return self.runtime / "interaction_receipts"

    @property
    def notification_root(self) -> Path:
        return self.runtime / "notifications"

    @property
    def notifications_new(self) -> Path:
        return self.notification_root / "new"

    @property
    def notifications_processing(self) -> Path:
        return self.notification_root / "processing"

    @property
    def notifications_delivered(self) -> Path:
        return self.notification_root / "delivered"

    @property
    def file_versions(self) -> Path:
        """Backups of prior file content, kept outside ``mind/`` so a mistaken
        overwrite (wrong mode, a partial read mistaken for the whole file,
        etc.) stays recoverable. See :func:`backup_before_overwrite`.
        """

        return self.runtime / "file-versions"

    @property
    def partial_reads_state(self) -> Path:
        """Which paths' most recent ``read_file`` in this runtime returned
        only a partial window, so a later overwrite of that path can warn
        before it clobbers content that was never actually seen. See
        :func:`record_partial_read`.
        """

        return self.runtime / "partial-reads.json"

    def fill_from_seed(self) -> list[str]:
        """Deliver shipped seed files the workspace has never received.

        Each file below ``app/seed/`` maps one-to-one onto
        ``workspace/mind/``.  Delivery is recorded, and a file is delivered at
        most once, so the three cases stay distinct:

        * a workspace that was deleted has no record and is rebuilt whole;
        * a memory the agent chose to discard stays discarded, because it was
          already delivered;
        * a memory a newer application adds arrives, because it never was.

        An existing file the agent has changed is never overwritten, and an
        application without a seed directory is left alone.  An existing file
        still byte-identical to what was shipped (the recorded delivery hash,
        or a known earlier release of that file) is refreshed to the newer
        seed, so an improved tool or harness memory reaches a mind that never
        touched it.  ``self.txt`` and ``meta_memory.md`` are exempt from both
        rules: the harness cannot start without them, so losing one is a
        failure to repair rather than a decision to respect, and they are
        never refreshed.
        """

        if not self.seed.is_dir():
            return []
        raw = read_json(self.seed_deliveries, []) or []
        # Older workspaces recorded names only; newer ones name -> sha256.
        delivered: dict[str, str | None] = (
            {str(k): v for k, v in raw.items()} if isinstance(raw, dict)
            else {str(name): None for name in raw}
        )
        required = {"self.txt", "meta_memory.md"}
        created: list[str] = []
        changed = False
        for source in sorted(self.seed.rglob("*")):
            if not source.is_file() or "__pycache__" in source.parts or source.suffix == ".pyc":
                continue
            name = source.relative_to(self.seed).as_posix()
            destination = self.mind / name
            shipped = sha256_file(source)
            if destination.exists():
                if name in required or not destination.is_file():
                    continue
                current = sha256_file(destination)
                if current == shipped:
                    if delivered.get(name) != shipped:
                        delivered[name] = shipped
                        changed = True
                    continue
                pristine = {delivered.get(name), *_EARLIER_SEED_RELEASES.get(name, ())}
                if current not in pristine:
                    continue
            elif name in delivered and name not in required:
                continue
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, destination)
            delivered[name] = shipped
            created.append(name)
            changed = True
        if changed:
            atomic_write_json(self.seed_deliveries, dict(sorted(delivered.items())))
        return created

    def ensure_layout(self) -> list[str]:
        directories = [
            self.install,
            self.root,
            self.overrides,
            self.prompt_overrides,
            self.code_overrides,
            self.mind,
            self.memory,
            self.streams,
            self.interactions,
            self.created_tools,
            self.space,
            self.logs,
            self.model_log,
            self.outputs,
            self.context_archive,
            self.feature_logs,
            self.runtime,
            self.self_history,
            self.receipts,
            self.scheduler_tasks,
            self.notifications_new,
            self.notifications_processing,
            self.notifications_delivered,
        ]
        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)
        for jsonl in (self.working_context, self.lifetime_log, self.life_loop_log):
            if not jsonl.exists():
                atomic_write_text(jsonl, "")
        return self.fill_from_seed()
