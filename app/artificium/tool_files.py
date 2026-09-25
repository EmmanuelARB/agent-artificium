from __future__ import annotations

import mimetypes
import os
import signal
import subprocess
from pathlib import Path
from typing import Any

from .filesystem import (
    atomic_write_text,
    backup_before_overwrite,
    check_shrink_guard,
    clear_partial_read,
    mind_prefix_hint,
    record_partial_read,
    sha256_file,
    utc_now,
)


class FileToolsMixin:
    """File, shell, and image tools mixed into :class:`ToolRegistry`.

    Every method here assumes it is mixed into ``ToolRegistry`` and freely
    uses attributes/helpers defined there (``self._resolve``,
    ``self._resolve_write``, ``self.config``, ``self.paths``, ``self.visual``).
    """

    def list_directory(
        self,
        path: str = "mind",
        depth: int = 1,
        include_hidden: bool = False,
        max_entries: int = 500,
    ) -> dict[str, Any]:
        target = self._resolve(path)
        if not target.is_dir():
            raise NotADirectoryError(
                f"{target} is not a directory.{mind_prefix_hint(self.paths, path)}"
            )
        depth = max(0, min(int(depth), 10))
        maximum = max(1, min(int(max_entries), 5_000))
        entries: list[dict[str, Any]] = []
        root_depth = len(target.parts)
        for child in sorted(target.rglob("*")):
            relative_parts = child.parts[root_depth:]
            if len(relative_parts) > depth:
                continue
            if not include_hidden and any(part.startswith(".") for part in relative_parts):
                continue
            entries.append(
                {
                    "path": str(child),
                    "type": "directory" if child.is_dir() else "file",
                    "size_bytes": child.stat().st_size if child.is_file() else None,
                }
            )
            if len(entries) >= maximum:
                break
        return {
            "status": "ok",
            "summary": f"listed {len(entries)} entries",
            "path": str(target),
            "entries": entries,
            "truncated": len(entries) >= maximum,
        }

    def read_file(
        self,
        path: str,
        start_line: int = 1,
        max_characters: int | None = None,
        start: int | None = None,
        max_chars: int | None = None,
    ) -> dict[str, Any]:
        target = self._resolve(path)
        if not target.is_file():
            raise FileNotFoundError(
                f"{target} does not exist.{mind_prefix_hint(self.paths, path)}"
            )
        size = target.stat().st_size
        explicit_bound = max_characters is not None or max_chars is not None or start is not None
        limit = min(
            int(max_characters or max_chars or self.config.max_direct_read_chars),
            self.config.max_direct_read_chars,
        )
        if not explicit_bound and int(start_line) <= 1 and size > limit:
            # Nothing was actually read; that is even less than a partial
            # window, so a later overwrite should get the same warning.
            record_partial_read(
                self.paths,
                target,
                {"start_byte": 0, "end_byte": 0, "size_bytes": size, "at": utc_now()},
            )
            return {
                "status": "requires_attention",
                "summary": "file is too large for a safe direct read",
                "path": str(target),
                "size_bytes": size,
                "direct_read_limit_characters": limit,
            }
        if start is not None:
            offset = max(0, int(start))
            with target.open("rb") as handle:
                handle.seek(offset)
                raw = handle.read(limit)
                end = handle.tell()
            content = raw.decode("utf-8", errors="replace")
        else:
            with target.open("r", encoding="utf-8", errors="replace") as handle:
                for _ in range(max(0, int(start_line) - 1)):
                    if not handle.readline():
                        break
                offset = handle.tell()
                content = handle.read(limit)
                end = handle.tell()
        complete = offset == 0 and end >= size
        if complete:
            clear_partial_read(self.paths, target)
        else:
            record_partial_read(
                self.paths,
                target,
                {"start_byte": offset, "end_byte": end, "size_bytes": size, "at": utc_now()},
            )
        return {
            "status": "ok",
            "summary": "bounded file content read",
            "path": str(target),
            "start_byte": offset,
            "end_byte": end,
            "size_bytes": size,
            "complete": complete,
            "truncated": end < size,
            "content": content,
        }

    # self.txt and mind/meta_memory.md are agent-authored maps/identity files
    # that are expected to be rewritten wholesale (including shrinking) as
    # part of ordinary upkeep; self.txt is separately versioned by
    # revise_self. The shrink guard exists for content files where a smaller
    # replacement is usually a mistake, not for these two.
    def _shrink_guard_exempt(self, target: Path) -> bool:
        return target in (self.paths.self_file.resolve(), self.paths.meta_memory.resolve())

    def write_file(
        self, path: str, content: str, mode: str = "create", allow_shrink: bool = False
    ) -> dict[str, Any]:
        target = self._resolve_write(path)
        if mode not in {"create", "overwrite", "append"}:
            raise ValueError("mode must be create, overwrite, or append")
        exists = target.exists()
        if mode == "create" and exists:
            size = target.stat().st_size
            raise FileExistsError(
                f"{target} already exists ({size} bytes). Read it fully first, then use "
                'mode="overwrite" to replace it, or mode="append" to add to it.'
            )
        target.parent.mkdir(parents=True, exist_ok=True)
        final = content
        backup_path = None
        if mode == "append" and exists:
            final = target.read_text(encoding="utf-8", errors="replace") + content
        if mode == "overwrite" and exists:
            if not self._shrink_guard_exempt(target):
                check_shrink_guard(
                    paths=self.paths,
                    target=target,
                    existing_size=target.stat().st_size,
                    new_size=len(final.encode("utf-8")),
                    allow_shrink=allow_shrink,
                )
            backup_path = backup_before_overwrite(self.paths, target)
        atomic_write_text(target, final)
        clear_partial_read(self.paths, target)
        result = {
            "status": "written",
            "summary": f"file {mode} complete",
            "path": str(target),
            "size_bytes": target.stat().st_size,
            "sha256": sha256_file(target),
        }
        if backup_path is not None:
            result["backup_path"] = str(backup_path)
        return result

    @staticmethod
    def _decode_shell_output(data: bytes | None) -> str:
        if not data:
            return ""
        return data.decode("utf-8", errors="replace")

    @staticmethod
    def _kill_shell_process_group(process: "subprocess.Popen[bytes]") -> None:
        """Kill the whole session/process group started for a shell command.

        The command runs via ``start_new_session=True`` so its pid is also its
        process group id; killing that group reaches any children it spawned
        (including backgrounded ones), not just the immediate shell process.
        """

        try:
            os.killpg(os.getpgid(process.pid), signal.SIGKILL)
        except (ProcessLookupError, PermissionError, OSError):
            try:
                process.kill()
            except Exception:
                pass

    def run_shell(
        self, command: str, cwd: str | None = None, timeout_seconds: float | None = None
    ) -> dict[str, Any]:
        if not command.strip():
            raise ValueError("command cannot be empty")
        timeout = min(
            float(timeout_seconds or self.config.shell_timeout_seconds),
            self.config.max_shell_timeout_seconds,
        )
        working = self._resolve(cwd) if cwd else self.paths.root
        with subprocess.Popen(
            command,
            cwd=working,
            shell=True,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            start_new_session=True,
        ) as process:
            try:
                raw_stdout, raw_stderr = process.communicate(timeout=timeout)
            except subprocess.TimeoutExpired as exc:
                self._kill_shell_process_group(process)
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    pass
                # Whatever had already been read from the pipes before the
                # timeout fired travels with the exception (``.stdout``/
                # ``.output`` are the same attribute; see cpython's
                # ``subprocess.run``, which relies on the same behavior on
                # POSIX); decode it so the caller is not left blind and
                # cannot be tripped up by binary output.
                exc.stdout = self._decode_shell_output(exc.stdout)
                exc.stderr = self._decode_shell_output(exc.stderr)
                raise
            return {
                "status": "ok" if process.returncode == 0 else "failed",
                "summary": f"shell exited {process.returncode}",
                "command": command,
                "cwd": str(working),
                "returncode": process.returncode,
                "stdout": self._decode_shell_output(raw_stdout),
                "stderr": self._decode_shell_output(raw_stderr),
            }

    def load_images(
        self,
        paths: list[str] | str,
        detail: str = "auto",
        retention: str = "once",
    ) -> dict[str, Any]:
        supplied = [paths] if isinstance(paths, str) else paths
        resolved = [str(self._resolve(path)) for path in supplied]
        return self.visual.load(resolved, detail=detail, retention=retention)

    def list_loaded_images(self) -> dict[str, Any]:
        images = self.visual.list()
        return {
            "status": "ok",
            "summary": f"{len(images)} image(s) are active in visual context",
            "active_count": len(images),
            "active_images": images,
        }

    def release_images(
        self,
        paths: list[str] | str | None = None,
        image_ids: list[str] | str | None = None,
        all_images: bool = False,
    ) -> dict[str, Any]:
        supplied_paths = [paths] if isinstance(paths, str) else (paths or [])
        supplied_ids = [image_ids] if isinstance(image_ids, str) else (image_ids or [])
        resolved = [str(self._resolve(path)) for path in supplied_paths]
        return self.visual.release(
            paths=resolved,
            image_ids=supplied_ids,
            all_images=all_images,
            reason="explicit_tool_release",
        )

    def load_attachment(
        self,
        path: str,
        detail: str = "auto",
        retention: str = "once",
    ) -> dict[str, Any]:
        target = self._resolve(path)
        if not target.is_file():
            raise FileNotFoundError(
                f"{target} does not exist.{mind_prefix_hint(self.paths, path)}"
            )
        mime = mimetypes.guess_type(target.name)[0] or "application/octet-stream"
        if mime.startswith("image/"):
            result = self.visual.load(
                [str(target)], detail=detail, retention=retention
            )
            result["compatibility_operation"] = "load_attachment"
            return result
        if mime.startswith("text/") or target.suffix.lower() in {".json", ".md", ".txt", ".py"}:
            return self.read_file(str(target))
        return {
            "status": "conversion_required",
            "summary": "attachment is durable but not directly supported by this engine adapter",
            "path": str(target),
            "mime": mime,
            "size_bytes": target.stat().st_size,
        }
