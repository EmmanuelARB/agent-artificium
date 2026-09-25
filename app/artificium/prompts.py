from __future__ import annotations

import hashlib
import json
import re
import tomllib
from pathlib import Path
from typing import Any

from .filesystem import Paths


PLACEHOLDER = re.compile(r"\{\{([A-Za-z0-9_]+)\}\}")


def _display(value: Any) -> str:
    if value is None:
        return "none"
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (dict, list, tuple)):
        return json.dumps(value, ensure_ascii=False, separators=(",", ":"))
    return str(value)


class PromptPack:
    """Loads every model-visible behavioral prompt from reviewable files."""

    def __init__(self, paths: Paths):
        self.paths = paths
        if not paths.prompt_manifest.is_file():
            raise FileNotFoundError(f"Prompt manifest is missing: {paths.prompt_manifest}")
        self.manifest = tomllib.loads(paths.prompt_manifest.read_text(encoding="utf-8"))
        self.validate()

    @property
    def version(self) -> str:
        return str(self.manifest.get("version") or "unknown")

    def _path(self, relative: str) -> Path:
        """Resolve one prompt, letting the workspace shadow the shipped file.

        ``workspace/overrides/prompts/<relative>`` wins over
        ``app/prompts/<relative>``.  Both sides are confined to
        their own directory, so a manifest entry can never escape either tree.
        """

        override = (self.paths.prompt_overrides / relative).resolve()
        override.relative_to(self.paths.prompt_overrides.resolve())
        if override.is_file():
            return override
        candidate = (self.paths.prompts / relative).resolve()
        candidate.relative_to(self.paths.prompts.resolve())
        return candidate

    def overridden(self) -> list[str]:
        """Manifest entries currently served from the workspace overlay."""

        shipped = self.paths.prompts.resolve()
        return [name for name in self._references()
                if not self._path(name).is_relative_to(shipped)]

    def _read(self, relative: str) -> str:
        return self._path(relative).read_text(encoding="utf-8", errors="replace").strip()

    def _references(self) -> list[str]:
        result: list[str] = []
        result.extend(str(item) for item in self.manifest.get("always", []))
        result.extend(str(item) for item in self.manifest.get("tool_catalog", []))
        for section in ("runtime", "events"):
            value = self.manifest.get(section, {})
            if isinstance(value, dict):
                result.extend(str(item) for item in value.values())
        return result

    def validate(self) -> None:
        missing = [item for item in self._references() if not self._path(item).is_file()]
        if missing:
            raise FileNotFoundError("Prompt manifest references missing files: " + ", ".join(missing))

    def render_text(self, text: str, values: dict[str, Any] | None = None) -> str:
        supplied = values or {}
        missing = sorted({name for name in PLACEHOLDER.findall(text) if name not in supplied})
        if missing:
            raise KeyError("Missing prompt values: " + ", ".join(missing))
        return PLACEHOLDER.sub(
            lambda match: _display(supplied[match.group(1)]),
            text,
        )

    def always(self) -> str:
        return "\n\n---\n\n".join(self._read(item) for item in self.manifest["always"])

    def tool_catalog(self) -> str:
        return "\n\n---\n\n".join(
            self._read(item) for item in self.manifest.get("tool_catalog", [])
        )

    def runtime(self, name: str, **values: Any) -> str:
        relative = self.manifest["runtime"][name]
        return self.render_text(self._read(relative), values)

    def event(self, name: str, **values: Any) -> str:
        relative = self.manifest["events"][name]
        return self.render_text(self._read(relative), values)

    def fingerprints(self) -> dict[str, str]:
        result: dict[str, str] = {}
        for relative in self._references():
            content = self._path(relative).read_bytes()
            result[relative] = hashlib.sha256(content).hexdigest()
        return result
