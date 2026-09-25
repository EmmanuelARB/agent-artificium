from __future__ import annotations

import mimetypes
from pathlib import Path
from typing import Any, Sequence

from .config import Config
from .filesystem import Paths, atomic_write_json, read_json, sortable_id, utc_now
from .records import Records


class VisualContext:
    """Explicit perceptual working set kept separate from textual working context."""

    RETENTIONS = {"once", "persistent"}
    DETAILS = {"auto", "low", "high"}

    def __init__(self, paths: Paths, config: Config, records: Records):
        self.paths = paths
        self.config = config
        self.records = records

    def _load(self) -> list[dict[str, Any]]:
        value = read_json(self.paths.visual_context, {})
        images = value.get("images", []) if isinstance(value, dict) else []
        return [dict(item) for item in images if isinstance(item, dict)]

    def _save(self, images: list[dict[str, Any]]) -> None:
        atomic_write_json(
            self.paths.visual_context,
            {"updated_at": utc_now(), "images": images},
        )

    def list(self) -> list[dict[str, Any]]:
        return self._load()

    def load(
        self,
        paths: Sequence[str],
        *,
        detail: str = "auto",
        retention: str = "once",
    ) -> dict[str, Any]:
        if detail not in self.DETAILS:
            raise ValueError("detail must be auto, low, or high")
        if retention not in self.RETENTIONS:
            raise ValueError("retention must be once or persistent")
        if not paths:
            raise ValueError("paths must contain at least one image")
        if self.config.vision == "no":
            return {
                "status": "vision_unavailable",
                "summary": "the configured engine is not expected to accept images",
                "paths": [str(Path(item).expanduser().resolve()) for item in paths],
            }

        active = self._load()
        by_path = {str(item.get("path")): item for item in active}
        loaded: list[dict[str, Any]] = []
        refreshed: list[dict[str, Any]] = []
        now = utc_now()
        for supplied in paths:
            target = Path(supplied).expanduser().resolve()
            if not target.is_file():
                raise FileNotFoundError(target)
            mime = mimetypes.guess_type(target.name)[0] or "application/octet-stream"
            if not mime.startswith("image/"):
                raise ValueError(f"not a supported image attachment: {target}")
            if target.stat().st_size > self.config.max_image_bytes:
                raise ValueError(f"image exceeds max_image_bytes: {target}")
            key = str(target)
            existing = by_path.get(key)
            if existing is None:
                item = {
                    "id": sortable_id("image_"),
                    "path": key,
                    "mime": mime,
                    "detail": detail,
                    "retention": retention,
                    "loaded_at": now,
                }
                active.append(item)
                by_path[key] = item
                loaded.append(dict(item))
            else:
                existing.update(
                    {
                        "mime": mime,
                        "detail": detail,
                        "retention": retention,
                        "loaded_at": now,
                    }
                )
                refreshed.append(dict(existing))

        self._save(active)
        result = {
            "status": "images_loaded",
            "summary": (
                f"{len(loaded)} image(s) loaded and {len(refreshed)} active image(s) "
                "refreshed without duplication"
            ),
            "retention": retention,
            "loaded": loaded,
            "refreshed": refreshed,
            "active_count": len(active),
            "active_images": [dict(item) for item in active],
        }
        self.records.emit("images_loaded", **result)
        return result

    def release(
        self,
        *,
        paths: Sequence[str] = (),
        image_ids: Sequence[str] = (),
        all_images: bool = False,
        reason: str = "explicit_release",
    ) -> dict[str, Any]:
        if not all_images and not paths and not image_ids:
            raise ValueError("provide paths, image_ids, or all_images=true")
        requested_paths = {
            str(Path(item).expanduser().resolve()) for item in paths if str(item).strip()
        }
        requested_ids = {str(item).strip() for item in image_ids if str(item).strip()}
        active = self._load()
        kept: list[dict[str, Any]] = []
        released: list[dict[str, Any]] = []
        for item in active:
            selected = (
                all_images
                or str(item.get("path")) in requested_paths
                or str(item.get("id")) in requested_ids
            )
            (released if selected else kept).append(item)
        self._save(kept)
        result = {
            "status": "images_released",
            "summary": f"released {len(released)} image(s) from active visual context",
            "reason": reason,
            "released": [dict(item) for item in released],
            "active_count": len(kept),
            "active_images": [dict(item) for item in kept],
        }
        self.records.emit("images_released", **result)
        return result

    def consume_once(self, *, request_id: str) -> dict[str, Any]:
        active = self._load()
        # Disabled images were not sent and must not be consumed by text inference.
        consumed = [item for item in active if item.get("retention") == "once" and self.config.vision != "no"]
        kept = [item for item in active if item not in consumed]
        if consumed:
            self._save(kept)
            self.records.emit(
                "images_consumed",
                request_id=request_id,
                count=len(consumed),
                images=consumed,
                active_count=len(kept),
            )
        return {
            "status": "images_consumed",
            "request_id": request_id,
            "consumed": [dict(item) for item in consumed],
            "active_count": len(kept),
            "active_images": [dict(item) for item in kept],
        }

    def request_message(self) -> dict[str, Any] | None:
        if self.config.vision == "no":
            return None
        active = self._load()
        if not active:
            return None
        summary = [
            {
                "id": item.get("id"),
                "path": item.get("path"),
                "retention": item.get("retention"),
                "detail": item.get("detail"),
            }
            for item in active
        ]
        content: list[dict[str, Any]] = [
            {
                "type": "text",
                "text": (
                    "[ACTIVE VISUAL CONTEXT]\n"
                    "These images were deliberately loaded by Artificium. Images with "
                    "retention `once` are released only after this inference succeeds. "
                    "Images with retention `persistent` are sent again on later "
                    "inferences until `release_images` or working-memory offloading "
                    "releases them. Original files remain durable.\n"
                    f"Active images: {summary}\n"
                    "[END ACTIVE VISUAL CONTEXT]"
                ),
            }
        ]
        content.extend(
            {
                "type": "artificium_image",
                "path": item["path"],
                "mime": item.get("mime"),
                "detail": item.get("detail", "auto"),
                "image_id": item.get("id"),
                "retention": item.get("retention", "once"),
            }
            for item in active
        )
        return {
            # Vision blocks belong to an ordinary user turn in both OpenAI-compatible
            # and Anthropic transports. Keep this independent from the configurable
            # role used for textual runtime notifications.
            "role": "user",
            "content": content,
            "_artificium": {
                "kind": "visual_context",
                "image_ids": [item.get("id") for item in active],
            },
        }
