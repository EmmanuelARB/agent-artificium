"""Load executable, agent-editable tools from ``mind/tools``."""

from __future__ import annotations

import hashlib
import importlib.util
import sys
from pathlib import Path
from typing import Any

from .filesystem import Paths


def load_mind_tool(paths: Paths, relative_path: str, symbol: str) -> Any:
    """Load one named symbol from an executable module below ``mind/tools``.

    The loader is the small architectural seam; the tool implementation remains
    in the mutable mind.  Tools are loaded once per runtime start, so edits take
    effect after Artificium is restarted.
    """

    root = paths.created_tools.resolve()
    source = (root / relative_path).resolve()
    source.relative_to(root)
    if not source.is_file():
        raise FileNotFoundError(
            f"Required mind tool is missing: {source}. Restore it from the release "
            "archive or provide a compatible replacement."
        )
    digest = hashlib.sha256(str(source).encode("utf-8")).hexdigest()[:16]
    module_name = f"_artificium_mind_tool_{source.stem}_{digest}"
    spec = importlib.util.spec_from_file_location(module_name, source)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load mind tool module: {source}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    try:
        spec.loader.exec_module(module)
    except Exception:
        sys.modules.pop(module_name, None)
        raise
    try:
        return getattr(module, symbol)
    except AttributeError as exc:
        raise ImportError(f"Mind tool {source} does not define {symbol}") from exc
