#!/usr/bin/env python3
"""Application launcher and import-compatible source-tree shim.

This file sits at the root of the project.  Run it as a script, or add the
project root (or ``app/``) to ``sys.path`` and ``import artificium`` to reach
the same harness from a client.  Both routes must agree about the workspace's code
overlay, so the shim applies it exactly as ``artificium/__init__.py`` does.
"""

import importlib.util
from pathlib import Path
import sys


CODE_ROOT = Path(__file__).resolve().parent / "app"
PACKAGE_ROOT = CODE_ROOT / "artificium"


if __name__ == "__main__":
    sys.path.insert(0, str(CODE_ROOT))
    from artificium.cli import main

    raise SystemExit(main())


def _code_overlay() -> str | None:
    """Ask the package's bootstrap, without importing the package itself."""

    spec = importlib.util.spec_from_file_location(
        "_artificium_shim_bootstrap", PACKAGE_ROOT / "_bootstrap.py"
    )
    if spec is None or spec.loader is None:
        return None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    overlay = module.code_overlay(PACKAGE_ROOT / "__init__.py")
    return None if overlay is None else str(overlay)


__path__ = [str(PACKAGE_ROOT)]
_overlay = _code_overlay()
if _overlay is not None:
    __path__.insert(0, _overlay)

from artificium.version import VERSION  # noqa: E402
from artificium.interactions import ArtificiumClient  # noqa: E402
from artificium.runtime import Artificium  # noqa: E402

__all__ = ["Artificium", "ArtificiumClient", "VERSION"]
