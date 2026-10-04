"""The application version, read from the prompt manifest.

The release and the prompt pack are versioned together, so the manifest's
``version = "Artificium-<codename>-<release>"`` is the one place a release
number is written.
"""

import re
import tomllib
from pathlib import Path

_MANIFEST = Path(__file__).resolve().parents[1] / "prompts" / "manifest.toml"


def _read() -> tuple[str, str]:
    try:
        declared = tomllib.loads(_MANIFEST.read_text(encoding="utf-8")).get("version")
    except (OSError, tomllib.TOMLDecodeError):
        declared = None
    match = re.fullmatch(r"Artificium-([A-Za-z0-9]+)-(\d+(?:\.\d+)*)", str(declared or ""))
    return (match.group(2), match.group(1)) if match else ("unknown", "revolution")


RELEASE, CODENAME = _read()
VERSION = f"{RELEASE}-{CODENAME}"
USER_AGENT = f"Artificium-{CODENAME}/{RELEASE}"
