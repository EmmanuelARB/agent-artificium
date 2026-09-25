"""Artificium-revolution — continual learning and Infinite Attention."""

from . import _bootstrap as _bootstrap
from .version import VERSION

# The workspace may shadow modules of this package.  The overlay is chosen and
# validated before the first submodule import, so `from .runtime import ...`
# below already sees it.
_overlay = _bootstrap.code_overlay(__file__)
if _overlay is not None:
    __path__.insert(0, str(_overlay))

from .interactions import ArtificiumClient
from .runtime import Artificium

__all__ = ["Artificium", "ArtificiumClient", "VERSION"]
