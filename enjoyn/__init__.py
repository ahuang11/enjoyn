from . import _version
from .animator import GifAnimator, Mp4Animator  # noqa
from .preprocessor import (  # noqa
    Preprocessor,
    MatplotlibPreprocessor,
    HoloViewsPreprocessor,
)

__version__ = _version.get_versions()["version"]
