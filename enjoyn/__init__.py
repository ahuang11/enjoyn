from . import _version
from .animator import GifAnimator, Mp4Animator  # noqa
from .preprocessor import Preprocessor, MatplotlibPreprocessor  # noqa

__version__ = _version.get_versions()["version"]
