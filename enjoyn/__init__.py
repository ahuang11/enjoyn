from . import _version
from .animator import GifAnimator, Mp4Animator  # noqa
from .preprocessor import Preprocessor  # noqa

__version__ = _version.get_versions()["version"]
