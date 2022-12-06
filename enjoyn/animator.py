"""
This module contains animators that join images into the desired animation format.
"""

import itertools
import shlex
import shutil
import subprocess
from abc import ABC, abstractmethod
from contextlib import contextmanager
from pathlib import Path
from tempfile import NamedTemporaryFile, TemporaryDirectory, mkstemp
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Union

import dask.bag
import dask.delayed
import dask.diagnostics
import imageio.v3 as iio
import numpy as np

try:
    from IPython.display import Image, Video
except ImportError:  # pragma: no cover
    Image = Video = None
try:
    import pygifsicle
except ImportError:  # pragma: no cover
    pygifsicle = None
try:
    import imageio_ffmpeg
except ImportError:  # pragma: no cover
    imageio_ffmpeg = None

from pydantic import BaseModel, Field, PrivateAttr, root_validator, validator

if TYPE_CHECKING:  # pragma: no cover
    from dask.distributed import Client
    from typing_extensions import Self

from .preprocessor import Preprocessor


# Do not converts Args section to Attributes
# for it to show up neatly in output docs.
class BaseAnimator(BaseModel, ABC):
    """
    The base animator containing most of the common inputs and methods used in
    other animators inheriting from this. Note, this should not to be used directly.

    Args:
        items: The items to animate; can be file names, bytes, numpy arrays, or
            anything that can be read with `imageio.imread`. If the `preprocessor` is
            provided, then the items can be anything the preprocessor function accepts.
        output_path: The path to save the output animation to.
        preprocessor: The preprocessor to apply to each item. More info can be
            found within the :meth:`enjoyn.Preprocessor` model's docstring.
        imwrite_kwds: Additional keywords to pass to `imageio.imwrite`.
        scratch_directory: The base directory to create the temporary directory
            for intermediary files.
        show_output: Whether to display the output inline; only available in an
            IPython environment.
    """

    items: List[Any] = Field(min_items=2)
    output_path: Path
    preprocessor: Optional[Union[Preprocessor, Callable]] = None

    imwrite_kwds: Optional[Dict[str, Any]] = None
    scratch_directory: Optional[Path] = None

    show_output: bool = True

    _output_extension: Optional[str] = PrivateAttr(None)
    _temporary_directory: Optional[str] = PrivateAttr(None)

    @root_validator(pre=True)
    @abstractmethod
    def _plugin_installed(cls, values):  # pragma: no cover
        """
        Check whether required libraries are installed.
        """
        pass

    @validator("items", pre=True)
    def _serialize_array(cls, value) -> List:
        """
        Serialize anything like `np.array` into a `list`.
        """
        return list(value)

    @validator("preprocessor", pre=True)
    def _serialize_callable(cls, value) -> Preprocessor:
        """
        Serialize callable into `Preprocessor`.
        """
        if callable(value):
            value = Preprocessor(func=value)
        return value

    @classmethod
    def from_directory(
        cls,
        directory: Union[Path, str],
        pattern: str = "*.*",
        limit: Optional[int] = None,
        **animator_kwds,
    ) -> "Self":
        """
        Searches a directory for file names that match the pattern and uses
        them as `items` in the animator.

        Args:
            directory: The directory to retrieve the file names.
            pattern: The pattern to subset the file names within the directory.
            limit: The maximum number of file names to use.
            **animator_kwds: Additional keywords to pass to the animator.

        Returns:
            An instantiated animator class.
        """
        directory = Path(directory)

        if not directory.exists():
            raise ValueError(f"`{directory}` does not exist")
        elif not directory.is_dir():
            raise ValueError(f"`{directory}` must be a directory")

        file_paths = sorted(path for path in directory.glob(pattern) if path.is_file())
        if limit:
            file_paths = file_paths[:limit]
        return cls(items=file_paths, **animator_kwds)

    def _serialize_item(self, item: Any) -> np.ndarray:
        """
        Applies the preprocessor to the item and serialize as a `np.array`.
        """
        if self.preprocessor:
            item = self.preprocessor.apply_on(item)
        image = iio.imread(item) if not isinstance(item, np.ndarray) else item
        if hasattr(item, "close"):
            item.close()
        return image

    def _create_temporary_path(self) -> Path:
        temporary_path = Path(
            mkstemp(
                prefix="enjoyn_",
                suffix=self._output_extension,
                dir=self._temporary_directory,
            )[1]
        )
        return temporary_path

    def _animate_images(self, partitioned_items: List[Any]) -> Path:
        """
        Serializes items in the partition and creates an incomplete animation.
        """
        if self._temporary_directory is None:
            raise RuntimeError(
                "Use the built-in `compute` method instead; the `plan` method "
                "does not have the temporary directory set internally yet."
            )

        images = [self._serialize_item(item) for item in partitioned_items]
        temporary_path = self._create_temporary_path()
        imwrite_kwds = self.imwrite_kwds or {}
        iio.imwrite(
            temporary_path,
            images,
            extension=self._output_extension,
            **imwrite_kwds,
        )
        return temporary_path

    @abstractmethod
    def _concat_animations(self, partitioned_animations: List[Path]) -> Path:
        """
        Concatenates the incomplete animations to create a more complete animation.
        """
        pass

    @dask.delayed
    def _transfer_output(self, temporary_path: Path) -> Path:
        """
        Transfers the final animation from a temporary
        directory to the desired output path.
        """
        output_path = self.output_path.absolute()
        shutil.move(temporary_path, output_path)
        return output_path

    def plan(
        self,
        partition_size: Optional[int] = None,
        split_every: int = None,
        visualize: bool = True,
        **compute_kwds: Dict[str, Any],  # noqa
    ) -> dask.delayed:
        """
        Assemble the plan to create the animation, partitioning items across workers,
        applying the preprocessor, if any, serializing the items into an incomplete
        animation, and progressively joining those animations into the final animation.

        Args:
            partition_size: The number of items per partition to pass to workers.
            split_every: The number of partitions per group while reducing.
            visualize: Returns a visual of how the items are delegated if True
                which could be useful when using a Jupyter notebook; otherwise,
                returns a Delayed object.
            **compute_kwds: Not used for anything in `plan`; exists so it's
                straightforward to swap `plan` out for `compute` when ready.

        Returns:
            A visualization if `visualize=True`, otherwise dask.delayed object.
        """
        extension = self.output_path.suffix.lower()
        if extension != self._output_extension:
            raise ValueError(
                f"The output path must end in '{self._output_extension}' "
                f"to use {self.__class__.__name__}, but got '{extension}'"
            )

        input_bag = dask.bag.from_sequence(self.items, partition_size=partition_size)
        temporary_file = input_bag.reduction(
            self._animate_images,
            self._concat_animations,
            split_every=split_every,
        )
        output_path = self._transfer_output(temporary_file)

        if visualize:
            return output_path.visualize()
        else:
            return output_path

    @staticmethod
    @contextmanager
    def _display_progress_bar(show_progress):
        """
        Toggles displaying a progress bar.
        """
        if show_progress:
            with dask.diagnostics.ProgressBar() as progress_bar:
                yield progress_bar
        else:
            yield

    def compute(
        self,
        partition_size: Optional[int] = None,
        split_every: Optional[int] = None,
        client: Optional["Client"] = None,
        scheduler: Optional[str] = None,
        show_progress: bool = True,
        **compute_kwds: Dict[str, Any],
    ) -> Union[Image, Video, Path]:
        """
        Execute the plan to create the animation, partitioning items across workers,
        applying the preprocessor, if any, serializing the items into an incomplete
        animation, and progressively joining those animations into the final animation.

        Args:
            partition_size: The number of items per partition to pass to workers.
            split_every: The number of partitions per group while reducing.
            client: If a distributed client is not provided, will use the local
                client, which has limited options.
            scheduler: Whether to use `threads` or `processes` workers; if unspecified,
                defaults to `processes` if a `preprocessor` is provided, else `threads`.
            show_progress: Whether to display the progress bar; if `client` is provided,
                the progress bar will not show.
            **compute_kwds: Additional keywords to pass to `dask.compute`,
                or if `client` is provided, `client.compute`.

        Returns:
            The path to the output animation.
        """
        plan = self.plan(
            partition_size=partition_size, split_every=split_every, visualize=False
        )

        if scheduler is None:
            scheduler = "processes" if self.preprocessor else "threads"
        compute_kwds["scheduler"] = scheduler

        with TemporaryDirectory(prefix="enjoyn_") as self._temporary_directory:
            if client is not None:
                output_path = client.compute(plan, **compute_kwds).result()
            else:
                with self._display_progress_bar(show_progress):
                    output_path = dask.compute(plan, **compute_kwds)[0]

        if self.show_output and (Image or Video):
            try:
                return Image(output_path)
            except ValueError:
                return Video(output_path)
        else:
            return output_path


class GifAnimator(BaseAnimator):
    """
    Used for animating images into a GIF animation.

    Args:
        gifsicle_options: A tuple of options to pass to `gifsicle`; see
            the [`gifsicle` manual](https://www.lcdf.org/gifsicle/man.html)
            for a full list of available options.
    """

    output_path: Path = Path("enjoyn.gif")
    gifsicle_options: List[str] = (
        "--optimize=2",
        "--loopcount=0",
        "--no-warnings",
        "--no-conserve-memory",
    )

    _output_extension: str = PrivateAttr(".gif")

    @root_validator(pre=True)
    def _plugin_installed(cls, values):  # pragma: no cover
        """
        Check whether required libraries are installed.
        """
        if pygifsicle is None:
            raise ImportError(
                "Ensure pygifsicle is installed with " "`pip install -U pygifsicle`"
            )
        if not shutil.which("gifsicle"):
            raise ImportError(
                "Ensure gifsicle is installed with the equivalent of "
                "`conda install -c conda-forge gifsicle`; "
                "visit the docs for other installation methods"
            )
        return values

    def _concat_animations(self, partitioned_animations: itertools.chain) -> Path:
        """
        Concatenates the incomplete animations to create a more complete animation.
        """
        temporary_path = self._create_temporary_path()
        partitioned_animations = list(partitioned_animations)
        pygifsicle.gifsicle(
            sources=list(partitioned_animations),
            destination=temporary_path,
            options=self.gifsicle_options,
        )
        if not temporary_path.stat().st_size:
            raise RuntimeError(
                "gifsicle failed somewhere; ensure that the inputs, "
                "`items`, `output_path`, `gifsicle_options` are valid"
            )
        return temporary_path


class Mp4Animator(BaseAnimator):
    """
    Used for animating images into a MP4 animation.

    Args:
        ffmpeg_options: A tuple of options to pass to `ffmpeg`; see
            the [`ffmpeg` manual](https://ffmpeg.org/ffmpeg.html#Options)
            for a full list of available options.
    """

    output_path: Path = Path("enjoyn.mp4")
    ffmpeg_options: List[str] = ("-loglevel warning",)

    _output_extension: str = PrivateAttr(".mp4")

    @root_validator(pre=True)
    def _plugin_installed(cls, values):  # pragma: no cover
        """
        Check whether required libraries are installed.
        """
        if imageio_ffmpeg is None:
            raise ImportError(
                "Ensure imageio_ffmpeg is installed with "
                "`pip install -U imageio_ffmpeg`"
            )
        if not shutil.which("ffmpeg"):
            raise ImportError(
                "Ensure ffmpeg is installed with the equivalent of "
                "`conda install -c conda-forge ffmpeg`; "
                "visit the docs for other installation methods"
            )
        return values

    def _concat_animations(self, partitioned_animations: itertools.chain) -> Path:
        """
        Concatenates the incomplete animations to create a more complete animation.
        """
        temporary_path = self._create_temporary_path()
        temporary_input_file = self._create_temporary_path()
        with NamedTemporaryFile(suffix=".txt") as temporary_input_file:
            input_text = "\n".join(
                f"file '{animation}'" for animation in partitioned_animations
            )
            with open(temporary_input_file.name, "w") as f:
                f.write(input_text)

            ffmpeg_options = " ".join(self.ffmpeg_options)
            cmd = shlex.split(
                f"ffmpeg -y -f concat {ffmpeg_options} -safe 0 "
                f"-i '{temporary_input_file.name}' -c copy '{temporary_path}'"
            )
            run = subprocess.run(cmd, capture_output=True)
            if run.returncode != 0:
                raise RuntimeError(f"{run.stderr.decode()}")
        return temporary_path
