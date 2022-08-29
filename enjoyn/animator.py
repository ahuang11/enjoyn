"""
This module contains animators that join images into the desired animation format.
"""

import inspect
import shlex
import subprocess
import uuid
from io import BytesIO
from pathlib import Path
from tempfile import TemporaryDirectory
from types import GeneratorType
from typing import TYPE_CHECKING, Any, Callable, Dict, Iterable, Optional, Tuple, Union

import dask.bag
import dask.delayed
import dask.diagnostics
import imageio.v3 as iio
import numpy as np
import pygifsicle
from pydantic import BaseModel, Extra, validator

if TYPE_CHECKING:
    from dask.distributed import Client
    from typing_extensions import Self

from .preprocessor import Preprocessor


class BaseAnimator(BaseModel, extra=Extra.allow):
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
    """

    items: Iterable[Any]
    output_path: Union[Path, str]
    preprocessor: Optional[Union[Callable, Preprocessor]] = None

    imwrite_kwds: Optional[Dict[str, Any]] = None
    scratch_directory: Optional[Union[Path, str]] = None

    _output_extension: Optional[str] = None
    _temporary_directory: Optional[Path] = None
    _debug: bool = False

    @validator("items", pre=True, always=True)
    def _disallow_items_generator(cls, value):
        """
        Prevent `items` from being a `generator` type.
        """
        if isinstance(value, GeneratorType) or inspect.isgeneratorfunction(value):
            raise ValueError("Input `items` cannot be a generator")
        return value

    @validator("items", pre=True, always=True)
    def _check_items_length(cls, value):
        """
        Prevent having only a single object in `items`.
        """
        if len(value) == 1:
            raise ValueError("Must have more than one object in `items`")
        return value

    @validator("output_path", pre=True, always=True)
    def _check_extension_matches(cls, value):
        """
        Prevent having only a single object in `items`.
        """
        suffix = Path(value).suffix
        if suffix != cls._output_extension:
            raise ValueError(
                f"The output path must end in '{cls._output_extension}' "
                f"to use {cls.__name__}, but got '{suffix}'"
            )
        return value

    @validator("preprocessor", pre=True)
    def _serialize_callable(cls, value):
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
        limit: int = None,
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
            args = self.preprocessor.args
            kwds = self.preprocessor.kwds or {}
            item = self.preprocessor.func(item, *args, **kwds)
            valid_types = (Path, str, BytesIO, bytes, np.ndarray)
            if not isinstance(item, valid_types):
                callable_name = self.preprocessor.func.__name__
                return_type = type(item).__name__
                valid_types_names = ", ".join(
                    f"`{type_.__name__}`" for type_ in valid_types
                )
                raise TypeError(
                    f"The '{callable_name}' callable returned an object with "
                    f"`{return_type}` type; update '{callable_name}' so the "
                    f"returned object is either {valid_types_names} type"
                )

        image = iio.imread(item) if not isinstance(item, np.ndarray) else item
        return image

    def _animate_images(self, partitioned_items: Iterable[Any]) -> Path:
        """
        Serializes items in the partition and creates an incomplete animation.
        """
        images = [self._serialize_item(item) for item in partitioned_items]
        try:
            intermediate_path = (
                self._temporary_directory / f"{uuid.uuid1()}{self._output_extension}"
            )
            imwrite_kwds = self.imwrite_kwds or {}
            if "loop" not in imwrite_kwds:
                imwrite_kwds["loop"] = 0
            iio.imwrite(
                intermediate_path,
                images,
                extension=self._output_extension,
                **imwrite_kwds,
            )
        except (TypeError, FileNotFoundError) as exc:
            raise RuntimeError(
                "Use the built-in `compute` method instead; the `plan` method "
                "does not have the temporary directory set internally yet"
            ) from exc
        return intermediate_path

    def _concat_animations(self, partitioned_animations) -> Path:
        """
        Concatenates the incomplete animations to create a more complete animation.
        """
        intermediate_path = (
            self._temporary_directory / f"{uuid.uuid1()}{self._output_extension}"
        )
        return intermediate_path

    @dask.delayed
    def _transfer_output(self, intermediate_path: Path) -> Path:
        """
        Transfers the final animation from a temporary
        directory to the desired output path.
        """
        output_path = Path(self.output_path).absolute()
        intermediate_path.rename(output_path)
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
            visualize: Returns a visual of how the items are delegated if True;
                otherwise returns a Delayed object.
            **compute_kwds: Not used for anything in `plan`; exists so it's
                straightforward to swap `plan` out for `compute` when ready.

        Returns:
            A visualization if `visualize=True`, otherwise dask.delayed object.
        """
        input_bag = dask.bag.from_sequence(self.items, partition_size=partition_size)
        intermediate_path = input_bag.reduction(
            self._animate_images,
            self._concat_animations,
            split_every=split_every,
        )
        output_path = self._transfer_output(intermediate_path)

        if visualize:
            return output_path.visualize()
        else:
            return output_path

    def compute(
        self,
        partition_size: Optional[int] = None,
        split_every: int = None,
        client: Optional["Client"] = None,
        scheduler: Optional[str] = None,
        **compute_kwds: Dict[str, Any],
    ) -> Path:
        """
        Execute the plan to create the animation, partitioning items across workers,
        applying the preprocessor, if any, serializing the items into an incomplete
        animation, and progressively joining those animations into the final animation.

        Args:
            partition_size: The number of items per partition to pass to workers.
            split_every: The number of partitions per group while reducing.
            client: If a distributed client is not provided, will use the local
                client, which has limited options.
            scheduler: Whether to use `threads` or `processes` workers.
            **compute_kwds: Additional keywords to pass to `dask.compute`,
                or if `client` is provided, `client.compute`.

        Returns:
            The path to the output animation.
        """
        with TemporaryDirectory(
            prefix="enjoyn_", dir=self.scratch_directory
        ) as temporary_directory:
            if self._debug:
                temporary_directory = "enjoyn_debug_workspace"
            self._temporary_directory = Path(temporary_directory)
            self._temporary_directory.mkdir(exist_ok=True)

            plan = self.plan(
                partition_size=partition_size, split_every=split_every, visualize=False
            )

            if scheduler is None:
                scheduler = "processes" if self.preprocessor else "threads"
            compute_kwds["scheduler"] = scheduler

            if client is not None:
                output_path = client.compute(plan, **compute_kwds).result()
            else:
                with dask.diagnostics.ProgressBar():
                    output_path = dask.compute(plan, **compute_kwds)[0]

        return output_path


class GifAnimator(BaseAnimator):
    """
    Used for animating images into a GIF animation.

    Args:
        gifsicle_options: A tuple of options to pass to `gifsicle`; see
            the [`gifsicle` manual](https://www.lcdf.org/gifsicle/man.html)
            for a full list of available options.
    """

    gifsicle_options: Tuple[str] = (
        "--optimize=2",
        "--no-conserve-memory",
        "--no-warnings",
    )

    _output_extension: str = ".gif"

    def _concat_animations(self, partitioned_animations) -> Path:
        """
        Concatenates the incomplete animations to create a more complete animation.
        """
        intermediate_path = super()._concat_animations(partitioned_animations)
        pygifsicle.gifsicle(
            sources=list(partitioned_animations),
            destination=intermediate_path,
            options=self.gifsicle_options,
        )
        return intermediate_path


class Mp4Animator(BaseAnimator):
    """
    Used for animating images into a GIF animation.

    Args:
        ffmpeg_options: A tuple of options to pass to `ffmpeg`; see
            the [`ffmpeg` manual](https://ffmpeg.org/ffmpeg.html#Options)
            for a full list of available options.
    """

    ffmpeg_options: Tuple[str] = "-loglevel quiet"

    _output_extension: str = ".mp4"

    def _concat_animations(self, partitioned_animations) -> Path:
        """
        Concatenates the incomplete animations to create a more complete animation.
        """
        intermediate_path = super()._concat_animations(partitioned_animations)
        input_path = self._temporary_directory / f"{uuid.uuid1()}.txt"
        input_text = "\n".join(
            f"file '{animation}'" for animation in partitioned_animations
        )
        with open(input_path, "w") as f:
            f.write(input_text)

        ffmpeg_options = " ".join(self.ffmpeg_options)
        cmd = shlex.split(
            f"ffmpeg -f concat {ffmpeg_options} "
            f"-i '{input_path}' -c copy '{intermediate_path}'"
        )
        subprocess.run(cmd)
        return intermediate_path
