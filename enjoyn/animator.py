import shlex
from abc import ABC
from pathlib import Path
from typing import (
    TYPE_CHECKING,
    Optional,
    Callable,
    Tuple,
    Union,
    List,
    Dict,
    Any,
    Iterable,
)
from typing_extensions import Self
from tempfile import TemporaryDirectory

import uuid
import dask.bag
import dask.delayed
from pydantic import BaseModel
import imageio.v3 as iio
import numpy as np
import pygifsicle
import subprocess

if TYPE_CHECKING:
    from dask.distributed import Client


class Preprocessor(BaseModel):

    func: Callable
    args: Tuple[Any] = ()
    kwds: Optional[Dict[str, Any]] = None


class BaseAnimator(BaseModel, ABC):

    items: Iterable[Any]
    output_path: Union[Path, str]
    preprocessor: Optional[Preprocessor] = None

    imwrite_kwds: Optional[Dict[str, Any]] = None
    scratch_directory: Optional[Union[Path, str]] = None

    _temporary_directory: Optional[Path] = None
    _debug: bool = False

    """
    Base animator class that holds most of the
    common methods that other animators use.

    Attributes:
        ...
    
    Returns:
        ...
    """

    class Config:
        extra = "allow"

    @classmethod
    def from_directory(
        cls,
        directory: Union[Path, str],
        pattern: str = "*.*",
        limit: int = None,
        **source_kwds,
    ) -> Self:
        directory = Path(directory)
        if not directory.exists():  # TODO: move to root validation
            raise ValueError(f"`{directory}` does not exist")
        elif not directory.is_dir():
            raise ValueError(f"`{directory}` must be a directory")
        file_paths = sorted(path for path in directory.glob(pattern) if path.is_file())
        if limit:
            file_paths = file_paths[:limit]
        return cls(items=file_paths, **source_kwds)

    def _serialize_item(self, item: Any) -> np.ndarray:
        if self.preprocessor:
            args = self.preprocessor.args
            kwds = self.preprocessor.kwds or {}
            item = self.preprocessor.func(
                item, *args, **kwds
            )
            valid_types = (Path, str, bytes)
            if not isinstance(item, valid_types):
                callable_name = self.preprocessor.func.__name__
                return_type = type(item).__name__
                valid_types_names = ", ".join(
                    f"`{type_.__name__}`" for type_ in valid_types
                )
                raise TypeError(
                    f"The '{callable_name}' callable returned an object with "
                    f"`{return_type}` type; update '{callable_name}' so that it returns "
                    f"an object with either {valid_types_names} type instead"
                )

        image = iio.imread(item) if not isinstance(item, np.ndarray) else item
        return image

    def _animate_images(self, partitioned_items: Iterable[Any]) -> Path:
        images = [self._serialize_item(item) for item in partitioned_items]
        intermediate_path = (
            self._temporary_directory / f"{uuid.uuid1()}{self.output_extension}"
        )
        imwrite_kwds = self.imwrite_kwds or {}
        iio.imwrite(
            intermediate_path, images, extension=self.output_extension, **imwrite_kwds
        )
        return intermediate_path

    def _concat_animations(self, partitioned_animations) -> Path:
        intermediate_path = (
            self._temporary_directory / f"{uuid.uuid1()}{self.output_extension}"
        )
        return intermediate_path

    @dask.delayed
    def _transfer_output(self, intermediate_path: Path):
        output_path = Path(self.output_path).absolute()
        intermediate_path.rename(output_path)
        return output_path

    def plan(
        self,
        partition_size: Optional[int] = None,
        split_every: int = None,
        display: bool = True,
        **compute_kwargs
    ) -> Path:
        input_bag = dask.bag.from_sequence(self.items, partition_size=partition_size)
        intermediate_path = input_bag.reduction(
            self._animate_images,
            self._concat_animations,
            split_every=split_every,
        )
        output_path = self._transfer_output(intermediate_path)
        if display:
            return output_path.visualize()
        else:
            return output_path

    def compute(
        self,
        partition_size: Optional[int] = None,
        split_every: int = None,
        client: Optional["Client"] = None,
        scheduler: str = "threads",
        **compute_kwds
    ) -> Path:
        with TemporaryDirectory(
            prefix="enjoyn_", dir=self.scratch_directory
        ) as temporary_directory:
            if self._debug:
                temporary_directory = "."
            self._temporary_directory = Path(temporary_directory)
            plan = self.plan(
                partition_size=partition_size, split_every=split_every
            )
            compute_kwds["scheduler"] = scheduler
            if client is not None:
                output_path = client.compute(plan, **compute_kwds).result()
            else:
                output_path = dask.compute(plan, **compute_kwds)[0]
        return output_path


class GifAnimator(BaseAnimator):

    gifsicle_options: Tuple[str] = ("--optimize=2", "--no-conserve-memory")
    output_extension: str = ".gif"

    def _concat_animations(self, partitioned_animations) -> Path:
        intermediate_path = super()._concat_animations(partitioned_animations)
        pygifsicle.gifsicle(
            sources=list(partitioned_animations),
            destination=intermediate_path,
            options=self.gifsicle_options,
        )
        return intermediate_path


class Mp4Animator(BaseAnimator):

    output_extension: str = ".mp4"

    def _concat_animations(self, partitioned_animations) -> Path:
        intermediate_path = super()._concat_animations(partitioned_animations)
        input_path = self._temporary_directory / f"{uuid.uuid1()}.txt"
        input_text = "\n".join(f"file '{animation}'" for animation in partitioned_animations)
        with open(input_path, "w") as f:
            f.write(input_text)
        cmd = shlex.split(f"ffmpeg -f concat -loglevel quiet -safe 0 -i '{input_path}' -c copy '{intermediate_path}'")
        subprocess.run(cmd)
        return intermediate_path
