from abc import ABC
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Optional, Callable, Tuple, Union, List, Dict, Any, Iterable

import uuid
import dask.bag
import dask.delayed
from pydantic import BaseModel
import imageio.v3 as iio
import numpy as np
import pygifsicle


class Preprocessor(BaseModel):

    func: Callable
    args: Tuple[Any] = ()
    kwds: Optional[Dict[str, Any]] = None


class Animator(BaseModel, ABC):

    items: Iterable[Any]
    output: Union[Path, str]
    preprocessor: Optional[Preprocessor] = None  # TODO: move to animate?
    partition_size: Optional[int] = None
    scheduler: str = "threads"
    # TODO: extension?

    @classmethod
    def from_directory(cls, directory: Union[Path, str], pattern: str = "*.*", **source_kwds):
        directory = Path(directory)
        if not directory.exists():  # TODO: move to root validation
            raise ValueError(f"`{directory}` does not exist")
        elif not directory.is_dir():
            raise ValueError(f"`{directory}` must be a directory")
        file_paths = sorted(path for path in directory.glob(pattern) if path.is_file())
        return cls(items=file_paths, **source_kwds)

    def _serialize_item(self, item: Any):
        if self.preprocessor:
            item = self.preprocessor.func(
                item, self.preprocessor.args, **self.preprocessor.kwds
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

        image = iio.imread(item, extension=".png") if not isinstance(item, np.ndarray) else item
        return image

    def _process_partition(self, partitioned_items: Iterable[Any], temporary_directory: Path):
        images = [self._serialize_item(item) for item in partitioned_items]
        temporary_path = temporary_directory / f"{uuid.uuid1()}.gif"
        iio.imwrite(temporary_path, images)
        return temporary_path

    def animate(self):
        ...


class GifAnimator(Animator):

    gifsicle_options: Tuple[str] = ("--optimize=2", "--no-conserve-memory")

    def animate(self):
        input_bag = dask.bag.from_sequence(
            self.items,
            partition_size=self.partition_size
        )
        with TemporaryDirectory() as temporary_directory:
            temporary_directory = Path(temporary_directory)
            partitioned_outputs = input_bag.map_partitions(
                self._process_partition,
                temporary_directory=temporary_directory
            ).compute(scheduler=self.scheduler)
            pygifsicle.gifsicle(
                sources=partitioned_outputs,
                destination=self.output,
                options=self.gifsicle_options,
            )
        return self.output

class VideoAnimator(Animator):
    ...
