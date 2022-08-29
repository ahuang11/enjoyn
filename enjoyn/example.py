"""
This module contains examples that loads data and generates images.
"""

import timeit
from contextlib import contextmanager
from io import BytesIO
from pathlib import Path
from shutil import rmtree
from tempfile import mkdtemp
from typing import List, Optional, Union
from uuid import uuid1

import matplotlib.pyplot as plt
import numpy as np
from pydantic import BaseModel, Extra

np.random.seed(20280808)


class Example(BaseModel, extra=Extra.allow):
    """
    The base example class containing most of the common inputs and methods used in
    other examples inheriting from this. Note, this should not to be used directly.

    Args:
        length: The number of items in the data.
        scratch_directory: The base directory to create the temporary directory
            for intermediary files.
    """

    length: int = 1000
    scratch_directory: Optional[Union[Path, str]] = None

    _temporary_directory: Path = None

    def __init__(self, **data):
        super().__init__(**data)
        self._temporary_directory = Path(
            mkdtemp(prefix="enjoyn_", dir=self.scratch_directory)
        )

    @contextmanager
    def time_run(self):
        """
        A context manager for tracking and printing the runtime.
        """
        start = timeit.default_timer()
        yield
        stop = timeit.default_timer()
        print(f"Runtime: {stop - start} seconds")

    def cleanup_images(self):
        """
        Deletes the temporary directory.
        """
        rmtree(self._temporary_directory)


class RandomWalkExample(Example):
    def load_data(self) -> np.ndarray:
        """
        Loads a `(self.length, 2)` shaped array.
        """
        start = np.random.random(2)
        steps = np.random.uniform(-0.2, 0.2, size=(self.length, 2))
        data = start + np.cumsum(steps, axis=0)
        return data

    def plot_image(
        self, data_subset: np.ndarray, in_memory: bool = False
    ) -> Union[BytesIO, Path]:
        """
        Plots an image from the data subset.

        Args:
            data_subset: The subset data array; should be shaped (n, 2).
            in_memory: If True, save output to `BytesIO`; if False, save to disk.

        Returns:
            The output image as `BytesIO` or `Path`.
        """
        x, y = zip(*data_subset)
        fig = plt.figure(figsize=(8, 8))
        ax = plt.axes()
        ax.plot(x, y)

        if in_memory:
            output = BytesIO()
        else:
            output = self._temporary_directory / f"{uuid1()}.png"
        fig.savefig(output, transparent=False, facecolor="white")

        plt.close()
        return output

    def output_images(self) -> List[Union[BytesIO, Path]]:
        """
        Outputs a list of images as `BytesIO` or `Path`.
        """
        data = self.load_data()
        outputs = [self.plot_image(data[:i]) for i in np.arange(1, len(data) + 1)]
        return outputs
