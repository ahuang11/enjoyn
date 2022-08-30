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
import xarray as xr
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

    def size_of(self, file: Union[Path, str]):
        """
        Gets the size of a file in MBs.
        """
        path = Path(file)
        file_size = path.stat().st_size / 1024 / 1024
        print(f"File size of {path.name}: {file_size:.2f} MBs")


class RandomWalkExample(Example):
    """
    An example related to a numpy array of random coordinates.

    Args:
        length: The number of items in the data.
        scratch_directory: The base directory to create the temporary directory
            for intermediary files.
    """

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
        fig = plt.figure(figsize=(8, 8))
        ax = plt.axes()

        x, y = zip(*data_subset)
        ax.plot(x, y)

        title = f"{len(data_subset):04d}"
        ax.set_title(title)

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


class AirTemperatureExample(Example):
    """
    An example related to an xarray Dataset of air temperatures.

    Args:
        length: The number of items in the data.
        scratch_directory: The base directory to create the temporary directory
            for intermediary files.
    """

    length: int = 2920

    def load_data(self) -> xr.Dataset:
        """
        Loads an xarray Dataset.
        """
        ds = xr.tutorial.open_dataset("air_temperature").chunk({"time": 10})
        return ds

    def plot_image(
        self, ds_sel: xr.Dataset, in_memory: bool = False
    ) -> Union[BytesIO, Path]:
        """
        Plots an image from the data subset.

        Args:
            data_subset: The subset dataset; should be shaped (x, y).
            in_memory: If True, save output to `BytesIO`; if False, save to disk.

        Returns:
            The output image as `BytesIO` or `Path`.
        """
        fig = plt.figure(figsize=(12, 8))
        ax = plt.axes()

        img = ax.contourf(
            ds_sel["lon"],
            ds_sel["lat"],
            ds_sel["air"],
            cmap="RdBu_r",
            levels=range(220, 320, 10),
        )
        plt.colorbar(img)

        title = ds_sel["time"].dt.strftime("%H:%MZ %Y-%m-%d").item()
        ax.set_title(title)

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
        ds = self.load_data()
        outputs = [
            self.plot_image(ds.sel(time=time))
            for time in ds["time"].values[: self.length]
        ]
        return outputs
