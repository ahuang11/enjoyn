from io import BytesIO
from pathlib import Path

import numpy as np
import pytest
import xarray as xr

from enjoyn.example import AirTemperatureExample, Example, RandomWalkExample


class StandardExampleSuite:
    @pytest.fixture
    def example(self, example_class):
        _example = example_class(length=8)
        try:
            yield _example
        finally:
            _example.cleanup_images()


class TestExample(StandardExampleSuite):
    @pytest.fixture
    def example_class(self):
        return Example

    def test_instantiation(self):
        example = Example()
        actual = example._temporary_directory
        assert isinstance(actual, Path)

    def test_time_run(self, example, capsys):
        with example.time_run() as actual:
            assert actual is None
        captured = capsys.readouterr()
        assert "seconds" in captured.out

    def test_cleanup_images(self, example):
        assert example.cleanup_images() is None
        assert not example._temporary_directory.exists()

    def test_size_of(self, example, capsys):
        assert example.size_of(__file__) is None
        captured = capsys.readouterr()
        assert "File size of " in captured.out


class RunnableExampleSuite(StandardExampleSuite):
    def test_load_data(self, example, data_type):
        actual = example.load_data()
        assert isinstance(actual, data_type)
        if isinstance(actual, xr.Dataset):
            actual = actual.to_array()[0]
        assert len(actual) == example.length

    @pytest.mark.parametrize("to_bytes_io", [True, False])
    def test_plot_image(self, example, to_bytes_io):
        example.to_bytes_io = to_bytes_io
        data = example.load_data()
        if isinstance(data, xr.Dataset):
            data_subset = data.isel(time=0)
        else:
            data_subset = data[:1]
        actual = example.plot_image(data_subset)
        if to_bytes_io:
            assert isinstance(actual, BytesIO)
        else:
            assert isinstance(actual, Path)
            assert actual.exists()

    def test_output_images(self, example):
        actual = example.output_images()
        assert isinstance(actual, list)
        assert len(actual) == example.length


class TestRandomWalkExample(RunnableExampleSuite):
    @pytest.fixture
    def example_class(self):
        return RandomWalkExample

    @pytest.fixture
    def data_type(self):
        return np.ndarray


class TestAirTemperatureExample(RunnableExampleSuite):
    @pytest.fixture
    def example_class(self):
        return AirTemperatureExample

    @pytest.fixture
    def data_type(self):
        return xr.Dataset
