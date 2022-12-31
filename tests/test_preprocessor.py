from io import BytesIO
from pathlib import Path

import holoviews as hv
import matplotlib.pyplot as plt
import numpy as np
import pytest

from enjoyn.preprocessor import (
    HoloViewsPreprocessor,
    MatplotlibPreprocessor,
    Preprocessor,
)


class TestPreprocessor:
    @pytest.mark.parametrize("args", [(), ("a",), {"a", "b"}, ["a", "b"]])
    @pytest.mark.parametrize("kwds", [None, {}, {"c": "C"}])
    def test_instantiation(self, args, kwds):
        func = lambda item, *args, **kwds: "irrelevant_output"
        preprocessor = Preprocessor(func=func, args=args, kwds=kwds)
        assert preprocessor.func == func
        assert preprocessor.args == list(args)
        assert preprocessor.kwds == kwds

    @pytest.mark.parametrize(
        "return_obj",
        [Path("path"), "string", BytesIO(b"bytes_io"), b"bytes", np.array([0])],
    )
    def test_apply_on(self, return_obj):
        func = lambda item: return_obj
        preprocessor = Preprocessor(func=func)
        with preprocessor.apply_on("irrelevant_item") as actual:
            assert actual == return_obj

    def test_apply_on_type_error(self):
        func = lambda item: 1234
        preprocessor = Preprocessor(func=func)
        with pytest.raises(TypeError, match="callable returned an object with"):
            with preprocessor.apply_on("irrelevant_item"):
                pass


class TestMatplotlibPreprocessor:
    @pytest.mark.parametrize("return_index", [0, 1, 2])
    def test_apply_on(self, return_index):
        def plot(i):
            fig = plt.figure()
            ax = plt.axes()
            img = ax.plot(range(i))
            return (fig, ax, img)[return_index]

        preprocessor = MatplotlibPreprocessor(func=plot)
        with preprocessor.apply_on(10) as actual:
            assert isinstance(actual, BytesIO)
        assert actual.closed

    def test_apply_on_plt(self):
        def plot(i):
            img = plt.plot(range(i))
            return img

        preprocessor = MatplotlibPreprocessor(func=plot)
        with preprocessor.apply_on(10) as actual:
            assert isinstance(actual, BytesIO)
        assert actual.closed

    def test_apply_not_matplotlib(self):
        func = lambda item: Path("path")
        preprocessor = MatplotlibPreprocessor(func=func)
        with preprocessor.apply_on("irrelevant_item") as actual:
            assert isinstance(actual, Path)


class TestHoloViewsPreprocessor:
    def test_apply_on(self):
        def plot(i):
            return hv.Curve(([0, 1, i], [0, 1, i]))

        preprocessor = HoloViewsPreprocessor(func=plot)
        with preprocessor.apply_on(10) as actual:
            assert isinstance(actual, BytesIO)
        assert actual.closed

    def test_apply_not_holoviews(self):
        func = lambda item: Path("path")
        preprocessor = HoloViewsPreprocessor(func=func)
        with preprocessor.apply_on("irrelevant_item") as actual:
            assert isinstance(actual, Path)
