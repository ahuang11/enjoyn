from io import BytesIO
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pytest

from enjoyn.preprocessor import MatplotlibPreprocessor, Preprocessor


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
        actual = preprocessor.apply_on("irrelevant_item")
        assert actual == return_obj

    def test_apply_on_type_error(self):
        func = lambda item: 1234
        preprocessor = Preprocessor(func=func)
        with pytest.raises(TypeError, match="callable returned an object with"):
            preprocessor.apply_on("irrelevant_item")


class TestMatplotlibPreprocessor:
    @pytest.mark.parametrize("return_index", [0, 1, 2])
    def test_apply_on(self, return_index):
        def plot(i):
            fig = plt.figure()
            ax = plt.axes()
            img = ax.plot(range(i))
            return (fig, ax, img)[return_index]

        preprocessor = MatplotlibPreprocessor(func=plot)
        actual = preprocessor.apply_on(10)
        assert isinstance(actual, BytesIO)

    def test_apply_on_plt(self):
        def plot(i):
            img = plt.plot(range(i))
            return img

        preprocessor = MatplotlibPreprocessor(func=plot)
        actual = preprocessor.apply_on(10)
        assert isinstance(actual, BytesIO)

    def test_apply_not_matplotlib(self):
        func = lambda item: Path("path")
        preprocessor = MatplotlibPreprocessor(func=func)
        actual = preprocessor.apply_on("irrelevant_item")
        assert isinstance(actual, Path)
