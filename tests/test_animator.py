from pathlib import Path

import imageio.v3 as iio
import numpy as np
import pandas as pd
import pytest
import xarray as xr
from dask import is_dask_collection
from dask.diagnostics import ProgressBar
from dask.distributed import Client
from IPython.core.display import Image, Video
from pydantic import ValidationError

from enjoyn.animator import GifAnimator, Mp4Animator, Preprocessor


class RunnableAnimatorSuite:
    @pytest.fixture
    def items(self):
        return ["imageio:astronaut.png", "imageio:astronaut.png"]

    @pytest.fixture
    def output_path(self, output_extension, tmp_path):
        return tmp_path / f"output_path.{output_extension}"

    @pytest.fixture
    def animator(self, animator_class, items, output_path):
        return animator_class(items=items, output_path=output_path, show_output=False)

    @pytest.mark.parametrize("scheduler", [None, "threads", "processes"])
    @pytest.mark.parametrize("client", [None, Client()])
    def test_compute(self, animator, client, scheduler):
        assert (
            animator.compute(client=client, scheduler=scheduler) == animator.output_path
        )

    def test_compute_debug(self, animator):
        animator._debug = True
        animator.compute()
        animator._temporary_directory.name == "enjoyn_debug_workspace"

    def test_instantiation_required_args(self, items, output_path, animator_class):
        instantiated_animator = animator_class(items=items, output_path=output_path)
        assert instantiated_animator.items == items
        assert instantiated_animator.output_path == Path(output_path)

    def test_instantiation_at_least_two_items(self, items, output_path, animator_class):
        items = items[:1]
        with pytest.raises(ValidationError, match="at least 2"):
            animator_class(
                items=items,
                output_path=output_path,
            )

    @pytest.mark.parametrize(
        "array",
        [np.array([0, 1, 2]), pd.Series([0, 1, 2]), xr.DataArray(data=[0, 1, 2])],
    )
    def test_instantiation_serialize_array(self, array, output_path, animator_class):
        instantiated_animator = animator_class(
            items=array,
            output_path=output_path,
        )
        print(type(instantiated_animator.items))
        assert instantiated_animator.items == list(array)

    @pytest.mark.parametrize(
        "preprocessor", [lambda item: item, Preprocessor(func=lambda item: item)]
    )
    def test_instantiation_serialize_callable(
        self, items, output_path, preprocessor, animator_class
    ):
        instantiated_animator = animator_class(
            items=items,
            preprocessor=preprocessor,
            output_path=output_path,
        )
        assert isinstance(instantiated_animator.preprocessor, Preprocessor)
        assert instantiated_animator.preprocessor.func("item") == "item"

    def test_from_directory_pattern(self, output_path, tmp_path, animator_class):
        paths = []
        for i in range(3):
            path = tmp_path / f"file_{i}.txt"
            path.write_text(f"item_{i}")
            paths.append(path)
        (tmp_path / "file_0.png").write_bytes(b"bytes")

        instantiated_animator = animator_class.from_directory(
            tmp_path, output_path=output_path, pattern="*.txt"
        )
        assert isinstance(instantiated_animator, animator_class)
        assert instantiated_animator.items == paths

    def test_from_directory_limit(self, output_path, tmp_path, animator_class):
        paths = []
        for i in range(3):
            path = tmp_path / f"file_{i}.txt"
            path.write_text(f"item_{i}")
            paths.append(path)

        limit = 2
        instantiated_animator = animator_class.from_directory(
            tmp_path, output_path=output_path, limit=limit
        )
        assert isinstance(instantiated_animator, animator_class)
        assert instantiated_animator.items == paths[:limit]

    def test_from_directory_not_exist(self, animator_class):
        with pytest.raises(ValueError, match="does not exist"):
            animator_class.from_directory(
                "not/an/existent/directory",
                output_path="output_path",
            )

    def test_from_directory_not_directory(self, tmp_path, animator_class):
        path = tmp_path / "file_0.txt"
        path.write_text("item_0")

        with pytest.raises(ValueError, match="must be a directory"):
            animator_class.from_directory(
                path,
                output_path="output_path",
            )

    @pytest.mark.parametrize("item", ["imageio:astronaut.png", np.array([0])])
    def test_serialize_item(self, animator, item):
        image = animator._serialize_item(item)
        assert isinstance(image, np.ndarray)

    def test_serialize_item_with_preprocessor(self, animator):
        preprocessor = Preprocessor(func=lambda item: "imageio:astronaut.png")
        animator.preprocessor = preprocessor

        image = animator._serialize_item("item")
        assert isinstance(image, np.ndarray)

    def test_animate_images(self, animator, tmp_path, items):
        animator._temporary_directory = tmp_path
        animator._output_extension = ".gif"
        intermediate_path = animator._animate_images(items)
        assert intermediate_path.exists()
        assert iio.improps(intermediate_path).shape == (1, 512, 512, 3)

    def test_animate_images_no_temporary_directory(self, animator, items):
        with pytest.raises(RuntimeError, match="Use the built-in"):
            animator._animate_images(items)

    def test_concat_animations(self, animator):
        with pytest.raises(TypeError, match="unsupported"):
            animator._concat_animations(["a", "b"])

    def test_transfer_output(self, animator):
        intermediate_path = Path("intermediate.txt")
        intermediate_path.write_bytes(b"bytes")
        assert intermediate_path.exists()
        animator._transfer_output(intermediate_path).compute()
        assert not intermediate_path.exists()
        assert animator.output_path.exists()

    @pytest.mark.parametrize("visualize", [True, False])
    def test_plan(self, animator, visualize):
        return_obj = animator.plan(visualize=visualize)
        if visualize:
            assert isinstance(return_obj, Image)
        else:
            assert is_dask_collection(return_obj)

    def test_plan_invalid_extension(self, animator):
        animator._output_extension = ".ext"
        with pytest.raises(ValueError, match="The output path must end in '.ext'"):
            animator.plan()

    @pytest.mark.parametrize("show_progress", [True, False, None])
    def test_toggle_progress(self, animator, show_progress):
        with animator._display_progress_bar(
            show_progress=show_progress
        ) as progress_bar:
            if show_progress:
                assert isinstance(progress_bar, ProgressBar)
            else:
                assert progress_bar is None


class TestGifAnimator(RunnableAnimatorSuite):
    @pytest.fixture
    def output_extension(self):
        return ".gif"

    @pytest.fixture
    def animator_class(self):
        return GifAnimator

    def test_compute_error(self, animator):
        animator.gifsicle_options = ("no-option-like-this True",)
        with pytest.raises(RuntimeError, match="gifsicle failed"):
            animator.compute()

    @pytest.mark.parametrize("show_output", [True, False])
    def test_compute_show_output(self, animator, show_output):
        animator.show_output = show_output
        actual = animator.compute()
        if show_output:
            assert isinstance(actual, Image)
        else:
            assert actual == animator.output_path


class TestMp4Animator(RunnableAnimatorSuite):
    @pytest.fixture
    def output_extension(self):
        return ".mp4"

    @pytest.fixture
    def animator_class(self):
        return Mp4Animator

    def test_compute_error(self, animator):
        animator.ffmpeg_options = ("no-option-like-this True",)
        with pytest.raises(RuntimeError, match="Invalid data found"):
            animator.compute()

    @pytest.mark.parametrize("show_output", [True, False])
    def test_compute_show_output(self, animator, show_output):
        animator.show_output = show_output
        actual = animator.compute()
        if show_output:
            assert isinstance(actual, Video)
        else:
            assert actual == animator.output_path
