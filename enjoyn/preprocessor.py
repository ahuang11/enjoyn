"""
This module contains preprocessors that store functions to
apply to each item of the animator.
"""

from contextlib import contextmanager
from io import BytesIO
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Type, Union

try:
    import matplotlib.pyplot as plt
except ImportError:  # pragma: no cover
    plt = None

try:
    import holoviews as hv
except ImportError:  # pragma: no cover
    hv = None

import numpy as np
from pydantic import BaseModel, Extra, PrivateAttr, root_validator


class Preprocessor(BaseModel, extra=Extra.forbid):
    """
    Used to store a function and its inputs.

    Args:
        func: The function to apply to each item of the animator; the function
            must accept the item to process as the first positional arg and
            must return either a `Path`, `str`, `BytesIO`, `bytes`, or `np.ndarray`
            type.
        args: The additional arguments to pass to the function.
        kwds: The additional keywords to pass to the function.
    """

    func: Callable
    args: Optional[List[Any]] = None
    kwds: Optional[Dict[str, Any]] = None

    _valid_return_types: Tuple[Type] = PrivateAttr(
        (Path, str, BytesIO, bytes, np.ndarray)
    )

    def _validate_item(self, item: Any, validate_type: bool):
        """
        Checks whether the item is of valid return type.
        """
        if validate_type and not isinstance(item, self._valid_return_types):
            callable_name = self.func.__name__
            return_type = type(item).__name__
            valid_return_types_names = ", ".join(
                f"`{type_.__name__}`" for type_ in self._valid_return_types
            )
            raise TypeError(
                f"The '{callable_name}' callable returned an object with "
                f"`{return_type}` type; update '{callable_name}' so the "
                f"returned object is either {valid_return_types_names} type"
            )

    @contextmanager
    def apply_on(
        self, item: Any, validate_type: bool = True
    ) -> Union[Path, str, BytesIO, bytes, np.ndarray]:
        """
        Applies the func, along with its args and kwds, to the item.

        Args:
            item: The item to apply the function on.
            validate_type: Whether to validate the preprocessed item is correct type.

        Yields:
            The preprocessed item.
        """
        args = self.args or ()
        kwds = self.kwds or {}
        item = self.func(item, *args, **kwds)

        self._validate_item(item, validate_type)
        yield item


class MatplotlibPreprocessor(Preprocessor):
    """
    Used to store a matplotlib function and its inputs.
    """

    @root_validator(pre=True)
    def _plugin_installed(cls, values):  # pragma: no cover
        """
        Check whether required libraries are installed.
        """
        if plt is None:
            raise ImportError(
                "Ensure matplotlib is installed with `pip install -U matplotlib`"
            )
        return values

    @contextmanager
    def apply_on(self, item: Any, validate_type: bool = True) -> BytesIO:
        """
        Applies the func, along with its args and kwds, to the item; additionally, if a
        matplotlib type is returned, automatically save the plot to memory and close.

        Args:
            item: The item to apply the function on.
            validate_type: Whether to validate the preprocessed item is correct type.

        Yields:
            The preprocessed item.
        """
        with super().apply_on(item, validate_type=False) as item:
            if plt.gcf().axes:  # if active axes
                with BytesIO() as buf:
                    plt.savefig(buf, format="png")
                    plt.close("all")
                    buf.seek(0)
                    yield buf
            else:
                yield item


class HoloViewsPreprocessor(Preprocessor):
    """
    Used to store a HoloViews function and its inputs.
    """

    @root_validator(pre=True)
    def _plugin_installed(cls, values):  # pragma: no cover
        """
        Check whether required libraries are installed.
        """
        if hv is None:
            raise ImportError(
                "Ensure holoviews is installed with `pip install -U holoviews`"
            )
        return values

    @contextmanager
    def apply_on(self, item: Any, validate_type: bool = True) -> BytesIO:
        """
        Applies the func, along with its args and kwds, to the item; additionally, if a
        HoloViews type is returned, automatically save the plot to memory and close.

        Args:
            item: The item to apply the function on.
            validate_type: Whether to validate the preprocessed item is correct type.

        Yields:
            The preprocessed item.
        """
        with super().apply_on(item, validate_type=False) as item:
            if isinstance(item, hv.Element):
                with BytesIO() as buf:
                    hv.save(item, buf, fmt="png")
                    buf.seek(0)
                    yield buf
            else:
                yield item


class NullPreprocessor(Preprocessor):
    """
    Used to simplify internal code; does nothing.
    """

    func: Callable = lambda item: item

    @contextmanager
    def apply_on(
        self, item: Any, validate_type: bool = True
    ) -> Union[Path, str, BytesIO, bytes, np.ndarray]:
        """
        Yields back the original item.

        Args:
            item: The item to apply the function on.
            validate_type: Whether to validate the preprocessed item is correct type.

        Yields:
            The item.
        """
        yield item
