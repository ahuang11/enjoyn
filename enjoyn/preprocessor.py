"""
This module contains preprocessors that store functions to
apply to each item of the animator.
"""

from io import BytesIO
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Type, Union

try:
    import matplotlib.pyplot as plt
except ImportError:  # pragma: no cover
    plt = None
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

    def apply_on(
        self, item: Any, validate_type: bool = True
    ) -> Union[Path, str, BytesIO, bytes, np.ndarray]:
        """
        Applies the func, along with its args and kwds, to the item.

        Args:
            item: The item to apply the function on.
            validate_type: Whether to validate the preprocessed item is correct type.

        Returns:
            The preprocessed item.
        """
        args = self.args or ()
        kwds = self.kwds or {}
        item = self.func(item, *args, **kwds)

        self._validate_item(item, validate_type)
        return item


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
                "Ensure matplotlib is installed with " "`pip install -U matplotlib`"
            )
        return values

    def apply_on(self, item: Any, validate_type: bool = True) -> BytesIO:
        """
        Applies the func, along with its args and kwds, to the item; additionally, if a
        matplotlib type is returned, automatically save the plot to memory and close.

        Args:
            item: The item to apply the function on.
            validate_type: Whether to validate the preprocessed item is correct type.

        Returns:
            The preprocessed item.
        """
        item = super().apply_on(item, validate_type=False)

        if plt.gcf().axes:
            item = BytesIO()
            plt.savefig(item)
            plt.close("all")

        self._validate_item(item, validate_type)
        return item
