"""
This module contains preprocessors that store functions to
apply to each item of the animator.
"""

from io import BytesIO
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Type, Union

import numpy as np
from pydantic import BaseModel, Extra


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

    _valid_return_types: Tuple[Type] = (Path, str, BytesIO, bytes, np.ndarray)

    def apply_on(self, item) -> Union[Path, str, BytesIO, bytes, np.ndarray]:
        """
        Applies the func, along with its args and kwds, to the item.

        Args:
            item The item to apply the function on.

        Returns:
            The preprocessed item.
        """
        args = self.args or ()
        kwds = self.kwds or {}
        item = self.func(item, *args, **kwds)
        if not isinstance(item, self._valid_return_types):
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
        return item
