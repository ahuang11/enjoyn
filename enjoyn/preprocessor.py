"""
This module contains preprocessors that store functions to
apply to each item of the animator.
"""

from typing import Any, Callable, Dict, Optional, Tuple

from pydantic import BaseModel


class Preprocessor(BaseModel):
    """
    Used to store a function and its inputs.

    Args:
        func: The function to apply to each item of the animator; the function
            must accept the item to process as the first positional arg and
            must return either a `Path`, `str`, `bytes`, or `np.ndarray` type.
        args: The additional arguments to pass to the function.
        kwds: The additional keywords to pass to the function.
    """

    func: Callable
    args: Tuple[Any] = ()
    kwds: Optional[Dict[str, Any]] = None
