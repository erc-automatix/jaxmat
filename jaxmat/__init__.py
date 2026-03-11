from importlib.metadata import version
from pathlib import Path

import jax

__version__ = version("jaxmat")

jax.config.update("jax_enable_x64", True)  # use double-precision
jax.config.update("jax_debug_nans", True)  # raise when encountering nan

def get_path(notebook_fallback: str | None = None) -> Path:
    """
    Return the directory of the current demo.
    - In scripts: based on __file__
    - In notebooks: fallback to Path.cwd() or a provided path
    """
    try:
        return Path(__file__).resolve().parent
    except NameError:
        # __file__ is not defined (e.g., in Jupyter)
        if notebook_fallback:
            return Path(notebook_fallback).resolve()
        return Path.cwd().resolve()


