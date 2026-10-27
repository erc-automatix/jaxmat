import functools
import json
import pathlib
import warnings
from collections.abc import Callable
from importlib.metadata import version

import equinox as eqx
import jax.numpy as jnp

from jaxmat import config


def assert_max_traces(max_traces: int = 1):
    """Decorator asserting that a function is traced at most ``max_traces`` times, only when
    the ``check_max_traces`` option of :mod:`jaxmat.config` is on (it is off by default).

    It is :func:`equinox.debug.assert_max_traces` behind a switch, and is placed the same
    way, inside the JIT wrapper:

    .. code-block:: python

        @eqx.filter_jit
        @assert_max_traces(max_traces=1)
        def constitutive_update(self, eps, state, dt): ...

    The option is read at trace time, so it can be switched at any point. Only traces
    happening while it is on are counted, and the count is shared by all instances.
    """

    def decorator(fn):
        checked = eqx.debug.assert_max_traces(fn, max_traces=max_traces)

        @functools.wraps(fn)
        def wrapper(*args, **kwargs):
            if config.check_max_traces:
                return checked(*args, **kwargs)
            return fn(*args, **kwargs)

        return wrapper

    return decorator


def initial_value(default, *, dtype=jnp.float64, **kwargs):
    """Declare an *internal-state* initial value.

    Cast to ``dtype`` like :func:`parameter`, but deliberately **not**
    parameter-capable: internal-state fields (accumulated plastic strain, viscous
    strain, back-stresses, ...) hold plain numerical data, never a material law.
    Passing a parameter/module here is a modelling error and is rejected.
    """

    def convert(x):
        if isinstance(x, eqx.Module):
            raise TypeError(
                "initial_value() fields hold plain internal-state data, not an "
                "AbstractParameter or module. Use parameter() for input-dependent "
                "coefficients, or default_factory for structured state."
            )
        return jnp.asarray(x, dtype=dtype)

    return eqx.field(converter=convert, default=default, **kwargs)


def enforce_dtype(dtype=jnp.float64, **kwargs):
    from .parameters import parameter

    warnings.warn(
        "'enforce_dtype' is deprecated and will be removed in v0.2. Use 'parameter()' instead "
        "to coerce dtypes, pass default values, etc.",
        FutureWarning,
        stacklevel=2,
    )
    """Deprecated alias for :func:`parameter` (coefficient field)."""
    return parameter(dtype=dtype, **kwargs)


def default_value(value, dtype=jnp.float64, **kwargs):
    """Deprecated alias for :func:`initial_value` (internal-state field)."""

    warnings.warn(
        "'default_value' is deprecated and will be removed in v0.2. Use 'initial_value' instead for"
        " state variable initialization and 'parameter' for physical parameters.",
        FutureWarning,
        stacklevel=2,
    )
    return initial_value(value, dtype=dtype, **kwargs)


def _rgetattr(obj, attr):
    """Like getattr, but supports dotted paths (e.g. 'layer.sub.weight')."""
    for name in attr.split("."):
        obj = getattr(obj, name)
    return obj


def partition_by_node_names(model, freeze_names):
    """
    Partition an ``equinox`` model into (trainable, static) where
    attributes listed in `freeze_names` are frozen (moved to static).
    """

    # Start with array-vs-nonarray partition
    trainable, static = eqx.partition(model, eqx.is_array)

    for name in freeze_names:

        def sel(m):
            return _rgetattr(m, name)

        # move out of trainable
        trainable = eqx.tree_at(sel, trainable, replace=None, is_leaf=lambda x: x is None)
        # copy original value into static
        static = eqx.tree_at(
            sel, static, replace=_rgetattr(model, name), is_leaf=lambda x: x is None
        )

    return trainable, static


def print_eqx_fields(obj, fields=None, indent=0):
    """
    Recursively print fields of an ``equinox`` module or dataclass-like object.

    Parameters
    ----------
    obj: eqx.Module
        The module or object to inspect.
    fields: list
        Optional list of field names (strings) to print.
        Supports nested paths like ``["layer1", "layer2.weight"]``.
        If ``None``, prints all fields recursively.
    indent: int
        Internal indentation level (used for recursion).
    """
    pad = " " * indent

    # Helper to match top-level field names
    def matches(field_name):
        if fields is None:
            return True
        # Match if this field or any nested subpath starts with it
        return any(f == field_name or f.startswith(f"{field_name}.") for f in fields)

    if isinstance(obj, eqx.Module):
        print(f"{pad}{obj.__class__.__name__}:")
        for k, v in obj.__dict__.items():
            if not matches(k):
                continue  # skip fields not requested

            # Extract subfields relevant to this nested module (if any)
            subfields = None
            if fields is not None:
                subfields = [f[len(k) + 1 :] for f in fields if f.startswith(f"{k}.")] or None

            if isinstance(v, eqx.Module):
                print(f"{pad}  {k}:")
                print_eqx_fields(v, fields=subfields, indent=indent + 4)
            else:
                print(f"{pad}  {k} = {v}")
    elif isinstance(obj, (list | tuple)):
        for i, v in enumerate(obj):
            print(f"{pad}[{i}]: {v}")
    else:
        print(f"{pad}{obj}")


_FILE_FORMAT = 1


def _model_name(model):
    return type(model).__qualname__


def save_model(
    filename: str | pathlib.Path,
    model: eqx.Module,
    hyperparameters: dict | None = None,
) -> None:
    """
    Serialize and save an Equinox Module in a file.

    The file starts with a one-line JSON header (file format, jaxmat version, model class and
    ``hyperparameters``), followed by the array and number leaves of the model, written with
    :func:`equinox.tree_serialise_leaves`. Only these leaves are saved: the structure of the
    model, its static fields (e.g. solvers) and functions (e.g. the transforms of constrained
    parameters) come from the template given to :func:`load_model`.

    Parameters
    ----------
    filename: str or pathlib.Path
        Path of the file to save the module.
    model: eqx.Module
        Model to save, e.g. a calibrated material or a material state.
    hyperparameters: dict, optional
        Hyperparameters of the model, that is parameters that describe the shape of the
        associated PyTree (e.g. a number of branches or layers). They must be JSON
        serializable, and are passed to the factory given to :func:`load_model`.
    """
    header = {
        "format": _FILE_FORMAT,
        "jaxmat": version("jaxmat"),
        "model": _model_name(model),
        "hyperparameters": hyperparameters,
    }
    with open(filename, "wb") as file:
        file.write((json.dumps(header) + "\n").encode())
        eqx.tree_serialise_leaves(file, model)


def load_model(filename: str | pathlib.Path, skeleton: eqx.Module | Callable) -> eqx.Module:
    """
    Load a serialized model saved with :func:`save_model`.

    Parameters
    ----------
    filename: str or pathlib.Path
        Path of the file to load the module from.
    skeleton: eqx.Module or Callable
        Either an equinox module with the same structure as the saved model, whose leaf
        values are replaced by the saved ones, or a function that takes the saved
        hyperparameters as keyword arguments and creates such a module.

    Raises
    ------
    ValueError
        If the file was not written by :func:`save_model`, or if it holds a model of another
        class than ``skeleton``.
    """
    with open(filename, "rb") as file:
        try:
            header = json.loads(file.readline())
        except ValueError:  # not text, or not JSON
            header = None
        if not isinstance(header, dict) or "model" not in header:
            raise ValueError(f"{filename} is not a model file written by save_model.")
        if isinstance(skeleton, eqx.Module):
            model = skeleton
        else:
            model = skeleton(**(header["hyperparameters"] or {}))
        if header["model"] != _model_name(model):
            raise ValueError(
                f"{filename} holds a {header['model']} model, but the template is a "
                f"{_model_name(model)}."
            )
        return eqx.tree_deserialise_leaves(file, model)
