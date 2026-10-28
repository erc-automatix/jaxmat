import dataclasses
import functools
import json
import pathlib
import warnings
from collections.abc import Callable
from importlib.metadata import version

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import paramax

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


def _format_value(value, spec):
    """Format a number or a numeric array with ``spec``, anything else with ``str``."""
    if not spec or isinstance(value, bool | np.bool_):
        return str(value)
    if isinstance(value, int | float | complex | np.number):
        return format(value, spec)
    if isinstance(value, np.ndarray | jax.Array) and jnp.issubdtype(value.dtype, jnp.number):
        if value.ndim == 0:
            return format(value.item(), spec)
        with np.printoptions(formatter={"all": lambda x: format(x, spec)}):
            return str(np.asarray(value))
    return str(value)


def _format_sequence(values, spec):
    """Format a list or tuple, with one ``spec`` for all elements or a list of them."""
    specs = spec if isinstance(spec, list | tuple) else [spec] * len(values)
    items = [_format_value(v, s) for v, s in zip(values, specs, strict=True)]
    if isinstance(values, tuple):
        return "(" + ", ".join(items) + ("," if len(items) == 1 else "") + ")"
    return "[" + ", ".join(items) + "]"


def print_eqx_fields(obj, fields=None, indent=0, format=""):
    """
    Recursively print fields of an ``equinox`` module or dataclass-like object.

    Constrained parameters (see :mod:`jaxmat.parameters`) are printed with their physical
    values, not with their latent representation.

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
    format: str, dict or list
        Format specification (syntax of ``str.format``) used to print numbers and numeric
        arrays, e.g. ``".3e"``; other values are printed as they are. A dict gives a
        specification per field name (for a nested module, a specification or a dict for
        its own fields), a list one per element of a list or tuple.
    """
    obj = paramax.unwrap(obj)
    pad = " " * indent

    # Helper to match top-level field names
    def matches(field_name):
        if fields is None:
            return True
        # Match if this field or any nested subpath starts with it
        return any(f == field_name or f.startswith(f"{field_name}.") for f in fields)

    if isinstance(obj, eqx.Module):
        print(f"{pad}{obj.__class__.__name__}:")
        # declared fields, in order: the instance __dict__ of a module rebuilt by a tree map
        # (e.g. eqx.combine) holds other attributes; init=False fields are class settings
        for field in dataclasses.fields(obj):
            k, v = field.name, getattr(obj, field.name)
            if not field.init or not matches(k):
                continue  # skip fields not requested

            # Extract subfields relevant to this nested module (if any)
            subfields = None
            if fields is not None:
                subfields = [f[len(k) + 1 :] for f in fields if f.startswith(f"{k}.")] or None

            spec = format.get(k, "") if isinstance(format, dict) else format

            if isinstance(v, eqx.Module):
                print(f"{pad}  {k}:")
                print_eqx_fields(v, fields=subfields, indent=indent + 4, format=spec)
            elif isinstance(v, list | tuple):
                print(f"{pad}  {k} = {_format_sequence(v, spec)}")
            else:
                print(f"{pad}  {k} = {_format_value(v, spec)}")
    elif isinstance(obj, list | tuple):
        specs = format if isinstance(format, list | tuple) else [format] * len(obj)
        for i, (v, spec) in enumerate(zip(obj, specs, strict=True)):
            print(f"{pad}[{i}]: {_format_value(v, spec)}")
    else:
        print(f"{pad}{_format_value(obj, format if isinstance(format, str) else '')}")


_FILE_FORMAT = 1


def _model_name(model):
    return type(model).__qualname__


def _saved_leaf_paths(model):
    """Paths of the leaves written by ``eqx.tree_serialise_leaves`` (arrays and numbers)."""
    leaves, _ = jax.tree_util.tree_flatten_with_path(model)
    return [jax.tree_util.keystr(path) for path, leaf in leaves if eqx.is_array_like(leaf)]


def save_model(
    filename: str | pathlib.Path,
    model: eqx.Module,
    hyperparameters: dict | None = None,
) -> None:
    """
    Serialize and save an Equinox Module in a file.

    The file starts with a one-line JSON header (file format, jaxmat version, model class,
    ``hyperparameters`` and paths of the saved leaves), followed by the array and number leaves
    of the model, written with :func:`equinox.tree_serialise_leaves`. Only these leaves are
    saved: the structure of the model, its static fields (e.g. solvers) and functions (e.g.
    the transforms of constrained parameters) come from the template given to
    :func:`load_model`. Constrained parameters are saved with their latent values.

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
        "leaves": _saved_leaf_paths(model),
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
        If the file was not written by :func:`save_model`, if it holds a model of another
        class than ``skeleton``, or if its leaves do not match those of the template, e.g. a
        model saved with constrained parameters unwrapped by ``paramax.unwrap``.
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
        saved, expected = header["leaves"], _saved_leaf_paths(model)
        if saved != expected:
            # first differing leaf, or the first leaf missing on one side
            i = next(
                (i for i, (a, b) in enumerate(zip(saved, expected)) if a != b),
                min(len(saved), len(expected)),
            )
            in_file = saved[i] if i < len(saved) else "absent"
            in_template = expected[i] if i < len(expected) else "absent"
            raise ValueError(
                f"{filename} does not match the structure of the template: leaf {i} is "
                f"{in_file} in the file and {in_template} in the template. Was the model saved "
                "in another form, e.g. with constrained parameters unwrapped by paramax.unwrap?"
            )
        return eqx.tree_deserialise_leaves(file, model)
