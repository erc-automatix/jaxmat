import warnings

import equinox as eqx
import jax.numpy as jnp


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
        "'enforce_dtype' is deprecated and will be removed in v1.1. Use 'parameter()' instead "
        "to coerce dtypes, pass default values, etc.",
        FutureWarning,
        stacklevel=2,
    )
    """Deprecated alias for :func:`parameter` (coefficient field)."""
    return parameter(dtype=dtype, **kwargs)


def default_value(value, dtype=jnp.float64, **kwargs):
    """Deprecated alias for :func:`initial_value` (internal-state field)."""

    warnings.warn(
        "'default_value' is deprecated and will be removed in v1.1. Use 'initial_value' instead for"
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
