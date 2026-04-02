import equinox as eqx
import jax.numpy as jnp


def default_value(value, dtype=jnp.float64, **kwargs):
    """Initialize and convert a field with default `value` of imposed `dtype`."""
    return eqx.field(
        converter=lambda x: jnp.asarray(x, dtype=dtype), default=value, **kwargs
    )


def enforce_dtype(dtype=jnp.float64, **kwargs):
    """Initialize and convert a field with default `value` of imposed `dtype`."""
    return eqx.field(converter=lambda x: jnp.asarray(x, dtype=dtype), **kwargs)


def _rgetattr(obj, attr):
    """Like getattr, but supports dotted paths (e.g. 'layer.sub.weight')."""
    for name in attr.split("."):
        obj = getattr(obj, name)
    return obj


def partition_by_node_names(model, freeze_names):
    """
    Partition an Equinox model into (trainable, static) where
    attributes listed in `freeze_names` are frozen (moved to static).
    """

    # Start with array-vs-nonarray partition
    trainable, static = eqx.partition(model, eqx.is_array)

    for name in freeze_names:

        def sel(m):
            return _rgetattr(m, name)

        # move out of trainable
        trainable = eqx.tree_at(
            sel, trainable, replace=None, is_leaf=lambda x: x is None
        )
        # copy original value into static
        static = eqx.tree_at(
            sel, static, replace=_rgetattr(model, name), is_leaf=lambda x: x is None
        )

    return trainable, static


def print_eqx_fields(obj, fields=None, indent=0, file=None, format=""):
    """
    Recursively print fields of an Equinox module or dataclass-like object.

    Args:
        obj: The Equinox module or object to inspect.
        fields: Optional list of field names (strings) to print.
                Supports nested paths like ["layer1", "layer2.weight"].
                If None, prints all fields recursively.
        indent: Internal indentation level (used for recursion).
        format: Formatter used to format Module element (syntax of str.format).
    """

    pad = " " * indent

    # Helper to match top-level field names
    def matches(field_name):
        if fields is None:
            return True
        # Match if this field or any nested subpath starts with it
        return any(f == field_name or f.startswith(f"{field_name}.") for f in fields)

    def format_list_tupple(to_format: list | tuple, formatter: str | list) -> str:
        """
        Format a list or a tuple using formatter. If formatter is a string, the
        same formatter is used for every element, if it is a list, the
        formatter can be specified for each element of the list/tuple.
        """
        # Define the pair of delimiters
        start: str
        end: str
        if isinstance(to_format, tuple):
            start, end = "(", ")"
        else:
            start, end = "[", "]"

        # Define the template
        template: str
        if isinstance(formatter, (list, tuple)):
            template = start + ", ".join([f"{{:{f}}}" for f in formatter]) + end
        else:
            template = start + ", ".join(len(to_format) * [f"{{:{formatter}}}"]) + end

        return template.format(*to_format)

    if isinstance(obj, eqx.Module):
        print(f"{pad}{obj.__class__.__name__}:", file=file)
        for k, v in obj.__dict__.items():
            if not matches(k):
                continue  # skip fields not requested

            # Extract subfields relevant to this nested module (if any)
            subfields = None
            if fields is not None:
                subfields = [
                    f[len(k) + 1 :] for f in fields if f.startswith(f"{k}.")
                ] or None

            v_formatter: str
            if isinstance(format, dict):
                if k in format:
                    v_formatter = format[k]
                else:
                    v_formatter = ""
            else:
                v_formatter = format

            if isinstance(v, eqx.Module):
                print(f"{pad}  {k}:", file=file)
                print_eqx_fields(
                    v,
                    fields=subfields,
                    indent=indent + 4,
                    format=v_formatter,
                    file=file,
                )
            elif isinstance(v, (list, tuple)):
                print(f"{pad}  {k} = {format_list_tupple(v, v_formatter)}", file=file)
            else:
                print(f"{pad}  {k} = {v:{v_formatter}}", file=file)
    elif isinstance(obj, (list, tuple)):
        for i, v in enumerate(obj):
            v_formatter: str
            if isinstance(format, (list, tuple)):
                v_formatter = format[i]
            else:
                v_formatter = format

            print(f"{pad}[{i}]: {v:{v_formatter}}", file=file)

    else:
        print(f"{pad}{obj}", file=file)
