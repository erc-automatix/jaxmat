"""Global configuration options of jaxmat.

Options are read as module attributes and set like JAX's own configuration:

.. code-block:: python

    import jaxmat

    jaxmat.config.update("check_max_traces", True)
    jaxmat.config.check_max_traces  # True

    with jaxmat.config.override(check_max_traces=True):  # temporarily
        ...

Each option can also be set before importing jaxmat with an environment variable named
``JAXMAT_<OPTION>`` in upper case, e.g. ``JAXMAT_CHECK_MAX_TRACES=1`` (accepted values:
``1``/``0``, ``true``/``false``, ``yes``/``no``, ``on``/``off``).
"""

import contextlib
import os

_TRUE = {"1", "true", "yes", "on"}
_FALSE = {"0", "false", "no", "off", ""}


def _from_environment(name, default):
    variable = f"JAXMAT_{name.upper()}"
    value = os.environ.get(variable)
    if value is None:
        return default
    if value.strip().lower() in _TRUE:
        return True
    if value.strip().lower() in _FALSE:
        return False
    raise ValueError(
        f"Invalid value {value!r} for {variable}: use 1/0, true/false, yes/no or on/off."
    )


_DEFAULTS = {"check_max_traces": False}

check_max_traces: bool = _from_environment("check_max_traces", _DEFAULTS["check_max_traces"])
"""Assert that the ``constitutive_update`` of jaxmat behaviors is traced (compiled) at most
once, to catch unintended retracing while developing. Off by default: the trace counter is
shared by all instances of a behavior class, so legitimate uses also retrace, e.g. two
materials of the same class with different hardening laws, or a behavior wrapped by
another one. Only traces happening while the option is on are counted. Environment
variable: ``JAXMAT_CHECK_MAX_TRACES``."""


def update(name: str, value: bool):
    """Set the option ``name`` to ``value``."""
    if name not in _DEFAULTS:
        raise ValueError(f"Unknown jaxmat option {name!r}. Options are {sorted(_DEFAULTS)}.")
    if not isinstance(value, bool):
        raise TypeError(f"Option {name!r} expects a bool, got {value!r}.")
    globals()[name] = value


@contextlib.contextmanager
def override(**options: bool):
    """Context manager setting options temporarily, e.g. ``override(check_max_traces=True)``."""
    previous = {name: globals().get(name) for name in options}
    try:
        for name, value in options.items():
            update(name, value)
        yield
    finally:
        for name, value in previous.items():
            if name in _DEFAULTS:
                globals()[name] = value
