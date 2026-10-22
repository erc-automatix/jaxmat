import collections
import dataclasses
import functools
import keyword
import operator
import typing

import equinox as eqx
import jax
import jax.numpy as jnp

from jaxmat.tensors import SymmetricTensor2, Tensor2


class AbstractState(eqx.Module):
    """Abstract base class representing mechanical states."""

    def _resolve_aliases(self, changes):
        alias_map = getattr(self, "__alias_targets__", {})
        field_names = self.__dict__
        resolved = {}
        for k, v in changes.items():
            field_name = alias_map.get(k, k)
            if field_name in field_names:
                resolved[field_name] = v
        return resolved

    def add(self, **changes):
        """Utility method to add values such as ``new_state = state.add(stress=dsig)``."""
        valid_changes = self._resolve_aliases(changes)
        return eqx.tree_at(
            lambda c: [getattr(c, k) for k in valid_changes],
            self,
            [getattr(self, k) + v for k, v in valid_changes.items()],
        )

    def update(self, **changes):
        """Utility method to update values such as ``new_state = state.update(stress=sig)``."""
        valid_changes = self._resolve_aliases(changes)
        return eqx.tree_at(
            lambda c: [getattr(c, k) for k in valid_changes],
            self,
            list(valid_changes.values()),
        )


class SmallStrainState(AbstractState):
    r"""
    State representation for small-strain behaviors.

    This class stores the current strain $\beps$ and stress $\bsig$ tensors under the assumption
    of infinitesimal (small) deformations. It also supports internal state variables
    such as plastic strain, damage, or other material history quantities through
    the `internal` attribute.

    Attributes
    ----------
    internal : :class:`AbstractState`, optional
        Nested state object representing internal variables (e.g., plastic strain,
        hardening variables, etc.). Defaults to None.
    strain : :class:`SymmetricTensor2`
        Symmetric second-rank strain tensor $\beps$ (small-strain assumption).
    stress : :class:`SymmetricTensor2`
        Symmetric second-rank Cauchy stress tensor $\bsig$.

    Notes
    -----
    eps : :class:`SymmetricTensor2`
        Alias for `strain`, allows accessing via `state.eps`.
    sig : :class:`SymmetricTensor2`
        Alias for `stress`, allows accessing via `state.sig`.
    """

    internal: AbstractState = None
    strain: SymmetricTensor2 = eqx.field(default_factory=SymmetricTensor2)
    stress: SymmetricTensor2 = eqx.field(default_factory=SymmetricTensor2)

    # define alias targets to authorize state updates with alias names
    __alias_targets__: typing.ClassVar[dict] = {"eps": "strain", "sig": "stress"}

    @property
    def eps(self):
        """Alias for ``strain``."""
        return self.strain

    @property
    def sig(self):
        """Alias for ``stress``."""
        return self.stress


def PK1_to_PK2(F, PK1):
    """
    Convert the first Piola-Kirchhoff stress tensor (PK1) to the
    second Piola-Kirchhoff stress tensor (PK2). Enforce symmetry explicitly.
    """
    return (F.inv @ PK1).sym


def PK1_to_Cauchy(F, PK1):
    """
    Convert the first Piola-Kirchhoff stress tensor (PK1) to the
    Cauchy stress tensor. Enforce symmetry explicitly.
    """
    return (PK1 @ F.T).sym / jnp.linalg.det(F)


class FiniteStrainState(AbstractState):
    r"""
    State representation for finite-strain continuum mechanics.

    This class encapsulates the deformation gradient $\bF$ (``F``) and first Piola-Kirchhoff
    stress $\bP$ (``PK1``), along with optional internal variables. It provides convenience
    properties for converting between common stress measures: second Piola-Kirchhoff
    $\bS$ (``PK2``) and Cauchy $\bsig$ (``sig``) stresses.

    Attributes
    ----------
    internal : AbstractState, optional
        Nested internal state representing material history or additional
        constitutive information. Defaults to None.
    F : :class:`Tensor2`
        Deformation gradient $\bF$. Initialized as the identity tensor.
    PK1 : :class:`Tensor2`
        First Piola-Kirchhoff stress tensor $\bP$.

    Notes
    -----
    PK2 : :class:`SymmetricTensor2`
        Second Piola-Kirchhoff stress tensor $\bS$, computed via :func:`PK1_to_PK2`.
    sig : :class:`SymmetricTensor2`
        Cauchy stress tensor $\bsig$, computed via :func:`PK1_to_Cauchy`.
    Cauchy : :class:`SymmetricTensor2`
        Alias for ``sig``.

    """

    internal: AbstractState = None
    F: Tensor2 = eqx.field(default_factory=Tensor2.identity)
    PK1: Tensor2 = eqx.field(default_factory=Tensor2)

    @property
    def PK2(self):
        vmap_axes = 0 if self.F.tensor.ndim == 3 else None
        return eqx.filter_vmap(PK1_to_PK2, in_axes=vmap_axes, out_axes=vmap_axes)(self.F, self.PK1)

    @property
    def sig(self):
        vmap_axes = 0 if self.F.tensor.ndim == 3 else None
        return eqx.filter_vmap(PK1_to_Cauchy, in_axes=vmap_axes, out_axes=vmap_axes)(
            self.F, self.PK1
        )

    @property
    def Cauchy(self):
        return self.sig


def make_batched(module: eqx.Module, Nbatch: int) -> eqx.Module:
    """Broadcasts all leaf arrays of a single unbatched module into a batched version.

    Parameters
    -----------
    module: eqx.module
        An instance of an equinox Module (e.g., `State`) with array leaves.
    Nbatch: int
        The number of batch items to broadcast.

    Returns:
        A new instance of the same class, with each array field having shape (Nbatch, ...).
    """

    def _broadcast(x):
        x_ = jnp.asarray(x)
        return jnp.broadcast_to(x_, (Nbatch, *x_.shape))

    batched_module = jax.tree.map(_broadcast, module)

    return batched_module


# ---------------------------------------------------------------------------------------------
#  Generic multiphysics states
# ---------------------------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class Scalar:
    """Initializer of a scalar state variable (hashable, so it can key the state factory)."""

    default: float = 0.0

    def __call__(self):
        return jnp.asarray(self.default, dtype=jnp.float64)


@dataclasses.dataclass(frozen=True)
class Vector:
    """Initializer of a vector state variable of dimension ``dim``."""

    dim: int = 3
    default: float = 0.0

    def __call__(self):
        return jnp.full((self.dim,), self.default, dtype=jnp.float64)


def _initializer(spec):
    """Normalize a variable declaration into a zero-argument, hashable initializer."""
    if isinstance(spec, bool):
        raise TypeError("A boolean is not a valid variable declaration.")
    if isinstance(spec, int | float):
        return Scalar(float(spec))
    if callable(spec):  # a class such as SymmetricTensor2, or a factory such as Tensor2.identity
        return spec
    raise TypeError(
        f"Cannot declare a variable from {spec!r}: use a tensor class (e.g. SymmetricTensor2), "
        "a zero-argument factory (e.g. Tensor2.identity), a number (scalar with that initial "
        "value) or Vector(dim)."
    )


def _as_items(variables):
    return tuple((name, _initializer(spec)) for name, spec in dict(variables or {}).items())


@dataclasses.dataclass(frozen=True)
class Physics:
    r"""Declaration of the variables exchanged by a (multiphysics) behavior.

    Following the terminology:

    - ``gradients`` are the driving inputs whose conjugates are computed by the behavior
      (strain $\beps$, deformation gradient $\bF$, pore pressure $p$, temperature gradient...);
    - ``fluxes`` are the computed outputs, the thermodynamic forces conjugate to the gradients
      (stress $\bsig$, fluid content $\zeta$, heat flux $\boldsymbol{q}$...);
    - ``external`` variables are known inputs with no conjugate output (temperature $T$...).

    Internal state variables are not declared here: they keep living in ``state.internal``.

    A ``Physics`` is immutable and hashable, and is built with :func:`make_physics`. The
    corresponding state class, input and flux containers are generated on first use and
    cached, so equal declarations always give the *same* classes (stable pytree structures for
    ``jit``, ``scan`` and serialization).
    """

    name: str
    gradients: tuple
    fluxes: tuple
    external: tuple = ()
    aliases: tuple = ()

    def __post_init__(self):
        names = [n for n, _ in self.gradients + self.external + self.fluxes]
        reserved = {"internal", "update", "add", "physics"}
        for n in names + [a for a, _ in self.aliases]:
            if not n.isidentifier() or keyword.iskeyword(n) or n.startswith("_"):
                raise ValueError(f"'{n}' is not a valid variable name.")
            if n in reserved:
                raise ValueError(f"'{n}' is a reserved name.")
        duplicates = {n for n in names if names.count(n) > 1}
        if duplicates:
            raise ValueError(f"Variables declared more than once: {sorted(duplicates)}.")
        for alias, target in self.aliases:
            if target not in names or alias in names:
                raise ValueError(f"Invalid alias '{alias}' -> '{target}'.")
        if not self.gradients or not self.fluxes:
            raise ValueError("A physics needs at least one gradient and one flux.")

    @property
    def gradient_names(self):
        return tuple(n for n, _ in self.gradients)

    @property
    def flux_names(self):
        return tuple(n for n, _ in self.fluxes)

    @property
    def external_names(self):
        return tuple(n for n, _ in self.external)

    @property
    def input_names(self):
        """Names of the inputs of ``constitutive_update``: gradients, then external variables."""
        return self.gradient_names + self.external_names

    @property
    def State(self):
        """The state class (an :class:`AbstractState`) storing all variables and ``internal``."""
        return _state_class(self)

    @property
    def Inputs(self):
        """Named tuple of the inputs (gradients, then external variables)."""
        return _named_tuple(f"{self.name}Inputs", self.input_names)

    @property
    def Fluxes(self):
        """Named tuple of the fluxes."""
        return _named_tuple(f"{self.name}Fluxes", self.flux_names)

    def make_inputs(self, inputs):
        """Accept ``Inputs``, a positional tuple or a dict, even for a single input."""
        return _pack(self.Inputs, inputs, cast=True)

    def make_fluxes(self, fluxes):
        """Accept ``Fluxes``, a positional tuple or a dict, even for a single flux."""
        return _pack(self.Fluxes, fluxes, cast=False)


def _cast(value):
    # numbers given by drivers become strong float64 arrays: no retrace, stable scan carries
    return value if isinstance(value, eqx.Module) else jnp.asarray(value, dtype=jnp.float64)


def _pack(container, values, cast):
    fields = container._fields
    if isinstance(values, container):
        packed = values
    elif isinstance(values, dict):
        if set(values) != set(fields):
            raise ValueError(f"{container.__name__} expects {fields}, got {tuple(values)}.")
        packed = container(**values)
    elif isinstance(values, tuple):
        if len(values) != len(fields):
            raise ValueError(f"{container.__name__} expects {fields}, got {len(values)} values.")
        packed = container(*values)
    else:
        raise TypeError(
            f"{container.__name__} expects a tuple or dict of {fields}, even for a single value."
        )
    return container(*map(_cast, packed)) if cast else packed


@functools.cache
def _named_tuple(typename, fields):
    return collections.namedtuple(typename, fields)


@functools.cache
def _state_class(physics: Physics):
    annotations = {"internal": AbstractState}
    namespace = {"internal": None}
    for name, init in physics.gradients + physics.external + physics.fluxes:
        annotations[name] = typing.Any
        namespace[name] = eqx.field(default_factory=init)
    annotations["__alias_targets__"] = typing.ClassVar[dict]
    namespace["__alias_targets__"] = dict(physics.aliases)
    annotations["physics"] = typing.ClassVar[Physics]
    namespace["physics"] = physics
    for alias, target in physics.aliases:
        namespace[alias] = property(operator.attrgetter(target), doc=f"Alias for ``{target}``.")
    name = f"{physics.name}State"
    namespace.update(
        __annotations__=annotations,
        __module__=__name__,
        __qualname__=name,
        __doc__=(
            f"State generated for the ``{physics.name}`` physics. Gradients: "
            f"{', '.join(physics.gradient_names)}; fluxes: {', '.join(physics.flux_names)}"
            + (f"; external: {', '.join(physics.external_names)}." if physics.external else ".")
        ),
    )
    return type(name, (AbstractState,), namespace)


def make_physics(
    name: str,
    *,
    gradients: dict[str, typing.Any],
    fluxes: dict[str, typing.Any],
    external: dict[str, typing.Any] | None = None,
    aliases: dict[str, str] | None = None,
) -> Physics:
    """Declare the variables of a behavior and get the matching state class for free.

    Each variable is declared by name with an initializer: a tensor class
    (``SymmetricTensor2``), a zero-argument factory (``Tensor2.identity``), a number (a scalar
    with that initial value) or :class:`Vector`. Gradients and fluxes are listed in matching
    (conjugate) order.

    .. code-block:: python

        POROELASTICITY = make_physics(
            "Poroelasticity",
            gradients={"strain": SymmetricTensor2, "p": 0.0},
            fluxes={"stress": SymmetricTensor2, "zeta": 0.0},
            aliases={"eps": "strain", "sig": "stress"},
        )
        state = POROELASTICITY.State()  # fields: internal, strain, p, stress, zeta
    """
    return Physics(
        name=name,
        gradients=_as_items(gradients),
        fluxes=_as_items(fluxes),
        external=_as_items(external),
        aliases=tuple(dict(aliases or {}).items()),
    )


def make_state(name: str, **variables) -> type[AbstractState]:
    """Shortcut for ``make_physics(name, **variables).State``."""
    return make_physics(name, **variables).State
