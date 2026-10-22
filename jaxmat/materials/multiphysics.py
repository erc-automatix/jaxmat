r"""Generic multiphysics behaviors.

A multiphysics behavior declares its variables once, as a :class:`~jaxmat.state.Physics`
built by :func:`~jaxmat.state.make_physics`, and implements a single ``response`` method. The
state class, the packing of inputs and outputs, the recording of every variable in the new
state, and the consistent tangent blocks are all derived from that declaration:

.. code-block:: python

    POROELASTICITY = make_physics(
        "Poroelasticity",
        gradients={"strain": SymmetricTensor2, "p": 0.0},
        fluxes={"stress": SymmetricTensor2, "zeta": 0.0},
    )

    class LinearBiot(MultiphysicsBehavior):
        physics = POROELASTICITY
        elasticity: AbstractLinearElastic
        b: float = parameter()
        M: float = parameter()

        def response(self, x, state, dt):
            sig = self.elasticity.C @ x.strain - self.b * x.p * Id
            zeta = self.b * x.strain.tr + x.p / self.M
            return (sig, zeta), state.internal

The library does not fix any multiphysics declaration: each behavior declares the variables
it exchanges, with the names and conventions of its own formulation. Coupled physics are
declared as a whole rather than composed: thermo-poromechanics, for instance, is one physics
with gradients ``(strain, p)``, fluxes ``(stress, zeta)`` and the external variable ``T``.

Material parameters may depend on any declared external variable (see
:mod:`jaxmat.parameters`): they are resolved at the current values before ``response``.
"""

from abc import abstractmethod
from typing import ClassVar

import equinox as eqx
import jax
import jax.numpy as jnp
from jax.flatten_util import ravel_pytree

from jaxmat.parameters import evaluate_parameters
from jaxmat.state import Physics

from .behavior import AbstractBehavior


class MultiphysicsBehavior(AbstractBehavior):
    """Behavior whose variables are declared by a :class:`~jaxmat.state.Physics`.

    Subclasses set the class attribute ``physics`` and implement :meth:`response`. They write
    no state class and no ``state.update`` boilerplate.

    ``constitutive_update(inputs, state, dt)`` takes the gradients followed by the external
    variables, as a tuple in declaration order, a dict, or the physics' ``Inputs`` named
    tuple, even when there is a single input. It always returns the fluxes as the physics'
    ``Fluxes`` named tuple, together with the new state in which every input and flux has
    been recorded.

    Variable-dependent parameters (:mod:`jaxmat.parameters`) of the behavior, including those
    of its sub-modules, are resolved at the current values of the declared external
    variables before :meth:`response` is called. A parameter depending on an undeclared
    variable raises an error.
    """

    physics: ClassVar[Physics]
    """Declaration of the gradients, fluxes and external variables."""

    def make_internal_state(self):
        """No internal state variables by default."""
        return None

    def init_state(self, Nbatch=None):
        """Initialize the state generated for ``physics``, optionally batched."""
        return self._init_state(self.physics.State, Nbatch)

    @abstractmethod
    def response(self, x, state, dt):
        """Compute the fluxes and the new internal state.

        Parameters
        ----------
        x : Inputs
            Named tuple of the current gradients and external variables, e.g. ``x.strain``,
            ``x.p``, ``x.T``.
        state : AbstractState
            State at the beginning of the step: previous values are ``state.strain``,
            ``state.stress``..., internal variables are ``state.internal``.
        dt : float
            Time increment.

        Returns
        -------
        fluxes : tuple, dict or Fluxes
            Fluxes in declaration order.
        internal : AbstractState or None
            New internal state.
        """

    @eqx.filter_jit
    def constitutive_update(self, inputs, state, dt):
        x = self.physics.make_inputs(inputs)
        external = {name: getattr(x, name) for name in self.physics.external_names}
        model = evaluate_parameters(self, **external)
        fluxes, internal = model.response(x, state, dt)
        fluxes = self.physics.make_fluxes(fluxes)
        changes = {**x._asdict(), **fluxes._asdict()}
        if internal is not None:
            changes["internal"] = internal
        return fluxes, state.update(**changes)

    def tangent(self, inputs, state, dt):
        """Consistent tangent blocks ``d flux / d gradient``, by forward-mode autodiff.

        Returns a dict keyed by ``(flux_name, gradient_name)``. Tensors are differentiated in
        their stored (Mandel) representation, so the strain-stress block of a small-strain
        behavior is the usual ``(6, 6)`` Mandel tangent; a block involving a scalar drops the
        corresponding dimension.
        """
        P = self.physics
        x = P.make_inputs(inputs)
        grads = tuple(getattr(x, n) for n in P.gradient_names)
        flat_grads, unravel = ravel_pytree(grads)

        def flat_fluxes(flat):
            x_ = x._replace(**dict(zip(P.gradient_names, unravel(flat), strict=True)))
            fluxes, _ = self.constitutive_update(x_, state, dt)
            return ravel_pytree(tuple(fluxes))[0]

        J = jax.jacfwd(flat_fluxes)(flat_grads)
        fluxes = tuple(self.constitutive_update(x, state, dt)[0])
        f_shapes, g_shapes = map(_shapes, (fluxes, grads))
        f_offsets, g_offsets = map(_offsets, (f_shapes, g_shapes))
        blocks = {}
        for fn, fs, fo in zip(P.flux_names, f_shapes, f_offsets, strict=True):
            for gn, gs, go in zip(P.gradient_names, g_shapes, g_offsets, strict=True):
                block = J[fo : fo + _size(fs), go : go + _size(gs)]
                blocks[(fn, gn)] = block.reshape(fs + gs)
        return blocks


def _shapes(variables):
    # tensors (modules) by their flat stored size, arrays by their own shape
    return [
        (ravel_pytree(v)[0].size,) if isinstance(v, eqx.Module) else jnp.shape(v) for v in variables
    ]


def _size(shape):
    n = 1
    for s in shape:
        n *= s
    return n


def _offsets(shapes):
    offsets, o = [], 0
    for s in shapes:
        offsets.append(o)
        o += _size(s)
    return offsets
