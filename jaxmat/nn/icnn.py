
import equinox as eqx
import jax
import jax.numpy as jnp


def positive_param(W: jnp.ndarray) -> jnp.ndarray:
    # Example: enforce nonnegativity
    return jax.nn.softplus(W)


class ICNN(eqx.Module):
    """Input Convex Neural Network (ICNN).

    Implements a variant of the convex neural network architecture of Amos et al.

    Parameters
    ----------
    in_dim : int
        Input dimension (e.g. number of invariants).
    hidden_dims : list[int]
        Sequence of hidden layer widths.
    key : jax.random.PRNGKey
        Random key for parameter initialization.
    scale : float, optional
        Scaling factor for random initialization (default: 1.0).

    Fields
    ------
    Ws : list[jax.Array]
        Trainable weight matrices for each hidden layer. Constrained to be
        nonnegative via ``positive_param``.
    bs : list[jax.Array]
        Trainable bias vectors for each hidden layer.
    final_W : jax.Array
        Final layer weights combining last hidden layer output and input.

    Call
    ----
    __call__(x: jax.Array) -> jax.Array
        Forward pass. Given input of shape ``(..., in_dim)``, returns a scalar
        convex output representing the learned function.
    """

    Ws: list[jnp.ndarray]
    bs: list[jnp.ndarray]
    final_W: jnp.ndarray

    def __init__(
        self,
        in_dim: int,
        hidden_dims: list[int],
        key: jax.Array,
        scale: float = 1.0,
    ):
        if in_dim == 0:
            in_dim = 1
        keys = jax.random.split(key, len(hidden_dims) + 1)
        self.Ws, self.bs = [], []

        prev_dim = in_dim
        for i, h in enumerate(hidden_dims):
            k1 = keys[i]
            # Nonnegative W (hidden-to-hidden)
            W: jnp.ndarray = jax.random.normal(k1, (h, prev_dim)) * scale
            b: jnp.ndarray = jnp.zeros((h,))
            self.Ws.append(W)
            self.bs.append(b)
            prev_dim = h

        # Final layer: combines last hidden z and input x
        self.final_W = jax.random.normal(keys[-1], (prev_dim,)) * scale

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        z = jnp.atleast_1d(x)
        for W, b in zip(self.Ws, self.bs):
            W_pos = positive_param(W)
            z = jax.nn.softplus(z @ W_pos.T + b)
        return z @ positive_param(self.final_W).T


class ICNNSkip(eqx.Module):
    """Input Convex Neural Network (ICNN) with skip connections.

    Implements the convex neural network architecture of Amos et al. (2017).

    Parameters
    ----------
    in_dim : int
        Input dimension.
    hidden_dims : list[int]
        Sequence of hidden layer widths.
    key : jax.random.PRNGKey
        Random key for parameter initialization.
    scale : float, optional
        Scaling factor for random initialization (default: 1.0).

    Notes
    -----
    - Nonnegative weights (W_k) ensure convexity.
    - Skip connections (U_k x) allow full convex function expressiveness.
    """

    Ws: list[jnp.ndarray]  # Nonnegative (hidden-to-hidden)
    Us: list[jnp.ndarray]  # Unconstrained (input skip)
    bs: list[jnp.ndarray]  # Biases
    final_W: jnp.ndarray  # Output weights
    final_U: jnp.ndarray  # Final input skip connection

    def __init__(
        self,
        in_dim: int,
        hidden_dims: list[int],
        key: jax.Array,
        scale: float = 1.0,
    ):
        if in_dim == 0:
            in_dim = 1

        keys = jax.random.split(key, len(hidden_dims) * 2 + 2)
        self.Ws, self.Us, self.bs = [], [], []

        prev_dim = in_dim
        for i, h in enumerate(hidden_dims):
            kW, kU = keys[2 * i], keys[2 * i + 1]
            # Nonnegative W (hidden-to-hidden)
            W = jax.random.normal(kW, (h, prev_dim)) * scale
            # Unconstrained U (skip from input)
            U = jax.random.normal(kU, (h, in_dim)) * scale
            b = jnp.zeros((h,))
            self.Ws.append(W)
            self.Us.append(U)
            self.bs.append(b)
            prev_dim = h

        # Final layer weights
        kW_final, kU_final = keys[-2], keys[-1]
        self.final_W = jax.random.normal(kW_final, (prev_dim,)) * scale
        self.final_U = jax.random.normal(kU_final, (in_dim,)) * scale

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        """Forward pass: returns scalar convex output f(x)."""
        x = jnp.atleast_1d(x)
        z = jnp.atleast_1d(x)

        for W, U, b in zip(self.Ws, self.Us, self.bs):
            W_pos = positive_param(W)
            U_pos = positive_param(U)
            z = jax.nn.softplus(
                z @ W_pos.T + x @ U_pos.T + b
            )  # convex, monotone nondecreasing activation

        Wf_pos = positive_param(self.final_W)
        Uf_pos = positive_param(self.final_U)
        out = z @ Wf_pos.T + x @ Uf_pos.T
        return out.squeeze()


class HomogeneousICNN(eqx.Module):
    """ICNN constrained to represent a convex, 1-homogeneous gauge function."""

    Ws: list[jnp.ndarray]  # hidden-to-hidden (constrained nonneg)
    Us: list[jnp.ndarray]  # input-to-hidden skip matrices (can be any real)
    final_W: jnp.ndarray

    def __init__(self, in_dim, hidden_dims, key, scale=1.0):
        keys = jax.random.split(key, len(hidden_dims) + 1)
        self.Ws = []
        self.Us = []
        prev = in_dim
        for i, h in enumerate(hidden_dims):
            W = jax.random.normal(keys[i], (h, prev)) * scale
            U = jax.random.normal(keys[i], (h, in_dim)) * scale
            self.Ws.append(W)
            self.Us.append(U)
            prev = h
        self.final_W = jax.random.normal(keys[-1], (prev,)) * scale

    def __call__(self, x):
        z = x
        for W, U in zip(self.Ws, self.Us):
            W_pos = positive_param(W)  # enforce W_z >= 0 for convexity
            z = jax.nn.relu(z @ W_pos.T + x @ U.T)  # no biases, ReLU is 1-homog
        return jnp.dot(z, positive_param(self.final_W))
