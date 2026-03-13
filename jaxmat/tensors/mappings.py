
import jax
import jax.numpy as jnp

IndexMap2 = tuple[jax.Array, jax.Array]
IndexMap4 = tuple[jax.Array, jax.Array, jax.Array, jax.Array]


def full_rank2_map(d: int) -> tuple[IndexMap2, jax.Array]:
    r"""
    Construct an index map for a full (non-symmetric) rank-2 tensor.

    The flat array ordering is:

    $$\{\boldsymbol{A}\} = [A_{11}, A_{22}, A_{33}, A_{12}, A_{21}, A_{13}, A_{31}, A_{23}, A_{32}]$$

    for $d=3$.

    Parameters
    ----------
    d : int
        Spatial dimension.

    Returns
    -------
    (I, J) : tuple[jax.Array, jax.Array]
        Index arrays of shape ``(d*d,)`` such that:

        ``tensor[..., I[k], J[k]] <-> array[..., k]``

    W : jax.Array
        Weight array of shape ``(d*d,)``, filled with ones.
        (Provided for compatibility with weighted indexed tensor backends.)
    """
    I = []
    J = []

    # Diagonal terms
    for i in range(d):
        I.append(i)
        J.append(i)

    # Off-diagonal symmetric pairs
    for i in range(d):
        for j in range(i + 1, d):
            I.append(i)
            J.append(j)
            I.append(j)
            J.append(i)

    I = jnp.array(I, dtype=jnp.int16)
    J = jnp.array(J, dtype=jnp.int16)
    W = jnp.ones_like(I, dtype=jnp.float32)

    return (I, J), W


def kelvin_rank2_map(d: int) -> tuple[IndexMap2, jax.Array]:
    r"""
    Construct an index map for a symmetric rank-2 tensor
    in Kelvin-Mandel notation.

    The Kelvin vector is defined as:

    $$\{\boldsymbol{A}\} = [A_{11}, A_{22}, A_{33}, \sqrt{2} A_{12}, \sqrt{2} A_{13}, \sqrt{2} A_{23}]$$

    for $d=3$.
    This scaling makes the Kelvin basis orthonormal, ensuring:

    $$\boldsymbol{A} : \boldsymbol{B} = \{\boldsymbol{A}\} \cdot \{\boldsymbol{B}\}$$

    Parameters
    ----------
    d : int
        Spatial dimension.

    Returns
    -------
    (I, J) : tuple[jax.Array, jax.Array]
        Index arrays of shape ``(d(d+1)/2,)`` mapping each Kelvin
        component to a tensor slot.

    W : jax.Array
        Weight array of shape ``(d(d+1)/2,)`` containing $1$ for diagonal components, $\sqrt{2}$ for off-diagonal components.
    """
    sqrt2 = jnp.sqrt(2.0)

    # Diagonal
    i_diag = jnp.arange(d)
    j_diag = i_diag

    # Upper triangular off-diagonal (i < j)
    i_off, j_off = jnp.triu_indices(d, k=1)

    I = jnp.concatenate([i_diag, i_off])
    J = jnp.concatenate([j_diag, j_off])

    W = jnp.concatenate(
        [
            jnp.ones_like(i_diag, dtype=float),
            jnp.full(i_off.shape, sqrt2),
        ]
    )

    return (I, J), W


def full_rank4_map(d: int) -> tuple[IndexMap4, jax.Array]:
    r"""
    Construct an index map for a full (non-symmetric) rank-4 tensor (Tensor4).

    The mapping is defined as:

    $$C_{\alpha\beta} = C_{ijkl}$$

    where $\alpha$ and $\beta$ are the Kelvin coefficients of the $(i,j)$ and $(k,l)$ index pairs.

    For $d = 3$, this produces a $9\times 9$ representation of a 4th-rank tensor

    Parameters
    ----------
    d : int
        Spatial dimension.

    Returns
    -------
    (I, J, K, L) : tuple[jax.Array, jax.Array, jax.Array, jax.Array]
        Index arrays of shape (N, N) where N = d*d, mapping:

        ``tensor[..., I[a,b], J[a,b], K[a,b], L[a,b]] <-> array[..., a, b]``

    W : jax.Array
        Weight array of shape (N, N), filled with ones.
    """

    # Rank-2 full (non-symmetric) map
    (I2, J2), W2 = full_rank2_map(d)
    N = I2.shape[0]  # N = d*d

    a, b = jnp.meshgrid(jnp.arange(N), jnp.arange(N), indexing="ij")

    I = I2[a]
    J = J2[a]
    K = I2[b]
    L = J2[b]

    # No Kelvin scaling for full Tensor4
    W = jnp.ones((N, N), dtype=jnp.float32)

    return (I, J, K, L), W


def kelvin_rank4_map(d: int) -> tuple[IndexMap4, jax.Array]:
    r"""
    Construct an index map for a symmetric rank-4 tensor in Kelvin-Mandel notation.

    The 4th-rank Kelvin tensor is defined as:

    $$C_{\alpha\beta} = s_{\alpha} s_{\beta} C_{ijkl}$$

    where $\alpha$ and $\beta$ are the Kelvin coefficients (weights $s_{\alpha},s_{\beta}$ are 1 and $\sqrt{2}$
    for shear components) of the $(i,j)$ and $(k,l)$ index pairs.

    For $d = 3$, this produces a $6\times 6$ representation of a 4th-rank tensor with minor symmetries.

    Parameters
    ----------
    d : int
        Spatial dimension.

    Returns
    -------
    (I, J, K, L) : tuple[jax.Array, jax.Array, jax.Array, jax.Array]
        Index arrays of shape (N, N) where N = d(d+1)/2, mapping:

        ``tensor[..., I[a,b], J[a,b], K[a,b], L[a,b]] <-> kelvin[..., a, b]``

    W : jax.Array
        Weight array of shape (N, N) with Kelvin scaling:

        ``W[a,b] = S[a] * S[b]``

    Notes
    -----
    This mapping enforces minor symmetry (ij) and (kl) by construction.
    """
    # Rank-2 Kelvin map
    (I2, J2), W2 = kelvin_rank2_map(d)
    N = I2.shape[0]

    # Build rank-4 index grids
    a, b = jnp.meshgrid(jnp.arange(N), jnp.arange(N), indexing="ij")

    I = I2[a]
    J = J2[a]
    K = I2[b]
    L = J2[b]

    # Kelvin–Mandel weights
    W = W2[a] * W2[b]

    return (I, J, K, L), W
