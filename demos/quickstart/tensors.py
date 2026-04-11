# ---
# jupyter:
#   jupytext:
#     default_lexer: ipython3
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.18.1
#   kernelspec:
#     display_name: jaxmat-env
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Working with tensors
#
# This notebook introduces how tensors are handled in `jaxmat`.

# %%
import jax

jax.config.update("jax_platform_name", "cpu")
import jax.numpy as jnp
import timeit
from jaxmat.tensors import SymmetricTensor4, Tensor4, SymmetricTensor2, Tensor, Tensor2

# %% [markdown]
# ## 2nd-rank tensors
#
# ### Tensorial and array representation
#
# `jaxmat` tensors are all sub-instances of the abstract `Tensor` class, which itself inherits from `equinox.Module`. Each tensor subclass corresponds to a specific symmetry class. For instance, 2nd-rank tensors are either `Tensor2` for non-symmetric or `SymmetricTensor2` for symmetric tensors. Tensor metadata contains for instance its dimension (`dim`, always fixed to 3 currently) and rank (`rank`).

# %%
T = Tensor2()
print(f"Tensor dimension = {T.dim}\nTensor rank = {T.rank}")

# %% [markdown]
# Importantly, tensor components are stored in a minimal format, accessible via the `array` property. For convenience, the `tensor` property provides a view to the corresponding tensorial form. For instance for a `SymmetricTensor2` `T`, `T.tensor` returns a (3,3) array, but its minimal components are stored in `T.array` which is a vector of length 6. For a `Tensor2`, `T.tensor` returns a (3,3) array as well but `T.array` is a vector of length 9. These array components correspond to the Kelvin-Mandel representation:
#
# - for symmetric 2nd-rank tensors:
#
# \begin{align*}
# \text{T.tensor} & =[\boldsymbol{T}]= \begin{bmatrix}T_{11} & T_{12} & T_{13} \\ T_{12} & T_{22} & T_{23} \\ T_{13} & T_{23} & T_{33}\end{bmatrix}\\
# \text{T.array} & =\{\boldsymbol{T}\}= \begin{Bmatrix}T_{11} & T_{22} & T_{33} & \sqrt{2}T_{12} & \sqrt{2}T_{13} & \sqrt{2}T_{23} \end{Bmatrix}^\text{T}
# \end{align*}
#
# - for generic 2nd-rank tensors:
#
# \begin{align*}
# \text{T.tensor} & =[\boldsymbol{T}]= \begin{bmatrix}T_{11} & T_{12} & T_{13} \\ T_{21} & T_{22} & T_{23} \\ T_{31} & T_{32} & T_{33}\end{bmatrix}\\
# \text{T.array} & =\{\boldsymbol{T}\}= \begin{Bmatrix}T_{11} & T_{22} & T_{33} & T_{12} & T_{21} & T_{13} & T_{31} & T_{23} & T_{32} \end{Bmatrix}^\text{T}
# \end{align*}
#
# The shape of the tensorial representation can be accessed from the `tensor_shape` property, while the shape of the array representation is given by the `shape` property.
#
# ### Instantiation
#
# As a result, tensors can be instantiated either via their tensorial or array representation. For instance, the symmetric tensor $\bT = \be_1\otimes\be_2+\be_2\otimes\be_1$ can be instantiated either by:

# %%
T = SymmetricTensor2(tensor=jnp.asarray([[0.0, 1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 0.0]]))
print(f"Tensor representation of shape {T.tensor_shape}:\n", T.tensor)
print(f"Array representation of shape {T.shape}:\n", T.array)

# %% [markdown]
# or by:

# %%
T_ = SymmetricTensor2(array=jnp.asarray([0.0, 0.0, 0.0, jnp.sqrt(2), 0.0, 0.0]))
print(f"Tensor representation of shape {T_.tensor_shape}:\n", T_.tensor)
print(f"Array representation of shape {T_.shape}:\n", T_.array)

# %% [markdown]
# and similarly for a the non-symmetric tensor $\bT = \be_1\otimes\be_2$:

# %%
T = Tensor2(tensor=jnp.asarray([[0.0, 1.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]))
print(f"Tensor representation of shape {T.tensor_shape}:\n", T.tensor)
print(f"Array representation of shape {T.shape}:\n", T.array)
# or
T_ = Tensor2(array=jnp.asarray([0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0]))
print(f"Tensor representation of shape {T_.tensor_shape}:\n", T_.tensor)
print(f"Array representation of shape {T_.shape}:\n", T_.array)

# %% [markdown]
# We provide a convenience intializer for the identity tensor $\boldsymbol{I}$, which can be materialized as a `SymmetricTensor2` or a `Tensor2`:

# %%
Id = SymmetricTensor2.identity()
print(Id.array)
Id = Tensor2.identity()
print(Id.array)

# %% [markdown]
# A non-symmetric tensor can be promoted to a `SymmetricTensor2` using the `sym` property, computing its symmetric part.

# %%
A = Tensor2(tensor=jnp.array([[1.0, 1.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]))
print(f"A = {A}")
print(f"sym(A) = {A.sym}")

# %% [markdown]
# ### Operations on tensors
#
# Attention must be paid when performing linear algebra on tensors. By default, when calling a mathematical operation on a tensor, it will behave like `jnp.Array` with respect to its **tensorial representation**. For instance:
#
# If a specific operation should be applied to its array presentation only, the user is required to explicitly call it on its `array` property. Transposition can be done using the `.T` property.

# %%
a = SymmetricTensor2(tensor=jnp.diag(jnp.asarray([1.0, 2.0, 3.0])))
print("Trace:", a.tr)
print(jnp.linalg.norm(a))
print(a.T)

# %% [markdown]
# Addition, substraction, scalar multiplication and division operators have been overloaded. Note that addition and substraction preserve symmetry of the tensors (weakest symmetry is used in case of two tensors with different symmetries).

# %%
b = SymmetricTensor2(tensor=jnp.ones((3, 3)))
print(type(a + b), a + b)
print(type(a - b), a - b)
print(type(a - Tensor2()))

# %% [markdown]
# Matrix multiplication operator `@` is also overloaded, understood here as a simple contraction. Generally, we can prefer `jnp.einsum` for linear algebra expressions. Double contraction is available via the `double_contract` method and is defined as $\bA:\bB=A_{ij}B_{ij}$.

# %%
print(Id @ a)
print(Id.double_contract(a))

# %% [markdown]
# `inv` and `det` properties allow to compute the inverse and determinant using custom expressions for $3\times 3$ matrices.

# %%
print("Inverse:", a.inv)
print("Determinant", a.det)

# %% [markdown]
# ## Batching
#
# Tensors can have any number of batch dimensions in addition to their base dimensions. By default, batch dimensions are always appended as first dimensions of the tensor, the last $r$ dimensions being the real tensorial dimensions for a tensor of rank $r$. For instance, the following tensor will be considered as a `SymmetricTensor2` of base tensor shape `(3,3)` and a single batch dimension of length 10. The total shape (array or tensorial) is always given by `batch_shape + base_shape`.

# %%
A = SymmetricTensor2(tensor=jnp.broadcast_to(jnp.eye(3), shape=(10, 3, 3)))
print(f"Tensorial shape = {A.tensor_shape}")
print(f"Array shape = {A.shape}")
print(f"Base tensorial shape = {A.base_tensor_shape}")
print(f"Base array shape = {A.base_array_shape}")
print(f"Batch shape = {A.batch_shape}")

# %% [markdown]
# Below, we consider the case of 2 batch dimensions of length 3 each, resulting in a tensor of total shape (3,3,3,3):

# %%
B = SymmetricTensor2(tensor=jnp.broadcast_to(jnp.eye(3), shape=(3, 3, 3, 3)))
print(f"Tensorial shape = {B.tensor_shape}")
print(f"Array shape = {B.shape}")
print(f"Base tensorial shape = {B.base_tensor_shape}")
print(f"Base array shape = {B.base_array_shape}")
print(f"Batch shape = {B.batch_shape}")

# %% [markdown]
# ## 4th-rank tensors
#
# 4th-rank tensors come in two flavors: `SymmetricTensor4` or `Tensor4`.
#
# `SymmetricTensor4` can be seen as a mapping from `SymmetricTensor2` to `SymmetricTensor2` objects. They satisfy minor symmetries: $C_{ijkl}=C_{jikl}=C_{ijlk}=C_{jilk}$. Using Kelvin-Mandel representation, they can be minimally stored as a $6x6$ matrix:
#
# $$[\mathbb{C}]=\begin{bmatrix}
# C_{1111} & C_{1122} & C_{1133} & \sqrt{2}\,C_{1112} & \sqrt{2}\,C_{1113} & \sqrt{2}\,C_{1123} \\
# C_{2211} & C_{2222} & C_{2233} & \sqrt{2}\,C_{2212} & \sqrt{2}\,C_{2213} & \sqrt{2}\,C_{2223} \\
# C_{3311} & C_{3322} & C_{3333} & \sqrt{2}\,C_{3312} & \sqrt{2}\,C_{3313} & \sqrt{2}\,C_{3323} \\
# \sqrt{2}\,C_{1211} & \sqrt{2}\,C_{1222} & \sqrt{2}\,C_{1233} & 2\,C_{1212} & 2\,C_{1213} & 2\,C_{1223} \\
# \sqrt{2}\,C_{1311} & \sqrt{2}\,C_{1322} & \sqrt{2}\,C_{1333} & 2\,C_{1312} & 2\,C_{1313} & 2\,C_{1323} \\
# \sqrt{2}\,C_{2311} & \sqrt{2}\,C_{2322} & \sqrt{2}\,C_{2333} & 2\,C_{2312} & 2\,C_{2313} & 2\,C_{2323}
# \end{bmatrix}$$
#
# ```{attention}
# Elastic sitffness tensors also satisfy major symmetry $C_{ijkl}=C_{klij}$, which results in $[\mathbb{C}]$ being a $6\times 6$ symmetric matrix. `SymmetricTensor4` objects do not necessarily assume major symmetry, as the latter can for instance be lost when considering tangent stiffness operators with non-associated plasticity.
# ```
#
# `Tensor4` can be seen as a mapping from `Tensor2` to `Tensor2` objects. No specific symmetry is verified in this case. They can be minimally stored as a $9\times 9$ matrix.
#
# $$[\mathbb{C}] =
# \begin{bmatrix}
# C_{1111} & C_{1122} & C_{1133} & C_{1112} & C_{1121} & C_{1113} & C_{1131} & C_{1123} & C_{1132} \\
# C_{2211} & C_{2222} & C_{2233} & C_{2212} & C_{2221} & C_{2213} & C_{2231} & C_{2223} & C_{2232} \\
# C_{3311} & C_{3322} & C_{3333} & C_{3312} & C_{3321} & C_{3313} & C_{3331} & C_{3323} & C_{3332} \\
# C_{1211} & C_{1222} & C_{1233} & C_{1212} & C_{1221} & C_{1213} & C_{1231} & C_{1223} & C_{1232} \\
# C_{2111} & C_{2122} & C_{2133} & C_{2112} & C_{2121} & C_{2113} & C_{2131} & C_{2123} & C_{2132} \\
# C_{1311} & C_{1322} & C_{1333} & C_{1312} & C_{1321} & C_{1313} & C_{1331} & C_{1323} & C_{1332} \\
# C_{3111} & C_{3122} & C_{3133} & C_{3112} & C_{3121} & C_{3113} & C_{3131} & C_{3123} & C_{3132} \\
# C_{2311} & C_{2322} & C_{2333} & C_{2312} & C_{2321} & C_{2313} & C_{2331} & C_{2323} & C_{2332} \\
# C_{3211} & C_{3222} & C_{3233} & C_{3212} & C_{3221} & C_{3213} & C_{3231} & C_{3223} & C_{3232}
# \end{bmatrix}$$
#
# The identity can be obtained from:

# %%
I4s = SymmetricTensor4.identity()
print(I4s.array)
print("Trace of I4s =", I4s.fourth_contract(I4s))
I4 = Tensor4.identity()
print(I4.array)
print("Trace of I4 =", I4.fourth_contract(I4))

# %% [markdown]
# ## Symmetry classes and projectors
#
# The `tensors.symmetry_classes` module provides functionalities for dealing with material symmteries, especialling regarding elasticity tensors. Below we can define the isotropic projectors $\mathbb{J}$ and $\mathbb{K}$. We can check that they are orthogonal and that $\mathbb{J}::\mathbb{J}=1$ and $\mathbb{K}::\mathbb{K}=5$.

# %%
from jaxmat.tensors.symmetry_classes import isotropic_projectors, IsotropicTensor4

J, K = isotropic_projectors()
print("J = ", J.array)
print("K = ", K.array)
print("J::J =", J.fourth_contract(J))
print("K::K =", K.fourth_contract(K))
print("K::J =", J.fourth_contract(K))

# %% [markdown]
# They can also be directly obtained as class attributes of the `IsotropicTensor4` class.

# %%
J = IsotropicTensor4.J
K = IsotropicTensor4.K

# %% [markdown]
# This allows to define an isotropic 4th-rank tensor which can always be expressed as $\mathbb{C}=3\kappa\mathbb{J}+2\mu\mathbb{K}$. The underlying coefficients being stored as $\{3\kappa,2\mu\}$ with respect to the $\mathbb{J}$, $\mathbb{K}$ basis.

# %%
kappa, mu = 1.0, 1.0
C = IsotropicTensor4(kappa=kappa, mu=mu)
print(C.array)
print("Coefficients =", C.coeffs)


# %% [markdown]
# Similar functionalities are available for cubic and transversely isotropic symmetries.
#
# ## Computing tangent stiffnesses
#
# Below, we define a Saint-Venant Kirchhoff potential and show how to derive the corresponding tangent stiffness. First, we compute the stress-strain relation using a first call to `jax.grad`.

# %%
def energy(eps):
    return 0.5 * eps.double_contract(C @ eps)


eps = SymmetricTensor2.identity()
sig = jax.grad(energy)
print(type(sig(eps)))

# %% [markdown]
# `sig(eps)` inherits from the type of `eps`. However, if we compute the jacobian once more, we will obtain a `SymmetricTensor2` with array shape `(6, 6)` and tensor shape `(6, 3, 3)`. This is because the PyTree is flattened when appending extra dimensions due to AD.

# %%
C_tang = jax.jacfwd(sig)
print(type(C_tang(eps)))
print(C_tang(eps).array.shape)

# %% [markdown]
#  If required, we need to explicitly promote it to a `SymmetricTensor4` object.

# %%
C_tang_tensor4 = lambda eps: SymmetricTensor4(array=jax.jacfwd(sig)(eps).array)
print(C_tang_tensor4(eps).array)
