# Conventions for representing tensors

## 2nd-rank tensors

2nd-rank tensors are represented as vectors using the Mandel representation.
Components ordering follow the conventions used by the MFront project
[described here](https://thelfer.github.io/tfel/web/tensors.html).

A symmetric 2nd-rank tensor $\boldsymbol{a}$ stored as follows in 3D:

$$
\{\boldsymbol{a}\} = \begin{Bmatrix}a_{11} & a_{22} & a_{33} & \sqrt{2}a_{12} & \sqrt{2}a_{13} & \sqrt{2}a_{23} \end{Bmatrix}^\text{T}
$$

and in 2D:

$$
\{\boldsymbol{a}\} = \begin{Bmatrix}a_{11} & a_{22} & \sqrt{2}a_{12} \end{Bmatrix}^\text{T}
$$

A non-symmetric 2nd-rank tensor $\boldsymbol{a}$ stored as follows in 3D:

$$
\{\boldsymbol{a}\} = \begin{Bmatrix}a_{11} & a_{22} & a_{33} & a_{12} & a_{21} & a_{13} & a_{31} & a_{23} & a_{32} \end{Bmatrix}^\text{T}
$$

and similarly in 2D:

$$
\{\boldsymbol{a}\} = \begin{Bmatrix}a_{11} & a_{22} & a_{12} & a_{21} \end{Bmatrix}^\text{T}
$$

## 4th-rank tensors

4th-rank tensors are represented as matrices with components complying with the
representation of 2nd-rank tensors.

For instance, a symmetric 4th-order tensor in 2D will be represented as:

$$
[\mathbf{C}] &= \begin{bmatrix}
C_{1111} & C_{1122} & \sqrt{2}C_{1112} \\
C_{2211} & C_{2222} & \sqrt{2}C_{2212} \\
\sqrt{2}C_{1112} & \sqrt{2}C_{2212} & 2C_{1212}
\end{bmatrix}
$$
