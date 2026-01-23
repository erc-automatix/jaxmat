# `jaxmat`: Automated material constitutive modeling in JAX

`jaxmat` is an open-source library for implementing material constitutive models in a way that integrates seamlessly with modern machine learning frameworks and existing finite element software.

## Aim and scope

The library is implemented in [JAX](https://github.com/google/jax) and heavily relies on additional features provided by the JAX ecosystem, in particular the [`equinox`](https://docs.kidger.site/equinox/), [`diffrax`](https://docs.kidger.site/diffrax/), and [`optimistix`](https://docs.kidger.site/optimistix/) libraries. The design choices are based on the following premises:

1. **User-friendliness**: `jaxmat` provides domain-specific abstractions that simplify the definition of complex material models. It is not a database of hard-coded material behaviors with fixed parameters. Instead, it offers a generic and extensible framework for users to define, compose, and calibrate their own constitutive models with minimal boilerplate.

2. **Differentiability**: We fully exploit JAX’s *Automatic Differentiation* (AD) capabilities to eliminate the need for manually deriving consistent tangent operators, Jacobians of implicit systems, normal vectors to plastic surfaces, hyperelastic stress expressions, and more.
In `jaxmat`, every input and parameter of a constitutive model is differentiable, making it easy to compute sensitivities with respect to material or even algorithmic parameters! These features are essential for tasks such as material parameter identification, solving inverse problems, and performing uncertainty quantification.

3. **Modularity**: The library is designed to be highly modular, allowing users to easily mix and match modeling components. For example, in a generic elastoviscoplastic model, you can independently swap out the plastic yield surface, the hardening laws or the viscous flow, independently. Each component can also be replaced with a data-driven alternative, such as a neural network, without disrupting the overall structure.

4. **Efficiency**: `jaxmat` makes full use of JAX features such as *Just-In-Time compilation* (`jax.jit`), *automatic vectorization* (`jax.vmap`) and hardware acceleration (CPU or GPU through the XLA compiler). Constitutive models can be evaluated extremely efficiently and in parallel across batches of material points. Additionally, we have designed and selected algorithms specifically suited to the needs of computational mechanics for maximum performance, robustness, and generality, see the [](sharp_bits.md) section.

## Installation

Simply run a `pip` install. Use `jaxmat[gpu]` for GPU support.

```bash
pip install jaxmat
```

You may need to update `nvidia-cublas-cu12` as there are some issues with the version shipped with `jax-0.6.1`.

```bash
pip install --upgrade nvidia-cublas-cu12
```

## Citation

The project can be cited from Zenodo:


[![DOI](https://zenodo.org/badge/1018223703.svg)](https://doi.org/10.5281/zenodo.17611833)



## Acknowledgements

<p align="center">
  <img src="images/banner_erc.png" width=800 />
</p>

*Funded by the European Union (ERC, Automatix, CoG project no 101229452). Views and opinions expressed are however those of the author(s) only and do not necessarily reflect those of the European Union or the European Research Council. Neither the European Union nor the granting authority can be held responsible for them.*
