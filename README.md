# PERK.jl

https://github.com/MagneticResonanceImaging/PERK.jl

[![action status][action-img]][action-url]
[![codecov][codecov-img]][codecov-url]
[![license][license-img]][license-url]
[![docs stable][docs-stable-img]][docs-stable-url]
[![docs dev][docs-dev-img]][docs-dev-url]

This package implements PERK, a parameter estimation technique presented in
[G. Nataraj, J.-F. Nielsen, C. Scott, and J. A. Fessler. Dictionary-free MRI PERK: Parameter estimation via regression with kernels. IEEE Trans. Med. Imag., 37(9):2103-14, September 2018](https://ieeexplore.ieee.org/document/8320384/).
This code was inspired by the MATLAB code written by Gopal Nataraj,
which can be found [here](https://github.com/gopal-nataraj/perk).

## Getting Started
At the Julia REPL, type `]` to enter the package prompt.
Then type `add https://github.com/MagneticResonanceImaging/PERK.jl`
to add the latest version.
(Append `#v0.3.2` to specify a specific version.)
Hit backspace to return to the normal Julia prompt,
and then type `using PERK` to load the package.

## Overview
The function `perk` provides the main functionality.
Training is done by generating synthetic data
using randomly generated parameters.
Distributions for these parameters can be passed directly to `perk`,
which will use them for training
and then estimate the latent parameters from the given test data.
Alternatively, one can pass the parameter distributions to `PERK.train`,
which will create a `TrainingData` object that can then be passed to `perk`
(and used multiple times, if desired).
One must also pass a `Kernel` object to `perk`.
Three are provided in this package:
`GaussianKernel`, `GaussianRFF`, and `EuclideanKernel`.
`GaussianKernel` and `GaussianRFF` are described in the paper.
Using `EuclideanKernel` indicates to solve ridge regression
instead of kernel ridge regression.

Because PERK utilizes kernel ridge regression at its core,
one can also use this package for solving kernel ridge regression.
(This can be useful if, e.g., one already has training data
and therefore does not need to generate some.)
Calling `PERK.krr_train` returns a `TrainingData` object
that can be passed to `PERK.krr`.
As with `perk`,
both `PERK.krr_train` and `PERK.krr` must be passed a `Kernel` object.

## Related packages

These are more general and not MRI specific.
- https://github.com/JuliaGaussianProcesses/KernelFunctions.jl
  [an example](https://juliagaussianprocesses.github.io/KernelFunctions.jl/dev/examples/kernel-ridge-regression)
- https://github.com/gdkrmr/KernelRidgeRegression.jl


## Status

This repo is not yet registered in part because its name
does not meet
[Julia naming guidelines](https://pkgdocs.julialang.org/v1/creating-packages/#Package-naming-guidelines).
If you want to be registered,
please post an issue and suggest a name.

On 2024-04-08,
the package was relocated here from
https://github.com/StevenWhitaker/PERK.jl
Some links may be broken by this transition.

<!-- URLs -->
[action-img]: https://github.com/MagneticResonanceImaging/PERK.jl/workflows/runtests/badge.svg
[action-url]: https://github.com/MagneticResonanceImaging/PERK.jl/actions
[codecov-img]: (https://codecov.io/github/MagneticResonanceImaging/PERK.jl/coverage.svg?branch=main
[codecov-url]: https://codecov.io/github/MagneticResonanceImaging/PERK.jl?branch=main
[docs-stable-img]: https://img.shields.io/badge/docs-stable-blue.svg
[docs-stable-url]: https://MagneticResonanceImaging.github.io/PERK.jl/stable
[docs-dev-img]: https://img.shields.io/badge/docs-dev-blue.svg
[docs-dev-url]: https://MagneticResonanceImaging.github.io/PERK.jl/dev
[license-img]: http://img.shields.io/badge/license-MIT-brightgreen.svg?style=flat
[license-url]: LICENSE
