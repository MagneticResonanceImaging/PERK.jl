# PERK.jl

[![Build Status](https://travis-ci.org/StevenWhitaker/PERK.jl.svg?branch=master)](https://travis-ci.org/StevenWhitaker/PERK.jl)
[![codecov](https://codecov.io/gh/StevenWhitaker/PERK.jl/branch/master/graph/badge.svg)](https://codecov.io/gh/StevenWhitaker/PERK.jl)

This package implements PERK, a parameter estimation technique presented in
[G. Nataraj, J.-F. Nielsen, C. Scott, and J. A. Fessler. Dictionary-free MRI PERK: Parameter estimation via regression with kernels. IEEE Trans. Med. Imag., 37(9):2103-14, September 2018](https://ieeexplore.ieee.org/document/8320384/).
This code was inspired by the MATLAB code written by Gopal Nataraj,
which can be found [here](https://github.com/gopal-nataraj/perk).

## Getting Started
At the Julia REPL, type `]` to enter the package prompt.
Then type `add https://github.com/StevenWhitaker/PERK.jl#v0.2.0`
to add PERK v0.2.0
(note that `v0.2.0` can be replaced with whatever version is needed).
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
