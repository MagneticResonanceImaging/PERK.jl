# PERK.jl
This package implements PERK, a parameter estimation technique presented in [G. Nataraj, J.-F. Nielsen, C. Scott, and J. A. Fessler. Dictionary-free MRI PERK: Parameter estimation via regression with kernels. IEEE Trans. Med. Imag., 37(9):2103-14, September 2018](https://ieeexplore.ieee.org/document/8320384/). This code was inspired by the MATLAB code written by Gopal Nataraj, which can be found [here](https://github.com/gopal-nataraj/perk).

## Getting Started
At the Julia REPL, type `]` to enter the package prompt. Then type `add https://github.com/StevenWhitaker/PERK.jl#v0.0.1` to add PERK v0.0.1 (note that `v0.0.1` can be replaced with whatever version is needed). Hit backspace to return to the normal Julia prompt, and then type `using PERK` to load the package.

## Overview
The function `perk` provides the main functionality. Training is done by generating synthetic data using randomly generated parameters. Distributions for these parameters can be passed directly to `perk`, which will use them for training and then estimate the latent parameters from the given test data. Alternatively, one can pass the parameter distributions to `train`, which will create a `TrainingData` object that can then be passed to `perk` (and used multiple times, if desired). One must also pass a `Kernel` object to `perk`. Two are provided in this package: `GaussianKernel` and `GaussianRFF`, both of which are described in the paper.
