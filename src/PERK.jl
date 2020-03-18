"""
    PERK

Module implementing parameter estimation via regression with kernels (PERK).

# Exports
- `EuclideanKernel`: Euclidean inner product kernel (for ridge regression
  instead of kernel ridge regression)
- `GaussianKernel`: Gaussian kernel used in kernel ridge regression
- `GaussianRFF`: Approximation of Gaussian kernel using random Fourier features
- `generatenoisydata`: Function for generating noisy data
- `perk`: Function for running PERK
"""
module PERK

using LinearAlgebra: I, Diagonal, norm
using Statistics: mean

include("utils.jl")
include("kernels.jl")
include("training.jl")
include("krr.jl")
include("estimation.jl")
include("holdout.jl")

export EuclideanKernel
export GaussianKernel
export GaussianRFF
export generatenoisydata
export perk

end
