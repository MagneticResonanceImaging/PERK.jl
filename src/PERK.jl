module PERK

using LinearAlgebra: I, Diagonal, norm
using Statistics: mean

include("utils.jl")
include("kernels.jl")
include("training.jl")
include("krr.jl")
include("estimation.jl")
include("holdout.jl")

export perk
export generatenoisydata
export GaussianKernel
export GaussianRFF
export EuclideanKernel

end
