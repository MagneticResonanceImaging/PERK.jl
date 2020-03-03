module PERK

using LinearAlgebra, Statistics, Random

include("utils.jl")
include("kernels.jl")
include("krr.jl")
include("training.jl")
include("estimation.jl")
include("holdout.jl")

export perk
export generatenoisydata
export GaussianKernel, GaussianRFF
export EuclideanKernel

end
