module PERK

using LinearAlgebra, Statistics, Random

include("utils.jl")
include("kernels.jl")
include("training.jl")
include("estimation.jl")
include("holdout.jl")

export perk
export train, generatenoisydata
export GaussianKernel, GaussianRFF

end
