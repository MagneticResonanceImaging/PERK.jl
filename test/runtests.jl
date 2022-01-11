using Distributions
using ForwardDiff
using LinearAlgebra
using PERK
using Random
using Statistics
using Test

@testset "PERK.jl" begin
    include("kernels.jl")
    include("krr.jl")
    include("estimation.jl")
    include("complex.jl")
    include("holdout.jl")
    include("forwarddiff.jl")
end
