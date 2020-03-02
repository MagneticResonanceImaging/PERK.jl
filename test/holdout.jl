using PERK, Test, Random, Distributions

function test_holdout_1()

    N = 10
    T = 200
    λvals = [1, Inf]
    ρvals = [1, Inf]
    weights = [1]
    xDistsTest = [Uniform(100, 100.000000001)]
    νDistsTest = [Uniform(30, 30.000000001)]
    xDistsTrain = [Uniform(10, 500)]
    νDistsTrain = νDistsTest
    noiseDist = Normal(0, 0.01)
    signalModels = [(x, ν) -> exp(-ν / x)]
    kernelgenerator = Λ -> GaussianKernel(Λ)
    (λ, ρ, Ψ) = PERK.holdout(N, T, λvals, ρvals, weights, xDistsTest,
                             νDistsTest, xDistsTrain, νDistsTrain, noiseDist,
                             signalModels, kernelgenerator, showprogress = true)
    @show Ψ
    return λ == 1 && ρ == 1

end

@testset "Holdout" begin

    @test test_holdout_1()

end
