using PERK, Test, Random, Distributions

function test_krr_1()

    Random.seed!(0)
    f = x -> x
    T = 100
    xtrain = LinRange(0, 1, T)
    ytrain = f.(xtrain) .+ 0.1 .* randn(T)
    kernel = EuclideanKernel()
    trainData = PERK.krr_train(xtrain, ytrain, kernel)
    N = 10
    ytest = LinRange(0.1, 0.9, N)
    ρ = 0.01
    xhat = PERK.krr(ytest, trainData, kernel, ρ)
    ytilde = [ytrain .- mean(ytrain); 0]
    Xtilde = [xtrain .- mean(xtrain); sqrt(T * ρ)]
    w = Xtilde \ ytilde
    b = mean(ytrain) - w * mean(xtrain)
    x_correct = collect(w .* ytest .+ b)
    @show xhat x_correct
    return xhat ≈ x_correct

end

@testset "Ridge Regression" begin

    @test test_krr_1()

end
