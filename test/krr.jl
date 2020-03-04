using PERK, Test, Random, LinearAlgebra, Statistics

# Make sure PERK.krr computes Eq. 11 in Nataraj et al. for exact kernels
function test_krr_1()

    Random.seed!(0)
    T = 100
    xtrain = LinRange(0, 1, T)
    ytrain = xtrain .+ 0.1 .* randn(T)
    kernel = EuclideanKernel()
    trainData = PERK.krr_train(xtrain, ytrain, kernel)
    ytest = 0.6
    ρ = 0.01
    xhat = PERK.krr(ytest, trainData, kernel, ρ)

    K = ytrain * ytrain'
    M = I - ones(T, T) / T
    k = ytest * conj(ytrain) - K * ones(T) / T
    x_correct = xtrain' * (ones(T) / T + M * ((M * K * M + T * ρ * I) \ k))

    return xhat ≈ x_correct

end

function test_krr_2()

    Random.seed!(0)
    T = 100
    D = 3
    xtrain = LinRange(0, 1, T)
    ytrain = xtrain' .+ 0.1 .* randn(D, T)
    Λ = ones(D)
    kernel = GaussianKernel(Λ)
    trainData = PERK.krr_train(xtrain, ytrain, kernel)
    ytest = transpose([0.6 0.57 0.67])
    ρ = 0.01
    xhat = PERK.krr(ytest, trainData, kernel, ρ)

    gauss = (p, q) -> exp(-0.5 * norm((1 ./ Λ) .* (p - q))^2)
    K = [gauss(ytrain[:,m], ytrain[:,n]) for m = 1:T, n = 1:T]
    M = I - ones(T, T) / T
    k = [gauss(vec(ytest), ytrain[:,n]) for n = 1:T] - K * ones(T) / T
    x_correct = xtrain' * (ones(T) / T + M * ((M * K * M + T * ρ * I) \ k))

    return xhat[] ≈ x_correct

end

# Make sure PERK.krr with EuclideanKernel produces the same result as ridge
# regression
function test_krr_3()

    Random.seed!(0)
    T = 100
    xtrain = LinRange(0, 1, T)
    ytrain = xtrain .+ 0.1 .* randn(T)
    kernel = EuclideanKernel()
    trainData = PERK.krr_train(xtrain, ytrain, kernel)
    N = 10
    ytest = LinRange(0.1, 0.9, N)
    ρ = 0.01
    xhat = PERK.krr(ytest, trainData, kernel, ρ)

    xtilde = [xtrain .- mean(xtrain); 0]
    Ytilde = [ytrain .- mean(ytrain); sqrt(T * ρ)]
    w = Ytilde \ xtilde
    b = mean(xtrain) - w * mean(ytrain)
    x_correct = w .* ytest .+ b

    return xhat ≈ x_correct

end

function test_krr_4()

    Random.seed!(0)
    T = 100
    D = 3
    xtrain = LinRange(0, 1, T)
    ytrain = xtrain' .+ 0.1 .* randn(D, T)
    kernel = EuclideanKernel()
    trainData = PERK.krr_train(xtrain, ytrain, kernel)
    N = 10
    ytest = LinRange(0.1, 0.9, N)' .+ 0.1 .* randn(D, N)
    ρ = 0.01
    xhat = PERK.krr(ytest, trainData, kernel, ρ)

    xtilde = [xtrain .- mean(xtrain); zeros(D)]
    Ytilde = [(ytrain .- mean(ytrain, dims = 2))'; sqrt(T * ρ) * I]
    w = Ytilde \ xtilde
    b = mean(xtrain) - w' * dropdims(mean(ytrain, dims = 2), dims = 2)
    x_correct = [w' * ytest[:,n] + b for n = 1:N]

    return xhat ≈ x_correct

end

@testset "Kernel Ridge Regression" begin

    @test test_krr_1()
    @test test_krr_2()
    @test test_krr_3()
    @test test_krr_4()

end
