using PERK, Test, Random, LinearAlgebra, Statistics, Distributions

# Make sure PERK.krr computes Eq. 11 in Nataraj et al. for exact kernels
function test_krr_1()

    Random.seed!(0)
    T = 100
    xtrain = LinRange(0, 1, T)
    ytrain = xtrain .+ 0.1 .* randn(T)
    kernel = EuclideanKernel()
    ρ = 0.01
    trainData = PERK.krr_train(xtrain, ytrain, kernel, ρ)
    ytest = 0.6
    xhat = PERK.krr(ytest, trainData, kernel)

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
    ρ = 0.01
    trainData = PERK.krr_train(xtrain, ytrain, kernel, ρ)
    ytest = transpose([0.6 0.57 0.67])
    xhat = PERK.krr(ytest, trainData, kernel)

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
    ρ = 0.01
    trainData = PERK.krr_train(xtrain, ytrain, kernel, ρ)
    N = 10
    ytest = LinRange(0.1, 0.9, N)
    xhat = PERK.krr(ytest, trainData, kernel)

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
    ρ = 0.01
    trainData = PERK.krr_train(xtrain, ytrain, kernel, ρ)
    N = 10
    ytest = LinRange(0.1, 0.9, N)' .+ 0.1 .* randn(D, N)
    xhat = PERK.krr(ytest, trainData, kernel)

    xtilde = [xtrain .- mean(xtrain); zeros(D)]
    Ytilde = [(ytrain .- mean(ytrain, dims = 2))'; sqrt(T * ρ) * I]
    w = Ytilde \ xtilde
    b = mean(xtrain) - w' * dropdims(mean(ytrain, dims = 2), dims = 2)
    x_correct = [w' * ytest[:,n] + b for n = 1:N]

    return xhat ≈ x_correct

end

# Other tests
function test_krr_5()

    Random.seed!(0)
    f = x -> exp(-30 / x)
    xtrue = 100
    y = f(xtrue)
    T = 200
    xDists = Uniform(10, 500)
    noiseDist = Normal(0, 0.01)
    signalModels = f
    λ = 2.0^-1.5
    H = 100
    kernel = GaussianRFF(H, λ * mean(y))
    ρ = 2.0^-20
    (ytrain, xtrain) = generatenoisydata(T, xDists, noiseDist, signalModels)
    trainData = PERK.krr_train(xtrain, ytrain, kernel, ρ, randn(H), rand(H))
    xhat = PERK.krr(y, trainData, kernel)

    error_rel = abs(xhat - xtrue) / xtrue
    return isapprox(error_rel, 0.05798742886313903, atol = 1e-6)

end

function test_krr_6()

    Random.seed!(0)
    f = x -> exp(-30 / x)
    xtrue = 100
    y = f(xtrue)
    T = 200
    xDists = Uniform(10, 500)
    noiseDist = Normal(0, 0.01)
    signalModels = f
    λ = 2.0^-1.5
    H = 100
    kernel = GaussianRFF(H, λ * mean(y))
    ρ = 2.0^-20
    (ytrain, xtrain) = generatenoisydata(T, xDists, noiseDist, signalModels)
    trainData = PERK.krr_train(xtrain, ytrain, kernel, ρ, randn(H,1), rand(H))
    xhat = PERK.krr(y, trainData, kernel)

    error_rel = abs(xhat - xtrue) / xtrue
    return isapprox(error_rel, 0.05798742886313903, atol = 1e-6)

end

@testset "Kernel Ridge Regression" begin

    @test test_krr_1()
    @test test_krr_2()
    @test test_krr_3()
    @test test_krr_4()
    @test test_krr_5()
    @test test_krr_6()

end
