using PERK, Test, Random, LinearAlgebra, Statistics, Distributions

function test_perk_1()

    Random.seed!(0)
    f = x -> exp(-30 / x)
    xtrue = 100
    y = f(xtrue)
    T = 200
    xDists = Uniform(10, 500)
    noiseDist = Normal(0, 0.01)
    signalModels = f
    λ = 2.0^-1.5
    kernel = GaussianKernel([λ * mean(y)])
    ρ = 2.0^-20
    xhat = perk(y, T, xDists, noiseDist, signalModels, kernel, ρ)

    error_rel = abs(xhat - xtrue) / xtrue
    return isapprox(error_rel, 0.02498437176144705, atol = 1e-6)

end

function test_perk_2()

    Random.seed!(0)
    f = x -> exp(-30 / x)
    xtrue = 20:490
    N = 1
    T = 200
    xDists = Uniform(10, 500)
    noiseDist = Normal(0, 0.01)
    signalModels = f
    λ = 2.0^-1.5
    ρ = 2.0^-20
    error_rel = zeros(length(xtrue))
    for i = 1:length(xtrue)
        y = fill(f(xtrue[i]), 1, N)
        kernel = GaussianKernel([λ * mean(y)])
        xhat = perk(y, T, xDists, noiseDist, signalModels, kernel, ρ)
        error_rel[i] = abs(xhat[] - xtrue[i]) / xtrue[i]
    end

    error_rel_avg = sum(error_rel) / length(error_rel)
    return isapprox(error_rel_avg, 0.045074705837992585, atol = 1e-6)

end

function test_perk_3()

    Random.seed!(0)
    f = x -> exp(-30 / x)
    xtrue = 100
    N = 1
    y = fill(f(xtrue), 1, N)
    T = 200
    xDists = [Uniform(10, 500)]
    noiseDist = Normal(0, 0.01)
    signalModels = f
    λ = 2.0^-1.5
    H = 100
    kernel = GaussianRFF(H, [λ * mean(y)])
    ρ = 2.0^-20
    xhat = perk(y, T, xDists, noiseDist, signalModels, kernel, ρ)

    error_rel = abs(xhat[] - xtrue) / xtrue
    return isapprox(error_rel, 0.05798742886313903, atol = 1e-6)

end

function test_perk_4()

    Random.seed!(0)
    f = x -> exp(-30 / x)
    xtrue = 20:490
    T = 200
    xDists = [Uniform(10, 500)]
    noiseDist = Normal(0, 0.01)
    signalModels = f
    λ = 2.0^-1.5
    ρ = 2.0^-20
    H = 100
    error_rel = zeros(length(xtrue))
    for i = 1:length(xtrue)
        y = f(xtrue[i])
        kernel = GaussianRFF(H, [λ * mean(y)])
        xhat = perk(y, T, xDists, noiseDist, signalModels, kernel, ρ)
        error_rel[i] = abs(xhat[] - xtrue[i]) / xtrue[i]
    end

    error_rel_avg = sum(error_rel) / length(error_rel)
    return isapprox(error_rel_avg, 0.05810872611593427, atol = 1e-6)

end

function test_perk_5()

    Random.seed!(0)
    f = (x, ν) -> exp(-ν / x)
    xtrue = 100
    ν = 30
    N = 1
    y = fill(f(xtrue, ν), 1, N)
    T = 200
    xDists = [Uniform(10, 500)]
    νDists = Uniform(30, 30.000000001)
    noiseDist = Normal(0, 0.01)
    signalModels = [f]
    λ = 2.0^-1.5
    kernel = GaussianKernel([λ * mean(y); λ * mean(ν)])
    ρ = 2.0^-20
    xhat = perk(y, ν, T, xDists, νDists, noiseDist, signalModels, kernel, ρ)

    error_rel = abs(xhat[] - xtrue) / xtrue
    return isapprox(error_rel, 0.011599576402842331, atol = 1e-6)

end

function test_perk_6()

    Random.seed!(0)
    f = (x, ν) -> [x + ν, exp(-ν / x)]
    xtrue = 100
    ν = 30
    D = 2
    N = 5
    y = f(xtrue, ν) .+ zeros(D, N)
    T = 200
    H = 100
    kernel = GaussianRFF(H, [vec(mean(y, dims = 2)); ν])
    xDists = Uniform(10, 500)
    νDists = Uniform(30, 30.000000001)
    noiseDist = Normal(0, 0.01)
    signalModels = f
    ρ = 0.001
    xhat = perk(y, fill(ν, 1, N), T, xDists, νDists, noiseDist, signalModels, kernel, ρ)

    error_rel = norm(xhat .- xtrue) / (sqrt(N) * xtrue)
    return isapprox(error_rel, 0.012756252384554273, atol = 1e-6)

end

@testset "PERK" begin

    @test test_perk_1()
    @test test_perk_2()
    @test test_perk_3()
    @test test_perk_4()
    @test test_perk_5()
    @test test_perk_6()

end
