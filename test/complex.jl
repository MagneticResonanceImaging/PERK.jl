function test_complex_1()

    Random.seed!(0)
    f = x -> [exp(-30 / x), exp(-30 / x)]
    xtrue = 100
    y = f(xtrue)
    T = 1
    xDists = Uniform(10, 500)
    noiseDist = Normal(0, 0.01)
    signalModels = f
    λ = 2.0^-1.5
    kernel = GaussianKernel(λ * y)
    ρ = 2.0^-20
    xhat = perk(reshape(y, :, 1), T, xDists, noiseDist, signalModels, kernel, ρ)
    error_rel_real = abs(xhat[] - xtrue) / xtrue

    Random.seed!(0)
    f = x -> complex(exp(-30 / x), exp(-30 / x))
    xtrue = 100
    y = f(xtrue)
    T = 1
    xDists = Uniform(10, 500)
    noiseDist = Normal(0, 0.01)
    signalModels = f
    λ = 2.0^-1.5
    kernel = GaussianKernel([λ * mean(y)])
    ρ = 2.0^-20
    xhat = perk(y, T, xDists, noiseDist, signalModels, kernel, ρ)
    error_rel_complex = abs(xhat[] - xtrue) / xtrue

    return error_rel_complex ≈ error_rel_real

end


function test_complex_2()

    rng = StableRNG(0)
    f = x -> complex(exp(-30 / x), exp(-30 / x))
    xtrue = 100
    y = f(xtrue)
    T = 200
    xDists = Uniform(10, 500)
    noiseDist = Normal(0, 0.01)
    signalModels = f
    λ = 2.0^-1.5
    kernel = GaussianKernel([λ * mean(y)])
    ρ = 2.0^-20
    xhat = perk(rng, y, T, xDists, noiseDist, signalModels, kernel, ρ)

    error_rel = abs(xhat[] - xtrue) / xtrue
    return error_rel ≈ 0.04464689602051635

end


function test_complex_3()

    Random.seed!(0)
    f = (x, ν) -> [exp(-ν / x), exp(-ν / x)]
    xtrue = 100
    ν = 30
    N = 1
    y = reshape(f(xtrue, ν), :, N)
    T = 1
    xDists = [Uniform(10, 500)]
    νDists = Uniform(30, 30.000000001)
    noiseDist = Normal(0, 0.01)
    signalModels = [f]
    λ = 2.0^-1.5
    kernel = GaussianKernel(λ * vec(y), [λ * mean(ν)])
    ρ = 2.0^-20
    xhat = perk(y, ν, T, xDists, νDists, noiseDist, signalModels, kernel, ρ)
    error_rel_real = abs(xhat[] - xtrue) / xtrue

    Random.seed!(0)
    f = (x, ν) -> complex(exp(-ν / x), exp(-ν / x))
    xtrue = 100
    ν = 30
    N = 1
    y = fill(f(xtrue, ν), 1, N)
    T = 1
    xDists = [Uniform(10, 500)]
    νDists = Uniform(30, 30.000000001)
    noiseDist = Normal(0, 0.01)
    signalModels = [f]
    λ = 2.0^-1.5
    kernel = GaussianKernel([λ * mean(y)], [λ * mean(ν)])
    ρ = 2.0^-20
    xhat = perk(y, ν, T, xDists, νDists, noiseDist, signalModels, kernel, ρ)
    error_rel_complex = abs(xhat[] - xtrue) / xtrue

    return error_rel_complex ≈ error_rel_real

end


function test_complex_4()

    rng = StableRNG(0)
    f = (x, ν) -> complex(exp(-ν / x), exp(-ν / x))
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
    kernel = GaussianKernel([λ * mean(y)], [λ * mean(ν)])
    ρ = 2.0^-20
    xhat = perk(rng, y, ν, T, xDists, νDists, noiseDist, signalModels, kernel, ρ)

    error_rel = abs(xhat[] - xtrue) / xtrue
    return error_rel ≈ 0.05111009427971681

end


function test_complex_5()

    Random.seed!(0)
    f = x -> [exp(-30 / x), exp(-30 / x)]
    xtrue = 100
    y = f(xtrue)
    T = 1
    xDists = Uniform(10, 500)
    noiseDist = Normal(0, 0.01)
    signalModels = f
    λ = 2.0^-1.5
    H = 2
    kernel = GaussianRFF(H, λ * y)
    ρ = 2.0^-20
    xhat = perk(reshape(y, :, 1), T, xDists, noiseDist, signalModels, kernel, ρ)
    error_rel_real = abs(xhat[] - xtrue) / xtrue

    Random.seed!(0)
    f = x -> complex(exp(-30 / x), exp(-30 / x))
    xtrue = 100
    y = f(xtrue)
    T = 1
    xDists = Uniform(10, 500)
    noiseDist = Normal(0, 0.01)
    signalModels = f
    λ = 2.0^-1.5
    H = 2
    kernel = GaussianRFF(H, [λ * mean(y)])
    ρ = 2.0^-20
    xhat = perk(y, T, xDists, noiseDist, signalModels, kernel, ρ)
    error_rel_complex = abs(xhat[] - xtrue) / xtrue

    return error_rel_complex ≈ error_rel_real

end


function test_complex_6()

    rng = StableRNG(0)
    f = x -> complex(exp(-30 / x), exp(-30 / x))
    xtrue = 100
    y = f(xtrue)
    T = 200
    xDists = Uniform(10, 500)
    noiseDist = Normal(0, 0.01)
    signalModels = f
    λ = 2.0^-1.5
    H = 40
    kernel = GaussianRFF(H, [λ * mean(y)])
    ρ = 2.0^-20
    xhat = perk(rng, y, T, xDists, noiseDist, signalModels, kernel, ρ)

    error_rel = abs(xhat[] - xtrue) / xtrue
    return error_rel ≈ 0.02231978386441426

end


function test_complex_7()

    Random.seed!(0)
    f = (x, ν) -> [exp(-ν / x), exp(-ν / x)]
    xtrue = 100
    ν = 30
    N = 1
    y = reshape(f(xtrue, ν), :, N)
    T = 1
    xDists = [Uniform(10, 500)]
    νDists = Uniform(30, 30.000000001)
    noiseDist = Normal(0, 0.01)
    signalModels = [f]
    λ = 2.0^-1.5
    H = 2
    kernel = GaussianRFF(H, λ * vec(y), [λ * mean(ν)])
    ρ = 2.0^-20
    xhat = perk(y, ν, T, xDists, νDists, noiseDist, signalModels, kernel, ρ)
    error_rel_real = abs(xhat[] - xtrue) / xtrue

    Random.seed!(0)
    f = (x, ν) -> complex(exp(-ν / x), exp(-ν / x))
    xtrue = 100
    ν = 30
    N = 1
    y = fill(f(xtrue, ν), 1, N)
    T = 1
    xDists = [Uniform(10, 500)]
    νDists = Uniform(30, 30.000000001)
    noiseDist = Normal(0, 0.01)
    signalModels = [f]
    λ = 2.0^-1.5
    H = 2
    kernel = GaussianRFF(H, [λ * mean(y)], [λ * mean(ν)])
    ρ = 2.0^-20
    xhat = perk(y, ν, T, xDists, νDists, noiseDist, signalModels, kernel, ρ)
    error_rel_complex = abs(xhat[] - xtrue) / xtrue

    return error_rel_complex ≈ error_rel_real

end


function test_complex_8()

    rng = StableRNG(0)
    f = (x, ν) -> complex(exp(-ν / x), exp(-ν / x))
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
    H = 40
    kernel = GaussianRFF(H, [λ * mean(y)], [λ * mean(ν)])
    ρ = 2.0^-20
    xhat = perk(rng, y, ν, T, xDists, νDists, noiseDist, signalModels, kernel, ρ)

    error_rel = abs(xhat[] - xtrue) / xtrue
    return error_rel ≈ 0.06272802910855091

end


@testset "Complex PERK" begin

    @test test_complex_1()
    @test test_complex_2()
    @test test_complex_3()
    @test test_complex_4()
    @test test_complex_5()
    @test test_complex_6()
    @test test_complex_7()
    @test test_complex_8()

end
