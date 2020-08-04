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

    Random.seed!(0)
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
    xhat = perk(y, T, xDists, noiseDist, signalModels, kernel, ρ)

    error_rel = abs(xhat[] - xtrue) / xtrue
    return isapprox(error_rel, 0.05211190543523628, atol = 1e-6)

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

    Random.seed!(0)
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
    xhat = perk(y, ν, T, xDists, νDists, noiseDist, signalModels, kernel, ρ)

    error_rel = abs(xhat[] - xtrue) / xtrue
    return isapprox(error_rel, 0.03526687067013938, atol = 1e-6)

end

function test_complex_5()

    Random.seed!(0)
    f = x -> complex(exp(-30 / x) + 1, exp(-30 / x) + 1)
    f_abs = x -> abs(f(x))
    xtrue = 100
    y = f(xtrue)
    y_abs = f_abs(xtrue)
    T = 200
    xDists = Uniform(10, 500)
    noiseDist = Normal(0, 0.01)
    λ = 2.0^-1.5
    kernel = GaussianKernel([λ * mean(y)])
    kernel_abs = GaussianKernel(λ * mean(y_abs))
    ρ = 2.0^-20
    xhat = perk(y, T, xDists, noiseDist, f, kernel, ρ)
    xhat_abs = perk(y_abs, T, xDists, noiseDist, f_abs, kernel_abs, ρ)

    error_rel = abs(xhat[] - xtrue) / xtrue
    error_rel_abs = abs(xhat_abs[] - xtrue) / xtrue
    return error_rel < error_rel_abs

end

function test_complex_6()

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

function test_complex_7()

    Random.seed!(0)
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
    xhat = perk(y, T, xDists, noiseDist, signalModels, kernel, ρ)

    error_rel = abs(xhat[] - xtrue) / xtrue
    return isapprox(error_rel, 0.01699497197550329, atol = 1e-2)

end

function test_complex_8()

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

function test_complex_9()

    Random.seed!(0)
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
    xhat = perk(y, ν, T, xDists, νDists, noiseDist, signalModels, kernel, ρ)

    error_rel = abs(xhat[] - xtrue) / xtrue
    return isapprox(error_rel, 0.0346180252991536, atol = 1e-2)

end

function test_complex_10()

    Random.seed!(0)
    f = x -> complex(exp(-30 / x) + 1, exp(-30 / x) + 1)
    f_abs = x -> abs(f(x))
    xtrue = 100
    y = f(xtrue)
    y_abs = f_abs(xtrue)
    T = 200
    xDists = Uniform(10, 500)
    noiseDist = Normal(0, 0.01)
    λ = 2.0^-1.5
    H = 40
    kernel = GaussianRFF(H, [λ * mean(y)])
    kernel_abs = GaussianRFF(H, λ * mean(y_abs))
    ρ = 2.0^-20
    xhat = perk(y, T, xDists, noiseDist, f, kernel, ρ)
    xhat_abs = perk(y_abs, T, xDists, noiseDist, f_abs, kernel_abs, ρ)

    error_rel = abs(xhat[] - xtrue) / xtrue
    error_rel_abs = abs(xhat_abs[] - xtrue) / xtrue
    return error_rel < error_rel_abs

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
    @test test_complex_9()
    @test test_complex_10()

end
