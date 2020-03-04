using PERK, Test

function test_EuclideanKernel_1()

    k = EuclideanKernel()
    p = rand()
    q = randn(10)
    x = k(p, q)
    x_correct = p * q
    return x ≈ x_correct
    
end

function test_EuclideanKernel_2()

    k = EuclideanKernel()
    p = rand()
    q = randn()
    x = k(p, q)
    x_correct = p * q
    return x ≈ x_correct
    
end

function test_GaussianKernel_1()

    k = GaussianKernel(2)
    p = rand()
    q = randn(10)
    return k(p, q) ≈ k(q, p)
    
end

function test_GaussianKernel_2()

    Λ = 2
    k = GaussianKernel(Λ)
    p = rand()
    q = randn()
    x = k(p, q)
    x_correct = exp(-0.5 * abs2((p - q) / Λ))
    return x ≈ x_correct
    
end

@testset "Kernels" begin

    @test test_EuclideanKernel_1()
    @test test_EuclideanKernel_2()
    @test test_GaussianKernel_1()
    @test test_GaussianKernel_2()

end
