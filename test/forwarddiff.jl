function test_forwarddiff_1()

    (Q, T, H) = (3, 2000, 40)
    p = randn(1, T)
    r = randn(H, Q)
    f1 = x -> begin
        y = p .* x
        A = r * y
        B = PERK.A_mul_A′(A)
        norm(B)
    end
    f2 = x -> begin
        y = p .* x
        A = r * y
        B = A * A'
        norm(B)
    end

    x = randn(Q)
    g1 = similar(x)
    g2 = similar(x)
    config1 = ForwardDiff.GradientConfig(f1, x, ForwardDiff.Chunk{Q}())
    config2 = ForwardDiff.GradientConfig(f2, x, ForwardDiff.Chunk{Q}())
    ForwardDiff.gradient!(g1, f1, x, config1)
    ForwardDiff.gradient!(g2, f2, x, config2)

    return g1 ≈ g2

end

@testset "ForwardDiff" begin

    @test test_forwarddiff_1()

end
