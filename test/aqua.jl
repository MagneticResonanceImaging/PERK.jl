using PERK: PERK
import Aqua
using Test: @testset

@testset "aqua" begin
    Aqua.test_all(
        PERK;
        deps_compat = (; ignore = [:LinearAlgebra, :Random]),
    )
end
