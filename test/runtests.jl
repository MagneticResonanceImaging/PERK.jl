tests = [
    "estimation"
]
for t in tests
    include("$(t).jl")
end
