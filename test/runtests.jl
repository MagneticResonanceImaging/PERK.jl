tests = [
    "kernels",
    "krr",
    "estimation",
    "holdout"
]
for t in tests
    include("$(t).jl")
end
