"""
    Kernel

Abstract type for representing kernel functions.
"""
abstract type Kernel end

"""
    ExactKernel <: Kernel

Abstract type for representing kernels that are evaluated exactly.
`ExactKernel`s must be callable with two inputs.
"""
abstract type ExactKernel <: Kernel end

"""
    RFFKernel <: Kernel

Abstract type for representing kernels that are approximated via random Fourier
features. `RFFKernel`s must be callable with one or three inputs.
"""
abstract type RFFKernel <: Kernel end

"""
    EuclideanKernel() <: ExactKernel

Create a kernel function where the kernel is just the Euclidean inner product
(i.e., ridge regression).
"""
struct EuclideanKernel <: ExactKernel end

"""
    (k::EuclideanKernel)(p, q)

Evaluate the Euclidean inner product between `p` and `q`.

# Arguments
- `p::Union{<:Real,<:AbstractVector{<:Real},<:AbstractMatrix{<:Real}}`: First
  kernel input [Q,M] or \\[M\\] (if Q = 1) or scalar (if Q = M = 1)
- `q::Union{<:Real,<:AbstractVector{<:Real},<:AbstractMatrix{<:Real}}`: Second
  kernel input [Q,N] or \\[N\\] (if Q = 1) or scalar (if Q = N = 1)

# Note
- Q is the number of features
- M is the number of feature vectors in the first input
- N is the number of feature vectors in the second input

# Return
- `K::Union{<:Real,<:AbstractVector{<:Real},<:AbstractMatrix{<:Real}}`: Kernel
  output [M,N] or \\[M\\] (if N = 1) or \\[N\\] (if M = 1) or scalar
  (if M = N = 1)
"""
function (k::EuclideanKernel)(
    p::AbstractMatrix{<:Real},
    q::AbstractMatrix{<:Real}
)

    return p' * q

end

function (k::EuclideanKernel)(
    p::AbstractVector{<:Real}, # [M]
    q::AbstractMatrix{<:Real} # [Q,N]
)

    return k(reshape(p, 1, :), q)

end

function (k::EuclideanKernel)(
    p::AbstractVector{<:Real},
    q::AbstractVector{<:Real}
)

    return k(reshape(p, 1, :), reshape(q, 1, :))

end

function (k::EuclideanKernel)(
    p::AbstractVector{<:Real},
    q::Real
)

    return conj(p) * q

end

function (k::EuclideanKernel)(
    p::Real,
    q::AbstractVector{<:Real}
)

    return conj(p) * q

end

function (k::EuclideanKernel)(
    p::Real,
    q::Real
)

    return conj(p) * q

end

function (k!::EuclideanKernel)(
    out::AbstractMatrix{<:Real},
    p::AbstractMatrix{<:Real},
    q::AbstractMatrix{<:Real}
)

    for i in CartesianIndices(out)
        @views out[i] = p[:,i[1]]' * q[:,i[2]]
    end

end

function (k!::EuclideanKernel)(
    out::AbstractMatrix{<:Real},
    p::AbstractVector{<:Real}, # [M]
    q::AbstractMatrix{<:Real} # [Q,N]
)

    k!(out, reshape(p, 1, :), q)

end

function (k!::EuclideanKernel)(
    out::AbstractMatrix{<:Real},
    p::AbstractVector{<:Real},
    q::AbstractVector{<:Real}
)

    k!(out, reshape(p, 1, :), reshape(q, 1, :))

end

function (k!::EuclideanKernel)(
    out::AbstractVector{<:Real},
    p::AbstractVector{<:Real},
    q::Real
)

    out .= conj.(p) .* q
    return nothing

end

function (k!::EuclideanKernel)(
    out::AbstractVector{<:Real},
    p::Real,
    q::AbstractVector{<:Real}
)

    out .= conj.(p) .* q
    return nothing

end

# Already no allocations, so in-place version not really needed
function (k!::EuclideanKernel)(
    ::Nothing,
    p::Real,
    q::Real
)

    return conj(p) * q

end

"""
    GaussianKernel(Λ) <: ExactKernel

Create a Gaussian kernel function.

# Properties
- `Λ::Union{<:Real,AbstractVector{<:Real}}`: Length scales \\[Q\\] or scalar
  (if Q = 1)
- `workspace::Union{<:AbstractVector{Float64},Nothing}`: Workspace for computing
  the Gaussian kernel; `nothing` if Q = 1

# Note
- Q is the number of features
"""
struct GaussianKernel{T<:Union{<:Real,<:AbstractVector{<:Real}},W<:Real} <: ExactKernel
    Λ::T
    workspace::Vector{W}

    GaussianKernel(Λ::Union{<:Real,<:AbstractVector{<:Real}}) = begin
        Q = length(Λ)
        Q == 1 && (Λ = Λ[])
        T = eltype(Λ)
        workspace = Vector{T}(undef, Q)
        new{typeof(Λ),T}(Λ, workspace)
    end
end

"""
    GuassianKernel(Λy, [Λν])

Create a Guassian kernel function.

# Arguments
- `Λy::Union{<:Number,<:AbstractVector{<:Number}}`: Length scales for features
  \\[Q\\] or scalar (if Q = 1)
- `Λν::Union{<:Real,<:AbstractVector{<:Real}}`: Length scales for known
  parameters \\[K\\] or scalar (if K = 1)
"""
function GaussianKernel(
    Λ::Union{<:Complex,<:AbstractVector{<:Complex}}
)

    Λ = reduce(vcat, ([real(Λ[i]), imag(Λ[i])] for i = 1:length(Λ)))

    return GaussianKernel(Λ)

end

function GaussianKernel(
    Λy::Union{<:Number,<:AbstractVector{<:Number}},
    Λν::Union{<:Real,<:AbstractVector{<:Real}}
)

    if eltype(Λy) <: Complex
        Λy = reduce(vcat, ([real(Λy[i]), imag(Λy[i])] for i = 1:length(Λy)))
    end

    return GaussianKernel([Λy; Λν])

end

"""
    (k::GaussianKernel)(p, q)

Evaluate the Gaussian kernel.

# Arguments
- `p::Union{<:Real,<:AbstractVector{<:Real},<:AbstractMatrix{<:Real}}`: First
  kernel input [Q,M] or \\[M\\] (if Q = 1) or scalar (if Q = M = 1)
- `q::Union{<:Real,<:AbstractVector{<:Real},<:AbstractMatrix{<:Real}}`: Second
  kernel input [Q,N] or \\[N\\] (if Q = 1) or scalar (if Q = N = 1)

## Note
- Q is the number of features
- M is the number of feature vectors in the first input
- N is the number of feature vectors in the second input

# Return
- `K::Union{<:Real,<:AbstractVector{<:Real},<:AbstractMatrix{<:Real}}`: Kernel
  output [M,N] or \\[M\\] (if N = 1) or \\[N\\] (if M = 1) or scalar
  (if M = N = 1)
"""
function (k::GaussianKernel)(
    p::AbstractMatrix{<:Real},
    q::AbstractMatrix{<:Real}
)

    # Get the sizes of p and q
    M = size(p, 2)
    N = size(q, 2)

    # Construct the square root of the Gaussian covariance matrix and invert it
    sqrtΣ = 1 ./ k.Λ

    # Compute the kernel function
    return [gaussiankernel!(p[:,m], q[:,n], sqrtΣ, k.workspace) for m = 1:M, n = 1:N]

end

function (k::GaussianKernel)(
    p::AbstractVector{<:Real}, # [M]
    q::AbstractMatrix{<:Real} # [Q,N]
)

    return k(reshape(p, 1, :), q)

end

function (k::GaussianKernel)(
    p::AbstractVector{<:Real},
    q::AbstractVector{<:Real}
)

    return k(reshape(p, 1, :), reshape(q, 1, :))

end

function (k::GaussianKernel)(
    p::AbstractVector{<:Real},
    q::Real
)

    # Construct the square root of the Gaussian covariance matrix and invert it
    sqrtΣ = 1 / k.Λ

    # Compute the kernel function
    return gaussiankernel.(p, q, sqrtΣ)

end

function (k::GaussianKernel)(
    p::Real,
    q::AbstractVector{<:Real}
)

    # Gaussian kernel is symmetric, so just swap the order of inputs
    return k(q, p)

end

function (k::GaussianKernel)(
    p::Real,
    q::Real
)

    # Construct the square root of the Gaussian covariance matrix and invert it
    sqrtΣ = 1 / k.Λ

    # Compute the kernel function
    return gaussiankernel(p, q, sqrtΣ)

end

function (k!::GaussianKernel)(
    out::AbstractMatrix{<:Real},
    p::AbstractMatrix{<:Real},
    q::AbstractMatrix{<:Real}
)

    # Construct the square root of the Gaussian covariance matrix and invert it
    sqrtΣ = 1 ./ k!.Λ

    for i in CartesianIndices(out)
        @views out[i] = gaussiankernel!(p[:,i[1]], q[:,i[2]], sqrtΣ, k!.workspace)
    end

end

function (k!::GaussianKernel)(
    out::AbstractMatrix{<:Real},
    p::AbstractVector{<:Real}, # [M]
    q::AbstractMatrix{<:Real} # [Q,N]
)

    k!(out, reshape(p, 1, :), q)

end

function (k!::GaussianKernel)(
    out::AbstractMatrix{<:Real},
    p::AbstractVector{<:Real},
    q::AbstractVector{<:Real}
)

    k!(out, reshape(p, 1, :), reshape(q, 1, :))

end

function (k!::GaussianKernel)(
    out::AbstractVector{<:Real},
    p::AbstractVector{<:Real},
    q::Real
)

    # Construct the square root of the Gaussian covariance matrix and invert it
    sqrtΣ = 1 / k!.Λ

    # Compute the kernel function
    out .= gaussiankernel.(p, q, sqrtΣ)
    return nothing

end

function (k!::GaussianKernel)(
    out::AbstractVector{<:Real},
    p::Real,
    q::AbstractVector{<:Real}
)

    # Gaussian kernel is symmetric, so just swap the order of inputs
    k!(out, q, p)

end

# Already no allocations, so in-place version not really needed
function (k::GaussianKernel)(
    ::Nothing,
    p::Real,
    q::Real
)

    # Construct the square root of the Gaussian covariance matrix and invert it
    sqrtΣ = 1 / k!.Λ

    # Compute the kernel function
    return gaussiankernel(p, q, sqrtΣ)

end

"""
    gaussiankernel(p, q, sqrtΣ)

Compute the Gaussian kernel with covariance matrix Σ evaluated at `p` and `q`.
Note that function input is `sqrtΣ`, i.e., the square root of Σ.

Currently, only diagonal covariance matrices are supported, and each must be
passed in as a vector of the diagonal elements.
"""
function gaussiankernel(p, q, sqrtΣ)

    return exp(-0.5sum(abs2, sqrtΣ .* (p .- q)))

end

function gaussiankernel!(p, q, sqrtΣ, workspace)

    workspace .= sqrtΣ .* (p .- q)
    return exp(-0.5sum(abs2, workspace))

end

"""
    GaussianRFF(H, Λ) <: RFFKernel

Create an approximate (via random Fourier features) Gaussian kernel function.

# Properties
- `H::Integer`: Approximation order
- `Λ::Union{<:Real,AbstractVector{<:Real}}`: Length scales \\[Q\\] or scalar (if
  Q = 1)

## Note
- Q is the number of features
"""
struct GaussianRFF{T1<:Integer,T2<:Union{<:Real,<:AbstractVector{<:Real}}} <: RFFKernel
    H::T1
    Λ::T2

    GaussianRFF(H::Integer, Λ::Union{<:Real,<:AbstractVector{<:Real}}) = begin
        length(Λ) == 1 && (Λ = Λ[])
        new{typeof(H),typeof(Λ)}(H, Λ)
    end
end

"""
    GuassianRFF(H, Λy, [Λν])

Create an approximate (via random Fourier features) Gaussian kernel function.

# Arguments
- `H::Integer`: Approximation order
- `Λy::Union{<:Number,<:AbstractVector{<:Number}}`: Length scales for features
  \\[Q\\] or scalar (if Q = 1)
- `Λν::Union{<:Real,<:AbstractVector{<:Real}}`: Length scales for known
  parameters \\[K\\] or scalar (if K = 1)
"""
function GaussianRFF(
    H::Integer,
    Λ::Union{<:Complex,<:AbstractVector{<:Complex}}
)

    Λ = reduce(vcat, ([real(Λ[i]), imag(Λ[i])] for i = 1:length(Λ)))

    return GaussianRFF(H, Λ)

end

function GaussianRFF(
    H::Integer,
    Λy::Union{<:Number,<:AbstractVector{<:Number}},
    Λν::Union{<:Real,<:AbstractVector{<:Real}}
)

    if eltype(Λy) <: Complex
        Λy = reduce(vcat, ([real(Λy[i]), imag(Λy[i])] for i = 1:length(Λy)))
    end

    return GaussianRFF(H, [Λy; Λν])

end

"""
    (k::GaussianRFF)(q, [f, phase])

Evaluate the approximate Gaussian kernel.

# Arguments
- `q::Union{<:Real,<:AbstractVector{<:Real},<:AbstractMatrix{<:Real}}`: Kernel
  input [Q,N] or \\[N\\] (if Q = 1) or scalar (if Q = N = 1)
- `f::Union{<:AbstractVector{<:Real},<:AbstractMatrix{<:Real}} = randn(k.H, Q)`:
  Unscaled random frequency values [H,Q] or \\[H\\] (if Q = 1)
- `phase::AbstractVector{<:Real} = rand(k.H)`: Random phase values \\[H\\]

## Note
- Q is the number of features
- N is the number of feature vectors in the input
- H is the approximation order for the random Fourier features

# Return
- `z::Union{<:AbstractVector{<:Real},<:AbstractMatrix{<:Real}}`:
  Higher-dimensional features [H,N] or \\[H\\] (if N = 1)
- `freq::Union{<:AbstractVector{<:Real},<:AbstractMatrix{<:Real}}`: Random
  frequency values [H,Q] or \\[H\\] (if Q = 1)
- `phase::AbstractVector{<:Real}`: Random phase values \\[H\\]
"""
function (k::GaussianRFF)(
    q::AbstractMatrix{<:Real}
)

    # Get the first dimension of q
    Q = size(q, 1)

    # Generate random phase and unscaled frequency values
    f = randn(k.H, Q)
    phase = rand(k.H)

    return k(q, f, phase)

end

function (k::GaussianRFF)(
    q::Union{<:Real,<:AbstractVector{<:Real}}
)

    # Generate random phase and unscaled frequency values
    f = randn(k.H)
    phase = rand(k.H)

    return k(q, f, phase)

end

function (k::GaussianRFF)(
    q::AbstractMatrix{<:Real},
    f::AbstractMatrix{<:Real},
    phase::AbstractVector{<:Real}
)

    # Construct the covariance matrix from which to draw the Gaussian samples
    # and take the square root
    sqrtΣ = Diagonal(div0.(1, 2π .* k.Λ))

    # Scale the random frequency values
    freq = f * sqrtΣ

    # Map the features to a higher dimensional space via random Fourier features
    # Also return freq and phase for use later
    return (rffmap(q, freq, phase), freq, phase)

end

function (k::GaussianRFF)(
    p::Union{<:Real,<:AbstractVector{<:Real}},
    f::AbstractMatrix{<:Real},
    phase::AbstractVector{<:Real}
)

    size(f, 2) == 1 ||
        throw(DimensionMismatch("p has feature dimension equal to 1, but f " *
                                "has a larger feature dimension"))

    return k(p, vec(f), phase)

end

function (k::GaussianRFF)(
    q::Union{<:Real,<:AbstractVector{<:Real}},
    f::AbstractVector{<:Real},
    phase::AbstractVector{<:Real}
)

    # Construct the covariance matrix from which to draw the Gaussian samples
    # and take the square root
    sqrtΣ = div0(1, 2π * k.Λ)

    # Scale the random frequency values
    freq = f * sqrtΣ

    # Map the features to a higher dimensional space via random Fourier features
    # Also return freq and phase for use later
    return (rffmap(q, freq, phase), freq, phase)

end

"""
    rffmap(q, freq, phase)

Map features to a higher dimensional space via random Fourier features.

# Arguments
- `q::Union{<:Real,<:AbstractVector{<:Real},<:AbstractMatrix{<:Real}}`:
  Lower-dimensional features [Q,N] or \\[N\\] (if Q = 1) or scalar
  (if Q = N = 1)
- `freq::Union{<:AbstractVector{<:Real},<:AbstractMatrix{<:Real}`: Random
  frequency values [H,Q] or \\[H\\] (if Q = 1)
- `phase::AbstractVector{<:Real}`: Random phase values \\[H\\]

## Note
- Q is the number of features
- N is the number of feature vectors in the input
- H is the approximation order for the random Fourier features

# Return
- `z::Union{<:AbstractVector{<:Real},<:AbstractMatrix{<:Real}}`:
  Higher-dimensional features [H,N] or \\[H\\] (if N = 1)
"""
function rffmap(
    q::AbstractMatrix{<:Real},
    freq::AbstractMatrix{<:Real},
    phase::AbstractVector{<:Real}
)

    return sqrt(div0(2, length(phase))) .* cos.(2π .* (freq * q .+ phase))

end

function rffmap(
    q::AbstractMatrix{<:Real},
    freq::AbstractVector{<:Real},
    phase::AbstractVector{<:Real}
)

    size(q, 1) == 1 ||
        throw(DimensionMismatch("freq has feature dimension equal to 1, but " *
                                "q has a larger feature dimension"))

    return rffmap(vec(q), freq, phase)

end

function rffmap(
    q::AbstractVector{<:Real},
    freq::AbstractVector{<:Real},
    phase::AbstractVector{<:Real}
)

    return sqrt(div0(2, length(phase))) .* cos.(2π .* (freq * transpose(q) .+ phase))

end

function rffmap(
    q::Real,
    freq::AbstractVector{<:Real},
    phase::AbstractVector{<:Real}
)

    return sqrt(div0(2, length(phase))) .* cos.(2π .* (freq .* q .+ phase))

end
