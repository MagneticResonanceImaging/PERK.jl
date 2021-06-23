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
features. `RFFKernel`s must be callable with one input. `RFFKernel`s must have
fields `freq` and `phase`.
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
- `workspace::Vector{<:Real}`: Workspace for computing the Gaussian kernel

# Note
- Q is the number of features
"""
struct GaussianKernel{T<:Union{<:Real,<:AbstractVector{<:Real}},W<:Real} <: ExactKernel
    Λ::T
    workspace::Vector{W}

    function GaussianKernel(Λ::Union{<:Real,<:AbstractVector{<:Real}})

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
    GaussianRFF(Λ, freq, phase) <: RFFKernel

Create an approximate (via random Fourier features) Gaussian kernel function.

# Properties
- `Λ::Union{<:Real,AbstractVector{<:Real}}`: Length scales \\[Q\\] or scalar (if
  Q = 1)
- `freq::Union{<:AbstractVector{<:Real},<:AbstractMatrix{<:Real}} = randn(k.H, Q)`:
  Scaled random frequency values [H,Q] or \\[H\\] (if Q = 1)
- `phase::AbstractVector{<:Real} = rand(k.H)`: Random phase values \\[H\\]
- `workspace::Vector{<:Real}`: Workspace for computing random Fourier features

## Note
- The `freq` input to `GaussianRFF` is *unscaled*, but is then scaled so that
  `getproperty(::GaussianRFF, :freq)` returns the scaled random frequency values
- H is the approximation order
- Q is the number of features
"""
struct GaussianRFF{T1<:Union{<:Real,<:AbstractVector{<:Real}},T2<:Union{<:AbstractVector{<:Real},<:AbstractMatrix{<:Real}},T3<:AbstractVector{<:Real},W<:Real} <: RFFKernel
    Λ::T1
    freq::T2
    phase::T3
    workspace::Vector{W}

    function GaussianRFF(
        Λ::Union{<:Real,<:AbstractVector{<:Real}},
        freq::Union{<:AbstractVector{<:Real},<:AbstractMatrix{<:Real}},
        phase::AbstractVector{<:Real}
    )

        Q = length(Λ)
        if Q == 1
            freq isa AbstractVector || (freq = dropdims(freq, dims = 2))
            Λ = Λ[]
            sqrtΣ = div0(1, 2π * Λ)
        else
            sqrtΣ = Diagonal(div0.(1, 2π * Λ))
        end
        # Scale freq by the square root of the inverse covariance matrix from
        # which to draw the Gaussian samples
        freq = freq * sqrtΣ
        T = eltype(Λ)
        # workspace must be vector because number of feature vectors is unknown
        # when creating GaussianRFF object
        workspace = Vector{T}(undef, length(phase))
        new{typeof(Λ),typeof(freq),typeof(phase),T}(Λ, freq, phase, workspace)

    end
end

"""
    GuassianRFF(Λy, [Λν], H)
    GuassianRFF(Λy, [Λν], freq, phase)

Create an approximate (via random Fourier features) Gaussian kernel function.

# Arguments
- `Λy::Union{<:Number,<:AbstractVector{<:Number}}`: Length scales for features
  \\[Q\\] or scalar (if Q = 1)
- `Λν::Union{<:Real,<:AbstractVector{<:Real}}`: Length scales for known
  parameters \\[K\\] or scalar (if K = 1)
- `H::Integer`: Approximation order
- `freq::Union{<:AbstractVector{<:Real},<:AbstractMatrix{<:Real}} = randn(k.H, Q)`:
  Unscaled random frequency values [H,Q] or \\[H\\] (if Q = 1)
- `phase::AbstractVector{<:Real} = rand(k.H)`: Random phase values \\[H\\]
"""
function GaussianRFF(
    Λ::Union{<:Number,<:AbstractVector{<:Number}},
    H::Integer
)

    if eltype(Λ) <: Complex
        Λ = reduce(vcat, ([real(Λ[i]), imag(Λ[i])] for i = 1:length(Λ)))
    end

    freq = randn(H, length(Λ))
    phase = rand(H)

    return GaussianRFF(Λ, freq, phase)

end

function GaussianRFF(
    Λy::Union{<:Number,<:AbstractVector{<:Number}},
    Λν::Union{<:Real,<:AbstractVector{<:Real}},
    H::Integer
)

    if eltype(Λy) <: Complex
        Λy = reduce(vcat, ([real(Λy[i]), imag(Λy[i])] for i = 1:length(Λy)))
    end

    Λ = [Λy; Λν]
    freq = randn(H, length(Λ))
    phase = rand(H)

    return GaussianRFF(Λ, freq, phase)

end

function GaussianRFF(
    Λ::Union{<:Complex,<:AbstractVector{<:Complex}},
    freq::Union{<:AbstractVector{<:Real},<:AbstractMatrix{<:Real}},
    phase::AbstractVector{<:Real}
)

    Λ = reduce(vcat, ([real(Λ[i]), imag(Λ[i])] for i = 1:length(Λ)))

    return GaussianRFF(Λ, freq, phase)

end

function GaussianRFF(
    Λy::Union{<:Number,<:AbstractVector{<:Number}},
    Λν::Union{<:Real,<:AbstractVector{<:Real}},
    freq::Union{<:AbstractVector{<:Real},<:AbstractMatrix{<:Real}},
    phase::AbstractVector{<:Real}
)

    if eltype(Λy) <: Complex
        Λy = reduce(vcat, ([real(Λy[i]), imag(Λy[i])] for i = 1:length(Λy)))
    end

    Λ = [Λy; Λν]

    return GaussianRFF(Λ, freq, phase)

end

"""
    (k::GaussianRFF)(q)

Evaluate the approximate Gaussian kernel.

# Arguments
- `q::Union{<:Real,<:AbstractVector{<:Real},<:AbstractMatrix{<:Real}}`: Kernel
  input [Q,N] or \\[N\\] (if Q = 1) or scalar (if Q = N = 1)

## Note
- Q is the number of features
- N is the number of feature vectors in the input
- H is the approximation order for the random Fourier features

# Return
- `z::Union{<:AbstractVector{<:Real},<:AbstractMatrix{<:Real}}`:
  Higher-dimensional features [H,N] or \\[H\\] (if N = 1)
"""
function (k::GaussianRFF)(
    q::Union{<:Real,<:AbstractVector{<:Real},<:AbstractMatrix{<:Real}}
)

    # Map the features to a higher dimensional space via random Fourier features
    return rffmap(q, k.freq, k.phase)

end

function (k!::GaussianRFF)(
    out::Union{<:AbstractVector{<:Real},<:AbstractMatrix{<:Real}},
    q::Union{<:Real,<:AbstractVector{<:Real},<:AbstractMatrix{<:Real}}
)

    rffmap!(out, q, k!.freq, k!.phase, k!.workspace)

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

function rffmap!(
    out::AbstractMatrix{<:Real},
    q::AbstractMatrix{<:Real},
    freq::AbstractMatrix{<:Real},
    phase::AbstractVector{<:Real},
    workspace::AbstractVector{<:Real}
)

    s = sqrt(div0(2, length(phase)))
    for n = 1:size(q, 2)
        @views mul!(workspace, freq, q[:,n])
        out[:,n] = s .* cos.(2π .* (workspace .+ phase))
    end

    return nothing

end

function rffmap!(
    out::AbstractMatrix{<:Real},
    q::AbstractMatrix{<:Real},
    freq::AbstractVector{<:Real},
    phase::AbstractVector{<:Real},
    workspace::AbstractVector{<:Real}
)

    size(q, 1) == 1 ||
        throw(DimensionMismatch("freq has feature dimension equal to 1, but " *
                                "q has a larger feature dimension"))

    return rffmap!(out, vec(q), freq, phase, workspace)

end

function rffmap!(
    out::AbstractMatrix{<:Real},
    q::AbstractVector{<:Real},
    freq::AbstractVector{<:Real},
    phase::AbstractVector{<:Real},
    ::AbstractVector{<:Real}
)

    s = sqrt(div0(2, length(phase)))
    for n = 1:length(q)
        out[:,n] = s .* cos.(2π .* (freq .* q[n] .+ phase))
    end

    return nothing

end

function rffmap!(
    out::AbstractVector{<:Real},
    q::Real,
    freq::AbstractVector{<:Real},
    phase::AbstractVector{<:Real},
    ::AbstractVector{<:Real}
)

    s = sqrt(div0(2, length(phase)))
    out .= s .* cos.(2π .* (freq .* q .+ phase))

    return nothing

end
