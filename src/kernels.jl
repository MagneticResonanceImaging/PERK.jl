"""
    Kernel

Abstract type for representing kernel functions.
"""
abstract type Kernel end

"""
    ExactKernel <: Kernel

Abstract type for representing kernels that are evaluated exactly.
`ExactKernel`s must be callable with two matrix inputs.
"""
abstract type ExactKernel <: Kernel end

"""
    RFFKernel <: Kernel

Abstract type for representing kernels that are approximated via random Fourier
features. `RFFKernel`s must be callable with one matrix input.
"""
abstract type RFFKernel <: Kernel end

"""
    GaussianKernel(Λ) <: ExactKernel

Create a Gaussian kernel function.

# Properties
- `Λ::Array{<:Any,1}`: Length scales [Q]
"""
struct GaussianKernel{T} <: ExactKernel
    Λ::Array{T,1}
end

"""
    (k::GaussianKernel)(q1, q2)

Evaluate the Gaussian kernel.

# Arguments
- `q1::AbstractArray{<:Number,2}`: First input to kernel [Q,M]; Q is D+K, i.e.,
  the number of data sets D plus the number of known parameters K
- `q2::AbstractArray{<:Number,2}`: Second input to kernel [Q,N]

# Return
- `K::Array{<:Real,2}`: Kernel output [M,N]
"""
function (k::GaussianKernel)(
    q1::AbstractArray{<:Number,2},
    q2::AbstractArray{<:Number,2}
)

    # Get the size of q1 and q2
    M = size(q1, 2)
    N = size(q2, 2)

    # Construct the square root of the Gaussian covariance matrix and invert it
    sqrtΣ = Diagonal(1 ./ k.Λ)

    # Compute the kernel function
    return [gaussiankernel(q1[:,m], q2[:,n], sqrtΣ) for m = 1:M, n = 1:N]

end

"""
    gaussiankernel(p, q, sqrtΣ)

Compute the Gaussian kernel with covariance matrix Σ evaluated at `p` and `q`.
Note that function input is `sqrtΣ`, i.e., the square root of Σ.
"""
function gaussiankernel(p, q, sqrtΣ)

    return exp(-0.5norm(sqrtΣ * (p - q))^2)

end

"""
    GaussianRFF(H, Λ) <: RFFKernel

Create an approximate (via random Fourier features) Gaussian kernel function.

# Properties
- `H::Integer`: Approximation order
- `Λ::Array{<:Any,1}`: Length scales [Q]
"""
struct GaussianRFF{T1<:Integer,T2} <: RFFKernel
    H::T1
    Λ::Array{T2,1}
end

"""
    (k::GaussianRFF)(q, [f, phase])

Evaluate the approximate Gaussian kernel.

# Arguments
- `q::AbstractArray{<:Number,2}`: Input to kernel [Q,N]; Q is D+K, i.e.,
  the number of data sets D plus the number of known parameters K
- `f::Array{<:Real,2} = randn(k.H, Q)`: Unscaled random frequency values [H,Q]
- `phase::Array{<:Real,1} = rand(k.H)`: Random phase values [H]

# Return
- `z::Array{<:Real,2}`: Higher-dimensional features [H,N]
- `freq::Array{<:Real,2}`: Random frequency values [H,Q]
- `phase::Array{<:Real,1}`: Random phase values [H]
"""
function (k::GaussianRFF)(q::AbstractArray{<:Number,2})

    # Get the first dimension of q
    Q = size(q, 1)

    # Generate random phase and unscaled frequency values
    f = randn(k.H, Q)
    phase = rand(k.H)

    return k(q, f, phase)

end

function (k::GaussianRFF)(
    q::AbstractArray{<:Number,2},
    f::Array{<:Real,2},
    phase::Array{<:Real,1}
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

"""
    rffmap(q, freq, phase)

Map features to a higher dimensional space via random Fourier features.

# Arguments
- `q::AbstractArray{<:Number,2}`: Lower-dimensional features [Q,N]
- `freq::AbstractArray{<:Real,2}`: Random frequency values [H,Q]
- `phase::AbstractArray{<:Real,1}`: Random phase values [H]

# Return
- `z::Array{<:Real,2}`: Higher-dimensional features [H,N]
"""
function rffmap(q::AbstractArray{<:Number,2},
                freq::AbstractArray{<:Real,2},
                phase::AbstractArray{<:Real,1})

    return sqrt(div0(2, length(phase))) .* cos.(2π .* (freq * q .+ phase))

end
