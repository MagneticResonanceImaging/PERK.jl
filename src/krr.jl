"""
    krr_train(xtrain, ytrain, kernel, ρ, [f, phase])

Train kernel ridge regression.

# Arguments
- `xtrain::Union{<:AbstractVector{<:Real},<:AbstractMatrix{<:Real}}`: Latent
  parameters for training data [L,T] or \\[T\\] (if L = 1)
- `ytrain::Union{<:AbstractVector{<:Real},<:AbstractMatrix{<:Real}}`: Features
  for training data [Q,T] or \\[T\\] (if Q = 1)
- `kernel::Kernel`: Kernel to use
- `ρ::Real`: Tikhonov regularization parameter
- `f::Union{<:AbstractVector{<:Real},AbstractMatrix{<:Real}} = randn(kernel.H, Q)`:
  Unscaled random frequency values [H,Q] or \\[H\\] (if Q = 1) (used when
  `kernel isa RFFKernel`)
- `phase::AbstractVector{<:Real} = rand(kernel.H)`: Random phase values \\[H\\]
  (used when `kernel isa RFFKernel`)

## Note
- L is the number of unknown or latent parameters to be predicted
- Q is the number of observed features per training sample
- T is the number of training samples
- H is approximation order for kernels that use random Fourier features

# Return
- `trainData::TrainingData`: `TrainingData` object to be passed to `krr`
"""
function krr_train(
    xtrain::AbstractVector{<:Real},
    ytrain::Union{<:AbstractVector{<:Real},<:AbstractMatrix{<:Real}},
    kernel::ExactKernel,
    ρ::Real
)

    # Grab the number of training points
    T = length(xtrain)

    # Evaluate the kernel on the training features
    K = kernel(ytrain, ytrain) # [T,T]

    # Calculate the sample mean and de-mean the latent parameters
    xm = mean(xtrain) # scalar (L = 1)
    xtrain = xtrain .- xm # [T]

    # De-mean the rows and columns of the kernel output
    Km = dropdims(mean(K, dims = 2), dims = 2) # [T]
    K = K .- Km # [T,T]
    K = K .- mean(K, dims = 1) # [T,T]

    # Compute the (regularized) inverse of K and multiply by xtrain
    xKinv = transpose(K + T * ρ * I) \ xtrain # [T]

    return ExactTrainingData(ytrain, xtrain, xm, K, Km, xKinv)

end

function krr_train(
    xtrain::AbstractMatrix{<:Real},
    ytrain::Union{<:AbstractVector{<:Real},<:AbstractMatrix{<:Real}},
    kernel::ExactKernel,
    ρ::Real
)

    # Grab the number of training points
    T = size(xtrain, 2)

    # Evaluate the kernel on the training features
    K = kernel(ytrain, ytrain) # [T,T]

    # Calculate the sample mean and de-mean the latent parameters
    xm = dropdims(mean(xtrain, dims = 2), dims = 2) # [L]
    xtrain = xtrain .- xm # [L,T]

    # De-mean the rows and columns of the kernel output
    Km = dropdims(mean(K, dims = 2), dims = 2) # [T]
    K = K .- Km # [T,T]
    K = K .- mean(K, dims = 1) # [T,T]

    # Compute the (regularized) inverse of K and multiply by xtrain
    xKinv = xtrain / (K + T * ρ * I) # [L,T]

    return ExactTrainingData(ytrain, xtrain, xm, K, Km, xKinv)

end

function krr_train(
    xtrain::Union{<:AbstractVector{<:Real},<:AbstractMatrix{<:Real}},
    ytrain::Union{<:AbstractVector{<:Real},<:AbstractMatrix{<:Real}},
    kernel::RFFKernel,
    ρ::Real
)

    # Use random Fourier features to approximate the kernel
    (z, freq, phase) = kernel(ytrain)

    return _krr_train(xtrain, z, ρ, freq, phase)

end

function krr_train(
    xtrain::Union{<:AbstractVector{<:Real},<:AbstractMatrix{<:Real}},
    ytrain::Union{<:AbstractVector{<:Real},<:AbstractMatrix{<:Real}},
    kernel::RFFKernel,
    ρ::Real,
    f::Union{<:AbstractVector{<:Real},<:AbstractMatrix{<:Real}},
    phase::AbstractVector{<:Real}
)

    # Use random Fourier features to approximate the kernel
    (z, freq, phase) = kernel(ytrain, f, phase)

    return _krr_train(xtrain, z, ρ, freq, phase)

end

function _krr_train(
    xtrain::AbstractVector{<:Real}, # [T]
    z::AbstractMatrix{<:Real}, # [H,T]
    ρ::Real,
    freq::Union{<:AbstractVector{<:Real},<:AbstractMatrix{<:Real}}, # [H,Q] or [H]
    phase::AbstractVector{<:Real} # [H]
)

    # Grab the number of training points
    T = size(z, 2)

    # Calculate sample means
    xm = mean(xtrain) # scalar (L = 1)
    zm = dropdims(mean(z, dims = 2), dims = 2) # [H]

    # Calculate sample covariances
    xtrain = xtrain .- xm # [T]
    z = z .- zm # [H,T]
    Czz = div0.(A_mul_A′(z), T) # [H,H]
    Cxz = div0.(z * xtrain, T) # [H]

    # Calculate the (regularized) inverse of Czz and multiply by Cxz
    CxzCzzinv = transpose(Czz + ρ * I) \ Cxz # [H]

    return RFFTrainingData(freq, phase, zm, xm, Czz, Cxz, CxzCzzinv)

end

function _krr_train(
    xtrain::AbstractMatrix{<:Real}, # [L,T]
    z::AbstractMatrix{<:Real}, # [H,T]
    ρ::Real,
    freq::Union{<:AbstractVector{<:Real},<:AbstractMatrix{<:Real}}, # [H,Q] or [H]
    phase::AbstractVector{<:Real} # [H]
)

    # Grab the number of training points
    T = size(z, 2)

    # Calculate sample means
    xm = dropdims(mean(xtrain, dims = 2), dims = 2) # [L]
    zm = dropdims(mean(z, dims = 2), dims = 2) # [H]

    # Calculate sample covariances
    xtrain = xtrain .- xm # [L,T]
    z = z .- zm # [H,T]
    Czz = div0.(A_mul_A′(z), T) # [H,H]
    Cxz = div0.(xtrain * z', T) # [L,H]

    # Calculate the (regularized) inverse of Czz and multiply by Cxz
    CxzCzzinv = Cxz / (Czz + ρ * I) # [L,H]

    return RFFTrainingData(freq, phase, zm, xm, Czz, Cxz, CxzCzzinv)

end

A_mul_A′(A) = A * A'

# See https://people.maths.ox.ac.uk/gilesm/files/NA-08-01.pdf
function A_mul_A′(D::AbstractMatrix{<:ForwardDiff.Dual{T}}) where T

    A = ForwardDiff.value.(D)
    AA′ = A_mul_A′(A)
    Δ = ForwardDiff.partials.(D)
    # dAA′ = Δ * A' + A * Δ'
    ΔA′ = Δ * A'
    dAA′ = [ΔA′[i,j] + conj(ΔA′[j,i]) for i in axes(ΔA′, 1), j in axes(ΔA′, 2)]
    return ForwardDiff.Dual{T}.(AA′, dAA′)

end

"""
    krr(ytest, trainData, kernel)

Predict latent parameters that generated `ytest` using kernel ridge regression.

# Arguments
- `ytest::Union{<:Real,<:AbstractVector{<:Real},<:AbstractMatrix{<:Real}}`:
  Observed test data [Q,N] or \\[N\\] (if Q = 1) or scalar (if Q = N = 1)
- `trainData::TrainingData`: Training data
- `kernel::Kernel`: Kernel to use

## Notes
- Q is the number of observed features per test sample
- N is the number of test samples

# Return
- `xhat::Union{<:Real,<:AbstractVector{<:Real},<:AbstractMatrix{<:Real}}`:
  Estimated latent parameters [L,N] or \\[N\\] (if L = 1) or \\[L\\] (if N = 1) or
  scalar (if L = N = 1)
"""
function krr(
    ytest::Union{<:Real,<:AbstractVector{<:Real},<:AbstractMatrix{<:Real}},
    trainData::ExactTrainingData,
    kernel::ExactKernel
)

    k = kernel(trainData.y, ytest) # [T,N] or [T]
    k = k .- trainData.Km # [T,N] or [T]

    # Check if L = 1
    if trainData isa ExactTrainingData{<:Any,<:AbstractVector,<:Any,<:Any,<:Any,<:Any}
        xhat = trainData.xm .+ transpose(k) * trainData.xKinv # [N] or scalar
    else
        xhat = trainData.xm .+ trainData.xKinv * k # [L,N] or [L]
    end

    return xhat

end

function krr(
    ytest::Union{<:Real,<:AbstractVector{<:Real},<:AbstractMatrix{<:Real}},
    trainData::RFFTrainingData,
    ::RFFKernel
)

    z = rffmap(ytest, trainData.freq, trainData.phase) # [H,N] or [H]
    z = z .- trainData.zm # [H,N] or [H]

    # Check if L = 1
    if trainData isa RFFTrainingData{<:Any,<:Any,<:Any,<:Any,<:Any,<:AbstractVector,<:Any}
        xhat = trainData.xm .+ transpose(z) * trainData.CxzCzzinv # [N] or scalar
    else
        xhat = trainData.xm .+ trainData.CxzCzzinv * z # [L,N] or [L]
    end

    return xhat

end
