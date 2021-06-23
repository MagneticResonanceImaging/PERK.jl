"""
    krr_train(xtrain, ytrain, kernel, ρ)

Train kernel ridge regression.

# Arguments
- `xtrain::Union{<:AbstractVector{<:Real},<:AbstractMatrix{<:Real}}`: Latent
  parameters for training data [L,T] or \\[T\\] (if L = 1)
- `ytrain::Union{<:AbstractVector{<:Real},<:AbstractMatrix{<:Real}}`: Features
  for training data [Q,T] or \\[T\\] (if Q = 1)
- `kernel::Kernel`: Kernel to use
- `ρ::Real`: Tikhonov regularization parameter

## Note
- L is the number of unknown or latent parameters to be predicted
- Q is the number of observed features per training sample
- T is the number of training samples

# Return
- `trainingdata::TrainingData`: `TrainingData` object to be passed to `krr`
"""
function krr_train(
    xtrain::AbstractVector{<:Real},
    ytrain::Union{<:AbstractVector{<:Real},<:AbstractMatrix{<:Real}},
    kernel::ExactKernel,
    ρ::Real
)

    T = promote_type(Float64, eltype(xtrain), eltype(ytrain))
    Ty = typeof(ytrain)
    if Ty <: AbstractVector
        trainingdata = ExactTrainingData(Ty, T, length(ytrain))
    else
        trainingdata = ExactTrainingData(Ty, T, size(ytrain)...)
    end
    krr_train!(trainingdata, xtrain, ytrain, kernel, ρ)
    return trainingdata

end

function krr_train!(
    trainingdata::ExactTrainingData,
    xtrain::AbstractVector{<:Real},
    ytrain::Union{<:AbstractVector{<:Real},<:AbstractMatrix{<:Real}},
    kernel!::ExactKernel,
    ρ::Real
)

    # Grab the number of training points
    T = length(xtrain)

    # Evaluate the kernel on the training features
    kernel!(trainingdata.K, ytrain, ytrain) # [T,T]

    # Calculate the sample mean and de-mean the latent parameters
    trainingdata.xm[] = mean(xtrain) # scalar (L = 1)
    trainingdata.x .= xtrain .- trainingdata.xm[] # [T]

    # De-mean the rows and columns of the kernel output
    for t = 1:T
        trainingdata.Km[t] = mean(trainingdata.K[t,i] for i = 1:T)
    end
    for t1 = 1:T
        tmp = trainingdata.Km[t1]
        for t2 = 1:T
            trainingdata.K[t1,t2] -= tmp
        end
    end
    for t2 = 1:T
        m = mean(trainingdata.K[i,t2] for i = 1:T)
        for t1 = 1:T
            trainingdata.K[t1,t2] -= m
        end
    end

    # Compute the (regularized) inverse of K and multiply by xtrain
    F = lu(transpose(trainingdata.K + T * ρ * I))
    copyto!(trainingdata.xKinv, trainingdata.x)
    ldiv!(F, trainingdata.xKinv) # [T]

    # Copy ytrain
    copyto!(trainingdata.y, ytrain)

    return nothing

end

function krr_train(
    xtrain::AbstractMatrix{<:Real},
    ytrain::Union{<:AbstractVector{<:Real},<:AbstractMatrix{<:Real}},
    kernel::ExactKernel,
    ρ::Real
)

    T = promote_type(Float64, eltype(xtrain), eltype(ytrain))
    Ty = typeof(ytrain)
    if Ty <: AbstractVector
        trainingdata = ExactTrainingData(Ty, T, size(xtrain)...)
    else
        trainingdata = ExactTrainingData(Ty, T, size(xtrain, 1), size(ytrain)...)
    end
    krr_train!(trainingdata, xtrain, ytrain, kernel, ρ)
    return trainingdata

end

function krr_train!(
    trainingdata::ExactTrainingData,
    xtrain::AbstractMatrix{<:Real},
    ytrain::Union{<:AbstractVector{<:Real},<:AbstractMatrix{<:Real}},
    kernel!::ExactKernel,
    ρ::Real
)

    # Grab the number of latent parameters and training points
    (L, T) = size(xtrain)

    # Evaluate the kernel on the training features
    kernel!(trainingdata.K, ytrain, ytrain) # [T,T]

    # Calculate the sample mean and de-mean the latent parameters
    for l = 1:L
        m = mean(xtrain[l,t] for t = 1:T)
        trainingdata.xm[l] = m
        for t = 1:T
            trainingdata.x[l,t] = xtrain[l,t] - m
        end
    end

    # De-mean the rows and columns of the kernel output
    for t = 1:T
        trainingdata.Km[t] = mean(trainingdata.K[t,i] for i = 1:T)
    end
    for t1 = 1:T
        tmp = trainingdata.Km[t1]
        for t2 = 1:T
            trainingdata.K[t1,t2] -= tmp
        end
    end
    for t2 = 1:T
        m = mean(trainingdata.K[i,t2] for i = 1:T)
        for t1 = 1:T
            trainingdata.K[t1,t2] -= m
        end
    end

    # Compute the (regularized) inverse of K and multiply by xtrain
    F = lu(trainingdata.K + T * ρ * I)
    copyto!(trainingdata.xKinv, trainingdata.x)
    rdiv!(trainingdata.xKinv, F) # [L,T]

    # Copy ytrain
    copyto!(trainingdata.y, ytrain)

    return nothing

end

function krr_train(
    xtrain::Union{<:AbstractVector{<:Real},<:AbstractMatrix{<:Real}}, # [T] or [L,T]
    ytrain::Union{<:AbstractVector{<:Real},<:AbstractMatrix{<:Real}},
    kernel::RFFKernel,
    ρ::Real
)

    T = promote_type(Float64, eltype(xtrain), eltype(ytrain))
    Tx = typeof(xtrain)
    H = length(kernel.phase)
    if Tx <: AbstractVector
        trainingdata = RFFTrainingData(Tx, T, length(xtrain), H)
    else
        trainingdata = RFFTrainingData(Tx, T, size(xtrain)..., H)
    end
    krr_train!(trainingdata, xtrain, ytrain, kernel, ρ)
    return trainingdata

end

function krr_train!(
    trainingdata::RFFTrainingData,
    xtrain::AbstractVector{<:Real}, # [T]
    ytrain::Union{<:AbstractVector{<:Real},<:AbstractMatrix{<:Real}},
    kernel!::RFFKernel,
    ρ::Real
)

    # Use random Fourier features to approximate the kernel
    kernel!(trainingdata.z, ytrain) # [H,T]

    # Grab the number of training points and approximation order
    (H, T) = size(trainingdata.z)

    # Calculate sample means
    trainingdata.xm[] = mean(xtrain) # scalar (L = 1)
    for h = 1:H
        trainingdata.zm[h] = mean(trainingdata.z[h,t] for t = 1:T)
    end

    # Calculate sample covariances
    trainingdata.x .= xtrain .- trainingdata.xm[] # [T]
    for h = 1:H
        tmp = trainingdata.zm[h]
        for t = 1:T
            trainingdata.z[h,t] -= tmp
        end
    end
    mul!(trainingdata.Czz, trainingdata.z, trainingdata.z')
    trainingdata.Czz .= div0.(trainingdata.Czz, T) # [H,H]
    mul!(trainingdata.Cxz, trainingdata.z, trainingdata.x)
    trainingdata.Cxz .= div0.(trainingdata.Cxz, T) # [H]

    # Calculate the (regularized) inverse of Czz and multiply by Cxz
    F = lu(transpose(trainingdata.Czz + ρ * I))
    copyto!(trainingdata.CxzCzzinv, trainingdata.Cxz)
    ldiv!(F, trainingdata.CxzCzzinv) # [H]

    return nothing

end

function krr_train!(
    trainingdata::RFFTrainingData,
    xtrain::AbstractMatrix{<:Real}, # [L,T]
    ytrain::Union{<:AbstractVector{<:Real},<:AbstractMatrix{<:Real}},
    kernel!::RFFKernel,
    ρ::Real
)

    # Use random Fourier features to approximate the kernel
    kernel!(trainingdata.z, ytrain) # [H,T]

    # Grab the number of training points and approximation order
    (H, T) = size(trainingdata.z)

    # Grab the number of latent parameters
    L = size(xtrain, 1)

    # Calculate sample means
    for l = 1:L
        m = mean(xtrain[l,t] for t = 1:T)
        trainingdata.xm[l] = m
        for t = 1:T
            trainingdata.x[l,t] = xtrain[l,t] - m
        end
    end
    for h = 1:H
        trainingdata.zm[h] = mean(trainingdata.z[h,t] for t = 1:T)
    end

    # Calculate sample covariances
    for h = 1:H
        tmp = trainingdata.zm[h]
        for t = 1:T
            trainingdata.z[h,t] -= tmp
        end
    end
    mul!(trainingdata.Czz, trainingdata.z, trainingdata.z')
    trainingdata.Czz .= div0.(trainingdata.Czz, T) # [H,H]
    mul!(trainingdata.Cxz, trainingdata.x, trainingdata.z')
    trainingdata.Cxz .= div0.(trainingdata.Cxz, T) # [L,H]

    # Calculate the (regularized) inverse of Czz and multiply by Cxz
    F = lu(trainingdata.Czz + ρ * I)
    copyto!(trainingdata.CxzCzzinv, trainingdata.Cxz)
    rdiv!(trainingdata.CxzCzzinv, F) # [L,H]

    return nothing

end

"""
    krr(ytest, trainingdata, kernel)

Predict latent parameters that generated `ytest` using kernel ridge regression.

# Arguments
- `ytest::Union{<:Real,<:AbstractVector{<:Real},<:AbstractMatrix{<:Real}}`:
  Observed test data [Q,N] or \\[N\\] (if Q = 1) or scalar (if Q = N = 1)
- `trainingdata::TrainingData`: Training data
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
    trainingdata::ExactTrainingData,
    kernel::ExactKernel
)

    k = kernel(trainingdata.y, ytest) # [T,N] or [T]
    k = k .- trainingdata.Km # [T,N] or [T]

    # Check if L = 1
    if trainingdata isa ExactTrainingData{<:Any,<:AbstractVector,<:Any,<:Any,<:Any,<:Any}
        xhat = trainingdata.xm .+ transpose(k) * trainingdata.xKinv # [N] or scalar
    else
        xhat = trainingdata.xm .+ trainingdata.xKinv * k # [L,N] or [L]
    end

    return xhat

end

function krr(
    ytest::Union{<:Real,<:AbstractVector{<:Real},<:AbstractMatrix{<:Real}},
    trainingdata::RFFTrainingData,
    kernel::RFFKernel
)

    z = rffmap(ytest, kernel.freq, kernel.phase) # [H,N] or [H]
    z = z .- trainingdata.zm # [H,N] or [H]

    # Check if L = 1
    if trainingdata isa RFFTrainingData{<:Any,<:Any,<:AbstractVector,<:Any,<:Any,<:Any,<:Any}
        xhat = trainingdata.xm[] .+ transpose(z) * trainingdata.CxzCzzinv # [N] or scalar
    else
        xhat = trainingdata.xm .+ trainingdata.CxzCzzinv * z # [L,N] or [L]
    end

    return xhat

end
