"""
    perk(y, ν, T, xDists, νDists, noiseDist, signalModels, kernel, ρ)

Train PERK and then estimate latent parameters.

# Arguments
- `y::AbstractArray{<:Real,2}`: Test data points [D,N]
- `ν::AbstractArray{<:AbstractArray{<:Real,1},1}`: Known parameters [K][N]
- `T::Integer`: Number of training points
- `xDists::AbstractArray{<:Any,1}`: Distributions of latent parameters [L]
- `νDists::AbstractArray{<:Any,1}`: Distributions of known parameters [K]
- `noiseDist`: Distribution of noise (assumes same noise distribution for both
  real and imaginary channels in complex case)
- `signalModels::AbstractArray{<:Function,1}`: Signal models used to generate
  noiseless data [numSignalModels]; each signal model accepts as inputs L
  latent parameters (scalars) first, then K known parameters (scalars);
  user-defined parameters (e.g., scan parameters in MRI) should be built into
  the signal model
- `kernel::Kernel`: Kernel to use
- `ρ::Real`: Regularization parameter

# Return
- `xhat::Array{<:Real,2}`: Estimated latent parameters [L,N]
- `trainData::TrainingData`: Training data
- `ttrain::Real`: Duration of training (s)
- `ttest::Real`: Duration of testing (s)
"""
function perk(
    y::AbstractArray{<:Real,2},
    ν::AbstractArray{<:AbstractArray{<:Real,1},1},
    T::Integer,
    xDists::AbstractArray{<:Any,1},
    νDists::AbstractArray{<:Any,1},
    noiseDist,
    signalModels::AbstractArray{<:Function,1},
    kernel::Kernel,
    ρ::Real
)

    ttrain = @elapsed begin
        trainData = train(T, xDists, νDists, noiseDist, signalModels, kernel)
    end

    (xhat, ttest) = perk(y, ν, trainData, kernel, ρ) # [L,N]

    return (xhat, trainData, ttrain, ttest)

end

function perk(
    y::AbstractArray{<:Real,2},
    ν::AbstractArray{<:AbstractArray{<:Real,1},1},
    trainData::TrainingData,
    kernel::Kernel,
    ρ::Real
)

    # Concatenate the data and the known parameters
    if isempty(ν)
        q = y # [D+K,N], K = 0
    else
        q = [y; transpose(hcat(ν...))] # [D+K,N]
    end

    (xhat, t) = perk(q, trainData, kernel, ρ) # [L,N]

    return (xhat, t)

end

"""
    perk(q, trainData, kernel, ρ)

Estimate latent parameters using the provided training data.

# Arguments
- `q::AbstractArray{<:Real,2}`: Test data points concatenated with known
  parameters [D+K,N]
- `trainData::TrainingData`: Training data
- `kernel::Kernel`: Kernel to use
- `ρ::Real`: Regularization parameter

# Return
- `xhat::Array{<:Real,2}`: Estimated latent parameters [L,N]
- `t::Real`: Duration of testing (s)
"""
function perk(
    q::AbstractArray{<:Real,2},
    trainData::ExactTrainingData,
    kernel::ExactKernel,
    ρ::Real
)

    t = @elapsed begin
        k = kernel(trainData.q, q) # [T,N]
        k = k .- trainData.Km # [T,N]
        tmp = (trainData.K + trainData.T * ρ * I) \ k # [T,N]
        xhat = trainData.xm .+ trainData.x * tmp # [L,N]
    end

    return (xhat, t)

end

function perk(
    q::AbstractArray{<:Real,2},
    trainData::RFFTrainingData,
    ::RFFKernel,
    ρ::Real
)

    t = @elapsed begin
        z = rffmap(q, trainData.freq, trainData.phase) # [H,N]
        z = z .- trainData.zm # [H,N]
        tmp = (trainData.Czz + ρ * I) \ z # [H,N]
        xhat = trainData.xm .+ trainData.Cxz * tmp # [L,N]
    end

    return (xhat, t)

end
