"""
    perk(y, T, xDists, noiseDist, signalModels, kernel, ρ)
    perk(y, ν, T, xDists, νDists, noiseDist, signalModels, kernel, ρ)

Train PERK and then estimate latent parameters.

# Arguments
- `y::Union{<:Real,<:AbstractVector{<:Real},<:AbstractMatrix{<:Real}}`: Test
  data points [D,N] or \\[N\\] (if D = 1) or scalar (if D = N = 1)
- `ν::Union{<:Real,<:AbstractVector{<:Real},<:AbstractMatrix{<:Real}}`: Known
  parameters [K,N] or \\[N\\] (if K = 1) or scalar (if K = N = 1); omit this
  parameter if K = 0
- `T::Integer`: Number of training points
- `xDists`: Distributions of latent parameters \\[L\\] or scalar (if L = 1);
  `xDists` can be any object such that `rand(xDists, ::Integer)` is defined (or
  a collection of such objects)
- `νDists`: Distributions of known parameters \\[K\\] or scalar (if K = 1);
  `νDists` can be any object such that `rand(νDists, ::Integer)` is defined (or
  a collection of such objects); omit this parameter if K = 0
- `noiseDist`: Distribution of noise (assumes same noise distribution for both
  real and imaginary channels in complex case); `noiseDist` can be any object
  such that `rand(noiseDist, ::Integer)` is defined
- `signalModels::Union{<:Function,<:AbstractVector{<:Function}}`: Signal models
  used to generate noiseless data \\[numSignalModels\\]; each signal model
  accepts as inputs L latent parameters (scalars) first, then K known parameters
  (scalars); user-defined parameters (e.g., scan parameters in MRI) should be
  built into the signal model
- `kernel::Kernel`: Kernel to use
- `ρ::Real`: Tikhonov regularization parameter

## Note
- D is the combined number of outputs from all signal models (e.g., number of
  scans in MRI)
- K is the number of known parameters
- L is the number of unknown or latent parameters to be estimated
- N is the number of test points (e.g., number of voxels in MRI)

# Return
- `xhat::Union{<:Real,<:AbstractVector{<:Real},<:AbstractMatrix{<:Real}}`:
  Estimated latent parameters [L,N] or \\[N\\] (if L = 1) or \\[L\\] (if N = 1)
  or scalar (if L = N = 1)
"""
function perk(
    y::Union{<:Real,<:AbstractVector{<:Real},<:AbstractMatrix{<:Real}},
    T::Integer,
    xDists,
    noiseDist,
    signalModels::Union{<:Function,<:AbstractVector{<:Function}},
    kernel::Kernel,
    ρ::Real
)

    trainData = train(T, xDists, noiseDist, signalModels, kernel, ρ)

    return perk(y, trainData, kernel)

end

function perk(
    y::Union{<:Real,<:AbstractVector{<:Real},<:AbstractMatrix{<:Real}},
    ν::Union{<:Real,<:AbstractVector{<:Real},<:AbstractMatrix{<:Real}},
    T::Integer,
    xDists,
    νDists,
    noiseDist,
    signalModels::Union{<:Function,<:AbstractVector{<:Function}},
    kernel::Kernel,
    ρ::Real
)

    trainData = train(T, xDists, νDists, noiseDist, signalModels, kernel, ρ)

    return perk(y, ν, trainData, kernel)

end

"""
    perk(y, trainData, kernel)
    perk(y, ν, trainData, kernel)

Estimate latent parameters using the provided training data.

# Arguments
- `y::Union{<:Real,<:AbstractVector{<:Real},<:AbstractMatrix{<:Real}}`: Test
  data points [D,N] or \\[N\\] (if D = 1) or scalar (if D = N = 1)
- `ν::Union{<:Real,<:AbstractVector{<:Real},<:AbstractMatrix{<:Real}}`: Known
  parameters [K,N] or \\[N\\] (if K = 1) or scalar (if K = N = 1); omit this
  parameter if K = 0
- `trainData::TrainingData`: Training data
- `kernel::Kernel`: Kernel to use

# Return
- `xhat::Union{<:Real,<:AbstractVector{<:Real},<:AbstractMatrix{<:Real}}`:
  Estimated latent parameters [L,N] or \\[N\\] (if L = 1) or \\[L\\] (if N = 1)
  or scalar (if L = N = 1)
"""
function perk(
    y::Union{<:Real,<:AbstractVector{<:Real},<:AbstractMatrix{<:Real}},
    trainData::TrainingData,
    kernel::Kernel
)

    return krr(y, trainData, kernel)

end

function perk(
    y::Union{<:Real,<:AbstractVector{<:Real},<:AbstractMatrix{<:Real}},
    ν::Union{<:Real,<:AbstractVector{<:Real},<:AbstractMatrix{<:Real}},
    trainData::TrainingData,
    kernel::Kernel
)

    q = combine(y, ν)

    return krr(q, trainData, kernel)

end
