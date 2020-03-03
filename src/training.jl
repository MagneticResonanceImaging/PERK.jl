"""
    TrainingData

Abstract type for representing training data.
"""
abstract type TrainingData end

"""
    ExactTrainingData(y, x, xm, K, Km) <: TrainingData

Create an object that contains the training data when using the full Gram matrix
K.

# Properties
- `y::Union{<:AbstractVector{<:Real},<:AbstractMatrix{<:Real}}`: Features for
  training data [Q,T] or [T] (if Q = 1)
- `x::Union{<:AbstractVector{<:Real},<:AbstractMatrix{<:Real}}`: Latent
  parameters for training data [L,T] or [T] (if L = 1)
- `xm::Union{<:Real,<:AbstractVector{<:Real}}`: Mean of latent parameters [L] or
  scalar (if L = 1)
- `K::AbstractMatrix{<:Real}`: De-meaned (both rows and columns) Gram matrix of
  the kernel evaluated on the training data features [T,T]
- `Km::AbstractVector{<:Real}`: Row means of `K` (before de-meaning) [T]
- `Q::Integer`: Number of training features
- `L::Integer`: Number of latent parameters
- `T::Integer`: Number of training points
"""
struct ExactTrainingData{
    T1<:Union{<:AbstractVector{<:Real},<:AbstractMatrix{<:Real}},
    T2<:Union{<:AbstractVector{<:Real},<:AbstractMatrix{<:Real}},
    T3<:Union{<:Real,<:AbstractVector{<:Real}},
    T4<:AbstractMatrix{<:Real},
    T5<:AbstractVector{<:Real}
} <: TrainingData
    y::T1
    x::T2
    xm::T3
    K::T4
    Km::T5
end

Base.getproperty(data::ExactTrainingData, s::Symbol) = begin
    if s == :Q
        y = getfield(data, :y)
        return ndims(y) == 1 ? 1 : size(y, 1)
    elseif s == :L
        x = getfield(data, :x)
        return ndims(x) == 1 ? 1 : size(x, 1)
    elseif s == :T
        return length(getfield(data, :Km))
    else
        return getfield(data, s)
    end
end

"""
    RFFTrainingData(freq, phase, zm, xm, Czz, Cxz) <: TrainingData

Create an object that contains the training data when using an approximation of
the Gram matrix K using random Fourier features.

# Properties
- `freq::Union{<:AbstractVector{<:Real},<:AbstractMatrix{<:Real}}`: Random
  frequency values for random Fourier features [H,Q] or [H] (if Q = 1)
- `phase::AbstractVector{<:Real}`: Random phase values for random Fourier
  features [H]
- `zm::AbstractVector{<:Real}`: Mean of feature maps [H]
- `xm::Union{<:Real,<:AbstractVector{<:Real}}`: Mean of latent parameters [L]
  or scalar (if L = 1)
- `Czz::AbstractMatrix{<:Real}`: Auto-covariance matrix of feature maps [H,H]
- `Cxz::Union{<:AbstractVector{<:Real},<:AbstractMatrix{<:Real}}`:
  Cross-covariance matrix between latent parameters and feature maps [L,H] or
  [H] (if L = 1)
- `Q::Integer`: Number of training features
- `L::Integer`: Number of latent parameters
- `H::Integer`: Kernel approximation order
"""
struct RFFTrainingData{
    T1<:Union{<:AbstractVector{<:Real},<:AbstractMatrix{<:Real}},
    T2<:AbstractVector{<:Real},
    T3<:AbstractVector{<:Real},
    T4<:Union{<:Real,<:AbstractVector{<:Real}},
    T5<:AbstractMatrix{<:Real},
    T6<:Union{<:AbstractVector{<:Real},<:AbstractMatrix{<:Real}}
} <: TrainingData
    freq::T1
    phase::T2
    zm::T3
    xm::T4
    Czz::T5
    Cxz::T6
end

Base.getproperty(data::RFFTrainingData, s::Symbol) = begin
    if s == :Q
        freq = getfield(data, :freq)
        return ndims(freq) == 1 ? 1 : size(freq, 2)
    elseif s == :L
        return length(getfield(data, :xm))
    elseif s == :H
        return length(getfield(data, :zm))
    else
        return getfield(data, s)
    end
end

"""
    train(T, xDists, νDists, noiseDist, signalModels, kernel)

Train PERK using simulated training data.

# Arguments
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

# Return
- `trainData::TrainingData`: TrainingData object to be passed to `perk`
"""
function train(
    T::Integer,
    xDists::AbstractArray{<:Any,1},
    νDists::AbstractArray{<:Any,1},
    noiseDist,
    signalModels::AbstractArray{<:Function,1},
    kernel::Kernel
)

    (y, x, ν) = generatenoisydata(T, xDists, νDists, noiseDist, signalModels)

    return train(y, x, ν, kernel)

end

function train(
    y::AbstractArray{<:Real,2},
    x::AbstractArray{<:AbstractArray{<:Number,1},1},
    ν::AbstractArray{<:AbstractArray{<:Number,1},1},
    kernel::Kernel,
    f::Union{Array{<:Real,2},Nothing} = nothing,
    phase::Union{Array{<:Real,1},Nothing} = nothing
)

    # Combine the training data with the known parameters
    if isempty(ν)
        q = y # [D+K,T], K = 0
    else
        q = [y; transpose(hcat(ν...))] # [D+K,T]
    end

    # Reshape x
    x = transpose(hcat(x...)) # [L,T]

    # Train
    if isnothing(f) || isnothing(phase)
        return train(x, q, kernel)
    else
        return train(x, q, kernel, f, phase)
    end

end

"""
    train(x, q, kernel, [f, phase])

Train PERK.

# Arguments
- `x::AbstractArray{<:Any,2}`: Latent parameters [L,T]
- `q::AbstractArray{<:Any,2}`: Training data concatenated with known
  parameters [D+K,T]
- `kernel::Kernel`: Kernel to use
- `f::Array{<:Real,2} = randn(kernel.H, Q)`: Unscaled random frequency values
  [H,Q] (used when `kernel isa RFFKernel`)
- `phase::Array{<:Real,1} = rand(kernel.H)`: Random phase values [H] (used when
  `kernel isa RFFKernel`)

# Return
- `trainData::TrainingData`: TrainingData object to be passed to `perk`
"""
function train(
    x::AbstractArray{<:Any,2},
    q::AbstractArray{<:Any,2},
    kernel::ExactKernel
)

    # Do the kernel calculation
    K = kernel(q, q) # [T,T]

    # Calculate sample mean and de-mean the latent parameters
    xm = dropdims(mean(x, dims = 2), dims = 2) # [L]
    x = x .- xm # [L,T]

    # De-mean the rows and columns of the kernel output
    Km = dropdims(mean(K, dims = 2), dims = 2) # [T]
    K = K .- Km # [T,T]
    K = K .- mean(K, dims = 1) # [T,T]

    return ExactTrainingData(q, x, xm, K, Km)

end

function train(
    x::AbstractArray{<:Any,2},
    q::AbstractArray{<:Any,2},
    kernel::RFFKernel,
    f::Union{Array{<:Real,2},Nothing} = nothing,
    phase::Union{Array{<:Real,1},Nothing} = nothing
)

    # Use random Fourier features to approximate the kernel
    if isnothing(f) || isnothing(phase)
        (z, freq, phase) = kernel(q)
    else
        (z, freq, phase) = kernel(q, f, phase)
    end

    return train(x, z, freq, phase)

end

function train(x, z, freq, phase)

    # Grab the number of training points
    T = size(z, 2)

    # Calculate sample means
    zm = dropdims(mean(z, dims = 2), dims = 2) # [H]
    xm = dropdims(mean(x, dims = 2), dims = 2) # [L]

    # Calculate sample covariances
    x = x .- xm # [L,T]
    z = z .- zm # [H,T]
    Czz = div0.(z * z', T) # [H,H]
    Cxz = div0.(x * z', T) # [L,H]

    return RFFTrainingData(freq, phase, zm, xm, Czz, Cxz)

end

"""
    generatenoisydata(N, xDists, νDists, noiseDist, signalModels)

Generate noisy data from unknown and known parameter distributions.

# Arguments
- `N::Integer`: Number of data points
- `xDists::AbstractArray{<:Any,1}`: Distributions of latent parameters [L]
- `νDists::AbstractArray{<:Any,1}`: Distributions of known parameters [K]
- `noiseDist`: Distribution of noise (assumes same noise distribution for both
  real and imaginary channels in complex case)
- `signalModels::AbstractArray{<:Function,1}`: Signal models used to generate
  noiseless data [numSignalModels]; each signal model accepts as inputs L
  latent parameters (scalars) first, then K known parameters (scalars);
  user-defined parameters (e.g., scan parameters in MRI) should be built into
  the signal model

# Return
- `y::Real`: Output magnitude data of all the (simulated) signals [D,N]
- `x::Number`: Randomly generated latent parameters [L][N]
- `ν::Number`: Randomly generated known parameters [K][N]
"""
function generatenoisydata(
    N::Integer,
    xDists::AbstractArray{<:Any,1},
    νDists::AbstractArray{<:Any,1},
    noiseDist,
    signalModels::AbstractArray{<:Function,1}
)

    # Sample the distributions of latent and known parameters
    x = rand.(xDists, N) # [L][N]
    ν = rand.(νDists, N) # [K][N]

    # Generate the data
    y = reduce(vcat, (reduce(hcat, signalModels[i].(x..., ν...))
        for i = 1:length(signalModels))) # [D,N]
    addnoise!(y, noiseDist)

    # Return magnitude data and random latent and known parameters
    return (abs.(y), x, ν)

end

"""
    addnoise!(y, noiseDist)

Add noise to `y`. If elements of `y` are complex-valued, then add independent
noise to both the real and imaginary parts of `y`.
"""
function addnoise!(y, noiseDist)

    if eltype(y) <: Complex
        n = complex.(rand(noiseDist, size(y)...), rand(noiseDist, size(y)...))
    else
        n = rand(noiseDist, size(y)...) # [D,N]
    end
    y .+= n # [D,N]

end
