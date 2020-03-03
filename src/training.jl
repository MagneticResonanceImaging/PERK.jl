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
    train(T, xDists, [νDists,] noiseDist, signalModels, kernel)

Train PERK using simulated training data.

# Arguments
- `T::Integer`: Number of training points
- `xDists`: Distributions of latent parameters [L] or scalar (if L = 1);
  `xDists` can be any object such that `rand(xDists, ::Integer)` is defined (or
  a collection of such objects)
- `νDists`: Distributions of known parameters [K] or scalar (if K = 1);
  `νDists` can be any object such that `rand(νDists, ::Integer)` is defined (or
  a collection of such objects); omit this parameter if K = 0
- `noiseDist`: Distribution of noise (assumes same noise distribution for both
  real and imaginary channels in complex case); `noiseDist` can be any object
  such that `rand(noiseDist, ::Integer)` is defined
- `signalModels::Union{<:Function,<:AbstractVector{<:Function}}`: Signal models
  used to generate noiseless data [numSignalModels]; each signal model accepts
  as inputs L latent parameters (scalars) first, then K known parameters
  (scalars); user-defined parameters (e.g., scan parameters in MRI) should be
  built into the signal model
- `kernel::Kernel`: Kernel to use

# Return
- `trainData::TrainingData`: TrainingData object to be passed to `perk`
"""
function train(
    T::Integer,
    xDists,
    noiseDist,
    signalModels::Union{<:Function,<:AbstractVector{<:Function}},
    kernel::Kernel
)

    (y, x) = generatenoisydata(T, xDists, noiseDist, signalModels)

    return krr_train(y, x, kernel)

end

function train(
    T::Integer,
    xDists,
    νDists,
    noiseDist,
    signalModels::Union{<:Function,<:AbstractArray{<:Function,1}},
    kernel::Kernel
)

    (y, x, ν) = generatenoisydata(T, xDists, νDists, noiseDist, signalModels)

    q = combine(y, ν) # [D+K,T]

    return krr_train(q, x, kernel)

end

"""
    combine(y, ν)

Combine the output of the signal models with the known parameters.
"""
function combine(
    y::AbstractVector{<:Real}, # [T]
    ν::AbstractVector{<:Real} # [T]
)

    return transpose([y ν])

end

function combine(
    y::AbstractMatrix{<:Real}, # [D,T]
    ν::AbstractVector{<:Real} # [T]
)

    return [y; transpose(ν)]

end

function combine(
    y::AbstractVector{<:Real}, # [T]
    ν::AbstractMatrix{<:Real} # [K,T]
)

    return [transpose(y); ν]

end

function combine(
    y::AbstractMatrix{<:Real}, # [D,T]
    ν::AbstractMatrix{<:Real} # [K,T]
)

    return [y; ν]

end

"""
    generatenoisydata(N, xDists, [νDists,] noiseDist, signalModels)

Generate noisy data from unknown (and possibly known) parameter distributions.

# Arguments
- `N::Integer`: Number of data points
- `xDists`: Distributions of latent parameters [L] or scalar (if L = 1);
  `xDists` can be any object such that `rand(xDists, ::Integer)` is defined (or
  a collection of such objects)
- `νDists`: Distributions of known parameters [K] or scalar (if K = 1);
  `νDists` can be any object such that `rand(νDists, ::Integer)` is defined (or
  a collection of such objects); omit this parameter if K = 0
- `noiseDist`: Distribution of noise (assumes same noise distribution for both
  real and imaginary channels in complex case); `noiseDist` can be any object
  such that `rand(noiseDist, ::Integer)` is defined
- `signalModels::Union{<:Function,<:AbstractVector{<:Function}}`: Signal models
  used to generate noiseless data [numSignalModels]; each signal model accepts
  as inputs L latent parameters (scalars) first, then K known parameters
  (scalars); user-defined parameters (e.g., scan parameters in MRI) should be
  built into the signal model

# Return
- `y::Union{<:AbstractVector{<:Real},<:AbstractMatrix{<:Real}}`: Output
  magnitude data of all the (simulated) signals [D,N] or [N] (if D = 1)
- `x::Union{<:AbstractVector{<:Real},<:AbstractMatrix{<:Real}}`: Randomly
  generated latent parameters [L,N] or [N] (if L = 1)
- `ν::Union{<:AbstractVector{<:Real},<:AbstractMatrix{<:Real}}`: Randomly
  generated known parameters [K,N] or [N] (if K = 1); not returned if `νDists`
  is omitted
"""
function generatenoisydata(
    N::Integer,
    xDists,
    noiseDist,
    signalModels::Union{<:Function,<:AbstractVector{<:Function}}
)

    # Sample the distributions of latent parameters
    if xDists isa AbstractVector
        x = rand.(xDists, N) # [L][N]
    else
        x = rand(xDists, N) # [N]
    end

    # Generate the data
    if signalModels isa AbstractVector
        y = reduce(vcat, (reduce(hcat, signalModels[i].(x...))
            for i = 1:length(signalModels))) # [D,N]
    else
        y = reduce(hcat, signalModels.(x...)) # [D,N]
    end
    size(y, 1) == 1 && (y = vec(y)) # [N] (if D = 1)
    addnoise!(y, noiseDist)

    # Reshape x
    xDists isa AbstractVector && (x = transpose(hcat(x...))) # [L,N]

    # Return magnitude data and random latent parameters
    return (abs.(y), x)

end

function generatenoisydata(
    N::Integer,
    xDists,
    νDists,
    noiseDist,
    signalModels::Union{<:Function,<:AbstractVector{<:Function}}
)

    # Sample the distributions of latent and known parameters
    if xDists isa AbstractVector
        x = rand.(xDists, N) # [L][N]
    else
        x = rand(xDists, N) # [N]
    end
    if νDists isa AbstractVector
        ν = rand.(νDists, N) # [K][N]
    else
        ν = rand(νDists, N) # [N]
    end

    # Generate the data
    if signalModels isa AbstractVector
        y = reduce(vcat, (reduce(hcat, signalModels[i].(x..., ν...))
            for i = 1:length(signalModels))) # [D,N]
    else
        y = reduce(hcat, signalModels.(x..., ν...)) # [D,N]
    end
    size(y, 1) == 1 && (y = vec(y)) # [N] (if D = 1)
    addnoise!(y, noiseDist)

    # Reshape x and ν
    xDists isa AbstractVector && (x = transpose(hcat(x...))) # [L,N]
    νDists isa AbstractVector && (ν = transpose(hcat(ν...))) # [K,N]

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
