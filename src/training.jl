"""
    TrainingData

Abstract type for representing training data.
"""
abstract type TrainingData end

"""
    ExactTrainingData(q, x, xm, K, Km) <: TrainingData

Create an object that contains the training data when using the full Gram matrix
K.

# Properties
- `q::Array{<:Any,2}`: Training values y concatenated with known parameters ν
    [Q,T]
- `x::Array{<:Any,2}`: De-meaned latent parameters [L,T]
- `xm::Array{<:Any,1}`: Mean of latent parameters [L]
- `K::Array{<:Any,2}`: De-meaned (both rows and columns) Gram matrix of the
    kernel evaluated on the training data `q` [T,T]
- `Km::Array{<:Any,1}`: Row means of `K` (before de-meaning) [T]
- `Q::Int`: Number of training values y plus the number of known parameters ν
- `L::Int`: Number of latent parameters x
- `T::Int`: Number of training points
"""
struct ExactTrainingData{T1,T2,T3,T4,T5} <: TrainingData
    q::Array{T1,2}
    x::Array{T2,2}
    xm::Array{T3,1}
    K::Array{T4,2}
    Km::Array{T5,1}
end

Base.getproperty(data::ExactTrainingData, s::Symbol) = begin
    if s == :Q
        return size(getfield(data, :q), 1)
    elseif s == :L
        return size(getfield(data, :x), 1)
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
- `freq::Array{<:Any,2}`: Random frequency values for random Fourier features
    [H,Q]
- `phase::Array{<:Any,1}`: Random phase values for random Fourier features [H]
- `zm::Array{<:Any,1}`: Mean of feature maps [H]
- `xm::Array{<:Any,1}`: Mean of latent parameters [L]
- `Czz::Array{<:Any,2}`: Auto-covariance matrix of feature maps [H,H]
- `Cxz::Array{<:Any,2}`: Cross-covariance matrix between latent parameters and
    feature maps [L,H]
- `Q::Int`: Number of training values y plus the number of known parameters ν
- `L::Int`: Number of latent parameters x
- `H::Int`: Kernel approximation order
"""
struct RFFTrainingData{T1,T2,T3,T4,T5,T6} <: TrainingData
    freq::Array{T1,2}
    phase::Array{T2,1}
    zm::Array{T3,1}
    xm::Array{T4,1}
    Czz::Array{T5,2}
    Cxz::Array{T6,2}
end

Base.getproperty(data::RFFTrainingData, s::Symbol) = begin
    if s == :Q
        return size(getfield(data, :freq), 2)
    elseif s == :L
        return size(getfield(data, :Cxz), 1)
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
		q = [y; transpose(reduce(hcat, ν))] # [D+K,T]
	end

	# Reshape x
	x = transpose(reduce(hcat, x)) # [L,T]

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
	K .-= Km # [T,T]
	K .-= mean(K, dims = 1) # [T,T]

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
	z .-= zm # [H,T]
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
