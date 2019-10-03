"""
    holdout(N, T, λvals, ρvals, weights, xDistsTest, νDistsTest, xDistsTrain,
			νDistsTrain, noiseDist,	signalModels, kernelgenerator; showprogress,
            logfile)

Select λ and ρ via a holdout process.

# Arguments
- `N::Integer`: Number of test points
- `T::Integer`: Number of training points
- `λvals::AbstractArray{<:Real,1}`: Values of λ to search over [nλ]
- `ρvals::AbstractArray{<:Real,1}`: Values of ρ to search over [nρ]
- `weights::AbstractArray{<:Real,1}`: Weights for calculating holdout cost [L]
- `xDistsTest::AbstractArray{<:Any,1}`: Distributions of latent parameters [L];
	used to generate test data
- `νDistsTest::AbstractArray{<:Any,1}`: Distributions of known parameters [K];
	used to generate test data
- `xDistsTrain::AbstractArray{<:Any,1}`: Distributions of latent parameters [L];
	used to generate training data
- `νDistsTrain::AbstractArray{<:Any,1}`: Distributions of known parameters [K];
	used to generate training data
- `noiseDist`: Distribution of noise (assumes same noise distribution for both
    real and imaginary channels in complex case)
- `signalModels::AbstractArray{<:Function,1}`: Signal models used to generate
	noiseless data [numSignalModels]; each signal model accepts as inputs L
	latent parameters (scalars) first, then K known parameters (scalars);
	user-defined parameters (e.g., scan parameters in MRI) should be built into
	the signal model
- `kernelgenerator::Function`: Function that creates a `Kernel` object given a
	vector `Λ` of lengthscales
- `showprogress::Bool = false`: Whether to show progress
- `logfile::String = ""`: File in which to output progress (if `showprogress` is
    set); default `""` indicates to output to the terminal

# Return
- `λ::Real`: Bandwidth scaling parameter
- `ρ::Real`: Regularization parameter
- `Ψ::Array{<:Real,2}`: Holdout costs for λvals and ρvals [nλ,nρ]
"""
function holdout(
	N::Integer,
	T::Integer,
	λvals::AbstractArray{<:Real,1},
	ρvals::AbstractArray{<:Real,1},
	weights::AbstractArray{<:Real,1},
	xDistsTest::AbstractArray{<:Any,1},
	νDistsTest::AbstractArray{<:Any,1},
	xDistsTrain::AbstractArray{<:Any,1},
	νDistsTrain::AbstractArray{<:Any,1},
	noiseDist,
	signalModels::AbstractArray{<:Function,1},
	kernelgenerator::Function;
    showprogress::Bool = false,
    logfile::String = ""
)

	# Generate synthetic test data
	(y, x, ν) = generatenoisydata(N, xDistsTest, νDistsTest, noiseDist,
								  signalModels)

	# Reshape x to make it easier to compare to the output of perk
	x = transpose(reduce(hcat, x)) # [L,N]

	# Loop through each value of λ and ρ
	nλ = length(λvals)
	nρ = length(ρvals)
	Ψ  = zeros(nλ, nρ)
	for idxλ = 1:nλ

        if showprogress
            if logfile == ""
		        println("idxλ = $idxλ/$nλ")
            else
                open(logfile, "a") do f
                    write(f, "idxλ = $idxλ/$nλ\n")
                end
            end
        end

		λ = λvals[idxλ]

		# Generate length scales
		if isempty(ν)
			q = y # [D,N]
		else
			q = [y; transpose(reduce(hcat, ν))]
		end
		Λ = λ * max.(dropdims(mean(abs.(q), dims = 2), dims = 2), eps()) # [D+K]

		# Create the kernel
		kernel = kernelgenerator(Λ)

		# Train PERK
		(trainData,) = train(T, xDistsTrain, νDistsTrain, noiseDist,
							 signalModels, kernel)

		for idxρ = 1:nρ

            if showprogress
                if logfile == ""
    		        println("    idxρ = $idxρ/$nρ")
                else
                    open(logfile, "a") do f
                        write(f, "    idxρ = $idxρ/$nρ\n")
                    end
                end
            end

			ρ = ρvals[idxρ]

			# Run PERK
			(xhat,) = perk(y, ν, trainData, kernel, ρ) # [L,N]

			# Calculate Ψ(λ,ρ), the holdout cost
			werr = ((xhat - x) ./ x) .* sqrt.(weights) # [L,N]
			Ψ[idxλ,idxρ] = sqrt(norm(werr) / N)

		end

	end

	# Return values of λ and ρ that minimize Ψ
	(idxλ, idxρ) = Tuple(argmin(Ψ))
	λ = λvals[idxλ]
	ρ = ρvals[idxρ]

	return (λ, ρ, Ψ)

end
