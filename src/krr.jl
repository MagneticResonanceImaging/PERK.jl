"""
    krr_train(xtrain, ytrain, kernel, [f, phase])

Train kernel ridge regression.

# Arguments
- `xtrain::AbstractVector{<:Union{<:Real,<:AbstractVector{<:Real}}}`: Latent
  parameters for training data [T]; each element of `xtrain` has length L for
  the number of latent parameters
- `ytrain::AbstractVector{<:Union{<:Real,<:AbstractVector{<:Real}}}`: Feature
  vectors for training data [T]; each element of `ytrain` has length Q for the
  number of features
- `kernel::Kernel`: Kernel to use
- `f::AbstractMatrix{<:Real} = randn(kernel.H, Q)`: TODO: Make this a vector?
"""
function krr_train()



end

"""
    krr(ytest, ytrain, xtrain, kernel, ρ)

Predict latent parameters that generated `ytest` using kernel ridge regression.

# Arguments
- `ytest::Union{<:Real,<:AbstractVector{<:Real}}`: Observed test data
"""
function krr()



end
