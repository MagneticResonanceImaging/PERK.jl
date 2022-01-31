#---------------------------------------------------------
# # [PERK overview](@id 01-overview)
#---------------------------------------------------------

#=
This page illustrates the Julia package
[`PERK`](https://github.com/StevenWhitaker/PERK.jl).

This page was generated from a single Julia file:
[01-overview.jl](@__REPO_ROOT_URL__/01-overview.jl).
=#

#md # In any such Julia documentation,
#md # you can access the source code
#md # using the "Edit on GitHub" link in the top right.

#md # The corresponding notebook can be viewed in
#md # [nbviewer](http://nbviewer.jupyter.org/) here:
#md # [`01-overview.ipynb`](@__NBVIEWER_ROOT_URL__/01-overview.ipynb),
#md # and opened in [binder](https://mybinder.org/) here:
#md # [`01-overview.ipynb`](@__BINDER_ROOT_URL__/01-overview.ipynb).


# ### Setup

# Packages needed here.

using PERK
using MIRTjim: jim, prompt
using Plots; default(markerstrokecolor = :auto, label="")
using InteractiveUtils: versioninfo


#src The following line is helpful when running this file as a script;
#src this way it will prompt user to hit a key after each figure is displayed.

#src isinteractive() ? jim(:prompt, true) : prompt(:draw);


# ### Overview

#=
Although neural networks are very popular,
low-dimensional nonlinear regression problems
can be handled quite efficiently
by kernel regression (KR).
Training KR does not require iterative algorithms
and is more interpretable than a deep network.
It is simply a nonlinear lifting
followed by ridge regression.
=#

#=
### Example

Here is an example of using KR
to learn the function ``y = x^3``
from noisy training data.
=#

fun(x) = x^3
xtrain = LinRange(-1, 1, 101) * 3
ytrain = fun.(xtrain) + 1 * randn(size(xtrain))
p0 = scatter(xtrain, ytrain, xlabel="x", ylabel="y", label="training data")
plot!(p0, fun, label="y = x^3", legend=:top, color=:black)

# Here is the key training step
ρ = 1e-5
λ = 0.5
kernel = GaussianKernel(λ)
train = PERK.krr_train(ytrain, xtrain, kernel, ρ);
#src jim(train.K, "PERK K")

# Now examine the fit using (exhaustive) test data.
# The fit is very good within the range of the training data,
# and regresses to the mean outside of that range.
xtest = LinRange(-1, 1, 200) * 4
yhat = PERK.krr(xtest, train, kernel) # todo: remove kernel eventually
plot!(p0, xtest, yhat, label="KR prediction", color=:magenta)


# The (only!) two parameters ρ and λ can be selected automatically
# using cross validation.
# todo: need example here


# ### Reproducibility

# This page was generated with the following version of Julia:

io = IOBuffer()
versioninfo(io)
split(String(take!(io)), '\n')


# And with the following package versions

import Pkg; Pkg.status()
