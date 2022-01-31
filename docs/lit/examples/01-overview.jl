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
using InteractiveUtils: versioninfo


#src The following line is helpful when running this example.jl file as a script;
#src this way it will prompt user to hit a key after each figure is displayed.

#src isinteractive() ? jim(:prompt, true) : prompt(:draw);


# ### Overview

#=
todo
=#


# ## Reproducibility

# This page was generated with the following version of Julia:

io = IOBuffer()
versioninfo(io)
split(String(take!(io)), '\n')


# And with the following package versions

import Pkg; Pkg.status()
