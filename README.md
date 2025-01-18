# Unet-Chernobyl-Wildfires
Repository for reproducible research of atmospheric inversion

<!--, as published in: [1] Brozova et. al. Spatial-temporal Source Term Estimation using Deep Neural Network Prior and its Application to Chernobyl Wildfires, Journal of Hazardous Materials...-->

This code base is using the [Julia Language](https://julialang.org/) and
[DrWatson](https://juliadynamics.github.io/DrWatson.jl/stable/)
to make a reproducible scientific project named
> Unet-Chernobyl-Wildfires

It is authored by Antonie Brozova.

To (locally) reproduce this project, do the following:

0. Download this code base.
1. Open a Julia console and do:
   ```
   julia> using Pkg
   julia> Pkg.add("DrWatson") # install globally, for using `quickactivate`
   julia> Pkg.activate("path/to/this/project")
   julia> Pkg.instantiate()
   ```

This will install all necessary packages for you to be able to run the scripts and
everything should work out of the box, including correctly finding local paths.

Run the code in unet_posterior.jl, starting with activation of the Unet-Chernobyl-Wildfires environment.

Our results are save as results15.mat. The file contains a tensor of size 11x11x3x7x28x15, with the last dimension coresponding to the seed. To obtain our final estimate, just compute mean over the last dimension.

