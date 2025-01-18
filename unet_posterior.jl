using DrWatson
@quickactivate "Unet-Chernobyl-Wildfires"

using Revise
using LinearAlgebra
using MAT
using Statistics
using Optimisers
using CUDA
using Measures

includet(srcdir("unet.jl"))
includet(srcdir("utils.jl"))
includet("src/inverst_struct.jl")
includet(srcdir("invert.jl"))
includet(srcdir("results.jl"))

#load the data
data = matread(datadir("data.mat"))
y = data["y"]; #measuments
lon = data["longitudes"]; #get longitudes
lat = data["latitudes"]; #get latitudes

#load SRS
file_names = ["SRS$i.mat" for i in 1:7];
Ms_list = [matread(datadir(fname))["M1"] for fname in file_names];
Ms = permutedims(cat(Ms_list...; dims=6), (1,2,3,6,4,5));

#set the parameters of the method
ω = 1e-2
K = 15

#prepase buffer for source terms
source_terms = []

#run the the ineversion K times
for run = 1 : K
    Random.seed!(run)

    #initialize the necesary structures
    prt = pretraining(Ms)
    invstruct = invert_var_cher(Ms, omega = ω)

    #preform pretraining
    pretrain!(invstruct, prt)
    #perform inversion
    invert!(invstruct, gpu(y), ind = run, k=K)

    #save the estimate
    push!(source_terms, cpu(invstruct.unet(invstruct.zx)[1]))
end

#get final estimate
μ = mean(source_terms);
σ = std(source_terms);

#get prediction
M = create_big2DM(Ms);
pred = predict(μ, M);

#show results
show_unet_results_3d(μ*data["koefBq"]/1e9, y, pred, lon, lat)

