using DrWatson
@quickactivate "Unet-Chernobyl-Wildfires"

using Revise
using LinearAlgebra
using MAT
using Statistics
using Optimisers
using Plots
using CUDA
using Measures

includet(srcdir("unet.jl"))
includet(srcdir("utils.jl"))
includet(srcdir("invert_struct.jl"))
includet(srcdir("invert.jl"))
includet(srcdir("results.jl"))

function run_experiment(p::Dict)
    @unpack y, Ms, epochs, lr, data, beta1, o, t, run, n_samples, lon, lat, x, s, dny, hing  = p
    
    Random.seed!(run)

    invstruct = invert_var_cher(Ms, epochs = epochs, lr = lr, β = (beta1, 0.999), nsamples = n_samples, omega = o)
    prt = pretraining_ch(Ms, s, x, epochs = 20000, lr = 0.01)

    pretrain(invstruct, prt)
    invert!(invstruct, gpu(y), hinged = hing)

    p1 = show_unet_results_3d(invstruct, vec(y), lon, lat, disp = false, days = vec(dny))

    results = copy(p)
    results["x_out"] = cpu(invstruct.unet(invstruct.zx)[1])
    results["std_out"] = cpu(invstruct.unet(invstruct.zx)[1])
    results["mse_loss"] = cpu(invstruct.tr.loss)
    results["elbo_loss"] = cpu(invstruct.elbo)
    results["pred"] = cpu(predict(cpu(invstruct.unet(invstruct.zx)[1]), invstruct.st.twoDM))
    #results["x_gt"] = x
    results["gpu"] = CUDA.name(CUDA.device())

    return p1, results
end

expr = "var_unet"
data = "Chernobyl2_noExcl"
dat = matread(string("/disk/brozova/brozova/Documents/plasty/spaciotemporal/data/Chernobyl_fires_2020_v2/SRS_data_EU_spatiotemporal_v2.mat"))
dt = matread("/disk/brozova/brozova/Documents/plasty/spaciotemporal/data/Chernobyl_fires_2020_v2/results_BLSAPCr.mat")
x = dt["hat_x_dnyGBq"]*1e9/dat["koefBq"]
std = dt["std_dsixGBq"]*1e9/dat["koefBq"]

x = permutedims(reshape(x, 28, 7, 3), (1,3,2)).*0 .+11*11
s = permutedims(reshape(std, 28, 7, 3), (1,3,2)).*0 .+11*11

y = dat["y_orig"][1:5:end, :]
indices = vec(Bool.(Int64.(abs.(dat["ind_cez"] .- 1))))
y = y[indices, :]
Ms = dat["Ms"][:, :, :, :, indices, 2:end-1]
lon = dat["longitudes"]
lat = dat["latitudes"]
dny = reshape(2:29, 28,1)

epochs = 20000
lr = 0.01f0
beta1 = 0.9f0
o = [1e-2]
run = [53;54]
n_samples = [1]
t = [0.0]
hing = true


params = @strdict y Ms epochs lr data beta1 expr o t run n_samples lon lat x s dny hing
settings = dict_list(params)

for (i, d) in enumerate(settings)
    p1, res = run_experiment(d)
    @tagsave(string("results/chernobyl_16plus/", savename(d, "jld2")), res)
    savefig(p1, string("results/chernobyl_16plus/", savename(d, "png")))
    #savefig(p2, string("/disk/brozova/brozova/Documents/plasty/spaciotemporal/projects/unet/results/chernobyl/variational_tau_omega/", savename(d), "_months.png"))
end