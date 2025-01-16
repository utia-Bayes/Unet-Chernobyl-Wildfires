using Flux
using Random
include(srcdir(unet.jl))
include(srcdir(utils.jl))

struct source_term_cher
    # structure for parameters of source term and unet representing it
    in_ch::Int64 #input channels (number of particle sizes)
    out_ch::Int64 #output channels (number of particle sizes)
    inner_ch::Int64 #channels inside unet (2*(number of paricle sizes))
    skip_ch::Int64 #channels in skip connections 
    kern_size::Int64 #size of convolutional ekrnels
    upsz_x::NTuple{2,Int64} #upsampling sizes of x (longitude) from top size to bottom
    upsz_y::NTuple{2,Int64} #upsampling sizes of y (latitude) from top size to bottom
    upsz_z::NTuple{2,Int64} #upsampling sizes of z (time) from top size to bottom
    twoDM #2D SRS matrix for the whole problem
end

function source_term_cher(Ms; in_ch=3, out_ch=3, inner_ch=6, skip_ch=2, kern_size=3)
    # Ms .................. SRS matrices, tensor of size lat x lon x fraction x altitude x measurements x time
    # in_ch, out_ch, inner_ch, skip_ch, kern_size ... parameters of unet 

    m, n, f, a, p, q = size(Ms)
    upsz = zeros(2, 3)
    upsz[1, :] = [m; n; q]
    for j = 2:2
        upsz[j, :] = Int.(div.(upsz[j-1, :] .- 1, 2) .+ 1)
    end
    upsz_x = tuple(Int.(upsz[:, 1])...)
    upsz_y = tuple(Int.(upsz[:, 2])...)
    upsz_z = tuple(Int.(upsz[:, 3])...)

    big2DM = create_big2DM(Ms)

    source_term_cher(in_ch, out_ch, inner_ch, skip_ch, kern_size, upsz_x, upsz_y, upsz_z, big2DM)
end

function Unet_var(st::source_term_cher)
    # initialize Unet from source term
    Unet_var(st.kern_size, st.in_ch, st.inner_ch, st.skip_ch, st.out_ch, st.upsz_x, st.upsz_y, st.upsz_z)
end

struct training
    # structure for parameters of training
    epochs::Int64 # number of epochs
    lr::Float32 # learning rate
    β::NTuple{2,Float32} # beta parameters of adam optimiser
    pred::Matrix{Float32} # for saving predictions
    report_step::Int64 # report loss and save progress every report_step
    mse_loss::Vector{Float32} # for storing current mse loss values
    elbo_loss::Vector{Float32} # for storing current elbo values
    clip_grad_tresh::Float32 # treshold for clipGard
    clip_norm_tresh::Float32 # treshold for clipNorm
end

function training(measure_N; epochs=2000, lr=0.01f0, β=(0.9, 0.999), report_step=100, cgt=Inf, cnt=Inf)
    # measure_N ...................... number of mesurements upsz_y
    # epochs, lr, β, report_step ..... parameters of training
    # cgt, cnt ....................... tresholds for gradient clipping of values, resp. norms, Inf means no clipping

    pred = zeros(measure_N, div(epochs, report_step) + 1)
    loss = zeros(div(epochs, report_step) + 1)
    elbo = zeros(div(tr.epochs, tr.report_step) + 1)
    training(epochs, lr, β, pred, report_step, loss, elbo, cgt, cnt)
end

mutable struct invert_var_cher
    # structure for the whole inversion process
    st::source_term_cher # parameters of source term
    unet::Unet_var_2l_3d # unet represneting the posterior parametrs
    zx::CuArray # input array for unet
    tr::training # parametrs of training
end

function invert_var_cher(Ms::Array{Float64, 6}; epochs=20000, lr=0.01f0, β=(0.9, 0.999), report_step=100, cgt=Inf, cnt=Inf, omega=1e-2)
    # Ms .................. SRS matrices, tensor of size lat x lon x fraction x altitude x measurements x time
    # epochs, lr, β, report-step, cgt, cnt ... parameters of training
    # omega .................................. precision of data

    m, n, f, a, p, q = size(Ms) 
    st = source_term_cher(Ms, in_ch=f, out_ch=f, inner_ch = 2*f, skip_ch = 2)
    unet = Unet_var(st) |> gpu
    if zxx == "rand"
        zx = rand(Float32, st.upsz_x[1], st.upsz_y[1], st.upsz_z[1], f, a) |> gpu
    elseif zxx == "randn"
        zx = randn(Float32, st.upsz_x[1], st.upsz_y[1], st.upsz_z[1], f, a) |> gpu
    else
        zx = zxx |> gpu
    end
    tr = training(p, epochs=epochs, lr=lr, β=β, report_step=report_step, cgt=cgt, cnt=cnt)

    invert_var_cher(st, unet, zx, tr)
end

struct pretraining
    epochs::Int64 #number of epochs
    lr::Float32 #learning rate
    β::NTuple{2,Float32} #beta parameters of adam optimiser
    pred::Matrix{Float32} #for saving predictions
    report_step::Int64 #report loss and save progress every report_step
    loss::Vector{Float32} #for storing current loss values
    std::CuArray{Float32} #target standard deviation
    mu::CuArray{Float32} #target mean
end

using ImageFiltering

function pretraining_ch(source_size_x, source_size_y, source_size_time, source_size_alt, source_size_frac, 
    measure_N, std, mean; epochs=20000, lr=0.01f0, β=(0.9, 0.999), report_step=100)
    #source_size_x .............. number of latitude steps 
    #source_size_y .............. number of longitude steps
    #source_size_time ........... number of time steps
    #source_size_alt ............ number of steps height levels
    #source_size_frac ........... numper of particle sizes
    #measure_N .................. number of measurements
    #std ........................ target std of shape time x particle size x height level
    #mean ....................... target mean of shape time x particle size x height level

    #initialize buffers for predistions and loss values
    pred = zeros(measure_N, div(epochs, report_step) + 1)
    loss = zeros(div(epochs, report_step) + 1)

    
    stand = ones(source_size_x, source_size_y, source_size_time, source_size_frac, source_size_alt)
    mn = ones(source_size_x, source_size_y, source_size_time, source_size_frac, source_size_alt)
    invkern = ones(source_size_x, source_size_y)
    for a = 1 : source_size_alt
        for f = 1 : source_size_frac
            for t = 1 : source_size_time 
                stand[:, :, t, f, a] = sqrt(std[t, f, a]^2/(source_size_x*source_size_y)).*invkern
                mn[:, :, t, f, a] .= mean[t, f, a]/(source_size_x*source_size_y)
            end
        end
    end
    stand = vec(stand) |> gpu
    mn  = vec(mn) |> gpu
    pretraining_ch(epochs, lr, β, pred, report_step, loss, stand, mn)
end

function pretraining_ch(Ms, std, mu; epochs=2000, lr=0.01f0, β=(0.9, 0.999), report_step=100, std_shape = "unif")
    m, n, f, a, p, t = size(Ms)
    pretraining_ch(m, n, t, a, f, p, std, mu; epochs=epochs, lr=lr, β=β, report_step=report_step, std_shape = std_shape)
end