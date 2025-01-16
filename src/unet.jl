using Flux
using LogExpFunctions

upsample3D(mode, up_sz_x, up_sz_y, up_sz_z) = Upsample(mode, size=(up_sz_x, up_sz_y, up_sz_z))

Down3D(kernel, in_ch, inner_ch, inits, act) = Chain(Conv((kernel, kernel, kernel), in_ch=>inner_ch; stride=2, pad=div(kernel, 2), init=inits),
    BatchNorm(inner_ch), 
    act,
    Conv((kernel, kernel, kernel), inner_ch=>inner_ch; stride=1, pad = div(kernel, 2), init=inits), 
    BatchNorm(inner_ch), 
    act)

Up3D(kernel, inner_ch, skip_ch, inits, act) = Chain(BatchNorm(inner_ch + skip_ch),
    Conv((kernel, kernel, kernel), inner_ch + skip_ch=>inner_ch; stride = 1, pad = div(kernel, 2), init=inits),
    BatchNorm(inner_ch),
    act,
    Conv((1,1,1), inner_ch=>inner_ch; stride = 1, pad = 0, init=inits),
    BatchNorm(inner_ch),
    act) 

Skip3D(in_ch, skip_ch, inits, act) = Chain(Conv((1,1,1), in_ch=>skip_ch, stride=1, pad = 0, init=inits),
    BatchNorm(skip_ch),
    act)

struct Unet_var
    #Structure for U-net with two levels and 3D convolution
    d1; d2 #down branches
    u1; u2 #up branches
    s1; s2 #skip connection
    out_mean; out_std #output: mean and std for VDIP
    up_sizes_x; up_sizes_y; up_sizes_z; up_mode #target sizes for upsampling, upsampling mode
end

Flux.@functor Unet_var
Flux.trainable(u::Unet_var) = (d1=u.d1, d2=u.d2, u1=u.u1, u2=u.u2, s1=u.s1, s2=u.s2, out_mean=u.out_mean, out_std=u.out_std) 

function Unet_var(kernel, in_ch, inner_ch, skip_ch, out_ch, up_sz_x, up_sz_y, upsz_z; inits_d=Flux.kaiming_uniform, inits_u=Flux.kaiming_uniform, 
    inits_s=Flux.kaiming_uniform, act=x->leakyrelu.(x, 0.2), up_mode=:bilinear)

    ##=============================##
    ##          Unet               ##
    ##=============================##                     
    # kernel ................... kernel size
    # in_ch .................... number of input channels of the unet
    # inner_ch ................. number of channels inside unet
    # skip_ch .................. number of output channels of skip connections
    # out_ch ................... number of output channels of the unet
    # up_sz_x .................. array of sizes of output from upsampling in frist dimension
    # up_sz_y .................. =||= second dimension
    # up_sz_z .................. =||= third dimension
    # inits_d .................. initialization of downsampling layers
    # inits_u .................. initialization of upsampling layers
    # inits_s .................. initialization of skip layers
    # act ...................... activation function used in the unet
    # up_mode .................. upsampling mode

    #down
    d1 = Down3D(kernel, in_ch, inner_ch, inits_d, act)
    d2 = Down3D(kernel, inner_ch, inner_ch, inits_d, act)
    #up
    u1 = Up3D(kernel, inner_ch, skip_ch, inits_u, act)
    u2 = Up3D(kernel, inner_ch, skip_ch, inits_u, act)
    #skip
    s1 = Skip3D(in_ch, skip_ch, inits_s, act)
    s2 = Skip3D(inner_ch, skip_ch, inits_s, act)
    #out
    out_mean = Chain(Conv((1,1,1), inner_ch=>out_ch; stride = 1, pad = 0), x->log1pexp.(x))
    out_std = Chain(Conv((1,1,1), inner_ch=>out_ch; stride = 1, pad = 0), x->log1pexp.(x))

    return Unet_var(d1, d2, u1, u2, s1, s2, out_mean, out_std, up_sz_x, up_sz_y, upsz_z, up_mode)
end

function (model::Unet_var)(x)
    d1_out = model.d1(x)
    d2_out = model.d2(d1_out)

    s1_out = model.s1(x)
    s2_out = model.s2(d1_out)

    u2_out = model.u2(cat(upsample3D(model.up_mode, model.up_sizes_x[2], model.up_sizes_y[2], model.up_sizes_z[2])(d2_out), s2_out, dims=4))
    u1_out = model.u1(cat(upsample3D(model.up_mode, model.up_sizes_x[1], model.up_sizes_y[1], model.up_sizes_z[1])(u2_out), s1_out, dims=4))

    return (model.out_mean(u1_out), model.out_std(u1_out))
end