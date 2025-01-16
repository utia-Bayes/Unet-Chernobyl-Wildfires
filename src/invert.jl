using Flux
using Random
include(srcdir(unet.jl))
include(srcdir(utils.jl))
include(srcdir(invert_struct.jl))

function pretrain(in::invert_var_cher, pt::pretraining_ch; log = true, scale = 1.0, logmodel = false)
    # pretrain the unet to output target values
    # in .............. inversion structure
    # pt .............. pretraining structure

    # set optimiser for unet
    opt = Optimisers.Adam(pt.lr, pt.β)
    state = Optimisers.setup(opt, in.unet)

    # set tragets 
    m, n, t, f, a = size(in.unet(in.zx)[1])
    target = vcat(vec(pt.mu), vec(pt.std))./scale
    if logmodel == true
        println("Log was set to false, because logmodel = true.")
        log = false
    end
    mu, sigma = vec.(in.unet(in.zx))
    max_m = zeros(div(pt.epochs, pt.report_step)+1)
    max_m[1] = maximum(mu)
    max_s = zeros(div(pt.epochs, pt.report_step)+1)
    max_s[1] = maximum(sigma)
    min_m = zeros(div(pt.epochs, pt.report_step)+1)
    min_m[1] = minimum(mu)
    min_s = zeros(div(pt.epochs, pt.report_step)+1)
    min_s[1] = minimum(sigma)
    for j = 1 : pt.epochs
        if log
            gsx, _ = gradient(in.unet, in.zx, target) do unet, in_array, tar # calculate the gradients
                Flux.mse(log10.(vcat(vec.(unet(in_array))...)), log10.(tar))
            end
        elseif logmodel
            gsx, _ = gradient(in.unet, in.zx, target) do unet, in_array, tar # calculate the gradients
                mu, S = vec.(unet(in_array))
                mean_x = exp.(mu.+S/2)
                std_x = sqrt.(exp.(S.^2 .-1).*exp.(2 .*mu .+ S.^2))
                Flux.mse(log10.(vcat(mean_x, std_x)), log10.(tar))
            end
        else
            gsx, _ = gradient(in.unet, in.zx, target) do unet, in_array, tar # calculate the gradients
                Flux.mse(vcat(vec.(unet(in_array))...), tar)
            end
        end

        state, in.unet = Optimisers.update!(state, in.unet, gsx)

        if div(j, pt.report_step)-(j)/pt.report_step == 0
            mu, sigma = vec.(in.unet(in.zx))
            if log
                curr_loss = Flux.mse(log10.(vcat(mu, sigma)), log10.(target))
            elseif logmodel
                mean_x = exp.(mu.+sigma/2)
                std_x = sqrt.(exp.(sigma.^2 .-1).*exp.(2 .*mu .+ sigma.^2))
                curr_loss = Flux.mse(log10.(vcat(mean_x, std_x)), log10.(target))
            else
                curr_loss = Flux.mse(vcat(mu, sigma), target)
            end
            y_pred = predict(cpu(mu), in.st.twoDM)
            println("Iteration ", j, ", Loss: ", curr_loss)
            pt.prog_m[:, div(j, pt.report_step)+1] = vec(sum(reshape(cpu(mu), m, n, t, f, a)[:, :, :,1,1], dims=3))
            pt.prog_s[:, div(j, pt.report_step)+1] = vec(sum(reshape(cpu(sigma), m, n, t, f, a)[:, :, :,1,1], dims=3))
            pt.pred[:, div(j, pt.report_step)+1] = cpu(y_pred)
            pt.loss[div(j, pt.report_step)+1] = curr_loss
            max_m[div(j, pt.report_step)+1] = maximum(mu)
            max_s[div(j, pt.report_step)+1] = maximum(sigma)
            min_m[div(j, pt.report_step)+1] = minimum(mu)
            min_s[div(j, pt.report_step)+1] = minimum(sigma)
            isnan(curr_loss) ? break : continue
        end
    end
    return max_m, max_s, min_m, min_s
end

function invert!(in::invert_var_cher, y::CuArray)
    # inversion 
    # in ........... inversion structure
    # y ............ measurements

    #create the optimiser for unet
    if in.tr.clip_grad_tresh == Inf && in.tr.clip_norm_tresh == Inf
        opt = Optimisers.Adam(in.tr.lr, in.tr.β)
    elseif in.tr.clip_grad_tresh == Inf
        opt = Optimisers.OptimiserChain(Optimisers.ClipNorm(in.tr.clip_norm_tresh), Optimisers.Adam(in.tr.lr, in.tr.β))
    elseif in.tr.clip_norm_tresh == Inf
        opt = Optimisers.OptimiserChain(Optimisers.ClipGrad(in.tr.clip_grad_tresh), Optimisers.Adam(in.tr.lr, in.tr.β))
    else
        println("Do you really want to clip both gradient value and norm of gradient?")
        opt = Optimisers.OptimiserChain(Optimisers.ClipGrad(in.tr.clip_grad_tresh), Optimisers.ClipNorm(in.tr.clip_norm_tresh), Optimisers.Adam(in.tr.lr, in.tr.β))
    end
    state_unet = Optimisers.setup(opt, in.unet)

    #get size of source term
    #m, n, q, ch, bs = size(in.unet(in.zx)[1])

    #precompute som eterms
    Mt = gpu(in.st.twoDM')
    Mty = vec(gpu(cpu(y)'*in.st.twoDM'))

    # get losses before optimization
    ls = size(in.st.twoDM, 1) #number of measurements
    epsilon = randn(ls, 1)|>gpu #sample epsilon
    in.tr.elbo_loss[1] = var_loss_rep_hinge(in.unet, in.varp.ω, in.zx,  Mty, Mt, epsilon) #compute elbo
    mu, sigma = vec.(in.unet(in.zx)) # get mu and sigma
    y_pred = predict(mu, gpu(in.st.twoDM)) # get prediction
    in.tr.mse_loss[1] = Flux.mse(y_pred, y) #compute MSE loss

    #run optimization
    for j = 1 : in.tr.epochs
        epsilon = randn(ls, in.varp.n_samples)|>gpu #sample epsilon

        #compute gradient
        gsx, _ = gradient(in.unet, in.varp.ω, in.zx, Mty, Mt, epsilon) do unet, om, in_array, omt, mtm, epsl
            var_loss_rep_hinge(unet, om, in_array, omt, mtm, epsl)
        end

        #update parameters of unet
        state_unet, in.unet = Optimisers.update!(state_unet, in.unet, gsx)

        #compute current loss
        curr_loss = var_loss_rep_hinge(in.unet, in.varp.ω, in.zx, Mty, Mt, epsilon)

        #check the current loss value
        if isnan(curr_loss)
            println("Not a number. Breaking in iteration ", j)
            break
        end

        # report statistics each report_step step
        if div(j, in.tr.report_step)-(j)/in.tr.report_step == 0
            mu, sigma = vec.(in.unet(in.zx))
            x_pred = mu
            y_pred = predict(x_pred, Mt')
            in.tr.mse_loss[div(j, in.tr.report_step)+1] = Flux.mse(y_pred, y)
            in.tr.pred[:, div(j, in.tr.report_step)+1] = cpu(y_pred)
            in.tr.elbo_loss[div(j, in.tr.report_step)+1] = curr_loss
            println("Iteration ", j, ", MSE Loss: ", curr_loss)
        end
    end
end

function var_loss_rep_hinge(Gx::Unet_var, omega::Float32, in_array::CuArray, MTy::CuArray, twoDMT, ϵs)
    # ELBO loss
    # Gx ............... variational unet
    # omega ............ precision of measurements
    # in_array ......... input of Gx
    # MTy .............. precomputed prooduct M'*y_pred
    # twoDMT ........... 2D SRS matrix M
    # ϵs ............... epsilon sample for reparametrization trick

    # get mu and sigma in 5D form with particle size in 4th dimension
    mum, sigmam = Gx(in_array)

    # create masks for prior on x
    mask1_up = (sign.(mum[:, :, :, 1, :] - 100*mum[:, :, :, 2, :]).+1)./2
    mask1_low = (sign.(mum[:, :, :, 1, :] - 0.01*mum[:, :, :, 2, :]).-1)./2
    mask2_up = (sign.(mum[:, :, :, 2, :] - 100*mum[:, :, :, 3, :]).+1)./2 
    mask2_low = (sign.(mum[:, :, :, 2, :] - 0.01*mum[:, :, :, 3, :]).-1)./2
    # apply the mask
    pen = sum(((mum[:, :, :, 1, :] .-  100*mum[:, :, :, 2, :]).*mask1_up).^2)/max(1,sum(mask1_up)) +
            sum(((mum[:, :, :, 1, :] .-  0.01*mum[:, :, :, 2, :]).*mask1_low).^2)/max(1,sum(mask1_low)) +
            sum(((mum[:, :, :, 2, :] .-  100*mum[:, :, :, 3, :]).*mask2_up).^2)/max(1,sum(mask2_up)) +
            sum(((mum[:, :, :, 2, :] .-  0.01*mum[:, :, :, 3, :]).*mask2_low).^2)/max(1,sum(mask2_low))

    # reshape mu and sigma into vector form
    mu = vec(mum)
    sigma = vec(sigmam)
    sigma2 = sigma.^2

    # first order term from likelihood
    m_term = mu'*(omega.*MTy)
    # second order term from likelihood that is reparametrized
    rep_term = -mean(map(x-> sum(omega/2 .* (twoDMT *(mu.+x.*sigma)).^2), eachcol(ϵs)))
    # ∑log(σ_i)
    s_term = 1/2*sum(log.(sigma2.+1e-8))

    return -sum(m_term .+ rep_term .+ s_term)+pen
end