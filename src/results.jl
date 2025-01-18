using Plots


function barplot(emission)
    categories = 2:29
    index = 1:28
    fig = plot(xlabel="Day", ylabel="Emission in GBq", title="Total emission")
    bar!(index, emission, lw=1, lc=:black, fillcolor=:limegreen, label="Unet posterior", margin = 5mm)
    labels = [i % 2 == 0 ? string(c) : "" for (i, c) in enumerate(categories)]
    xticks!(index, labels)
    return fig
end

function show_unet_results_3d(x, y_gt, pred, lon, lat, disp = true, days = 2:29)
    p1 = heatmap(vec(lon), vec(lat), cpu(dropdims(sum(x, dims=(3,4,5)), dims=(3, 4, 5))), aspect_ratio = 1, xticks = vec(lon)[1:2:end], yticks = vec(lat)[1:2:end], 
    title = "Source term summed over time, \n altitude and particle size \n Emission in GBq", xlabel = "Longitude", ylabel  = "Latitude")
    p2 = barplot(dropdims(sum(x,dims=(1,2,4,5)), dims=(1,2,4,5)))
    if sum(isnan.(x)) == 0
        mn  = log10(min(minimum(y_gt[y_gt.>0]), minimum(pred[pred.>0])))
        mx  = log10(max(maximum(y_gt[y_gt.>0]), maximum(pred[pred.>0])))
        p4 = plot([mn, mx], [mn, mx], label = "")
        plot!(log10.(cpu(y_gt)), log10.(cpu(pred)), ylabel = "Prediction", xlabel = "Observation", title = "Correspondence with data",
            seriestype = :scatter, label = "",  margin = 5mm, xlim = (-2, 3), ylim = (-2, 3), color = :chartreuse4)
        xticks!(-2:3, ["0.01", "0.1", "1.0", "10", "100", "1000"])
        yticks!(-2:3, ["0.01", "0.1", "1.0", "10", "100", "1000"])
    else
        p4 = plot(title = "NaN")
    end
    l = @layout [a ; b]
    pp = plot(p2, p4, layout = l, margin = 5mm, size = (1200, 600))
    p = plot(p1, pp, margin = 5mm, size = (1200, 600))
    if disp
        display(p)
    end
    return p
end

