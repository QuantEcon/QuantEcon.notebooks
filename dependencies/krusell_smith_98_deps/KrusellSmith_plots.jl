using PlotlyJS

function plot_2dobj(gridk::Vector{Float64}, obj::Matrix{Float64}, objname::AbstractString, yaxis::AbstractString)

    pl_full = plot([scatter(; x = gridk, y = obj[:, 1], name = "Unemployed", line_color = "blue");
            scatter(; x = gridk, y = obj[:, 2], name = "Employed", line_color = "orange")],
            Layout(title = "$(objname)", xaxis_title = "Current capital", yaxis_title = "$(yaxis)"))
    pl_det = plot([scatter(; x = gridk[1:35], y = (obj[:, 1])[1:35], showlegend = false, line_color = "blue");
            scatter(; x = gridk[1:35], y = (obj[:, 2])[1:35], showlegend = false, line_color = "orange")],
            Layout(title = "$(objname), Detail", xaxis_title = "Current capital", yaxis_title = "$(yaxis)"))

    [pl_full; pl_det]
end

function plot_measure(m::KrusellSmith, a::Approx, μ::Matrix{Float64})
    ϵl, ϵh = m.ϵl, m.ϵh
    gridkf = a.gridkf

    pl_μk = plot(scatter(; x = gridkf, y = vec(sum(μ, 2)), line_color = "blue", showlegend = false),
            Layout(title = "Marginal Stationary Density, Capital Holding", xaxis_title = "Capital holding", yaxis_title = "Density"))
    pl_μϵ = plot(scatter(; x = [ϵl; ϵh], y = vec(sum(μ, 1)), line_color = "blue", mode = "markers", marker_size = 12, showlegend = false),
            Layout(title = "Marginal Stationary Density, Employment State", xaxis_title = "Employment state", yaxis_title = "Probability"))

    [pl_μk; pl_μϵ]
end

function plot_3dobj(a::Approx, obj::Array{Float64, 4}, objname::AbstractString, yaxis::AbstractString)
    gridk = a.gridk
    obj = squeeze(obj, 3)

    pl_full = plot([scatter(; x = gridk, y = obj[:, 1, 1], name = "Unempl., z = bad", line_color = "blue");
            scatter(; x = gridk, y = obj[:, 2, 1], name = "Empl., z = bad", line_color = "orange");
            scatter(; x = gridk, y = obj[:, 1, 2], name = "Unempl., z = good", line_color = "green");
            scatter(; x = gridk, y = obj[:, 2, 2], name = "Empl., z = good", line_color = "red")],
            Layout(title = "$(objname)", xaxis_title = "Current capital", yaxis_title = "$(yaxis)"))
    pl_det = plot([scatter(; x = gridk[1:35], y = (obj[:, 1, 1])[1:35], showlegend = false, line_color = "blue");
            scatter(; x = gridk[1:35], y = (obj[:, 2, 1])[1:35], showlegend = false, line_color = "orange");
            scatter(; x = gridk[1:35], y = (obj[:, 1, 2])[1:35], showlegend = false, line_color = "green");
            scatter(; x = gridk[1:35], y = (obj[:, 2, 2])[1:35], showlegend = false, line_color = "red")],
            Layout(title = "$(objname), Detail", xaxis_title = "Current capital", yaxis_title = "$(yaxis)"))

    [pl_full; pl_det]
end

function plot_lorenz(pop_frac::Vector{Float64}, lorenz::Vector{Float64})
    plot([scatter(; x = pop_frac, y = lorenz, line_color = "blue", name = "Lorenz Curve"),
        scatter(; x = pop_frac, y = pop_frac, line_color = "black", mode = "lines", name = "45° Line")],
    Layout(title = "Lorenz Curve for Wealth Holdings", xaxis_title = "Fraction of population",
    yaxis_title = "Fraction of wealth"))
end
