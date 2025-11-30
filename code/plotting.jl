# plotting.jl
module RCPlotting

using LinearAlgebra, Statistics, DataFrames, Plots


import ..RCIO

export plot_matrix_percentile, omega_eigen_tables, 
       block_eigen_summary, block_eigen_table,
       apply_plot_theme!, sample_sizes_from_results,
       plot_estimator_distribution, plot_asymptotic, plot_multi_variance, plot_variance_ratio,
       savefig_distribution!,savefig_heatmap!, savefig_asymptotic!, savefig_multi_variance!, savefig_variance_ratio!,
       make_result_plots

"Percentile heatmap of a matrix (Spectral colormap, no ticks)."
function plot_matrix_percentile(Ω::AbstractMatrix; title::AbstractString="Matrix Percentile Heatmap")
    A = Array(Ω)
    flat = vec(A)

    # 0:100 percentiles (101 points)
    qs = quantile(flat, 0:0.01:1.0)

    # Map entries to percentile ranks
    ranks = similar(A, Float64)
    @inbounds for idx in eachindex(A)
        ranks[idx] = searchsortedfirst(qs, A[idx]) / length(qs)
    end

    plt = heatmap(
        ranks;
        c = :Spectral,
        colorbar_title = "Percentile Rank",
        title = title,
        xticks = false,
        yticks = false,
        yflip = true,
        size = (900, 700)
    )
    return plt
end

"Summary stats for eigenvalues of Ωα, Ωγ, Ωλ."
function block_eigen_summary(Ωa::AbstractMatrix, Ωg::AbstractMatrix, Ωl::AbstractMatrix)
    ea = eigvals(Symmetric(Array(Ωa)))
    eg = eigvals(Symmetric(Array(Ωg)))
    el = eigvals(Symmetric(Array(Ωl)))
    make(block, v) = (; block,
        n = length(v),
        min = minimum(v),
        max = maximum(v),
        negatives = count(<(0), v),
        mean = mean(v),
        median = median(v))
    rows = [ make("alpha", ea), make("gamma", eg), make("lambda", el) ]
    return DataFrame(rows)
end
"""
Eigenvalue tables for Ω:
- `summary`: n, min, max, #negatives, mean, median
- `extrema`: smallest/largest k eigenvalues (default 10)
"""
function omega_eigen_tables(Ω::AbstractMatrix; k::Int=10)
    vals = eigvals(Symmetric(Array(Ω)))
    n = length(vals)
    sorted = sort(vals)
    k = min(k, n)

    summary = DataFrame(
        stat  = ["n", "min", "max", "negatives", "mean", "median"],
        value = [n,
                 minimum(vals),
                 maximum(vals),
                 count(<(0), vals),
                 mean(vals),
                 median(vals)]
    )

    extrema = DataFrame(
        smallest = sorted[1:k],
        largest  = sorted[end-k+1:end]
    )

    return summary, extrema
end

"Full list of eigenvalues in a single table (block, idx, eig)."
function block_eigen_table(Ωa::AbstractMatrix, Ωg::AbstractMatrix, Ωl::AbstractMatrix)
    ea = eigvals(Symmetric(Array(Ωa)))
    eg = eigvals(Symmetric(Array(Ωg)))
    el = eigvals(Symmetric(Array(Ωl)))
    dfa = DataFrame(block = fill("alpha", length(ea)), idx = eachindex(ea), eig = ea)
    dfg = DataFrame(block = fill("gamma", length(eg)), idx = eachindex(eg), eig = eg)
    dfl = DataFrame(block = fill("lambda", length(el)), idx = eachindex(el), eig = el)
    return vcat(dfa, dfg, dfl)
end

# ---- Theme ----
"Apply a plotting theme; e.g. :ggplot2, :default, :gruvboxdark, etc."
function apply_plot_theme!(theme_sym::Union{Nothing,Symbol})
    if theme_sym === nothing
        return nothing
    end
    try
        theme(theme_sym)
    catch
        @warn "Unknown theme $(theme_sym); keeping current theme."
    end
    return nothing
end

# ---- Helpers over the mc_estimate_over_sizes() structure ----
"Return vector of sample sizes n = N1*N2*T from `est_res`."
function sample_sizes_from_results(est_res::Vector)
    return [r.size.N1 * r.size.N2 * r.size.T for r in est_res]
end

"Get estimates vector (per rep) by estimator + size index. Returns (vec, n)."
function select_estimates(est_res::Vector, size_index::Integer, estimator::AbstractString)
    @assert 1 ≤ size_index ≤ length(est_res) "size_index out of range"
    r = est_res[size_index]
    n = r.size.N1 * r.size.N2 * r.size.T
    est = lowercase(estimator)
    if est in ["ols"]
        return (r.β_ols, n)
    elseif est in ["ols fe", "fe", "fe-ols", "fe_ols"]
        return (r.β_fe, n)
    elseif est in ["fgls1"]
        return (r.β_fgls1, n)
    elseif est in ["fgls2"]
        return (r.β_fgls2, n)
    elseif est in ["gls"]
        return (r.β_gls, n)
    else
        error("Unknown estimator: $estimizer")
    end
end

"Get estimated-variance vectors (per rep) by estimator + size index."
function select_variances(est_res::Vector, size_index::Integer, estimator::AbstractString)
    @assert 1 ≤ size_index ≤ length(est_res) "size_index out of range"
    r = est_res[size_index]
    est = lowercase(estimator)
    if est in ["ols"]
        return r.v_ols
    elseif est in ["ols fe", "fe", "fe-ols", "fe_ols"]
        return r.v_fe
    elseif est in ["fgls1"]
        return r.v_fgls1
    elseif est in ["fgls2"]
        return r.v_fgls2
    elseif est in ["gls"]
        return r.v_gls
    else
        error("Unknown estimator: $estimator")
    end
end

# ---- Simple KDE for overlay (no extra deps) ----
"Return (grid, density) using a Gaussian KDE (Silverman's bandwidth)."
function _kde_gaussian(x::AbstractVector{<:Real}; points::Int=200)
    x = collect(skipmissing(x))
    n = length(x)
    if n < 2
        return (range(minimum(x), stop=maximum(x), length=2), [1.0, 1.0])
    end
    s = std(x)
    h = 1.06 * s * n^(-1/5)
    h = h > 0 ? h : 1e-6
    xmin, xmax = minimum(x), maximum(x)
    pad = 0.1 * (xmax - xmin + eps())
    grid = range(xmin - pad, xmax + pad, length=points)
    dens = similar(collect(grid), Float64)
    invh = 1 / h
    const_norm = invh / sqrt(2π)
    for (j, g) in enumerate(grid)
        z = (g .- x) .* invh
        dens[j] = const_norm * mean(exp.(-0.5 .* (z .^ 2)))
    end
    return (grid, dens)
end

# ---- Plots (creation only; saving is separate) ----
"""
plot_estimator_distribution(est_res; size_index, estimator, true_beta=missing,
                            bins=30, kde=true, hist=true)

Returns a Plots.Plot object. Select size by position (e.g. params.smoke_test_size).
"""
function plot_estimator_distribution(est_res::Vector;
    size_index::Integer,
    estimator::AbstractString,
    true_beta = missing,
    bins::Int=30,
    kde::Bool=true,
    hist::Bool=true
)
    vals, n = select_estimates(est_res, size_index, estimator)
    data = collect(skipmissing(vals))

    plt = plot()  # empty

    if hist
        histogram!(plt, data; bins=bins, normalize=:pdf, label="hist",
                   linealpha=0.8, fillalpha=0.4, legend=:topright)
    end
    if kde && length(data) ≥ 2
        xx, yy = _kde_gaussian(data)
        plot!(plt, xx, yy; label="kde", lw=2)
    end
    if !(true_beta === missing)
        vline!(plt, [true_beta]; ls=:dash, color=:red, label="true β")
    end

    xlabel!(plt, "Estimate")
    ylabel!(plt, "Density")
    title!(plt, "Distribution: $(estimator)  (n=$(n), idx=$(size_index))")
    plot(plt; grid=true)
    return plt
end

"""
plot_asymptotic(est_res; choice=:bias, estimators, true_beta=0,
                size_range=nothing)

choice ∈ (:bias, :variance). `size_range` may be a UnitRange of indices to keep.
"""
function plot_asymptotic(est_res::Vector;
    choice::Symbol=:bias,
    estimators::Vector{<:AbstractString}=String["OLS","OLS FE","FGLS1","FGLS2","GLS"],
    true_beta::Real=0,
    size_range::Union{Nothing,UnitRange{Int}}=nothing,
    fig_size=(1280,720)
)
    ns_all = sample_sizes_from_results(est_res)
    idxs = size_range === nothing ? eachindex(est_res) : size_range
    ns = ns_all[idxs]

    plt = plot(; legend=:topleft, size=fig_size)
    for est in estimators
        vals = Float64[]
        for i in idxs
            x, _ = select_estimates(est_res, i, est)
            data = collect(skipmissing(x))
            if isempty(data)
                push!(vals, NaN)
            else
                if choice === :bias
                    push!(vals, mean(data .- true_beta))
                elseif choice === :variance
                    push!(vals, var(data; corrected=true))
                else
                    error("choice must be :bias or :variance")
                end
            end
        end
        plot!(plt, ns, vals; label=est, lw=2)
    end
    xlabel!(plt, "n")
    ylabel!(plt, choice === :bias ? "Bias" : "Variance")
    title!(plt, "Asymptotic $(choice === :bias ? "Bias" : "Variance")")
    plot!(plt; grid=true, legend=:best)
    return plt
end

"""
plot_multi_variance(est_res; estimators, size_range=nothing, log_scale=false)

2×2 grid: per estimator plot empirical variance vs mean estimated variance with 95% CI.
"""
function plot_multi_variance(est_res::Vector;
    estimators::Vector{<:AbstractString}=String["OLS FE","FGLS1","FGLS2", "GLS"],
    size_range::Union{Nothing,UnitRange{Int}}=nothing,
    log_scale::Bool=false,
    fig_size=(1280, 1280)
)
    @assert length(estimators) == 4 "Provide exactly four estimators."
    ns_all = sample_sizes_from_results(est_res)
    idxs = size_range === nothing ? collect(eachindex(est_res)) : collect(size_range)
    ns = ns_all[idxs]

    lay = @layout [a b; c d]
    suptitle = "Asymptotic Variance" * (log_scale ? " (log scale)" : "")
    plt = plot(; layout=lay, size=fig_size, plot_title=suptitle)

    for (k, est) in enumerate(estimators)
        emp = Float64[]      # empirical variances
        meanv = Float64[]    # mean estimated variance
        low = Float64[]      # 2.5% quantile
        high = Float64[]     # 97.5% quantile

        for i in idxs
            x, _ = select_estimates(est_res, i, est)
            v    = select_variances(est_res, i, est)
            xdat = collect(skipmissing(x))
            vdat = collect(skipmissing(v))

            push!(emp, isempty(xdat) ? NaN : var(xdat; corrected=true))
            if isempty(vdat)
                push!(meanv, NaN); push!(low, NaN); push!(high, NaN)
            else
                push!(meanv, mean(vdat))
                sort!(vdat)
                lo = vdat[max(1, round(Int, 0.025*length(vdat)))]
                hi = vdat[min(length(vdat), round(Int, 0.975*length(vdat)))]
                push!(low, lo); push!(high, hi)
            end
        end

        if log_scale
            emp   = log.(emp)
            meanv = log.(meanv)
            low   = log.(low)
            high  = log.(high)
        end

        y = meanv
        loff = y .- low
        uoff = high .- y

        plot!(plt[k], ns, emp; label="Empirical", ls=:dash, lw=2)
        plot!(plt[k], ns, y;   label="Estimated", lw=2, fillalpha=0.2,
               ribbon=(loff, uoff))
        xlabel!(plt[k], "n")
        ylabel!(plt[k], log_scale ? "log Variance" : "Variance")
        title!(plt[k], est)
        plot!(plt[k]; grid=true, legend=:best)
    end

    return plt
end

"""
plot_variance_ratio(est_res; estimators, size_range=nothing)

Plots mean(estimated variance) / empirical variance vs n.
"""
function plot_variance_ratio(est_res::Vector;
    estimators::Vector{<:AbstractString}=String["OLS","OLS FE","FGLS1","FGLS2","GLS"],
    size_range::Union{Nothing,UnitRange{Int}}=nothing,
    fig_size=(1280, 720)
)
    ns_all = sample_sizes_from_results(est_res)
    idxs = size_range === nothing ? eachindex(est_res) : size_range
    ns = ns_all[idxs]

    plt = plot(; legend=:best, size=fig_size)
    for est in estimators
        ratios = Float64[]
        for i in idxs
            x, _ = select_estimates(est_res, i, est)
            v    = select_variances(est_res, i, est)
            xdat = collect(skipmissing(x))
            vdat = collect(skipmissing(v))
            empv = isempty(xdat) ? NaN : var(xdat; corrected=true)
            meanv = isempty(vdat) ? NaN : mean(vdat)
            push!(ratios, meanv / max(empv, eps()))
        end
        plot!(plt, ns, ratios; label="ratio $(est)", lw=2)
    end
    xlabel!(plt, "n")
    ylabel!(plt, "Mean est. var / Empirical var")
    title!(plt, "Variance Ratios")
    plot!(plt; grid=true)
    return plt
end

# ---- Saving wrappers (use RCIO naming) ----
savefig_distribution!(plt, params; estimator::AbstractString, sample_n::Integer) =
    savefig(plt, RCIO.plot_path_estimator_dist(params; estimator=estimator, sample_n=sample_n))

savefig_heatmap!(plt, params; estimator::AbstractString) =
    savefig(plt, RCIO.plot_path_omega_heatmap(params; estimator=estimator))

savefig_asymptotic!(plt, params; choice::Symbol) =
    savefig(plt, RCIO.plot_path_asymptotic(params; choice=choice))

savefig_multi_variance!(plt, params) =
    savefig(plt, RCIO.plot_path_multi_variance(params))

savefig_variance_ratio!(plt, params) =
    savefig(plt, RCIO.plot_path_variance_ratio(params))

# ---- Orchestrator ------------------------------------------------------------

"""
make_result_plots(; params, est_res=nothing, save=true)

- If `est_res === nothing`, loads the compact estimation results from disk
  using `RCIO.load_estimation_results(params)`.
- Applies theme from `params.plot_theme` (default :ggplot2).
- Creates/saves plots per the switches in `params`:
    - plot_dist_estimators::Vector{String} (default ["FGLS1","FGLS2"])
    - smoke_test_size::Int (which size index to use for the distribution plot)
    - make_asym_bias_plot::Bool (default true)
    - make_asym_variance_plot::Bool (default true)
    - make_multi_variance_plot::Bool (default true)
    - make_variance_ratio_plot::Bool (default true)
  Optional range by sample size via:
    - asym_min_n, asym_max_n (if present in params)

Saves using RCIO path helpers when `save=true`, otherwise displays.
"""
function make_result_plots(; params::NamedTuple,
                            est_res::Union{Nothing,Vector}=nothing,
                            save::Bool=true,
                            show::Bool=true)

    p = params

    # Theme (if param absent, keep default)
    apply_plot_theme!(get(p, :plot_theme, nothing))

    # Prefer param-driven "show" if present
    show = get(p, :plot_show, show)

    # Load results if not provided
    if est_res === nothing
        loaded = RCIO.load_estimation_results(params)
        est_res = loaded.est_res
    end

    # --- Bounds by *n* for asymptotic plots (optional) ---
    ns_all  = sample_sizes_from_results(est_res)
    min_n   = get(p, :asym_min_n, nothing)
    max_n   = get(p, :asym_max_n, nothing)

    size_range = nothing
    if (min_n !== nothing) || (max_n !== nothing)
        mask = trues(length(ns_all))
        if min_n !== nothing
            mask .&= ns_all .>= min_n
        end
        if max_n !== nothing
            mask .&= ns_all .<= max_n
        end
        idxs = findall(mask)
        size_range = isempty(idxs) ? nothing : (first(idxs):last(idxs))
    end

    # Choices from params (with safe defaults)
    size_idx    = get(p, :plot_dist_size_index, get(p, :smoke_test_size, 1))
    true_beta   = get(p, :beta_true, missing)
    dist_ests   = get(p, :plot_dist_estimators, ["FGLS1","FGLS2"])
    asym_ests   = get(p, :asym_estimators,      ["OLS","OLS FE","FGLS1","FGLS2","GLS"])
    var_ratio_ests = get(p, :var_ratio_estimators, ["OLS","OLS FE","FGLS1","GLS"])
    log_varplot = get(p, :plot_log_variance, false)

    # 1) Distribution plots for the chosen size index (toggleable)
    if get(p, :make_dist_plots, true)
        for est in dist_ests
            plt = plot_estimator_distribution(est_res;
                                              size_index=size_idx,
                                              estimator=est,
                                              true_beta=true_beta,
                                              bins=30, kde=true, hist=true)
            if save
                ns = sample_sizes_from_results(est_res); n = ns[size_idx]
                savefig_distribution!(plt, params; estimator=est, sample_n=n)
            end
            if show
                display(plt)
            end
        end
    end

    # 2) Asymptotic bias
    plt_bias = plot_asymptotic(est_res; choice=:bias,
                            estimators=asym_ests,
                            true_beta=true_beta,
                            size_range=size_range);
    if save; savefig_asymptotic!(plt_bias, params; choice=:bias); end
    if show; display(plt_bias); end

    # 3) Asymptotic variance
    plt_var = plot_asymptotic(est_res; choice=:variance,
                              estimators=asym_ests,
                              true_beta=true_beta,
                              size_range=size_range);
    if save; savefig_asymptotic!(plt_var, params; choice=:variance); end
    if show; display(plt_var); end

    # 4) 2×2 multi-variance panel
    @assert length(var_ratio_ests) == 4 "var_ratio_estimators must have four entries"
    plt_mv = plot_multi_variance(est_res; estimators=var_ratio_ests,
                                size_range=size_range,
                                log_scale=log_varplot);
    if save; savefig_multi_variance!(plt_mv, params); end
    if show; display(plt_mv); end

    # 5) Variance ratio
    plt_vr = plot_variance_ratio(est_res; estimators=asym_ests,
                                 size_range=size_range);
    if save; savefig_variance_ratio!(plt_vr, params); end
    if show; display(plt_vr); end

    return nothing
end

end # module
