module RCDiagnostics

using DataFrames, LinearAlgebra, Statistics, LaTeXStrings
import ..RCParams
import ..RCIO
import ..RCBetaEstimators
import ..RCOmegaEstimators
import ..RCPlotting

export smoke_diagnosis!, find_outlier_estimates


"""
smoke_diagnosis!(; params)

Loads the smoke-test dataset (suffix "_ST"), builds y, sorts rows to [:t,:j,:i],
then:
  • constructs FGLS1 Ω̂ (no GLS run), plots heatmap + prints eigen tables
  • constructs FGLS2 Ω̂ (no GLS run), plots heatmap + prints eigen tables
  • constructs Oracle Ω★ from true small blocks, plots heatmap + prints eigen tables

Returns NamedTuple (; Ωhat = Ω̂₁, Ωhat2 = Ω̂₂, Ωstar, sizes, path).
"""
function smoke_diagnosis!(; params::NamedTuple)
    p = params

    # --- Load smoke-test dataset ---
    if !RCIO.dataset_exists(p; suffix="_ST")
        error("No smoke-test file found for current PARAMS at $(RCIO.output_path(p; suffix="_ST")). Run smoke test first.")
    end
    st = RCIO.load_dataset(p; suffix="_ST")
    df = st.df
    Ωi_true, Ωj_true, Ωt_true = st.Ωi, st.Ωj, st.Ωt
    N1, N2, T = p.N1, st.sizes.N2, st.sizes.T

    # --- Build y, correct ordering (i fastest) ---
    RCBetaEstimators.add_y!(df; beta=p.beta_true, constant=p.c_true, x_col=:x)
    sort!(df, [:t, :j, :i])

    # --- FGLS1 Ω̂₁ (no GLS run) ---
    _, _, _, Ω̂ = RCBetaEstimators.fgls1(
        df, N1, N2, T;
        x_col=:x, y_col=:y,
        i_block_est = p.i_block_est,
        j_block_est = p.j_block_est,
        t_block_est = p.t_block_est,
        repeat_alpha  = p.repeat_alpha_fgls,
        repeat_gamma  = p.repeat_gamma_fgls,
        repeat_lambda = p.repeat_lambda_fgls,
        subtract_sigma_u2_fgls1 = p.subtract_sigma_u2_fgls1,
        run_gls   = false,
        shrinkage = p.fgls_shrinkage,
        project_spd = (:fgls_project_spd ∈ propertynames(p) ? p.fgls_project_spd : false),
        spd_floor   = (:fgls_spd_floor   ∈ propertynames(p) ? p.fgls_spd_floor   : 1e-8)
    )

    do_plots = get(p, :smoke_plot_omega_heatmaps, true)
    if do_plots
        mode_str = "blocks(i=$(p.i_block_est), j=$(p.j_block_est), t=$(p.t_block_est))"
        ttl1 = latexstring("\$\\mathrm{FGLS1}\\ \\hat{\\Omega}_{\\mathrm{$mode_str}}\\ \\mid\\ \\mathrm{rep}\\ \\alpha=$(p.repeat_alpha_fgls),\\ \\gamma=$(p.repeat_gamma_fgls),\\ \\lambda=$(p.repeat_lambda_fgls)\$")
        display(RCPlotting.plot_matrix_percentile(Ω̂; title=ttl1))
    end
    sum1, ext1 = RCPlotting.omega_eigen_tables(Ω̂; k=10)
    println("\n[FGLS1] eigenvalue summary:"); println(sum1)
    println("\n[FGLS1] smallest/largest:");   println(ext1)

    # --- FGLS2 Ω̂₂ (no GLS run) ---
    _, _, _, Ω̂2 = RCBetaEstimators.fgls2(
        df, N1, N2, T;
        x_col=:x, y_col=:y,
        i_block_est = p.i_block_est,
        j_block_est = p.j_block_est,
        t_block_est = p.t_block_est,
        repeat_alpha  = p.repeat_alpha_fgls2,
        repeat_gamma  = p.repeat_gamma_fgls2,
        repeat_lambda = p.repeat_lambda_fgls2,
        subtract_sigma_u2_fgls2 = p.subtract_sigma_u2_fgls2,
        run_gls   = false,
        shrinkage = p.fgls_shrinkage,
        project_spd = (:fgls_project_spd ∈ propertynames(p) ? p.fgls_project_spd : false),
        spd_floor   = (:fgls_spd_floor   ∈ propertynames(p) ? p.fgls_spd_floor   : 1e-8),
        # --- debug only ---
        debug=p.smoke_test_debug_print,
        debug_truth=(; Ωi_star=Ωi_true, Ωj_star=Ωj_true, Ωt_star=Ωt_true),
        debug_digits=3
        )

    if do_plots
        mode_str2 = "blocks(i=$(p.i_block_est), j=$(p.j_block_est), t=$(p.t_block_est))"
        ttl2f = latexstring("\$\\mathrm{FGLS2}\\ \\hat{\\Omega}_{\\mathrm{$mode_str2}}\\ \\mid\\ \\mathrm{rep}\\ \\alpha=$(p.repeat_alpha_fgls2),\\ \\gamma=$(p.repeat_gamma_fgls2),\\ \\lambda=$(p.repeat_lambda_fgls2)\$")
        display(RCPlotting.plot_matrix_percentile(Ω̂2; title=ttl2f))
    end
    sum2f, ext2f = RCPlotting.omega_eigen_tables(Ω̂2; k=10)
    println("\n[FGLS2] eigenvalue summary:"); println(sum2f)
    println("\n[FGLS2] smallest/largest:");   println(ext2f)

    # --- Oracle GLS Ω★ from true blocks (with repeats) ---
    Sα, Sγ, Sλ = RCOmegaEstimators.make_S_matrices(
        N1, N2, T;
        repeat_alpha = p.repeat_alpha_gls,
        repeat_gamma = p.repeat_gamma_gls,
        repeat_lambda= p.repeat_lambda_gls
    )
    Ωa, Ωg, Ωl = Ωi_true, Ωj_true, Ωt_true
    if p.repeat_alpha_gls; Ωa = RCOmegaEstimators.repeat_block(Ωa, T); end
    if p.repeat_gamma_gls; Ωg = RCOmegaEstimators.repeat_block(Ωg, T);  end
    if p.repeat_lambda_gls; Ωl = RCOmegaEstimators.repeat_block(Ωl, N2); end

    Ωstar = RCOmegaEstimators.construct_omega(Ωa, Ωg, Ωl, Sα, Sγ, Sλ, p.sigma_u^2)
    if (:gls_shrinkage ∈ propertynames(p)) && p.gls_shrinkage != 1.0
        Ωstar = RCOmegaEstimators.shrink_offdiagonal!(Ωstar, p.gls_shrinkage)
    end
    if (:gls_project_spd ∈ propertynames(p)) && p.gls_project_spd
        Ωstar = RCOmegaEstimators.project_to_spd(Ωstar; floor=(
            :gls_spd_floor ∈ propertynames(p) ? p.gls_spd_floor : 1e-8
        ))
    end

    if do_plots
        ttl3 = latexstring("\$\\mathrm{Oracle}\\ \\Omega^{\\star}\\ \\mid\\ \\mathrm{rep}\\ \\alpha=$(p.repeat_alpha_gls),\\ \\gamma=$(p.repeat_gamma_gls),\\ \\lambda=$(p.repeat_lambda_gls)\$")
        display(RCPlotting.plot_matrix_percentile(Ωstar; title=ttl3))
    end
    sum3, ext3 = RCPlotting.omega_eigen_tables(Ωstar; k=10)
    println("\n[GLS] eigenvalue summary:"); println(sum3)
    println("\n[GLS] smallest/largest:");   println(ext3)

    return (; Ωhat = Ω̂, Ωhat2 = Ω̂2, Ωstar, sizes = st.sizes, path = st.path)
end

"""
find_outlier_estimates(est_res, size_index, estimator; cutoff, direction=:above)

Return Vector{Tuple{Int,Float64}} of (rep_index, value) for estimates that
are > cutoff (direction=:above) or < cutoff (direction=:below) at the given size.
`est_res` is the Vector returned by mc_estimate_over_sizes.
Valid estimators: "OLS", "OLS FE", "FGLS1", "FGLS2", "GLS".
"""
function find_outlier_estimates(est_res::Vector, size_index::Integer, estimator::AbstractString;
                                cutoff::Real, direction::Symbol=:above)
    @assert 1 ≤ size_index ≤ length(est_res) "size_index out of range"
    row = est_res[size_index]

    est_map = Dict(
        "OLS"    => row.β_ols,
        "OLS FE" => row.β_fe,
        "FGLS1"  => row.β_fgls1,
        "FGLS2"  => row.β_fgls2,
        "GLS"    => row.β_gls,
    )
    @assert haskey(est_map, estimator) "Unknown estimator: $estimator"

    vals = est_map[estimator]
    data = collect(skipmissing(vals))

    if direction === :above
        return [(i, v) for (i, v) in enumerate(data) if v > cutoff]
    elseif direction === :below
        return [(i, v) for (i, v) in enumerate(data) if v < cutoff]
    else
        error("direction must be :above or :below")
    end
end

end # module