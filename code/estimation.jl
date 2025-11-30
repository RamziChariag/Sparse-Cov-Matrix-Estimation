module RCEstimation

using DataFrames, LinearAlgebra, Statistics, ProgressMeter, Random
using Printf: @printf

import ..RCDGP
import ..RCBetaEstimators
import ..RCOmegaEstimators

export estimate_all, print_estimate_summary, mc_estimate_over_sizes

"""
    estimate_all(df; kwargs...) -> NamedTuple

Runs OLS, FE-OLS, FGLS1, FGLS2, and (optionally) Oracle GLS on `df`.

Required sizes:
- N1::Int, N2::Int, T::Int

Estimator controls (pull these from params.jl):
- beta_true, c_true
- vcov_ols::Symbol, vcov_fe::Symbol
- cluster_col_ols::Union{Nothing,Symbol} = nothing
- cluster_col_fe::Union{Nothing,Symbol}  = nothing

FGLS controls (per-dimension Ω choice for estimation):
- i_block_est::Bool=true   # true => full SPD for i; false => homoskedastic diagonal
- j_block_est::Bool=false
- t_block_est::Bool=false
- rep_a_fgls::Bool=false, rep_g_fgls::Bool=false, rep_l_fgls::Bool=false
- rep_a_fgls2::Bool=false, rep_g_fgls2::Bool=false, rep_l_fgls2::Bool=false
- fgls_shrinkage::Real=1.0
- fgls_project_spd::Bool=false, fgls_spd_floor::Real=1e-8

Oracle GLS (optional): provide Ωi_star, Ωj_star, Ωt_star to enable
- rep_a_gls::Bool=true, rep_g_gls::Bool=false, rep_l_gls::Bool=false
- gls_shrinkage::Real=1.0
- gls_project_spd::Bool=false, gls_spd_floor::Real=1e-8
- sigma_u2_oracle::Real = 1.0

Other:
- sort_for_gls::Bool=true   # sorts as [:t,:j,:i] so that i is fastest to match S

Returns NamedTuple with:
  β_ols, V_ols, se_ols  (β_ols is length-2: intercept, slope)
  β_fe,  V_fe,  se_fe   (single coefficient)
  β_fgls1, V_fgls1, se_fgls1, Ωhat1
  β_fgls2, V_fgls2, se_fgls2, Ωhat2
  β_gls,  V_gls,  se_gls,  Ωstar   (only if true blocks are provided)
"""
function estimate_all(df::DataFrame;
    # sizes
    N1::Int, N2::Int, T::Int,

    # truth & variance choices
    beta_true::Real, c_true::Real,
    vcov_ols::Symbol, vcov_fe::Symbol,
    cluster_col_ols::Union{Nothing,Symbol}=nothing,
    cluster_col_fe::Union{Nothing,Symbol}=nothing,

    # FGLS (estimation-side Ω choices)
    i_block_est::Bool=true, j_block_est::Bool=false, t_block_est::Bool=false,
    rep_a_fgls::Bool=false, rep_g_fgls::Bool=false, rep_l_fgls::Bool=false,
    rep_a_fgls2::Bool=false, rep_g_fgls2::Bool=false, rep_l_fgls2::Bool=false,
    subtract_sigma_u2_fgls1::Bool=false, subtract_sigma_u2_fgls2::Bool=false,
    iterate_fgls2::Bool=false,
    fgls_shrinkage::Real=1.0, fgls_project_spd::Bool=false, fgls_spd_floor::Real=1e-8,

    # Oracle GLS (pass true blocks to enable)
    Ωi_star::Union{Nothing,AbstractMatrix}=nothing,
    Ωj_star::Union{Nothing,AbstractMatrix}=nothing,
    Ωt_star::Union{Nothing,AbstractMatrix}=nothing,
    rep_a_gls::Bool=true, rep_g_gls::Bool=false, rep_l_gls::Bool=false,
    gls_shrinkage::Real=1.0, gls_project_spd::Bool=false, gls_spd_floor::Real=1e-8,
    sigma_u2_oracle::Real=1.0,
)
    # 0) Build y (idempotent)
    RCBetaEstimators.add_y!(df; beta=beta_true, constant=c_true, x_col=:x)

    # 1) OLS (raw, with intercept)
    β_ols, e_ols, V_ols = RCBetaEstimators.ols(df; x_col=:x, y_col=:y,
                                              vcov=vcov_ols, cluster_col=cluster_col_ols)
    se_ols = sqrt.(max.(diag(V_ols), 0))  # guard print

    # 2) FE-OLS (within i,j,t; no intercept)
    β_fe, e_fe, V_fe = RCBetaEstimators.fe_ols(df; x_col=:x, y_col=:y,
                                               group_vars=[:i,:j,:t],
                                               vcov=vcov_fe, cluster_col=cluster_col_fe)
    se_fe = sqrt.(max.(diag(V_fe), 0))

    # Prepare a view with correct row order for (F)GLS (i fastest ⇒ [:t,:j,:i])
    df_gls = sort(df, [:t,:j,:i])

    # 3a) FGLS1
    β_fgls1 = nothing; e_fgls1 = nothing; V_fgls1 = nothing; Ωhat1 = nothing; se_fgls1 = nothing
    try
        β_fgls1, e_fgls1, V_fgls1, Ωhat1 = RCBetaEstimators.fgls1(
            df_gls, N1, N2, T;
            x_col=:x, y_col=:y,
            i_block_est = i_block_est,
            j_block_est = j_block_est,
            t_block_est = t_block_est,
            repeat_alpha  = rep_a_fgls,
            repeat_gamma  = rep_g_fgls,
            repeat_lambda = rep_l_fgls,
            subtract_sigma_u2_fgls1 = subtract_sigma_u2_fgls1,
            run_gls       = true,
            shrinkage     = fgls_shrinkage,
            project_spd   = fgls_project_spd,
            spd_floor     = fgls_spd_floor
        )
    catch err
        @warn "FGLS1 failed" exception=(err, catch_backtrace()) n=nrow(df) p=ncol(df) N1=N1 N2=N2 T=T
    end
    if V_fgls1 !== nothing
        se_fgls1 = sqrt.(max.(diag(V_fgls1), 0))
    end

    # 3b) FGLS2
    # 3b) FGLS2 
    β_fgls2 = nothing; e_fgls2 = nothing; V_fgls2 = nothing; Ωhat2 = nothing; se_fgls2 = nothing
    try
        if iterate_fgls2
            β_fgls2, e_fgls2, V_fgls2, Ωhat2 = RCBetaEstimators.ifgls2(
                df_gls, N1, N2, T;
                x_col=:x, y_col=:y,
                i_block_est = i_block_est,
                j_block_est = j_block_est,
                t_block_est = t_block_est,
                repeat_alpha  = rep_a_fgls2,
                repeat_gamma  = rep_g_fgls2,
                repeat_lambda = rep_l_fgls2,
                subtract_sigma_u2_fgls2 = subtract_sigma_u2_fgls2,
                run_gls       = true,
                shrinkage     = fgls_shrinkage,
                project_spd   = fgls_project_spd,
                spd_floor     = fgls_spd_floor
            )
        else
            β_fgls2, e_fgls2, V_fgls2, Ωhat2 = RCBetaEstimators.fgls2(
                df_gls, N1, N2, T;
                x_col=:x, y_col=:y,
                i_block_est = i_block_est,
                j_block_est = j_block_est,
                t_block_est = t_block_est,
                repeat_alpha  = rep_a_fgls2,
                repeat_gamma  = rep_g_fgls2,
                repeat_lambda = rep_l_fgls2,
                subtract_sigma_u2_fgls2 = subtract_sigma_u2_fgls2,
                run_gls       = true,
                shrinkage     = fgls_shrinkage,
                project_spd   = fgls_project_spd,
                spd_floor     = fgls_spd_floor
            )
        end
    catch err
        @warn "FGLS2 failed" exception=(err, catch_backtrace()) n=nrow(df) p=ncol(df) N1=N1 N2=N2 T=T
    end

    # 4) Oracle GLS (if true small blocks provided)
    β_gls = nothing; e_gls = nothing; V_gls = nothing; Ωstar = nothing
    if !(Ωi_star === nothing || Ωj_star === nothing || Ωt_star === nothing)
        try
            β_gls, e_gls, V_gls, Ωstar = RCBetaEstimators.oracle_gls(
                df_gls, Ωi_star, Ωj_star, Ωt_star, N1, N2, T;
                x_col=:x, y_col=:y,
                repeat_alpha = rep_a_gls,
                repeat_gamma = rep_g_gls,
                repeat_lambda= rep_l_gls,
                sigma_u2     = sigma_u2_oracle,
                shrinkage    = gls_shrinkage,
                project_spd  = gls_project_spd,
                spd_floor    = gls_spd_floor
            )
        catch err
            @warn "Oracle GLS failed" exception=(err, catch_backtrace()) n=nrow(df) p=ncol(df) N1=N1 N2=N2 T=T
        end
    end
    se_gls = V_gls === nothing ? nothing : sqrt.(max.(diag(V_gls), 0))

    return (
        # OLS
        β_ols = β_ols, V_ols = V_ols, se_ols = se_ols,
        # FE-OLS
        β_fe  = β_fe,  V_fe  = V_fe,  se_fe  = se_fe,
        # FGLS1
        β_fgls1 = β_fgls1, V_fgls1 = V_fgls1, se_fgls1 = se_fgls1, Ωhat1 = Ωhat1,
        # FGLS2
        β_fgls2 = β_fgls2, V_fgls2 = V_fgls2, se_fgls2 = se_fgls2, Ωhat2 = Ωhat2,
        # Oracle GLS
        β_gls = β_gls, V_gls = V_gls, se_gls = se_gls, Ωstar = Ωstar
    )
end

"Pretty-print a short bias/variance style summary for one dataset."
function print_estimate_summary(res::NamedTuple; beta_true::Real)
    # pull slope components (index 2 for OLS/GLS, 1 for FE/FGLS)
    β̂_ols  = res.β_ols === nothing ? nothing : res.β_ols[2]
    v_ols   = res.V_ols === nothing ? nothing : res.V_ols[2,2]

    β̂_fe   = res.β_fe  === nothing ? nothing : res.β_fe[1]
    v_fe    = res.V_fe  === nothing ? nothing : res.V_fe[1,1]

β̂_fgls1 = res.β_fgls1 === nothing ? nothing : res.β_fgls1[2]
    v_fgls1  = res.V_fgls1 === nothing ? nothing : res.V_fgls1[2,2]

    β̂_fgls2 = res.β_fgls2 === nothing ? nothing : res.β_fgls2[2]
    v_fgls2  = res.V_fgls2 === nothing ? nothing : res.V_fgls2[2,2]


    β̂_gls  = res.β_gls === nothing ? nothing : res.β_gls[2]
    v_gls   = res.V_gls === nothing ? nothing : res.V_gls[2,2]

    fmt(x) = x === nothing ? "—" : string(round(x, digits=6))
    println("\n== Single Estimation summary ==")
    println("Using smoke test data, if it does not exist, using first MC bundle data...")
    println("True β: ", beta_true)
    println("OLS:     β̂ = ", fmt(β̂_ols),  "   Var = ", fmt(v_ols))
    println("FE-OLS:  β̂ = ", fmt(β̂_fe),   "   Var = ", fmt(v_fe))
    println("FGLS1:   β̂ = ", fmt(β̂_fgls1), "   Var = ", fmt(v_fgls1))
    println("FGLS2:   β̂ = ", fmt(β̂_fgls2), "   Var = ", fmt(v_fgls2))
    println("Oracle:  β̂ = ", fmt(β̂_gls),  "   Var = ", fmt(v_gls))
    return nothing
end

""" 
    mc_estimate_over_sizes(; params, reps=params.num_reps,
                           progress_sizes=true, progress_reps=true,
                           print_each=true, bundle=nothing, keep_vectors=false)

Runs Monte Carlo estimation:
- sequential over sample sizes (from `params`)
- parallel over reps (Threads.@threads)

If `keep_vectors=false` (default), the return is compact: `(size, reps, stats)` per sample size.
"""
function mc_estimate_over_sizes(; params::NamedTuple,
                                reps::Int=params.num_reps,
                                progress_sizes::Bool=true,
                                progress_reps::Bool=true,
                                print_each::Bool=true,
                                bundle::Union{Nothing,Vector}=nothing,
                                keep_vectors::Bool=false)

    p = params
    sizes = [(N2 = p.start_N2 + (k-1)*p.N2_increment,
              T  = p.start_T  + (k-1)*p.T_increment) for k in 1:p.num_sample_sizes]

    out = Vector{NamedTuple}(undef, length(sizes))
    outer = progress_sizes ? Progress(length(sizes); dt=0.2, desc="Sample sizes") : nothing

    for (idx, sz) in enumerate(sizes)
        N1 = p.N1; N2 = sz.N2; T = sz.T
        n  = N1 * N2 * T

        # --- Draw Ω once per size and reuse across reps ---
        rng_Ω = MersenneTwister(p.seed + 1_000_000*idx)
        Ωi_fixed, Ωj_fixed, Ωt_fixed = RCDGP.build_cov_mats(
            N1, N2, T;
            i_block=p.i_block, j_block=p.j_block, t_block=p.t_block,
            sigma_i=p.sigma_i, sigma_j=p.sigma_j, sigma_t=p.sigma_t,
            rng=rng_Ω
        )
        # -------------------------------------------------------

        # possibly override reps if a bundle of this size is provided
        local_datasets = bundle === nothing ? nothing : bundle[idx]
        reps_here = bundle === nothing ? reps :
                    (local_datasets isa Vector ? length(local_datasets) : 1)

        β_ols   = Vector{Float64}(undef, reps_here)
        v_ols   = Vector{Float64}(undef, reps_here)
        β_fe    = Vector{Float64}(undef, reps_here)
        v_fe    = Vector{Float64}(undef, reps_here)
        β_fgls1 = Vector{Union{Missing,Float64}}(undef, reps_here)
        v_fgls1 = Vector{Union{Missing,Float64}}(undef, reps_here)
        β_fgls2 = Vector{Union{Missing,Float64}}(undef, reps_here)
        v_fgls2 = Vector{Union{Missing,Float64}}(undef, reps_here)
        β_gls   = Vector{Union{Missing,Float64}}(undef, reps_here)
        v_gls   = Vector{Union{Missing,Float64}}(undef, reps_here)

        inner = progress_reps ? Progress(reps_here; dt=0.1, desc="N2=$N2, T=$T (n=$n)") : nothing
        lk = ReentrantLock()

        Threads.@threads for r in 1:reps_here
            # pick data
            df = nothing; Ωi_true=nothing; Ωj_true=nothing; Ωt_true=nothing
            if bundle === nothing
                seed_r = p.seed + 10_000*idx + r
                # --- changed: pass fixed Ω into generate_dataset ---
                df, meta = RCDGP.generate_dataset(
                    N1=N1, N2=N2, T=T,
                    i_block=p.i_block, j_block=p.j_block, t_block=p.t_block,
                    i_draw_mode=p.i_draw_mode, j_draw_mode=p.j_draw_mode, t_draw_mode=p.t_draw_mode,
                    E_i=p.E_i, E_j=p.E_j, E_t=p.E_t,
                    sigma_i=p.sigma_i, sigma_j=p.sigma_j, sigma_t=p.sigma_t,
                    mu_x=p.mu_x, sigma_x=p.sigma_x,
                    mu_u=p.mu_u, sigma_u=p.sigma_u,
                    seed=seed_r,
                    Ωi_fixed=Ωi_fixed, Ωj_fixed=Ωj_fixed, Ωt_fixed=Ωt_fixed
                )
                Ωi_true, Ωj_true, Ωt_true = Ωi_fixed, Ωj_fixed, Ωt_fixed
            else
                d = local_datasets isa Vector ? local_datasets[r] : local_datasets
                df = d.df; Ωi_true, Ωj_true, Ωt_true = d.Ωi, d.Ωj, d.Ωt
            end

            res = estimate_all(df;
                N1=N1, N2=N2, T=T,
                beta_true=p.beta_true, c_true=p.c_true,
                vcov_ols=p.vcov_ols, vcov_fe=p.vcov_fe,
                cluster_col_ols=p.cluster_col_ols, cluster_col_fe=p.cluster_col_fe,
                # estimation-side Ω toggles
                i_block_est=p.i_block_est, j_block_est=p.j_block_est, t_block_est=p.t_block_est,
                # FGLS controls
                rep_a_fgls=p.repeat_alpha_fgls, rep_g_fgls=p.repeat_gamma_fgls, rep_l_fgls=p.repeat_lambda_fgls,
                rep_a_fgls2=p.repeat_alpha_fgls2, rep_g_fgls2=p.repeat_gamma_fgls2, rep_l_fgls2=p.repeat_lambda_fgls2,
                subtract_sigma_u2_fgls1 = p.subtract_sigma_u2_fgls1,
                subtract_sigma_u2_fgls2 = p.subtract_sigma_u2_fgls2,
                iterate_fgls2 = p.iterate_fgls2,
                # FGLS shrinkage
                fgls_shrinkage=p.fgls_shrinkage,
                fgls_project_spd = (:fgls_project_spd ∈ propertynames(p) ? p.fgls_project_spd : false),
                fgls_spd_floor   = (:fgls_spd_floor   ∈ propertynames(p) ? p.fgls_spd_floor   : 1e-8),
                # oracle
                Ωi_star=Ωi_true, Ωj_star=Ωj_true, Ωt_star=Ωt_true,
                rep_a_gls=p.repeat_alpha_gls, rep_g_gls=p.repeat_gamma_gls, rep_l_gls=p.repeat_lambda_gls,
                gls_shrinkage=p.gls_shrinkage,
                gls_project_spd = (:gls_project_spd ∈ propertynames(p) ? p.gls_project_spd : false),
                gls_spd_floor   = (:gls_spd_floor   ∈ propertynames(p) ? p.gls_spd_floor   : 1e-8),
                sigma_u2_oracle=p.sigma_u^2
            )

            β_ols[r] = res.β_ols[2];  v_ols[r] = res.V_ols[2,2]
            β_fe[r]  = res.β_fe[1];   v_fe[r]  = res.V_fe[1,1]

                        if res.β_fgls1 === nothing
                β_fgls1[r] = missing; v_fgls1[r] = missing
            else
                β_fgls1[r] = res.β_fgls1[2]; v_fgls1[r] = res.V_fgls1[2,2]
            end

            if res.β_fgls2 === nothing
                β_fgls2[r] = missing; v_fgls2[r] = missing
            else
                β_fgls2[r] = res.β_fgls2[2]; v_fgls2[r] = res.V_fgls2[2,2]
            end

            if res.β_gls === nothing
                β_gls[r] = missing; v_gls[r] = missing
            else
                β_gls[r] = res.β_gls[2];  v_gls[r] = res.V_gls[2,2]
            end

            
            if inner !== nothing
                lock(lk) do; next!(inner); end
            end
        end # threads

        if inner !== nothing; finish!(inner); end

        # per-size summary + printing
        βf1 = collect(skipmissing(β_fgls1)); vf1 = collect(skipmissing(v_fgls1))
        βf2 = collect(skipmissing(β_fgls2)); vf2 = collect(skipmissing(v_fgls2))
        βg = collect(skipmissing(β_gls));  vg = collect(skipmissing(v_gls))

        bias_ols = mean(β_ols .- p.beta_true)
        bias_fe  = mean(β_fe  .- p.beta_true)
        bias_f1  = isempty(βf1) ? NaN : mean(βf1 .- p.beta_true)
        bias_f2  = isempty(βf2) ? NaN : mean(βf2 .- p.beta_true)
        bias_g   = isempty(βg) ? NaN : mean(βg .- p.beta_true)

        var_emp_ols = var(β_ols; corrected=true)
        var_emp_fe  = var(β_fe;  corrected=true)
        var_emp_f1  = isempty(βf1) ? NaN : var(βf1; corrected=true)
        var_emp_f2  = isempty(βf2) ? NaN : var(βf2; corrected=true)
        var_emp_g   = isempty(βg) ? NaN : var(βg; corrected=true)

        var_est_ols = mean(v_ols)
        var_est_fe  = mean(v_fe)
        var_est_f1  = isempty(vf1) ? NaN : mean(vf1)
        var_est_f2  = isempty(vf2) ? NaN : mean(vf2)
        var_est_g   = isempty(vg) ? NaN : mean(vg)

        ratio_ols = var_est_ols / max(var_emp_ols, eps(Float64))
        ratio_fe  = var_est_fe  / max(var_emp_fe,  eps(Float64))
        ratio_f1  = isempty(βf1) ? NaN : (var_est_f1 / max(var_emp_f1, eps(Float64)))
        ratio_f2  = isempty(βf2) ? NaN : (var_est_f2 / max(var_emp_f2, eps(Float64)))
        ratio_g   = isempty(βg) ? NaN : (var_est_g / max(var_emp_g, eps(Float64)))

        if print_each
            @printf("\n--- Results for Sample Size: %d (N2=%d, T=%d) ---\n", n, N2, T)
            @printf("Bias (OLS):                %.4f\n", bias_ols)
            @printf("Bias (OLS FE):             %.4f\n", bias_fe)
            @printf("Bias (FGLS1):              %.4f\n", bias_f1)
            @printf("Bias (FGLS2):              %.4f\n", bias_f2)
            @printf("Bias (GLS):                %.4f\n", bias_g)
            @printf("Empirical Var (OLS):       %.4f\n", var_emp_ols)
            @printf("Empirical Var (OLS FE):    %.4f\n", var_emp_fe)
            @printf("Empirical Var (FGLS1):     %.4f\n", var_emp_f1)
            @printf("Empirical Var (FGLS2):     %.4f\n", var_emp_f2)
            @printf("Empirical Var (GLS):       %.4f\n", var_emp_g)
            @printf("Estimated/Empirical Var (OLS):       %.4f\n", ratio_ols)
            @printf("Estimated/Empirical Var (OLS FE):    %.4f\n", ratio_fe)
            @printf("Estimated/Empirical Var (FGLS1):     %.4f\n", ratio_f1)
            @printf("Estimated/Empirical Var (FGLS2):     %.4f\n", ratio_f2)
            @printf("Estimated/Empirical Var (GLS):       %.4f\n", ratio_g)
        end

        stats_tuple = (
            bias_ols=bias_ols, bias_fe=bias_fe, bias_fgls1=bias_f1, bias_fgls2=bias_f2, bias_gls=bias_g,
            var_emp_ols=var_emp_ols, var_emp_fe=var_emp_fe,
            var_emp_fgls1=var_emp_f1, var_emp_fgls2=var_emp_f2, var_emp_gls=var_emp_g,
            var_est_ols=var_est_ols, var_est_fe=var_est_fe,
            var_est_fgls1=var_est_f1, var_est_fgls2=var_est_f2, var_est_gls=var_est_g,
            ratio_est_emp_ols=ratio_ols, ratio_est_emp_fe=ratio_fe,
            ratio_est_emp_fgls1=ratio_f1, ratio_est_emp_fgls2=ratio_f2, ratio_est_emp_gls=ratio_g
        )

        if keep_vectors
            out[idx] = (
                size = (N1=N1, N2=N2, T=T),
                reps = reps_here,
                β_ols = β_ols, v_ols = v_ols,
                β_fe  = β_fe,  v_fe  = v_fe,
                β_fgls1 = β_fgls1, v_fgls1 = v_fgls1,
                β_fgls2 = β_fgls2, v_fgls2 = v_fgls2,
                β_gls  = β_gls,  v_gls  = v_gls,
                stats = stats_tuple
            )
        else
            out[idx] = (
                size = (N1=N1, N2=N2, T=T),
                reps = reps_here,
                stats = stats_tuple
            )
        end

        if outer !== nothing; next!(outer); end
    end

    if outer !== nothing; finish!(outer); end
    return out
end

end # module