module RCEstimation

using DataFrames, LinearAlgebra, Statistics, ProgressMeter
using Printf: @printf

import ..RCDGP
import ..RCBetaEstimators
import ..RCOmegaEstimators

export estimate_all, print_estimate_summary, mc_estimate_over_sizes

"""
    estimate_all(df; kwargs...) -> NamedTuple

Runs OLS, FE-OLS, FGLS, and (optionally) Oracle GLS on `df`.

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
  β_fgls, V_fgls, se_fgls, Ωhat
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
    fgls_shrinkage::Real=1.0,
    fgls_project_spd::Bool=false, fgls_spd_floor::Real=1e-8,

    # Oracle GLS (pass true blocks to enable)
    Ωi_star::Union{Nothing,AbstractMatrix}=nothing,
    Ωj_star::Union{Nothing,AbstractMatrix}=nothing,
    Ωt_star::Union{Nothing,AbstractMatrix}=nothing,
    rep_a_gls::Bool=true, rep_g_gls::Bool=false, rep_l_gls::Bool=false,
    gls_shrinkage::Real=1.0, gls_project_spd::Bool=false, gls_spd_floor::Real=1e-8,
    sigma_u2_oracle::Real=1.0,

    # ordering
    sort_for_gls::Bool=true
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
    df_gls = sort_for_gls ? sort(df, [:t,:j,:i]) : df

    # 3) FGLS
    β_fgls = nothing; e_fgls = nothing; V_fgls = nothing; Ωhat = nothing; se_fgls = nothing
    try
        β_fgls, e_fgls, V_fgls, Ωhat = RCBetaEstimators.fgls(
            df_gls, N1, N2, T;
            x_col=:x, y_col=:y,
            i_block_est = i_block_est,
            j_block_est = j_block_est,
            t_block_est = t_block_est,
            repeat_alpha  = rep_a_fgls,
            repeat_gamma  = rep_g_fgls,
            repeat_lambda = rep_l_fgls,
            run_gls       = true,
            shrinkage     = fgls_shrinkage,
            project_spd   = fgls_project_spd,
            spd_floor     = fgls_spd_floor
        )
    catch err
        @warn "FGLS failed" exception=(err, catch_backtrace()) n=nrow(df) p=ncol(df) N1=N1 N2=N2 T=T
    end
    if V_fgls !== nothing
        se_fgls = sqrt.(max.(diag(V_fgls), 0))
    end

    # (no second FGLS variant)

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
        # FGLS
        β_fgls = β_fgls, V_fgls = V_fgls, se_fgls = se_fgls, Ωhat = Ωhat,
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

    β̂_fgls = res.β_fgls === nothing ? nothing : res.β_fgls[2]
    v_fgls  = res.V_fgls === nothing ? nothing : res.V_fgls[2,2]

    β̂_gls  = res.β_gls === nothing ? nothing : res.β_gls[2]
    v_gls   = res.V_gls === nothing ? nothing : res.V_gls[2,2]

    fmt(x) = x === nothing ? "—" : string(round(x, digits=6))
    println("\n== Single Estimation summary ==")
    println("Using smoke test data, if it does not exist, using first MC bundle data...")
    println("True β: ", beta_true)
    println("OLS:     β̂ = ", fmt(β̂_ols),  "   Var = ", fmt(v_ols))
    println("FE-OLS:  β̂ = ", fmt(β̂_fe),   "   Var = ", fmt(v_fe))
    println("FGLS:    β̂ = ", fmt(β̂_fgls), "   Var = ", fmt(v_fgls))
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

        # possibly override reps if a bundle of this size is provided
        local_datasets = bundle === nothing ? nothing : bundle[idx]
        reps_here = bundle === nothing ? reps :
                    (local_datasets isa Vector ? length(local_datasets) : 1)

        β_ols  = Vector{Float64}(undef, reps_here)
        v_ols  = Vector{Float64}(undef, reps_here)
        β_fe   = Vector{Float64}(undef, reps_here)
        v_fe   = Vector{Float64}(undef, reps_here)
        β_fgls = Vector{Union{Missing,Float64}}(undef, reps_here)
        v_fgls = Vector{Union{Missing,Float64}}(undef, reps_here)
        β_gls  = Vector{Union{Missing,Float64}}(undef, reps_here)
        v_gls  = Vector{Union{Missing,Float64}}(undef, reps_here)

        inner = progress_reps ? Progress(reps_here; dt=0.1, desc="N2=$N2, T=$T (n=$n)") : nothing
        lk = ReentrantLock()

        Threads.@threads for r in 1:reps_here
            # pick data
            df = nothing; Ωi_true=nothing; Ωj_true=nothing; Ωt_true=nothing
            if bundle === nothing
                seed_r = p.seed + 10_000*idx + r
                df, meta = RCDGP.generate_dataset(
                    N1=N1, N2=N2, T=T,
                    i_block=p.i_block, j_block=p.j_block, t_block=p.t_block,
                    i_draw_mode=p.i_draw_mode, j_draw_mode=p.j_draw_mode, t_draw_mode=p.t_draw_mode,
                    E_i=p.E_i, E_j=p.E_j, E_t=p.E_t,
                    sigma_i=p.sigma_i, sigma_j=p.sigma_j, sigma_t=p.sigma_t,
                    mu_x=p.mu_x, sigma_x=p.sigma_x,
                    mu_u=p.mu_u, sigma_u=p.sigma_u,
                    seed=seed_r
                )
                Ωi_true, Ωj_true, Ωt_true = meta.Ωi, meta.Ωj, meta.Ωt
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
                fgls_shrinkage=p.fgls_shrinkage,
                fgls_project_spd = (:fgls_project_spd ∈ propertynames(p) ? p.fgls_project_spd : false),
                fgls_spd_floor   = (:fgls_spd_floor   ∈ propertynames(p) ? p.fgls_spd_floor   : 1e-8),
                # oracle
                Ωi_star=Ωi_true, Ωj_star=Ωj_true, Ωt_star=Ωt_true,
                rep_a_gls=p.repeat_alpha_gls, rep_g_gls=p.repeat_gamma_gls, rep_l_gls=p.repeat_lambda_gls,
                gls_shrinkage=p.gls_shrinkage,
                gls_project_spd = (:gls_project_spd ∈ propertynames(p) ? p.gls_project_spd : false),
                gls_spd_floor   = (:gls_spd_floor   ∈ propertynames(p) ? p.gls_spd_floor   : 1e-8),
                sigma_u2_oracle=p.sigma_u^2,
                sort_for_gls=true
            )

            β_ols[r] = res.β_ols[2];  v_ols[r] = res.V_ols[2,2]
            β_fe[r]  = res.β_fe[1];   v_fe[r]  = res.V_fe[1,1]

            if res.β_fgls === nothing
                β_fgls[r] = missing; v_fgls[r] = missing
            else
                β_fgls[r] = res.β_fgls[2]; v_fgls[r] = res.V_fgls[2,2]
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
        βf = collect(skipmissing(β_fgls)); vf = collect(skipmissing(v_fgls))
        βg = collect(skipmissing(β_gls));  vg = collect(skipmissing(v_gls))

        bias_ols = mean(β_ols .- p.beta_true)
        bias_fe  = mean(β_fe  .- p.beta_true)
        bias_f   = isempty(βf) ? NaN : mean(βf .- p.beta_true)
        bias_g   = isempty(βg) ? NaN : mean(βg .- p.beta_true)

        var_emp_ols = var(β_ols; corrected=true)
        var_emp_fe  = var(β_fe;  corrected=true)
        var_emp_f   = isempty(βf) ? NaN : var(βf; corrected=true)
        var_emp_g   = isempty(βg) ? NaN : var(βg; corrected=true)

        var_est_ols = mean(v_ols)
        var_est_fe  = mean(v_fe)
        var_est_f   = isempty(vf) ? NaN : mean(vf)
        var_est_g   = isempty(vg) ? NaN : mean(vg)

        ratio_ols = var_est_ols / max(var_emp_ols, eps(Float64))
        ratio_fe  = var_est_fe  / max(var_emp_fe,  eps(Float64))
        ratio_f   = isempty(βf) ? NaN : (var_est_f / max(var_emp_f, eps(Float64)))
        ratio_g   = isempty(βg) ? NaN : (var_est_g / max(var_emp_g, eps(Float64)))

        if print_each
            @printf("\n--- Results for Sample Size: %d (N2=%d, T=%d) ---\n", n, N2, T)
            @printf("Bias (OLS):                %.4f\n", bias_ols)
            @printf("Bias (OLS FE):             %.4f\n", bias_fe)
            @printf("Bias (FGLS Arithmetic):    %.4f\n", bias_f)
            @printf("Bias (GLS):                %.4f\n", bias_g)
            @printf("Empirical Var (OLS):       %.4f\n", var_emp_ols)
            @printf("Empirical Var (OLS FE):    %.4f\n", var_emp_fe)
            @printf("Empirical Var (FGLS AM):   %.4f\n", var_emp_f)
            @printf("Empirical Var (GLS):       %.4f\n", var_emp_g)
            @printf("Estimated/Empirical Var (OLS):       %.4f\n", ratio_ols)
            @printf("Estimated/Empirical Var (OLS FE):    %.4f\n", ratio_fe)
            @printf("Estimated/Empirical Var (FGLS AM):   %.4f\n", ratio_f)
            @printf("Estimated/Empirical Var (GLS):       %.4f\n", ratio_g)
        end

        stats_tuple = (
            bias_ols=bias_ols, bias_fe=bias_fe, bias_fgls=bias_f, bias_gls=bias_g,
            var_emp_ols=var_emp_ols, var_emp_fe=var_emp_fe,
            var_emp_fgls=var_emp_f, var_emp_gls=var_emp_g,
            var_est_ols=var_est_ols, var_est_fe=var_est_fe,
            var_est_fgls=var_est_f, var_est_gls=var_est_g,
            ratio_est_emp_ols=ratio_ols, ratio_est_emp_fe=ratio_fe,
            ratio_est_emp_fgls=ratio_f, ratio_est_emp_gls=ratio_g
        )

        if keep_vectors
            out[idx] = (
                size = (N1=N1, N2=N2, T=T),
                reps = reps_here,
                β_ols = β_ols, v_ols = v_ols,
                β_fe  = β_fe,  v_fe  = v_fe,
                β_fgls = β_fgls, v_fgls = v_fgls,
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
