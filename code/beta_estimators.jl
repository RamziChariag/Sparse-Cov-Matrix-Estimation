# beta_estimators.jl
module RCBetaEstimators

using LinearAlgebra, Statistics, DataFrames
using Printf: @printf

import ..RCOmegaEstimators

export add_y!, within_transform!, ols_estimation, ols, fe_ols, gls, fgls1, fgls2, oracle_gls

# -----------------------------
# Helpers
# -----------------------------

"Compute y = c + β·x + fe_i + fe_j + fe_t + u_ijt (adds/overwrites :y)."
function add_y!(df::DataFrame; beta::Real, constant::Real=0.0, x_col::Symbol=:x)
    @assert all(Symbol.([:fe_i,:fe_j,:fe_t,:u_ijt]) .∈ Ref(Symbol.(names(df)))) "Missing FE/u columns"
    df[!, :y] = constant .+ beta .* df[!, x_col] .+ df.fe_i .+ df.fe_j .+ df.fe_t .+ df.u_ijt
    return df
end

# -----------------------------
# Within (demeaning) transform
# -----------------------------

"""
Within-transform x and y along group_vars ⊆ [:i,:j,:t].

For k dims, uses x̃ = x + (k-1)·x̄ − Σ_d x̄_d (same for y).
Creates x_bar_{dim}, y_bar_{dim} and x_tilde_{suffix}, y_tilde_{suffix},
where suffix maps :i→alpha, :j→gamma, :t→lambda.
Returns the suffix String.
"""
function within_transform!(
    df::DataFrame;
    x_col::Symbol=:x,
    y_col::Symbol=:y,
    group_vars::Vector{Symbol} = [:i,:j,:t]
)
    @assert !isempty(group_vars) "group_vars must not be empty"
    for v in group_vars
        @assert v in (:i,:j,:t) "group_vars may only contain :i, :j, :t"
    end
    dim_map = Dict(:i=>"alpha", :j=>"gamma", :t=>"lambda")
    suffix  = join(getindex.(Ref(dim_map), group_vars), "_")

    # Per-dimension means (replicated to rows)
    x_bar_cols = Symbol[]
    y_bar_cols = Symbol[]
    for v in group_vars
        xbar = Symbol(string(x_col), "_bar_", String(v))
        ybar = Symbol(string(y_col), "_bar_", String(v))
        transform!(groupby(df, v), x_col => mean => xbar, y_col => mean => ybar)
        push!(x_bar_cols, xbar); push!(y_bar_cols, ybar)
    end

    x_mean = mean(df[!, x_col]); y_mean = mean(df[!, y_col]); k = length(group_vars)

    n = nrow(df)
    sx = zeros(n); for c in x_bar_cols; sx .+= df[!, c]; end
    sy = zeros(n); for c in y_bar_cols; sy .+= df[!, c]; end

    xtilde = Symbol(string(x_col), "_tilde_", suffix)
    ytilde = Symbol(string(y_col), "_tilde_", suffix)
    df[!, xtilde] = df[!, x_col] .+ (k - 1) * x_mean .- sx
    df[!, ytilde] = df[!, y_col] .+ (k - 1) * y_mean .- sy

    return suffix
end

# -----------------------------
# OLS (with robust / cluster-robust vcov)
# -----------------------------

"""
Low-level OLS on given X and y. **X must include any intercept column if desired.**

Keyword options:
- vcov::Symbol = :HC1      # :none | :HC0 | :HC1 | :cluster
- cluster::Union{Nothing,AbstractVector} = nothing
Returns (beta::Vector, resid::Vector, V::Matrix).
"""
function ols_estimation(X::AbstractMatrix, y::AbstractVector;
                        vcov::Symbol=:HC0,
                        cluster::Union{Nothing,AbstractVector}=nothing)
    n = length(y); @assert size(X,1) == n "Rows of X must match length of y"
    k = size(X,2)

    β = X \ y
    e = y .- X*β

    XtX = X' * X
    XtX_inv = inv(XtX)

    if vcov == :none
        σ2 = sum(abs2, e) / max(n - k, 1)
        V  = σ2 * XtX_inv

    elseif vcov == :HC0 || vcov == :HC1
        w = e .^ 2
        meat = X' * (X .* w)
        V = XtX_inv * meat * XtX_inv
        if vcov == :HC1
            V .*= n / max(n - k, 1)
        end

    elseif vcov == :cluster
        @assert cluster !== nothing "Provide `cluster` for vcov=:cluster"
        @assert length(cluster) == n "cluster length must equal n"

        keys = unique(cluster); G = length(keys)
        meat = zeros(eltype(X), k, k)
        for g in keys
            idx = findall(c -> c == g, cluster)
            Xg = @view X[idx, :]; eg = @view e[idx]
            xte = Xg' * eg
            meat .+= xte * xte'
        end
        scale = (G / max(G - 1, 1)) * (n - 1) / max(n - k, 1)  # CR1
        V = XtX_inv * (scale .* meat) * XtX_inv

    else
        error("Unknown vcov: $vcov")
    end

    return β, e, V
end

"Raw-data OLS on a DataFrame (adds an intercept automatically)."
function ols(df::DataFrame; x_col::Symbol=:x, y_col::Symbol=:y,
             vcov::Symbol=:HC1, cluster_col::Union{Nothing,Symbol}=nothing)
    n = nrow(df)
    x = df[!, x_col]                  # Vector
    X = hcat(ones(eltype(x), n), x)   # n×2 Matrix (intercept + x)
    y = Vector(df[!, y_col])
    cl = cluster_col === nothing ? nothing : Vector(df[!, cluster_col])
    return ols_estimation(X, y; vcov=vcov, cluster=cl)
end

"Three-way FE OLS on a DataFrame (within-transform, then OLS without intercept)."
function fe_ols(df::DataFrame; x_col::Symbol=:x, y_col::Symbol=:y,
                group_vars::Vector{Symbol} = [:i,:j,:t],
                vcov::Symbol=:HC1, cluster_col::Union{Nothing,Symbol}=nothing)
    suffix = within_transform!(df; x_col=x_col, y_col=y_col, group_vars=group_vars)
    xt = Symbol(string(x_col), "_tilde_", suffix)
    yt = Symbol(string(y_col), "_tilde_", suffix)
    X = reshape(df[!, xt], :, 1)      # n×1 Matrix (no intercept after within)
    y = Vector(df[!, yt])
    cl = cluster_col === nothing ? nothing : Vector(df[!, cluster_col])
    return ols_estimation(X, y; vcov=vcov, cluster=cl)
end

# -----------------------------
# GLS / FGLS
# -----------------------------

"GLS on a DataFrame: uses a given full Ω (n×n). Includes intercept like raw OLS."
function gls(df::DataFrame, Ω::AbstractMatrix; x_col::Symbol=:x, y_col::Symbol=:y)
    n = nrow(df)
    x = df[!, x_col]
    X = hcat(ones(eltype(x), n), x)   # intercept + x
    y = Vector(df[!, y_col])

    # Solve Ω^{-1}X and Ω^{-1}y (prefer Cholesky if SPD)
    Zx = nothing; Zy = nothing
    try
        F = cholesky(Symmetric(Ω))
        Zx = F \ X
        Zy = F \ y
    catch
        Zx = Ω \ X
        Zy = Ω \ y
    end

    XtΩinvX = X' * Zx
    XtΩinvy = X' * Zy
    β = XtΩinvX \ XtΩinvy
    e = y .- X*β
    V = inv(XtΩinvX)   # GLS variance
    return β, e, V
end

"""
FGLS1 on a DataFrame:
1) OLS for β̂
2) σ²_u, σ²_α/γ/λ via within-residuals
3) Ωα/Ωγ/Ωλ blocks per `covariance_mode` (:homoskedastic, :mixed_i, :mixed_j, :mixed_t, :full_mixed)
4) Expand blocks if `repeat_*` are true; build S-matrices
5) Ω̂ = SαΩαSα' + SγΩγSγ' + SλΩλSλ' + σ²_u I
6) GLS with Ω̂

Returns (β̂_fgls1, ê, V̂_fgls1, Ω̂).
"""
function fgls1(df::DataFrame, N1::Int, N2::Int, T::Int;
              x_col::Symbol=:x, y_col::Symbol=:y,
              i_block_est::Bool=true,
              j_block_est::Bool=true,
              t_block_est::Bool=true,
              repeat_alpha::Bool=false, repeat_gamma::Bool=false, repeat_lambda::Bool=false,
              subtract_sigma_u2_fgls1::Bool=false,
              run_gls::Bool=true, print_omega::Bool=false,
              shrinkage::Real=1.0,            # keep your shrinkage if you pass it
              project_spd::Bool=false, spd_floor::Real=1e-8)

    # 1) OLS for initial β̂
    β̂_ols, _, _ = ols(df; x_col=x_col, y_col=y_col, vcov=:none)

    # 2) σ² components
    sigmas = RCOmegaEstimators.estimate_homoskedastic_component_variances(
        df, N1, N2, T, β̂_ols[2]; x_col=x_col, y_col=y_col)
    σu2 = sigmas.sigma_u2

    # 3) base blocks Ω̂ᵢ, Ω̂ⱼ, Ω̂ₜ
    blocks = RCOmegaEstimators.estimate_omegas(
        df, N1, N2, T, sigmas, β̂_ols[2];
        x_col=x_col,                  
        i_block_est=i_block_est,
        j_block_est=j_block_est,
        t_block_est=t_block_est,
        subtract_sigma_u2 = subtract_sigma_u2_fgls1,
        two_step=false
    )
    Ωa, Ωg, Ωl = blocks.Ωa, blocks.Ωg, blocks.Ωl

    # 4) S matrices and optional block repeats
    Sα, Sγ, Sλ = RCOmegaEstimators.make_S_matrices(
        N1, N2, T; repeat_alpha=repeat_alpha, repeat_gamma=repeat_gamma, repeat_lambda=repeat_lambda)

    if repeat_alpha; Ωa = RCOmegaEstimators.repeat_block(Ωa, T); end
    if repeat_gamma; Ωg = RCOmegaEstimators.repeat_block(Ωg, T);  end
    if repeat_lambda; Ωl = RCOmegaEstimators.repeat_block(Ωl, N2); end

    # 5) Ω̂ and optional conditioning
    Ω̂ = RCOmegaEstimators.construct_omega(Ωa, Ωg, Ωl, Sα, Sγ, Sλ, σu2)

    if shrinkage != 1.0
        Ω̂ = RCOmegaEstimators.shrink_offdiagonal!(Ω̂, shrinkage)
    end
    
    if project_spd
        Ω̂ = RCOmegaEstimators.project_to_spd(Ω̂; floor=spd_floor)
    end

    if print_omega
        io = IOContext(stdout, :limit=>false, :compact=>false)
        println("\nΩ̂ (", size(Ω̂,1), "×", size(Ω̂,2), "):")
        show(io, "text/plain", Matrix(Ω̂))
        println()
    end

    if !run_gls
        return nothing, nothing, nothing, Ω̂
    end

    β̂_fgls, ê, V̂ = gls(df, Ω̂; x_col=x_col, y_col=y_col)
    return β̂_fgls, ê, V̂, Ω̂
end

"""
FGLS2 on a DataFrame using two-step averaging of residual vectors.

Returns (β̂_fgls2, ê, V̂_fgls2, Ω̂).
"""
function fgls2(df::DataFrame, N1::Int, N2::Int, T::Int;
              x_col::Symbol=:x, y_col::Symbol=:y,
              i_block_est::Bool=true,
              j_block_est::Bool=true,
              t_block_est::Bool=true,
              repeat_alpha::Bool=false, repeat_gamma::Bool=false, repeat_lambda::Bool=false,
              subtract_sigma_u2_fgls2::Bool=false,
              run_gls::Bool=true, print_omega::Bool=false,
              shrinkage::Real=1.0,
              project_spd::Bool=false, spd_floor::Real=1e-8,
              # --- debug-only ---
              debug::Bool=false,
              debug_truth::Union{Nothing,NamedTuple}=nothing,  # e.g., (Ωi_star=..., Ωj_star=..., Ωt_star=...)
              debug_digits::Int=3)

    # 0) Sort data with i fastest, then j then t
    df = RCOmegaEstimators.sort_for_dim(df, :i)  

    # 1) OLS for initial β̂
    β̂_ols, _, _ = ols(df; x_col=x_col, y_col=y_col, vcov=:none)

    # 2) σ² Homoskedastic components (needed if any block is not estimated,
    #    and for homoskedastic diagonals in those cases)
    sigmas = RCOmegaEstimators.estimate_homoskedastic_component_variances(
        df, N1, N2, T, β̂_ols[2]; x_col=x_col, y_col=y_col
    )

    # 3) get blocks + pooled σ̂_u^2 via two-step
    blocks = RCOmegaEstimators.estimate_omegas(
        df, N1, N2, T, sigmas, β̂_ols[2];
        x_col=x_col,
        i_block_est=i_block_est,
        j_block_est=j_block_est,
        t_block_est=t_block_est,
        two_step=true,
        return_sigma=true,
        subtract_sigma_u2 = subtract_sigma_u2_fgls2
    )
    Ωi, Ωj, Ωt = blocks.Ωa, blocks.Ωg, blocks.Ωl
    # This uses pooled σ̂_u^2 from two-step (not per-dim σ̂²_u)
    σ2_u = blocks.sigma_u2    
    #σ2_u = sigmas.sigma_u2

         # ===== DEBUG PRINT (small blocks + sigma pieces) =====
    if debug
        io = IOContext(stdout, :limit=>false, :compact=>false)

        # Accept multiple naming conventions for truth, if provided
        Ωi_star = debug_truth === nothing ? nothing :
                  get(debug_truth, :Ωi_star, get(debug_truth, :omega_alpha_star, nothing))
        Ωj_star = debug_truth === nothing ? nothing :
                  get(debug_truth, :Ωj_star, get(debug_truth, :omega_gamma_star, nothing))
        Ωt_star = debug_truth === nothing ? nothing :
                  get(debug_truth, :Ωt_star, get(debug_truth, :omega_lambda_star, nothing))

        local function show_difference(name::AbstractString, A::AbstractMatrix, B)
            io = IOContext(stdout, :limit=>false, :compact=>false)
            println("→ $(name) (hat) vs $(name)★ (true):")

            A_m = Matrix(A)
            show(io, "text/plain", round.(A_m; digits=debug_digits)); println()

            if B === nothing
                println("   (no truth provided for $(name)★)")
                println()
                return
            end

            B_m = Matrix(B)
            show(io, "text/plain", round.(B_m; digits=debug_digits)); println()

            Δ = A_m .- B_m
            println("   Δ = (hat − true):")
            show(io, "text/plain", round.(Δ; digits=debug_digits)); println()
            println()
        end


        println("\n========== FGLS2 diagnostics (small blocks) ==========")
        show_difference("Ω_i", Ωi, Ωi_star)
        show_difference("Ω_j", Ωj, Ωj_star)
        show_difference("Ω_t", Ωt, Ωt_star)

        println("σ (single-component pieces):")
        # from estimate_homoskedastic_component_variances (NamedTuple)
        @printf("   %-18s = %.6g\n", "sigma_alpha2", getfield(sigmas, :sigma_alpha2))
        @printf("   %-18s = %.6g\n", "sigma_gamma2", getfield(sigmas, :sigma_gamma2))
        @printf("   %-18s = %.6g\n", "sigma_lambda2", getfield(sigmas, :sigma_lambda2))
        @printf("   %-18s = %.6g\n", "sigma_u2 (two-step pooled)", σ2_u)
        println("======================================================\n")
    end
    # ===== END DEBUG PRINT =====

    # 4) Repeat expansion if requested
    if repeat_alpha;  Ωi = RCOmegaEstimators.repeat_block(Ωi, T); end
    if repeat_gamma;  Ωj = RCOmegaEstimators.repeat_block(Ωj, T);  end
    if repeat_lambda; Ωt = RCOmegaEstimators.repeat_block(Ωt, N2); end

    # 5) Build S matrices
    Sα, Sγ, Sλ = RCOmegaEstimators.make_S_matrices(N1, N2, T;
        repeat_alpha=repeat_alpha, repeat_gamma=repeat_gamma, repeat_lambda=repeat_lambda
    )

    # 6) Assemble Ω with σ̂_u^2 I
    Ω̂ = RCOmegaEstimators.construct_omega(Ωi, Ωj, Ωt, Sα, Sγ, Sλ, σ2_u)
    
         # ===== DEBUG PRINT (full Ω̂ diagnostics) =====

    if debug
        println("\n========== FGLS2 diagnostics (full Ω̂) ==========")
        @info "diag Ω̂ (mean)" mean(diag(Ω̂))
        @info "diag Ω̂i (mean)" mean(diag(Ωi))
        @info "diag Ω̂i★ (mean)" mean(diag(Ωi_star))
        @info "diag Ω̂j (mean)" mean(diag(Ωj))
        @info "diag Ω̂j★ (mean)" mean(diag(Ωj_star))
        @info "diag Ω̂t (mean)" mean(diag(Ωt))
        @info "diag Ω̂t★ (mean)" mean(diag(Ωt_star))
        @info "trace Ω̂" tr(Ω̂)
        @info "σ2_u used" σ2_u
        @info "trace α/γ/λ parts" (tr(Sα*Ωi*Sα'), tr(Sγ*Ωj*Sγ'), tr(Sλ*Ωt*Sλ'))

        println("======================================================\n")
    end

    # 7) Optional shrinkage / SPD floor/projection
    if shrinkage != 1.0
        Ω̂ = RCOmegaEstimators.shrink_offdiagonal!(Ω̂, shrinkage)
    end
    if project_spd
        Ω̂ = RCOmegaEstimators.project_to_spd(Ω̂; floor=spd_floor)
    end

    if print_omega
        io = IOContext(stdout, :limit=>false, :compact=>false)
        println("\nΩ̂ (", size(Ω̂,1), "×", size(Ω̂,2), "):")
        show(io, "text/plain", Matrix(Ω̂))
        println()
    end

    if !run_gls
        return nothing, nothing, nothing, Ω̂
    end

    β̂_fgls, ê, V̂ = gls(df, Ω̂; x_col=x_col, y_col=y_col)
    return β̂_fgls, ê, V̂, Ω̂
end


"""
Oracle GLS using true small blocks (Ωᵢ, Ωⱼ, Ωₜ) from the DGP.

Returns (β̂_gls, ê_gls, V̂_gls, Ω★).
Assumes rows are ordered with i fastest (sort!(df, [:t,:j,:i])) to match S.
"""
function oracle_gls(df::DataFrame,
                    Ωi::AbstractMatrix, Ωj::AbstractMatrix, Ωt::AbstractMatrix,
                    N1::Int, N2::Int, T::Int;
                    x_col::Symbol=:x, y_col::Symbol=:y,
                    repeat_alpha::Bool=true, repeat_gamma::Bool=false, repeat_lambda::Bool=false,
                    sigma_u2::Real=1.0,
                    shrinkage::Real=1.0, project_spd::Bool=false, spd_floor::Real=1e-8)

    # Base blocks
    Ωa, Ωg, Ωl = Ωi, Ωj, Ωt

    # S matrices per repeat pattern
    Sα, Sγ, Sλ = RCOmegaEstimators.make_S_matrices(
        N1, N2, T; repeat_alpha=repeat_alpha, repeat_gamma=repeat_gamma, repeat_lambda=repeat_lambda)

    # Expand blocks if repeating
    if repeat_alpha
        Ωa = RCOmegaEstimators.repeat_block(Ωa, T)   # repeat along t
    end
    if repeat_gamma
        Ωg = RCOmegaEstimators.repeat_block(Ωg, T)    # repeat along t
    end
    if repeat_lambda
        Ωl = RCOmegaEstimators.repeat_block(Ωl, N2)   # repeat along j
    end

    # Full Ω★
    Ωstar = RCOmegaEstimators.construct_omega(Ωa, Ωg, Ωl, Sα, Sγ, Sλ, sigma_u2)

    # Optional off-diagonal shrinkage + SPD projection
    if shrinkage != 1.0
        Ωstar = RCOmegaEstimators.shrink_offdiagonal!(Ωstar, shrinkage)
    end
    if project_spd
        Ωstar = RCOmegaEstimators.project_to_spd(Ωstar; floor=spd_floor)
    end

    # GLS
    β̂, ê, V̂ = gls(df, Ωstar; x_col=x_col, y_col=y_col)
    return β̂, ê, V̂, Ωstar
end

end # module
