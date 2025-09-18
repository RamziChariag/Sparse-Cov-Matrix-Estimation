# omega_estimators.jl
module RCOmegaEstimators

using LinearAlgebra, Statistics, DataFrames

export plot_matrix_percentile, omega_eigen_tables,
       residuals_from_suffix, ensure_tilde_columns!,
       estimate_homoskedastic_component_variances,
       sort_for_dim, arithmetic_mean_outer_products,
       generate_threeway_omegas, generate_single_component_omega,
       estimate_omegas, make_S_matrices, construct_omega, repeat_block,
       shrink_offdiagonal!, make_psd_by_minimal_ridge

# --- helpers ---

"""
Return a Symmetric copy of A after adding the *smallest* uniform ridge c*I
so the matrix is PSD. If `strict=true`, make it SPD by adding a tiny buffer.
Returns (A_psd, c).
"""
function make_psd_by_minimal_ridge(A::AbstractMatrix; strict::Bool=true, eps_rel::Real=1e-12)
    S = Symmetric(Matrix(A))
    λmin = eigmin(S)
    if strict
        # tiny, scale-aware buffer for strict PD
        δ = eps_rel * max(1.0, maximum(abs, S))
        c = max(0.0, -λmin + δ)
    else
        c = max(0.0, -λmin)
    end
    if c > 0.0
        @inbounds for i in 1:size(S,1)
            S[i,i] += c
        end
    end
    return S
end

"Repeat a covariance block along the diagonal `rep` times (kron(I(rep), Ω))."
repeat_block(Ω::AbstractMatrix, rep::Integer) = kron(I(rep), Ω)

"Sort so that the requested dim is the inner-most (fastest)."
function sort_for_dim(df::DataFrame, dim::Symbol)
    @assert dim ∈ (:i,:j,:t)
    if dim === :i
        return sort(df, [:t, :j, :i])  # i fastest
    elseif dim === :j
        return sort(df, [:t, :i, :j])  # j fastest
    else
        return sort(df, [:i, :j, :t])  # t fastest
    end
end

"Ensure within-tilde columns for needed suffix exist (computes them if missing)."
function ensure_tilde_columns!(df::DataFrame; x_col::Symbol=:x, y_col::Symbol=:y,
                               suffixes::Vector{String}=String[])
    needed = Set(suffixes)
    # what we might need
    candidates = Dict(
        "alpha_gamma_lambda" => [:i,:j,:t],
        "alpha_gamma"        => [:i,:j],
        "alpha_lambda"       => [:i,:t],
        "gamma_lambda"       => [:j,:t]
    )
    # see what’s missing and compute
    for (suf, gv) in candidates
        if suf in needed
            xt = Symbol(string(x_col), "_tilde_", suf)
            yt = Symbol(string(y_col), "_tilde_", suf)
            if !(xt in propertynames(df)) || !(yt in propertynames(df))
                # Bring in the within transformer from beta_estimators
                # (we don't import it here to avoid a circular dep; call via Main module path)
                if isdefined(Main, :RCBetaEstimators)
                    Main.RCBetaEstimators.within_transform!(df; x_col=x_col, y_col=y_col, group_vars=gv)
                else
                    error("within_transform! not available. Include beta_estimators.jl before calling Ω-estimators.")
                end
            end
        end
    end
    return df
end

"Residuals from tilde columns with `suffix` (ỹ - β̂ x̃)."
function residuals_from_suffix(df::DataFrame, suffix::AbstractString, beta_hat::Real;
                               x_col::Symbol=:x, y_col::Symbol=:y)
    xt = Symbol(string(x_col), "_tilde_", suffix)
    yt = Symbol(string(y_col), "_tilde_", suffix)
    @assert xt in propertynames(df) && yt in propertynames(df) "Missing tilde columns for suffix=$suffix"
    return df[!, yt] .- beta_hat .* df[!, xt]
end

"Arithmetic mean of clusterwise outer products of length `block_size` chunks."
function arithmetic_mean_outer_products(resid::AbstractVector, block_size::Int)
    n = length(resid)
    @assert n % block_size == 0 "Residual length $n not divisible by block_size=$block_size"
    m = div(n, block_size)
    M = zeros(eltype(resid), block_size, block_size)
    @inbounds for b in 0:m-1
        r = @view resid[b*block_size + 1 : (b+1)*block_size]
        M .+= r * r'
    end
    M ./= m
    return M
end

"Centered, Bessel-corrected mean of clusterwise outer products of length `block_size` chunks."
function arithmetic_mean_outer_products_without_demeaning(resid::AbstractVector, block_size::Int)
    n = length(resid)
    @assert n % block_size == 0
    m = div(n, block_size)

    μ = zeros(eltype(resid), block_size)
    @inbounds for b in 0:m-1
        μ .+= @view resid[b*block_size + 1 : (b+1)*block_size]
    end
    μ ./= m

    M = zeros(eltype(resid), block_size, block_size)
    @inbounds for b in 0:m-1
        r = @view resid[b*block_size + 1 : (b+1)*block_size]
        d = r .- μ
        M .+= d * d'
    end
    M ./= (m - 1)
    return M
end

"""
Two-step averaging: average residual vectors across the *constant* dimension
before forming outer products, then average the resulting matrices.

Assumes resid is ordered with i fastest, then j, then t.
So reshape(resid, N1, N2, T) gives A[i,j,t].
"""
function two_step_mean_outer_products(resid::AbstractVector, component::Symbol,
                                      N1::Int, N2::Int, T::Int)

    A = reshape(resid, N1, N2, T)  # i, j, t

    if component === :i
        X = dropdims(mean(A; dims=3), dims=3)           # N1×N2  (columns = j replicates)
        return (X * X') / N2

    elseif component === :j
        Y = dropdims(mean(A; dims=1), dims=1)           # N2×T   (columns = t replicates)
        return (Y * Y') / T

    elseif component === :t
        Y = dropdims(mean(A; dims=1), dims=1)           # N2×T
        W = permutedims(Y, (2, 1))                      # T×N2   (columns = j replicates)
            return (W * W') / N2
    else
        error("component must be :i, :j, or :t")
    end
end 

function two_step_mean_outer_products_with_demeaning(resid::AbstractVector, component::Symbol,
                                      N1::Int, N2::Int, T::Int)

    @assert length(resid) == N1 * N2 * T "resid length must be N1*N2*T"
    @assert component === :i || component === :j || component === :t
    A = reshape(resid, N1, N2, T)  # i, j, t

    if component === :i
        # Average across t to get N2 replicate i-vectors (columns = j)
        X = dropdims(mean(A; dims=3), dims=3)           # N1×N2
        μ = mean(X; dims=2)                             # N1×1 (mean across columns j)
        Xc = X .- μ                                     # column-centered
        @assert N2 ≥ 2 "Need at least 2 j-replicates for demeaning"
        return (Xc * Xc') / (N2 - 1)

    elseif component === :j
        # Average across i to get T replicate j-vectors (columns = t)
        Y = dropdims(mean(A; dims=1), dims=1)           # N2×T
        μ = mean(Y; dims=2)                             # N2×1 (mean across columns t)
        Yc = Y .- μ
        @assert T ≥ 2 "Need at least 2 t-replicates for demeaning"
        return (Yc * Yc') / (T - 1)

    else # component === :t
        # Average across i to get N2 replicate t-vectors (columns = j after permute)
        Y = dropdims(mean(A; dims=1), dims=1)           # N2×T
        W = permutedims(Y, (2, 1))                      # T×N2
        μ = mean(W; dims=2)                             # T×1 (mean across columns j)
        Wc = W .- μ
        @assert N2 ≥ 2 "Need at least 2 j-replicates for demeaning"
        return (Wc * Wc') / (N2 - 1)
    end
end

"σ²_u and σ²_α/γ/λ via differences of within-residual means."
function estimate_homoskedastic_component_variances(df::DataFrame, N1::Int, N2::Int, T::Int,
                                                    beta_hat::Real; x_col::Symbol=:x, y_col::Symbol=:y)
    ensure_tilde_columns!(df; x_col=x_col, y_col=y_col,
                          suffixes=["alpha_gamma_lambda","alpha_gamma","alpha_lambda","gamma_lambda"])
    res_full = residuals_from_suffix(df, "alpha_gamma_lambda", beta_hat; x_col=x_col, y_col=y_col)
    σu2 = mean(res_full .^ 2)

    res_l = residuals_from_suffix(df, "alpha_gamma", beta_hat; x_col=x_col, y_col=y_col)
    σλ2 = mean(res_l .^ 2) - σu2

    res_g = residuals_from_suffix(df, "alpha_lambda", beta_hat; x_col=x_col, y_col=y_col)
    σγ2 = mean(res_g .^ 2) - σu2

    res_a = residuals_from_suffix(df, "gamma_lambda", beta_hat; x_col=x_col, y_col=y_col)
    σα2 = mean(res_a .^ 2) - σu2

    return (; sigma_u2 = σu2, sigma_alpha2 = σα2, sigma_gamma2 = σγ2, sigma_lambda2 = σλ2)
end


"""
    generate_single_component_omega(df, component, N1, N2, T, sigma_u2, beta_hat;
                                    x_col=:x, y_col=:y,
                                    subtract_sigma=false, do_ridge=false)

Generate one base Ω_block for `component ∈ (:i, :j, :t)` **without sorting df**.
Assumes `df` rows are already ordered so that when you vectorize, `i` varies fastest,
then `j`, then `t` (i.e., consistent with your DGP and `reshape(res, N1, N2, T)`).

Estimator: raw moment (divide by the number of “other-dimension” combinations).

- :i → Ω_i = (B * B') / (N2*T), where B = reshape(A, N1, N2*T)
- :j → Ω_j = (B * B') / (N1*T), where B = reshape(permutedims(A,(2,1,3)), N2, N1*T)
- :t → Ω_t = (B * B') / (N1*N2), where B = reshape(permutedims(A,(3,2,1)), T, N2*N1)
"""
function generate_single_component_omega(df::DataFrame, component::Symbol, N1::Int, N2::Int, T::Int,
                                         sigma_u2::Real, beta_hat::Real;
                                         x_col::Symbol=:x, y_col::Symbol=:y,
                                         do_ridge::Bool=false)

    # Pick the residual suffix for the component (no sorting; use df as-is)
    suffix = component === :i ? "gamma_lambda" :
             component === :j ? "alpha_lambda" :
             component === :t ? "alpha_gamma" :
             error("component must be :i, :j, or :t")

    ensure_tilde_columns!(df; x_col=x_col, y_col=y_col, suffixes=[suffix])
    res = residuals_from_suffix(df, suffix, beta_hat; x_col=x_col, y_col=y_col)
    @assert length(res) == N1 * N2 * T "Residual vector has wrong length."

    # Reshape to A[i,j,t] under the canonical DGP order (i fastest → j → t)
    A = reshape(res, N1, N2, T)

    # Raw-moment estimator via matrix products (fast, no chunking, no sorting)
    Ω = if component === :i
        # Stack (j,t) along columns: N1 × (N2*T)
        B = reshape(A, N1, N2*T)
        (B * B') / (N2 * T)
    elseif component === :j
        # Move j to rows: N2 × (N1*T)
        B = reshape(permutedims(A, (2, 1, 3)), N2, N1*T)
        (B * B') / (N1 * T)
    else # :t
        # Move t to rows: T × (N2*N1)
        B = reshape(permutedims(A, (3, 2, 1)), T, N2*N1)
        (B * B') / (N1 * N2)
    end

    return Ω
end

"""
two_step_sigma_u(res, component, N1, N2, T)

Compute σ̂_u^2 for a component using the two-step idea:
  - Average-of-vectors is done over a 'repeat' dimension R (i: R=T; j: R=N1; t: R=N1 under current design).
  - Here we estimate σ̂_u^2 as the average of within-cell sample variances across that repeat dimension.

Assumes df has already been sorted for `component` before residuals `res` were built:
  - :i  → sort! by [:t, :j, :i] ⇒ reshape (N1, N2, T), take var over dim=3 (t)
  - :j  → sort! by [:t, :i, :j] ⇒ reshape (N2, N1, T), take var over dim=2 (i)
  - :t  → sort! by [:j, :i, :t] ⇒ reshape (T,  N2, N1), take var over dim=3 (i)

Returns: (σ2::Float64, R::Int, df_weight::Int)
"""
function two_step_sigma_u(res::AbstractVector{<:Real}, component::Symbol, N1::Int, N2::Int, T::Int)
    @assert length(res) == N1*N2*T "res length mismatch with N1*N2*T"

    if component === :i
        # shape: [i, j, t], variance over t
        A = reshape(res, N1, N2, T)
        # sample var across t for each (i,j)
        # var over last dim: mean((x - mean)^2) * R/(R-1)
        σ2_cells = map(1:N1, 1:N2) do i, j
            x = @view A[i, j, :]
            T > 1 ? var(x; corrected=true) : 0.0
        end
        σ2_hat = mean(σ2_cells)
        R = T
        dfw = (N1*N2)*max(T-1, 0)
        return (max(σ2_hat, 0.0), R, dfw)

    elseif component === :j
        # shape: [j, i, t], variance over i
        A = reshape(res, N2, N1, T)
        σ2_cells = map(1:N2, 1:N1, 1:T) do j, i, t
            # We'll compute var across i by building all i for each (j,t) below
            nothing
        end
        # Build (j,t) panels and take var across i
        σsum = 0.0; cnt = 0
        if N1 > 1
            for t in 1:T, j in 1:N2
                x = @view A[j, :, t]
                σsum += var(x; corrected=true)
                cnt += 1
            end
        end
        σ2_hat = (cnt > 0 ? σsum / cnt : 0.0)
        R = N1
        dfw = (N2*T)*max(N1-1, 0)
        return (max(σ2_hat, 0.0), R, dfw)

    elseif component === :t
        # shape: [t, j, i], variance over i (your current design averages over i)
        A = reshape(res, T, N2, N1)
        σsum = 0.0; cnt = 0
        if N1 > 1
            for t in 1:T, j in 1:N2
                x = @view A[t, j, :]
                σsum += var(x; corrected=true)
                cnt += 1
            end
        end
        σ2_hat = (cnt > 0 ? σsum / cnt : 0.0)
        R = N1
        dfw = (N2*T)*max(N1-1, 0)
        return (max(σ2_hat, 0.0), R, dfw)

    else
        error("Unknown component: $component. Use :i, :j, or :t.")
    end
end

"""
generate_single_component_omega_twostep(df, component, N1, N2, T; x_col, y_col, beta_hat, return_sigma=false)

Build the two-step block Ω_component AND (optionally) return (σ̂2_component, R_component, df_weight_component)
computed from the same residuals used for the two-step averaging.

If return_sigma=false (default) → returns Ω
If return_sigma=true          → returns (Ω, σ̂2_component, R_component, df_weight_component)
"""
function generate_single_component_omega_twostep(
    df::DataFrame, component::Symbol, N1::Int, N2::Int, T::Int;
    x_col::Symbol, y_col::Symbol, beta_hat::Real, return_sigma::Bool=false
)
    # Always sort rows so that i is fastest in the flattened vector:
    dfo = sort(df, [:t, :j, :i])

    if component === :i
        ensure_tilde_columns!(dfo; x_col=x_col, y_col=y_col, suffixes=["gamma_lambda"])
        res = residuals_from_suffix(dfo, "gamma_lambda", beta_hat; x_col=x_col, y_col=y_col)

    elseif component === :j
        ensure_tilde_columns!(dfo; x_col=x_col, y_col=y_col, suffixes=["alpha_lambda"])
        res = residuals_from_suffix(dfo, "alpha_lambda", beta_hat; x_col=x_col, y_col=y_col)

    elseif component === :t
        ensure_tilde_columns!(dfo; x_col=x_col, y_col=y_col, suffixes=["alpha_gamma"])
        res = residuals_from_suffix(dfo, "alpha_gamma", beta_hat; x_col=x_col, y_col=y_col)

    else
        error("Unknown component: $component")
    end

    Ω = two_step_mean_outer_products(res, component, N1, N2, T)

    if return_sigma
        σ2, R, dfw = two_step_sigma_u(res, component, N1, N2, T)
        return (Ω, σ2, R, dfw)
    else
        return Ω
    end
end


"""
Estimate base blocks (Ωα, Ωγ, Ωλ) using per-dimension switches.

Single-step (two_step=false):
  - Uses residual outer-products (no pre-averaging).
  - Returns base-sized blocks WITHOUT repeat expansion.

Two-step (two_step=true):
  - For each estimated component, average residual vectors over the repeated
    dimension R (i: R=T; j: R=N1; t: R=N1), form outer products of those means,
    then average matrices.
  - Computes a df-weighted pooled σ̂_u^2 from per-cell sample variances across
    the repeated dimension.
  - Subtracts (sigma_damp * σ̂_u^2 / R) from the diagonal of each estimated
    base block BEFORE any SPD projection/repeats.
  - If `return_sigma=true`, also returns the pooled σ̂_u^2.

Signature kept backward-compatible with your old calls.
"""
function estimate_omegas(df::DataFrame, N1::Int, N2::Int, T::Int,
                         sigmas::NamedTuple, beta_hat::Real;
                         x_col::Symbol=:x, y_col::Symbol=:y,
                         i_block_est::Bool=true,
                         j_block_est::Bool=true,
                         t_block_est::Bool=true,
                         two_step::Bool=false,
                         return_sigma::Bool=false,
                         subtract_sigma_u2::Bool=true,
                         sigma_damp::Real=1.0)

    if !two_step
        # --- single-step procedure ---
        σu2 = sigmas.sigma_u2

        Ωa = 
            if i_block_est
                Ωα = generate_single_component_omega(df, :i, N1, N2, T, σu2, beta_hat; x_col=x_col, y_col=y_col)
                if subtract_sigma_u2
                    # subtract σ̂_u^2 / T from diag
                    @views Ωα[diagind(Ωα)] .-= sigma_damp * (σu2 / T)
                end
                Symmetric(Matrix(Ωα))
            else
                sigmas.sigma_alpha2 * I(N1)
            end

        Ωg = 
            if j_block_est
                Ωγ = generate_single_component_omega(df, :j, N1, N2, T, σu2, beta_hat; x_col=x_col, y_col=y_col)
                if subtract_sigma_u2
                    # subtract σ̂_u^2 / N1 from diag
                    @views Ωγ[diagind(Ωγ)] .-= sigma_damp * (σu2 / N1)
                end
                Symmetric(Matrix(Ωγ))
            else
                sigmas.sigma_gamma2 * I(N2)
            end

        Ωl = 
            if t_block_est
                Ωλ = generate_single_component_omega(df, :t, N1, N2, T, σu2, beta_hat; x_col=x_col, y_col=y_col)
                if subtract_sigma_u2
                    # subtract σ̂_u^2 / N1 from diag
                    @views Ωλ[diagind(Ωλ)] .-= sigma_damp * (σu2 / N1)
                end
                Symmetric(Matrix(Ωλ))
            else
                sigmas.sigma_lambda2 * I(T)
            end

        return (; Ωa, Ωg, Ωl)
    end

    # =========================
    # Two-step path (FGLS2)
    # =========================

    # Accumulators for pooled σ̂_u^2
    σ2_parts = Float64[]   # component-specific σ̂_u^2
    dfw      = Int[]       # df weights used for pooling

    # Keep component-specific sigmas in scope for debug/diagonal subtraction
    σ2α = 0.0; σ2γ = 0.0; σ2λ = 0.0

    var_over_last(x) = var(x; corrected=true)  # sample variance

    # i / α : mean over t for each (i,j); outer product averaged over j
    Ωa = if i_block_est
        dfi = sort_for_dim(df, :i)
        ensure_tilde_columns!(dfi; x_col=x_col, y_col=y_col, suffixes=["gamma_lambda"])
        res = residuals_from_suffix(dfi, "gamma_lambda", beta_hat; x_col=x_col, y_col=y_col)

        Ωα = two_step_mean_outer_products(res, :i, N1, N2, T)

        # σ̂_{u,α}^2: avg sample variance across t within (i,j)
        σsum = 0.0; cells = 0
        if T > 1
            A = reshape(res, N1, N2, T)
            @inbounds for i in 1:N1, j in 1:N2
                σsum += var_over_last(@view A[i, j, :])
                cells += 1
            end
        end
        σ2α = cells > 0 ? σsum / cells : 0.0
        push!(σ2_parts, max(σ2α, 0.0))
        push!(dfw, (N1*N2)*max(T-1, 0))
        if subtract_sigma_u2 && σ2α > 0
            @views Ωα[diagind(Ωα)] .-= sigma_damp * (σ2α / T)
        end
        Symmetric(Matrix(Ωα))
    else
        sigmas.sigma_alpha2 * I(N1)
    end

    # j / γ : mean over i for each (j,t); outer product averaged over t
    Ωg = if j_block_est
        dfj = sort_for_dim(df, :j)
        ensure_tilde_columns!(dfj; x_col=x_col, y_col=y_col, suffixes=["alpha_lambda"])
        res = residuals_from_suffix(dfj, "alpha_lambda", beta_hat; x_col=x_col, y_col=y_col)

        Ωγ = two_step_mean_outer_products(res, :j, N1, N2, T)

        σsum = 0.0; cells = 0
        if N1 > 1
            A = reshape(res, N2, N1, T)  # [j, i, t]
            @inbounds for t in 1:T, j in 1:N2
                σsum += var_over_last(@view A[j, :, t])  # variance across i
                cells += 1
            end
        end
        σ2γ = cells > 0 ? σsum / cells : 0.0
        push!(σ2_parts, max(σ2γ, 0.0))
        push!(dfw, (N2*T)*max(N1-1, 0))
        if subtract_sigma_u2 && σ2γ > 0
            @views Ωγ[diagind(Ωγ)] .-= sigma_damp * (σ2γ / N1)
        end
        Symmetric(Matrix(Ωγ))
    else
        sigmas.sigma_gamma2 * I(N2)
    end

    # t / λ : mean over i for each (t,j); outer product averaged over j
    Ωl = if t_block_est
        dft = sort_for_dim(df, :t)
        ensure_tilde_columns!(dft; x_col=x_col, y_col=y_col, suffixes=["alpha_gamma"])
        res = residuals_from_suffix(dft, "alpha_gamma", beta_hat; x_col=x_col, y_col=y_col)

        Ωλ = two_step_mean_outer_products(res, :t, N1, N2, T)

        σsum = 0.0; cells = 0
        if N1 > 1
            A = reshape(res, T, N2, N1)  # [t, j, i]
            @inbounds for t in 1:T, j in 1:N2
                σsum += var_over_last(@view A[t, j, :])  # variance across i
                cells += 1
            end
        end
        σ2λ = cells > 0 ? σsum / cells : 0.0
        push!(σ2_parts, max(σ2λ, 0.0))
        push!(dfw, (N2*T)*max(N1-1, 0))
        if subtract_sigma_u2 && σ2λ > 0
            @views Ωλ[diagind(Ωλ)] .-= sigma_damp * (σ2λ / N1)
        end
        Symmetric(Matrix(Ωλ))
    else
        sigmas.sigma_lambda2 * I(T)
    end

    # --- pooled σ̂_u^2 (df-weighted across available components) ---
    total_df = sum(dfw)
    σ2_u = total_df > 0 ? sum(σ2_parts .* dfw) / total_df : 0.0
    σ2_u = max(σ2_u, 0.0)
    

    return return_sigma ? (; Ωa, Ωg, Ωl, sigma_u2 = σ2_u, sigma_alpha2 = σ2α, sigma_gamma2 = σ2γ, sigma_lambda2 = σ2λ) : (; Ωa, Ωg, Ωl)
end

"Build Sα, Sγ, Sλ per your repeat rules (n × cols)."
function make_S_matrices(N1::Int, N2::Int, T::Int;
                         repeat_alpha::Bool, repeat_gamma::Bool, repeat_lambda::Bool)
    n = N1 * N2 * T
    # α
    Sα = repeat_alpha ?
         kron(kron(ones(T,1), I(N2)), I(N1)) :
         kron(kron(ones(T,1), ones(N2,1)), I(N1))
    @assert size(Sα,1) == n
    # γ
    Sγ = repeat_gamma ?
         kron(kron(I(T), I(N2)), ones(N1,1)) :
         kron(kron(ones(T,1), I(N2)), ones(N1,1))
    @assert size(Sγ,1) == n
    # λ
    Sλ = repeat_lambda ?
         kron(kron(I(T), I(N2)), ones(N1,1)) :
         kron(kron(I(T), ones(N2,1)), ones(N1,1))
    @assert size(Sλ,1) == n
    return Sα, Sγ, Sλ
end

"Construct Ω = Sα Ωα Sα' + Sγ Ωγ Sγ' + Sλ Ωλ Sλ' + σ²_u I."
function construct_omega(Ωa::AbstractMatrix, Ωg::AbstractMatrix, Ωl::AbstractMatrix,
                         Sα::AbstractMatrix, Sγ::AbstractMatrix, Sλ::AbstractMatrix,
                         sigma_u2::Real)
    n = size(Sα,1)
    Ω = Sα * Ωa * Sα'
    Ω .+= Sγ * Ωg * Sγ'
    Ω .+= Sλ * Ωl * Sλ'
    Ω .+= sigma_u2 .* I(n)
    return Symmetric(Ω)  
end

function shrink_offdiagonal!(Ω::AbstractMatrix, α::Real)
    @assert 0.0 ≤ α ≤ 1.0 "fgls_shrinkage α must be in [0,1]"
    # If Ω is Symmetric, operate on its parent data
    A = Ω isa Symmetric ? parent(Ω) : Ω
    n, m = size(A); @assert n == m "Ω must be square"

    # save diagonal, scale everything, then restore diagonal
    d = copy(@view A[diagind(A)])
    @. A = α * A
    @view(A[diagind(A)]) .= d

    return Symmetric(A)
end

"Project Ω to SPD by flooring eigenvalues at `floor`."
function project_to_spd(Ω::AbstractMatrix; floor::Real=1e-8)
    F = eigen(Symmetric(Matrix(Ω)))
    d = map(x -> max(x, floor), F.values)
    return F.vectors * Diagonal(d) * F.vectors'
end

end # module
