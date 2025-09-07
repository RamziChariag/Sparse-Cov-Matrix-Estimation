# omega_estimators.jl
module RCOmegaEstimators

using LinearAlgebra, Statistics, DataFrames

export plot_matrix_percentile, omega_eigen_tables,
       residuals_from_suffix, ensure_tilde_columns!,
       estimate_homoskedastic_component_variances,
       sort_for_dim, arithmetic_mean_outer_products,
       generate_threeway_omegas, generate_single_component_omega,
       estimate_omegas, make_S_matrices, construct_omega, repeat_block,
       shrink_offdiagonal!

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


"Generate only one base block (no repeat expansion). component ∈ (:i,:j,:t)."
function generate_single_component_omega(df::DataFrame, component::Symbol, N1::Int, N2::Int, T::Int,
                                         sigma_u2::Real, beta_hat::Real; x_col::Symbol=:x, y_col::Symbol=:y)
    if component === :i
        dfi = sort_for_dim(df, :i)
        ensure_tilde_columns!(dfi; x_col=x_col, y_col=y_col, suffixes=["gamma_lambda"])
        res = residuals_from_suffix(dfi, "gamma_lambda", beta_hat; x_col=x_col, y_col=y_col)
        Ω = arithmetic_mean_outer_products(res, N1)
        #@views Ω[diagind(Ω)] .-= sigma_u2
        #Ω = make_psd_by_minimal_ridge(Ω; strict=true)
    elseif component === :j
        dfj = sort_for_dim(df, :j)
        ensure_tilde_columns!(dfj; x_col=x_col, y_col=y_col, suffixes=["alpha_lambda"])
        res = residuals_from_suffix(dfj, "alpha_lambda", beta_hat; x_col=x_col, y_col=y_col)
        Ω = arithmetic_mean_outer_products(res, N2)
        #@views Ω[diagind(Ω)] .-= sigma_u2
        #Ω = make_psd_by_minimal_ridge(Ω; strict=true)
    elseif component === :t
        dft = sort_for_dim(df, :t)
        ensure_tilde_columns!(dft; x_col=x_col, y_col=y_col, suffixes=["alpha_gamma"])
        res = residuals_from_suffix(dft, "alpha_gamma", beta_hat; x_col=x_col, y_col=y_col)
        Ω = arithmetic_mean_outer_products(res, T)
        #@views Ω[diagind(Ω)] .-= sigma_u2
        #Ω = make_psd_by_minimal_ridge(Ω; strict=true)
    else
        error("component must be :i, :j, or :t")
    end

    return Ω
end

"""
Estimate base blocks (Ωα, Ωγ, Ωλ) using per-dimension switches:

- i_block_est=true  ⇒ estimate full SPD block for i (via residual outer products)
                    false ⇒ homoskedastic diagonal (σ²_α I)
- j_block_est=true  ⇒ full SPD for j; false ⇒ σ²_γ I
- t_block_est=true  ⇒ full SPD for t; false ⇒ σ²_λ I

Returns the base-sized blocks WITHOUT repeat expansion.
"""
function estimate_omegas(df::DataFrame, N1::Int, N2::Int, T::Int,
                         sigmas::NamedTuple, beta_hat::Real;
                         x_col::Symbol=:x,
                         i_block_est::Bool=true,
                         j_block_est::Bool=true,
                         t_block_est::Bool=true)

    σu2 = sigmas.sigma_u2

    # i / α
    Ωa = if i_block_est
        generate_single_component_omega(df, :i, N1, N2, T, σu2, beta_hat; x_col=x_col)
    else
        sigmas.sigma_alpha2 * I(N1)
    end

    # j / γ
    Ωg = if j_block_est
        generate_single_component_omega(df, :j, N1, N2, T, σu2, beta_hat; x_col=x_col)
    else
        sigmas.sigma_gamma2 * I(N2)
    end

    # t / λ
    Ωl = if t_block_est
        generate_single_component_omega(df, :t, N1, N2, T, σu2, beta_hat; x_col=x_col)
    else
        sigmas.sigma_lambda2 * I(T)
    end

    return (; Ωa, Ωg, Ωl)
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
    return Symmetric(Ω)  # helps the solver
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
