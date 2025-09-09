# dgp.jl
module RCDGP

using Random, LinearAlgebra, Distributions, DataFrames
import ..RCFunctions: random_spd, homoskedastic_diag, mvn_draw, build_ids, assert_drawmode_invariance

export generate_dataset, build_cov_mats

"Build Ω_i, Ω_j, Ω_t based on {i,j,t}_block flags and σ_* scales."
function build_cov_mats(N1::Int, N2::Int, T::Int;
        i_block::Bool, j_block::Bool, t_block::Bool,
        sigma_i::Real, sigma_j::Real, sigma_t::Real,
        rng::AbstractRNG)

    Ωi = i_block ? random_spd(N1; variance=sigma_i^2, rng=rng) : homoskedastic_diag(N1; σ=sigma_i)
    Ωj = j_block ? random_spd(N2; variance=sigma_j^2, rng=rng) : homoskedastic_diag(N2; σ=sigma_j)
    Ωt = t_block ? random_spd(T ; variance=sigma_t^2, rng=rng) : homoskedastic_diag(T ; σ=sigma_t)
    return Ωi, Ωj, Ωt
end

"""
Generate one full i–j–t panel dataset with:

- ids (i,j,t) in lexicographic order (i fastest),
- x ~ i.i.d. Normal(mu_x, sigma_x),
- u ~ i.i.d. Normal(mu_u, sigma_u),
- fe_i, fe_j, fe_t drawn according to covariance and draw modes.

Returns (df, meta) where meta contains Ωi, Ωj, Ωt and the RNG seed used.
"""
function generate_dataset(; 
        N1::Int, N2::Int, T::Int,
        i_block::Bool, j_block::Bool, t_block::Bool,
        i_draw_mode::Symbol, j_draw_mode::Symbol, t_draw_mode::Symbol,
        E_i::Real, E_j::Real, E_t::Real,
        sigma_i::Real, sigma_j::Real, sigma_t::Real,
        mu_x::Real, sigma_x::Real,
        mu_u::Real, sigma_u::Real,
        seed::Integer = 42
    )

    rng = MersenneTwister(seed)
    n = N1 * N2 * T

    # IDs
    i_vec, j_vec, t_vec = build_ids(N1, N2, T)

    # Covariance matrices
    Ωi, Ωj, Ωt = build_cov_mats(N1,N2,T;
        i_block=i_block, j_block=j_block, t_block=t_block,
        sigma_i=sigma_i, sigma_j=sigma_j, sigma_t=sigma_t,
        rng=rng)

    # FE means
    μi = fill(E_i, N1)
    μj = fill(E_j, N2)
    μt = fill(E_t, T)

    # --- fe_i draws: i fastest, then j, then t ---
    fe_i_tensor = zeros(Float64, N1, N2, T)
    if i_draw_mode == :draw_once
        v = mvn_draw(μi, Ωi; rng=rng)          # length N1
        @inbounds for t in 1:T, j in 1:N2, i in 1:N1
            fe_i_tensor[i, j, t] = v[i]
        end
    elseif i_draw_mode == :mixed
        @inbounds for j in 1:N2
            v = mvn_draw(μi, Ωi; rng=rng)      # redraw per j
            for t in 1:T, i in 1:N1
                fe_i_tensor[i, j, t] = v[i]
            end
        end
    elseif i_draw_mode == :full_redraw
        @inbounds for t in 1:T, j in 1:N2
            v = mvn_draw(μi, Ωi; rng=rng)      # redraw per (j,t)
            for i in 1:N1
                fe_i_tensor[i, j, t] = v[i]
            end
        end
    else
        error("Invalid i_draw_mode = $i_draw_mode")
    end

    # --- fe_j draws: i fastest, then j, then t ---
    fe_j_tensor = zeros(Float64, N1, N2, T)
    if j_draw_mode == :draw_once
        v = mvn_draw(μj, Ωj; rng=rng)          # length N2
        @inbounds for t in 1:T, j in 1:N2, i in 1:N1
            fe_j_tensor[i, j, t] = v[j]
        end
    elseif j_draw_mode == :mixed
        @inbounds for t in 1:T
            v = mvn_draw(μj, Ωj; rng=rng)      # redraw per t
            for j in 1:N2, i in 1:N1
                fe_j_tensor[i, j, t] = v[j]
            end
        end
    elseif j_draw_mode == :full_redraw
        @inbounds for t in 1:T, i in 1:N1
            v = mvn_draw(μj, Ωj; rng=rng)      # redraw per (i,t)
            for j in 1:N2
                fe_j_tensor[i, j, t] = v[j]
            end
        end
    else
        error("Invalid j_draw_mode = $j_draw_mode")
    end

    # --- fe_t draws: i fastest, then j, then t ---
    fe_t_tensor = zeros(Float64, N1, N2, T)
    if t_draw_mode == :draw_once
        v = mvn_draw(μt, Ωt; rng=rng)          # length T
        @inbounds for t in 1:T, j in 1:N2, i in 1:N1
            fe_t_tensor[i, j, t] = v[t]
        end
    elseif t_draw_mode == :mixed
        @inbounds for j in 1:N2
            v = mvn_draw(μt, Ωt; rng=rng)      # redraw per j
            for t in 1:T, i in 1:N1
                fe_t_tensor[i, j, t] = v[t]
            end
        end
    elseif t_draw_mode == :full_redraw
        @inbounds for j in 1:N2, i in 1:N1
            v = mvn_draw(μt, Ωt; rng=rng)      # redraw per (i,j)
            for t in 1:T
                fe_t_tensor[i, j, t] = v[t]
            end
        end
    else
        error("Invalid t_draw_mode = $t_draw_mode")
    end


    # Flatten tensors to vectors in (i fast, then j, then t) order
    to_vec(A) = vec(reshape(A, :, 1))
    fe_i_vec = to_vec(fe_i_tensor)
    fe_j_vec = to_vec(fe_j_tensor)
    fe_t_vec = to_vec(fe_t_tensor)

    # x and u
    x = rand(rng, Normal(mu_x, sigma_x), n)
    u = rand(rng, Normal(mu_u, sigma_u), n)

    df = DataFrame(i=i_vec, j=j_vec, t=t_vec,
                   x=x, u_ijt=u,
                   fe_i=fe_i_vec, fe_j=fe_j_vec, fe_t=fe_t_vec)

    meta = (; Ωi=Ωi, Ωj=Ωj, Ωt=Ωt, seed=seed)
    return df, meta
end

end # module
