# rc_functions.jl
module RCFunctions

using Random, LinearAlgebra, Distributions, DataFrames

export random_spd, homoskedastic_diag, mvn_draw, build_ids,
       assert_drawmode_invariance

"Make a random SPD matrix with average diagonal scaled to `variance`."
function random_spd(n::Int; variance::Real=1.0, rng::AbstractRNG=Random.GLOBAL_RNG)
    A = randn(rng, n, n)
    Σ = (A * A') / n
    s = variance / mean(diag(Σ))
    Σ .*= s
    return Symmetric(Σ)
end

"Return homoskedastic diagonal covariance σ²·Iₙ."
homoskedastic_diag(n::Int; σ::Real=1.0) = Diagonal(fill(σ^2, n))

"Draw from MVN(mean_vec, Σ). Accepts Σ as SPD or diagonal."
function mvn_draw(mean_vec::AbstractVector, Σ::AbstractMatrix;
                  rng::AbstractRNG=Random.GLOBAL_RNG)
    d = MvNormal(mean_vec, Σ)
    return rand(rng, d)
end

"Build lexicographic ids with i varying fastest, then j, then t."
function build_ids(N1::Int, N2::Int, T::Int)
    i_vec = repeat(collect(1:N1), N2*T)
    j_vec = repeat(repeat(collect(1:N2), inner=N1), T)
    t_vec = repeat(collect(1:T), inner=N1*N2)
    return i_vec, j_vec, t_vec
end

"""
Assert that the realized FE columns satisfy the draw-mode invariances.

- For :draw_once: FE for a given key is constant across the other two dims.
- For :mixed:
    i: constant over t within (i,j)
    j: constant over i within (j,t)
    t: constant over i within (j,t)  (i.e., per j)
- For :full_redraw: no invariance across other dims.
"""
function assert_drawmode_invariance(df::DataFrame;
        i_draw_mode::Symbol, j_draw_mode::Symbol, t_draw_mode::Symbol)

    # true if `col` is constant within each group defined by `keys`
    constant_within(df, keys::Vector{Symbol}, col::Symbol; atol=1e-10) = begin
        all([all(isapprox.(g[!, col], first(g[!, col]); atol=atol)) for g in groupby(df, keys)])
    end

    # i
    if i_draw_mode == :draw_once
        @assert constant_within(df, [:i], :fe_i) "i:draw_once failed (fe_i not constant across j,t for each i)"
    elseif i_draw_mode == :mixed
        @assert constant_within(df, [:i, :t], :fe_i) "i:mixed failed (fe_i not constant over j for each (i,t))"
    elseif i_draw_mode == :full_redraw
        # no constraint
    else
        error("Unknown i_draw_mode = $i_draw_mode")
    end

    # j
    if j_draw_mode == :draw_once
        @assert constant_within(df, [:j], :fe_j) "j:draw_once failed (fe_j not constant across i,t for each j)"
    elseif j_draw_mode == :mixed
        @assert constant_within(df, [:j, :t], :fe_j) "j:mixed failed (fe_j not constant over i for each (j,t))"
    elseif j_draw_mode == :full_redraw
        # no constraint
    else
        error("Unknown j_draw_mode = $j_draw_mode")
    end

    # t
    if t_draw_mode == :draw_once
        @assert constant_within(df, [:t], :fe_t) "t:draw_once failed (fe_t not constant across i,j for each t)"
    elseif t_draw_mode == :mixed
        @assert constant_within(df, [:j, :t], :fe_t) "t:mixed failed (fe_t not constant over i for each (j,t))"
    elseif t_draw_mode == :full_redraw
        # no constraint
    else
        error("Unknown t_draw_mode = $t_draw_mode")
    end

    return true
end

end # module
