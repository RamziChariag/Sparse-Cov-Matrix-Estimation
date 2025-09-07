# smoke_test.jl
module RCSmokeTest

using DataFrames, LinearAlgebra, Statistics, LaTeXStrings

import ..RCParams
import ..RCIO
import ..RCFunctions: assert_drawmode_invariance
import ..RCDGP: generate_dataset

export run_smoke_test!


_sample_size_grid(p) = [(N2 = p.start_N2 + (k-1)*p.N2_increment,
                         T  = p.start_T  + (k-1)*p.T_increment) for k in 1:p.num_sample_sizes]
diagmean(A) = mean(@view A[diagind(A)])

# Pretty-print a matrix without truncation
function _print_omega(name::AbstractString, Ω)
    io = IOContext(stdout, :limit=>false, :compact=>false)
    println("\n", name, " (", size(Ω,1), "×", size(Ω,2), "):")
    show(io, "text/plain", Matrix(Ω))   # ensure full matrix shown
    println()
end

function run_smoke_test!(; params::NamedTuple = RCParams.PARAMS, head::Int=10, save::Bool=false, s::Int=1)
    p = params
    grid = _sample_size_grid(p)
    N2, T = grid[s].N2, grid[s].T   #s is the sample size and can go from 1 to p.num_sample_sizes
    println("Testing DGP with N1=$(p.N1), N2=$N2, T=$T, seed=$(p.seed)")

    df, meta = generate_dataset(
        N1=p.N1, N2=N2, T=T,
        i_block=p.i_block, j_block=p.j_block, t_block=p.t_block,
        i_draw_mode=p.i_draw_mode, j_draw_mode=p.j_draw_mode, t_draw_mode=p.t_draw_mode,
        E_i=p.E_i, E_j=p.E_j, E_t=p.E_t,
        sigma_i=p.sigma_i, sigma_j=p.sigma_j, sigma_t=p.sigma_t,
        mu_x=p.mu_x, sigma_x=p.sigma_x,
        mu_u=p.mu_u, sigma_u=p.sigma_u,
        seed=p.seed
    )

    println("\nHead of generated data:")
    first(df, min(head, nrow(df))) |> println

    println("\nChecking draw-mode invariances...")
    @assert assert_drawmode_invariance(df;
        i_draw_mode=p.i_draw_mode, j_draw_mode=p.j_draw_mode, t_draw_mode=p.t_draw_mode)
    println("✓ invariance checks passed.")

    println("\nDimension checks:")
    @assert nrow(df) == p.N1 * N2 * T "Row count mismatch: got $(nrow(df)), expected $(p.N1 * N2 * T)"
    required = Set(Symbol.([:i, :j, :t, :x, :u_ijt, :fe_i, :fe_j, :fe_t]))
    present  = Set(Symbol.(names(df)))
    missing  = collect(setdiff(required, present))
    @assert isempty(missing) "Missing columns: $(missing)"
    println("✓ rows = $(nrow(df))")
    println("Columns present: ", Symbol.(names(df)))

    println("\nDiagonal means:")
    println("Ω_i ≈ ", diagmean(meta.Ωi), "   Ω_j ≈ ", diagmean(meta.Ωj), "   Ω_t ≈ ", diagmean(meta.Ωt))

    # === Print the three Ω blocks ===
    _print_omega("Ω_i", meta.Ωi)
    _print_omega("Ω_j", meta.Ωj)
    _print_omega("Ω_t", meta.Ωt)
    
    if save
           sizes = (; N1 = p.N1, N2 = N2, T = T)
       path = RCIO.save_dataset!(df, meta, sizes; params=p, suffix="_ST")
       println("\nSaved dataset to: ", path)
    end

    println("\nSmoke test complete.")
    return nothing
end

end # module
