# mc_driver.jl (replace the file with this version or patch run_mc! accordingly)
module RCMonteCarlo

using Random, Statistics, LinearAlgebra, DataFrames
import ..RCParams
import ..RCDGP
import ..RCIO

export run_mc!

# Build the (N2, T) grid from params
function _sample_size_grid(p)
    [(N2 = p.start_N2 + (k-1)*p.N2_increment,
      T  = p.start_T  + (k-1)*p.T_increment) for k in 1:p.num_sample_sizes]
end

"""
    run_mc!(; params=RCParams.PARAMS, save=true)

Generates *all reps* for each sample size (N2,T) in `params`, stores them as
a vector-of-vectors:
    bundle[k] = Vector{NamedTuple}(reps)
where each element is:
    (; df, Ωi, Ωj, Ωt, sizes=(; N1, N2, T))

If `save=true`, writes to the standard JLD2 path (via RCIO.save_mc_bundle!) and
returns that path. Always returns the in-memory `bundle`.
"""
function run_mc!(; params::NamedTuple = RCParams.PARAMS, save::Bool=true)
    grid = _sample_size_grid(params)
    println("== Running Monte Carlo (sizes=$(params.num_sample_sizes), reps=$(params.num_reps)) ==")

    bundle = Vector{Vector{NamedTuple}}(undef, length(grid))

    for (k, g) in enumerate(grid)
        N2, T = g.N2, g.T
        N  = params.N1 * N2 * T
        R  = params.num_reps

        # ---------- Draw Ω once per size ----------
        # Use a deterministic seed for Ω at this size so reruns are stable
        rng_Ω = MersenneTwister(hash((params.seed, :OMEGA, k, N2, T)))
        Ωi_fix, Ωj_fix, Ωt_fix = RCDGP.build_cov_mats(
            params.N1, N2, T;
            i_block=params.i_block, j_block=params.j_block, t_block=params.t_block,
            sigma_i=params.sigma_i, sigma_j=params.sigma_j, sigma_t=params.sigma_t,
            rng=rng_Ω
        )
        # ------------------------------------------------

        sims_k = Vector{NamedTuple}(undef, R)

        Threads.@threads for r in 1:R
            # Different seed for *data* (x, FE, u) each rep, but Ω is fixed
            seed_r = hash((params.seed, :DATA, k, N2, T, r))

            df, meta = RCDGP.generate_dataset(
                N1=params.N1, N2=N2, T=T,
                i_block=params.i_block, j_block=params.j_block, t_block=params.t_block,
                i_draw_mode=params.i_draw_mode, j_draw_mode=params.j_draw_mode, t_draw_mode=params.t_draw_mode,
                E_i=params.E_i, E_j=params.E_j, E_t=params.E_t,
                sigma_i=params.sigma_i, sigma_j=params.sigma_j, sigma_t=params.sigma_t,
                mu_x=params.mu_x, sigma_x=params.sigma_x,
                mu_u=params.mu_u, sigma_u=params.sigma_u,
                seed=seed_r,
                # ---------- NEW: pin Ω across reps ----------
                Ωi_fixed=Ωi_fix, Ωj_fixed=Ωj_fix, Ωt_fixed=Ωt_fix
                # --------------------------------------------
            )

            # Store the fixed Ω in every rep for clarity/reproducibility
            sims_k[r] = (; df=df, Ωi=Ωi_fix, Ωj=Ωj_fix, Ωt=Ωt_fix,
                          sizes=(; N1=params.N1, N2=N2, T=T))
        end

        bundle[k] = sims_k
        println("Sample Size $(N): done"); flush(stdout)
    end

    if save
        path = RCIO.save_mc_bundle!(bundle; params=params)
        println("Saved full MC bundle (all reps per size) to: ", path)
        return bundle, path
    else
        return bundle, nothing
    end
end

end # module
