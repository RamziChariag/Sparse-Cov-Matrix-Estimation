# main.jl
using Random, LinearAlgebra, Distributions, DataFrames, Statistics, LaTeXStrings
using Revise
using JLD2

includet("params.jl")
includet("rc_functions.jl")
includet("dgp.jl")
includet("io.jl")
includet("mc_driver.jl")
includet("smoke_test.jl")
includet("omega_estimators.jl")
includet("beta_estimators.jl")
includet("estimation.jl")
includet("plotting.jl")
includet("diagnostics.jl")

using .RCParams, .RCFunctions, .RCDGP, .RCIO, .RCMonteCarlo, .RCSmokeTest
using .RCBetaEstimators, .RCOmegaEstimators, .RCEstimation, .RCPlotting, .RCDiagnostics

function main()
    # ---------------- Switches ----------------
    DO_SMOKE_TEST  = get(ENV, "RC_SMOKE", "1") == "1"           # run smoke test
    DO_SAVE_SMOKE  = get(ENV, "SAVE_SMOKE", "1") == "1"         # save smoke test results
    DO_SMOKE_DIAG  = get(ENV, "RC_SMOKE_DIAG", "1") == "1"      # run smoke test diagnostics
    DO_TEST_RUN    = get(ENV, "DO_TEST_RUN", "1") == "1"        # run test estimation
    DO_MC          = get(ENV, "RC_MC", "0") == "1"              # run DGP
    DO_ESTIMATE    = get(ENV, "RC_ESTIMATE", "0") == "1"        # run full estimation
    DO_SAVE_ESTIMATION = get(ENV, "RC_SAVE_EST", "0") == "1"    # save estimation results, if we ran it
    DO_MAKE_PLOTS      = get(ENV, "RC_PLOTS", "0") == "1"       # create simulation results plots
    # ------------------------------------------

    p = RCParams.PARAMS

    # (A) Optional smoke test and diagnostics
    if DO_SMOKE_TEST
        println("== Running DGP smoke test ==")
        RCSmokeTest.run_smoke_test!(; params=p, save=DO_SAVE_SMOKE, s=p.smoke_test_size)
    end
    if DO_SMOKE_DIAG
        println("== Smoke-test diagnostics ==")
        RCDiagnostics.smoke_diagnosis!(; params=p)
    end

    # (B) Prepare MC bundle:
    #     - If DO_MC=1: generate and save bundle, then load it from disk
    #     - Else: try to load an existing bundle from disk
    bundle = nothing  # Vector{Vector{NamedTuple}} expected by mc_estimate_over_sizes
    if DO_MC
        println("== Generating Datasets ==")
        RCMonteCarlo.run_mc!(; params=p, save=true)   # saves to RCIO.output_path(p)
        println("Saved bundle to: ", RCIO.output_path(p))
        bundle = RCIO.load_mc_bundle(p)               # always load to get consistent shape
    else
        if isfile(RCIO.output_path(p))
            bundle = RCIO.load_mc_bundle(p)
            println("Loaded bundle from: ", RCIO.output_path(p))
        else
            # No bundle on disk; we can still do a single TEST run from smoke test if available
            if !DO_TEST_RUN || !RCIO.dataset_exists(p; suffix="_ST")
                error("No MC bundle found at $(RCIO.output_path(p)). Run with RC_MC=1 or save a smoke test (RC_SMOKE=1,SAVE_SMOKE=1).")
            end
        end
    end

    # (C) Optional single-dataset estimation (prefers smoke test if present)
    if DO_TEST_RUN
        df = nothing; Ωi_true = nothing; Ωj_true = nothing; Ωt_true = nothing
        N2 = nothing; T = nothing; loaded_source = ""

        if RCIO.dataset_exists(p; suffix="_ST")
            st = RCIO.load_dataset(p; suffix="_ST")
            df = st.df
            Ωi_true, Ωj_true, Ωt_true = st.Ωi, st.Ωj, st.Ωt
            N2, T = st.sizes.N2, st.sizes.T
            loaded_source = "smoke-test (_ST) @ $(st.path)"
        else
            # fall back to the first rep of the first size in the bundle
            first_rep = bundle[1][1]  # Vector{Vector{NamedTuple}} → size idx 1, rep idx 1
            df = first_rep.df
            Ωi_true, Ωj_true, Ωt_true = first_rep.Ωi, first_rep.Ωj, first_rep.Ωt
            N2, T = first_rep.sizes.N2, first_rep.sizes.T
            loaded_source = "MC bundle (first size/rep)"
        end

        println("== Using dataset from: $loaded_source ==")

        res = RCEstimation.estimate_all(
            df;
            # sizes
            N1 = p.N1, N2 = N2, T = T,
            # truth / vcov choices
            beta_true = p.beta_true, c_true = p.c_true,
            vcov_ols = p.vcov_ols, vcov_fe = p.vcov_fe,
            cluster_col_ols = p.cluster_col_ols, cluster_col_fe = p.cluster_col_fe,
            # FGLS (estimation-side Ω choices)
            i_block_est = p.i_block_est,
            j_block_est = p.j_block_est,
            t_block_est = p.t_block_est,
            rep_a_fgls      = p.repeat_alpha_fgls,
            rep_g_fgls      = p.repeat_gamma_fgls,
            rep_l_fgls      = p.repeat_lambda_fgls,
            rep_a_fgls2     = p.repeat_alpha_fgls2,
            rep_g_fgls2     = p.repeat_gamma_fgls2,
            rep_l_fgls2     = p.repeat_lambda_fgls2,
            subtract_sigma_u2_fgls1 = p.subtract_sigma_u2_fgls1,
            subtract_sigma_u2_fgls2 = p.subtract_sigma_u2_fgls2,
            iterate_fgls2   = p.iterate_fgls2,
            fgls_shrinkage  = p.fgls_shrinkage,
            fgls_project_spd= p.fgls_project_spd,
            fgls_spd_floor  = p.fgls_spd_floor,
            # Oracle GLS (if true blocks exist)
            Ωi_star = Ωi_true, Ωj_star = Ωj_true, Ωt_star = Ωt_true,
            rep_a_gls      = p.repeat_alpha_gls,
            rep_g_gls      = p.repeat_gamma_gls,
            rep_l_gls      = p.repeat_lambda_gls,
            gls_shrinkage  = p.gls_shrinkage,
            gls_project_spd= p.gls_project_spd,
            gls_spd_floor  = p.gls_spd_floor,
            sigma_u2_oracle= p.sigma_u^2
        )
        RCEstimation.print_estimate_summary(res; beta_true=p.beta_true)
    end

    # (D) MC estimation across sizes/reps
    est_res = nothing
    if DO_ESTIMATE
        println("== Running estimation MC (sizes=$(p.num_sample_sizes), reps=$(p.num_reps)) ==")
        est_res = RCEstimation.mc_estimate_over_sizes(; params=p,
                                                      reps=p.num_reps,
                                                      progress_sizes=true,
                                                      progress_reps=true,
                                                      print_each=true,
                                                      bundle=bundle,
                                                      keep_vectors=true)   
        # (E) Optional save of compact results
        if DO_SAVE_ESTIMATION
            out_path = RCIO.save_estimation_results!(est_res, p)
            println("Saved estimation results to: ", out_path)
        end
    end

    # (E) Plots
    if DO_MAKE_PLOTS
        RCPlotting.make_result_plots(; params=p,
                                    est_res=est_res,   # if you just computed it; otherwise it loads
                                    save=true,
                                    show=p.plot_show)  
    end

end

main()
