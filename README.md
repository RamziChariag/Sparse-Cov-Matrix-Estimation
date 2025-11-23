# Three-Way Covariance Monte Carlo (Julia)

Monte Carlo playground for linear regression with three random-effect dimensions (i, j, t) and flexible covariance structures. Everything is plain Julia code under `code/`, orchestrated by `code/main.jl`.

## What this code does
- Generates i–j–t panel data with configurable Ω blocks (full SPD vs diagonal) and draw modes (:draw_once, :mixed, :full_redraw) while pinning Ω per sample size for repeatable reps.
- Runs OLS, three-way FE-OLS, two FGLS variants with block/repeat/shrinkage options, and an oracle GLS that uses the true small blocks when available.
- Drives Monte Carlo experiments over a user-defined grid of (N2, T) sizes, multi-threaded across reps, with compact JLD2 outputs.
- Provides smoke tests, invariance checks, Ω diagnostics (heatmaps + eigenvalue tables), and plotting for distributions, asymptotic bias/variance, multi-variance panels, and variance ratios.
- Saves datasets, MC bundles, estimation results, and plots with consistent naming helpers in `code/io.jl`.

## Dependencies
Julia 1.9+ is recommended. Add these packages to your environment:
- DataFrames
- Distributions
- JLD2
- Plots
- LaTeXStrings
- ProgressMeter
- Revise (optional, used by `main.jl`)

Example setup:
```julia
julia> ] activate .
(.) pkg> add DataFrames Distributions JLD2 Plots LaTeXStrings ProgressMeter Revise
```
Set `JULIA_NUM_THREADS` (e.g., `export JULIA_NUM_THREADS=auto`) to parallelize reps.

## Layout
- `code/main.jl` – top-level orchestrator driven by environment switches.
- `code/params.jl` – single source of configuration (`RCParams.PARAMS`).
- `code/rc_functions.jl` – shared helpers (random SPD draws, invariance checks, id builders).
- `code/dgp.jl` – data generator and Ω builders.
- `code/mc_driver.jl` – Monte Carlo dataset generator (Ω fixed per size, threaded reps).
- `code/beta_estimators.jl` – OLS/FE-OLS/FGLS/GLS estimators and within transforms.
- `code/omega_estimators.jl` – Ω estimators, S-matrices, shrink/SPD tools.
- `code/estimation.jl` – wrapper that runs all estimators; MC estimation loop.
- `code/smoke_test.jl` – DGP smoke test and invariance checks.
- `code/diagnostics.jl` – Ω diagnostics for the smoke dataset and outlier helpers.
- `code/plotting.jl` – plotting utilities and the results plot orchestrator.
- `code/io.jl` – JLD2 I/O, path helpers, and plot naming.
- `generated data/` – datasets and estimation results (JLD2).
- `output/plots/` – generated PNG plots.

## Configure in `code/params.jl`
`RCParams.PARAMS` is a NamedTuple; adjust it to control everything:
- Panel grid & Monte Carlo: `N1`, `start_N2`, `N2_increment`, `start_T`, `T_increment`, `num_sample_sizes`, `num_reps`, `seed`, `smoke_test_size`.
- DGP: block toggles `i_block`, `j_block`, `t_block`; draw modes `*_draw_mode`; means `E_*`; scales `sigma_*`; x/u settings `mu_x`, `sigma_x`, `mu_u`, `sigma_u`.
- Estimation & GLS/FGLS: `vcov_ols`, `vcov_fe`, `cluster_col_*`; estimation-side block choices `i_block_est`, `j_block_est`, `t_block_est`; repeat flags (`repeat_alpha_*`, `repeat_gamma_*`, `repeat_lambda_*`); `subtract_sigma_u2_fgls1/2`; shrink/projection controls `fgls_shrinkage`, `fgls_project_spd`, `fgls_spd_floor` and GLS counterparts; oracle repeats `repeat_alpha_gls`, `repeat_gamma_gls`, `repeat_lambda_gls`.
- Plotting: `plot_theme`, `plot_show`, `make_dist_plots`, `plot_dist_size_index`, `plot_dist_estimators`, `asym_estimators`, `var_ratio_estimators`, `plot_log_variance`, `asym_min_n`, `asym_max_n`, `dist_bins`, `dist_kde`, `dist_hist`.
- Diagnostics toggles: `smoke_plot_omega_heatmaps`, `print_omegas_post_dgp`, `print_generated_data_head`, `generated_data_rows_to_check`, `smoke_test_debug_print`.
Tip: lower `num_sample_sizes` and `num_reps` for quicker runs before scaling up.

## How to run `code/main.jl`
Switches are read from environment variables (default is "1" for all):
- `RC_SMOKE` – run the DGP smoke test.
- `SAVE_SMOKE` – persist the smoke dataset as `<base>_ST.jld2`.
- `RC_SMOKE_DIAG` – run Ω diagnostics on the smoke dataset.
- `DO_TEST_RUN` – single estimation using the smoke data if present, else the first MC dataset.
- `RC_MC` – build the full MC bundle (all sizes × reps) and save it.
- `RC_ESTIMATE` – run Monte Carlo estimation over sizes/reps.
- `RC_SAVE_EST` – save estimation results to `generated data/`.
- `RC_PLOTS` – create plots (loads results from disk if `est_res` is not in memory).

Examples:
- Quick smoke + diagnostics only:
  ```bash
  JULIA_NUM_THREADS=auto RC_MC=0 RC_ESTIMATE=0 RC_PLOTS=0 julia code/main.jl
  ```
- Full pipeline (honors your params; adjust `num_reps`/`num_sample_sizes` first):
  ```bash
  JULIA_NUM_THREADS=auto julia code/main.jl
  ```
- Reuse an existing MC bundle and only estimate/plot:
  ```bash
  JULIA_NUM_THREADS=auto RC_MC=0 RC_SMOKE=0 RC_SMOKE_DIAG=0 DO_TEST_RUN=0 RC_ESTIMATE=1 RC_SAVE_EST=1 RC_PLOTS=1 julia code/main.jl
  ```
If `RC_MC=0` and no bundle exists, `main.jl` will stop with guidance to generate data or save a smoke test first.

## REPL snippets
```julia
include("code/main.jl")
using .RCParams, .RCDGP, .RCEstimation, .RCMonteCarlo, .RCIO, .RCPlotting
p = RCParams.PARAMS

# 1) Generate one dataset at a chosen size from your grid
N2 = p.start_N2; T = p.start_T
df, meta = RCDGP.generate_dataset(
    N1=p.N1, N2=N2, T=T,
    i_block=p.i_block, j_block=p.j_block, t_block=p.t_block,
    i_draw_mode=p.i_draw_mode, j_draw_mode=p.j_draw_mode, t_draw_mode=p.t_draw_mode,
    E_i=p.E_i, E_j=p.E_j, E_t=p.E_t,
    sigma_i=p.sigma_i, sigma_j=p.sigma_j, sigma_t=p.sigma_t,
    mu_x=p.mu_x, sigma_x=p.sigma_x,
    mu_u=p.mu_u, sigma_u=p.sigma_u,
    seed=p.seed
)

# 2) Estimate all models on that dataset
res = RCEstimation.estimate_all(df;
    N1=p.N1, N2=N2, T=T,
    beta_true=p.beta_true, c_true=p.c_true,
    vcov_ols=p.vcov_ols, vcov_fe=p.vcov_fe,
    cluster_col_ols=p.cluster_col_ols, cluster_col_fe=p.cluster_col_fe,
    i_block_est=p.i_block_est, j_block_est=p.j_block_est, t_block_est=p.t_block_est,
    rep_a_fgls=p.repeat_alpha_fgls, rep_g_fgls=p.repeat_gamma_fgls, rep_l_fgls=p.repeat_lambda_fgls,
    rep_a_fgls2=p.repeat_alpha_fgls2, rep_g_fgls2=p.repeat_gamma_fgls2, rep_l_fgls2=p.repeat_lambda_fgls2,
    subtract_sigma_u2_fgls1=p.subtract_sigma_u2_fgls1, subtract_sigma_u2_fgls2=p.subtract_sigma_u2_fgls2,
    fgls_shrinkage=p.fgls_shrinkage, fgls_project_spd=p.fgls_project_spd, fgls_spd_floor=p.fgls_spd_floor,
    Ωi_star=meta.Ωi, Ωj_star=meta.Ωj, Ωt_star=meta.Ωt,
    rep_a_gls=p.repeat_alpha_gls, rep_g_gls=p.repeat_gamma_gls, rep_l_gls=p.repeat_lambda_gls,
    gls_shrinkage=p.gls_shrinkage, gls_project_spd=p.gls_project_spd, gls_spd_floor=p.gls_spd_floor,
    sigma_u2_oracle=p.sigma_u^2
)
RCEstimation.print_estimate_summary(res; beta_true=p.beta_true)

# 3) Full MC estimation (optionally pass a prebuilt bundle)
est_res = RCEstimation.mc_estimate_over_sizes(; params=p, reps=50, keep_vectors=true)
RCIO.save_estimation_results!(est_res, p)
RCPlotting.make_result_plots(; params=p, est_res=est_res, save=true, show=true)
```

## Outputs and naming
- Datasets and MC bundles live in `generated data/`. Base name comes from `RCIO.build_output_basename(params)` and encodes block/draw choices. Smoke datasets add `_ST`.
- MC bundles are JLD2 files with a vector-of-vectors (`bundle[size_idx][rep_idx]`).
- Estimation results are saved as `<base>_block_est_<dimcode>.jld2`, where `<dimcode>` reflects estimation-side block toggles (`i`, `ij`, `ijt`, or `none`).
- Plots are stored under `output/plots/<results basename>/` using `RCIO` helpers, e.g. `<base>_bias_asym_analysis.png`, `<base>_mult_asym_var.png`, `<base>_<est>_<n>_beta_hat_dist.png`.

## Diagnostics and plotting
- `RC_SMOKE_DIAG=1` loads the smoke dataset, builds Ω̂ from FGLS1/FGLS2 without running GLS, shows percentile heatmaps, and prints eigenvalue tables alongside the oracle Ω★.
- `RCPlotting.make_result_plots` respects `plot_theme`, `plot_dist_size_index` (or `smoke_test_size`), `plot_dist_estimators`, `asym_estimators`, `var_ratio_estimators`, `plot_log_variance`, and optional `asym_min_n`/`asym_max_n` bounds.

## Notes
- Ω blocks are drawn once per sample size and reused for every rep to isolate sampling noise from covariance draws.
- Monte Carlo loops are threaded over reps; sizes are processed sequentially.
- FGLS/GLS SPD projection and shrinkage options are available if an Ω estimate is not positive definite.

## License
This project is licensed under the Unlicense (public domain). See `LICENSE` for details.
