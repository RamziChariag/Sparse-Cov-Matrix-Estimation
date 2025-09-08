# Three‑Way Covariance Estimation & Monte Carlo (Julia)

This repository contains a reproducible Monte Carlo framework to study linear regression with three sets of random effects (i, j, t) and structured covariance. It provides a data‑generating process (DGP), several beta estimators (OLS, three‑way FE‑OLS, two FGLS variants, and an oracle GLS), tools to estimate block‑structured covariance matrices, and plotting/diagnostics utilities. All components are written in plain Julia (no package wrapping) and orchestrated via `code/main.jl`.

Core ideas:
- Flexible DGP for an i–j–t panel: draws x, u, and three FE blocks (αᵢ, γⱼ, λₜ) under user‑specified covariance and draw modes.
- Estimators: OLS, FE‑OLS (within along i/j/t), FGLS1 (single‑step), FGLS2 (two‑step), and Oracle GLS (using true small blocks).
- Covariance structure: per‑dimension toggle between full SPD block vs homoskedastic diagonal; repeat/stacking rules to build full Ω; optional off‑diagonal shrinkage and SPD projection.
- Monte Carlo across growing sample sizes with multi‑threaded reps; compact results saved to disk and visualized via plots.


## Requirements

- Julia 1.9+ (earlier versions may work)
- Packages (add to your environment):
  - DataFrames, Distributions, JLD2, Plots, LaTeXStrings, ProgressMeter

Example setup:

```julia
julia> ] activate .
(.) pkg> add DataFrames Distributions JLD2 Plots LaTeXStrings ProgressMeter 
```

For best performance, enable threads. The main estimation loop is parallel over Monte Carlo reps, but sequential over sample sizes. Adjust `JULIA_NUM_THREADS` as needed, e.g., `export JULIA_NUM_THREADS=4` or `auto` for all cores.


## Repository Layout

- `code/main.jl` — entry point wiring everything together via switches (ENV vars).
- `code/params.jl` — single source of configuration (`RCParams.PARAMS`).
- `code/dgp.jl` — DGP: builds Ωᵢ, Ωⱼ, Ωₜ and generates one dataset.
- `code/mc_driver.jl` — Monte Carlo driver (`RCMonteCarlo.run_mc!`).
- `code/estimation.jl` — runs OLS, FE‑OLS, FGLS1, FGLS2, Oracle GLS; MC estimator.
- `code/beta_estimators.jl` — within transform, OLS/FE‑OLS/GLS/FGLS.
- `code/omega_estimators.jl` — block Ω estimators, S‑matrices, assembly, shrink/SPD.
- `code/io.jl` — JLD2 I/O for datasets, MC bundles, results, and plot paths.
- `code/plotting.jl` — result plots (bias/variance curves, distributions, ratios).
- `code/diagnostics.jl` — smoke‑test diagnostics and Ω eigenvalue summaries.
- `generated data/` — JLD2 datasets and estimation results.
- `output/plots/` — saved PNG plots.


## Quick Start

All flows are controlled by environment variables read by `code/main.jl`.

Available switches ("1" to enable, anything else to disable):
- `RC_SMOKE`: run a small smoke test DGP; `SAVE_SMOKE`: save it as `..._ST.jld2`.
- `RC_SMOKE_DIAG`: run smoke diagnostics (Ω̂/Ω★ heatmaps + eigen tables).
- `DO_TEST_RUN`: run a single estimation using the smoke dataset (if present) or the first MC dataset.
- `RC_MC`: generate the full Monte Carlo bundle (all sizes × reps) and save it.
- `RC_ESTIMATE`: run Monte Carlo estimation over sizes/reps.
- `RC_SAVE_EST`: save estimation results to `generated data/`.
- `RC_PLOTS`: create plots from the saved/just‑computed results.

Typical runs:

1) Smoke test + save dataset
```bash
JULIA_NUM_THREADS=auto RC_SMOKE=1 SAVE_SMOKE=1 julia code/main.jl
```

2) Full MC, estimation, and plots (using defaults in `params.jl`)
```bash
JULIA_NUM_THREADS=auto RC_MC=1 RC_ESTIMATE=1 RC_SAVE_EST=1 RC_PLOTS=1 julia code/main.jl
```

Notes
- If `RC_MC=0` and no bundle exists on disk, `main.jl` will error with guidance to run the MC or save a smoke test first.
- Multi‑threading is used for reps; adjust `JULIA_NUM_THREADS` for speed.


## Configuration (`code/params.jl`)

Edit `RCParams.PARAMS` to control both the DGP and estimation. Key groups:

- Panel sizes and Monte Carlo:
  - `N1` (i size), grid over `(N2, T)` via `start_N2`, `N2_increment`, `start_T`, `T_increment`, `num_sample_sizes`.
  - `num_reps`, `seed`.
- DGP covariance and draw modes:
  - Per dimension: `i_block`, `j_block`, `t_block` (true=full SPD, false=diag σ²I).
  - Draw modes: `i_draw_mode`, `j_draw_mode`, `t_draw_mode` ∈ `:draw_once | :mixed | :full_redraw`.
  - Scales: `sigma_i`, `sigma_j`, `sigma_t`; means: `E_i`, `E_j`, `E_t`.
  - X and U: `mu_x`, `sigma_x`, `mu_u`, `sigma_u`.
- Estimation controls:
  - `vcov_ols`, `vcov_fe` ∈ `:none | :HC0 | :HC1 | :cluster` (+ optional `cluster_col_*`).
  - FGLS block choices: `i_block_est`, `j_block_est`, `t_block_est` (estimation‑side).
  - Repeat/stacking for FGLS: `repeat_alpha_fgls`, `repeat_gamma_fgls`, `repeat_lambda_fgls` and for FGLS2 similarly (`*_fgls2`).
  - Oracle GLS repeat: `repeat_alpha_gls`, `repeat_gamma_gls`, `repeat_lambda_gls`.
  - Optional conditioning: `fgls_shrinkage`, `fgls_project_spd`, `fgls_spd_floor` (and analogs for GLS).
- Plotting:
  - `plot_theme`, `plot_show`, `make_dist_plots`, `plot_dist_size_index`, `plot_dist_estimators`, `asym_estimators`, `var_ratio_estimators`, `plot_log_variance`, `asym_min_n`, `asym_max_n`.

Tip: for quick iterations, reduce `num_sample_sizes` and `num_reps`.


## Usage

You can call the modules directly from the REPL or your scripts.

```julia
include("code/main.jl")  # brings modules into scope via includet/using
p = RCParams.PARAMS

# 1) Generate a single dataset
df, meta = RCDGP.generate_dataset(
    N1=p.N1, N2=8, T=4,
    i_block=p.i_block, j_block=p.j_block, t_block=p.t_block,
    i_draw_mode=p.i_draw_mode, j_draw_mode=p.j_draw_mode, t_draw_mode=p.t_draw_mode,
    E_i=p.E_i, E_j=p.E_j, E_t=p.E_t,
    sigma_i=p.sigma_i, sigma_j=p.sigma_j, sigma_t=p.sigma_t,
    mu_x=p.mu_x, sigma_x=p.sigma_x,
    mu_u=p.mu_u, sigma_u=p.sigma_u,
    seed=p.seed)

# 2) Estimate with all estimators on that dataset
res = RCEstimation.estimate_all(df;
    N1=p.N1, N2=8, T=4,
    beta_true=p.beta_true, c_true=p.c_true,
    vcov_ols=p.vcov_ols, vcov_fe=p.vcov_fe,
    i_block_est=p.i_block_est, j_block_est=p.j_block_est, t_block_est=p.t_block_est,
    rep_a_fgls=p.repeat_alpha_fgls, rep_g_fgls=p.repeat_gamma_fgls, rep_l_fgls=p.repeat_lambda_fgls,
    rep_a_fgls2=p.repeat_alpha_fgls2, rep_g_fgls2=p.repeat_gamma_fgls2, rep_l_fgls2=p.repeat_lambda_fgls2,
    Ωi_star=meta.Ωi, Ωj_star=meta.Ωj, Ωt_star=meta.Ωt,
    rep_a_gls=p.repeat_alpha_gls, rep_g_gls=p.repeat_gamma_gls, rep_l_gls=p.repeat_lambda_gls,
    sigma_u2_oracle=p.sigma_u^2)
RCEstimation.print_estimate_summary(res; beta_true=p.beta_true)

# 3) Full MC across sizes (optionally pass a pre‑generated bundle)
est_res = RCEstimation.mc_estimate_over_sizes(; params=p, reps=p.num_reps)
RCIO.save_estimation_results!(est_res, p)
RCPlotting.make_result_plots(; params=p, est_res=est_res, save=true, show=true)
```

Structure of `est_res` (compact mode): a Vector over size grid where each element is a NamedTuple with
`size = (N1,N2,T)`, `reps`, and `stats` (bias, empirical/estimated variances, and variance ratios for each estimator). If you pass `keep_vectors=true`, per‑rep vectors of estimates/variances are also retained.


## Outputs and Naming

Naming is fully automated. Datasets and results are saved under `generated data/` using `RCIO` helpers. The nomenclature tules are as follows:

- Base MC bundle name: `output_data[_<blocks>][_full_fe|_full_re|_re_<dims>].jld2`, where
  - `<blocks>` reflects true DGP blocks enabled: `i_block`, `j_block`, `t_block`.
  - draw‑mode suffix encodes whether dimensions are `:draw_once` (FE) vs with repeats (RE).
  - Append `"_ST"` for smoke test datasets.
- Estimation results: `<base>_block_est_<dimcode>.jld2` where `<dimcode>` ∈ `{none,i,ij,ijt,…}` reflects estimation‑side Ω block toggles.
- Plots: `output/plots/<results base>/` containing PNGs for distributions, asymptotic bias/variance, a 2×2 multi‑variance panel, and variance ratios.


## Troubleshooting

- “No MC bundle found …”: run with `RC_MC=1` to generate, or save a smoke test via `RC_SMOKE=1 SAVE_SMOKE=1`.
- Ensure the required packages are added to your active environment (see Requirements).
- For faster tests, lower `num_sample_sizes` and `num_reps` in `params.jl`.


## License

This project is licensed under the Unlicense (public domain). See `LICENSE` for details.

