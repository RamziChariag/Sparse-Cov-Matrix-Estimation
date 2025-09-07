# params.jl
# All configuration in one spot.

module RCParams

export PARAMS

const PARAMS = (; 

    # --- Estimation parameters ---
    beta_true = 2.0,          # true β used to form y
    c_true    = 0.0,          # true constant in y-generation
    vcov_ols  = :HC0,         # :none | :HC0 | :HC1 | :cluster
    vcov_fe   = :HC0,         # same choices
    # If you later want cluster-robust, set vcov_*=:cluster and add these:
    cluster_col_ols = nothing,  # e.g., :i or nothing
    cluster_col_fe  = nothing,  # e.g., :i or nothing

    # --- FGLS controls ---
    # --- Estimation-side Ω block choices (can differ from DGP) ---
    i_block_est = true,   # true ⇒ estimate full SPD Ωα; false ⇒ diagonal σ²_α I
    j_block_est = true,
    t_block_est = true,
    # --- Repeat patterns for estimation-side Ω ---
    repeat_alpha_fgls = false,
    repeat_gamma_fgls = false,
    repeat_lambda_fgls = false,
    # --- FGLS shrinkage controls ---
    fgls_shrinkage   = 1.0,     # off-diag shrink (1.0 = none)
    fgls_project_spd = false,   # clip eigvals ≥ fgls_spd_floor
    fgls_spd_floor   = 1e-8,

    # --- GLS (oracle) controls ---
    repeat_alpha_gls = true,
    repeat_gamma_gls = true,
    repeat_lambda_gls = true,
    gls_shrinkage    = 1.0,    # off-diag shrink; 1.0 = none
    gls_project_spd  = false,    # clip eigvals ≥ gls_spd_floor
    gls_spd_floor    = 1e-8,

    # --- Smoke Test Sample Size ---
    smoke_test_size = 3,                # which sample size to use for smoke tests
    smoke_plot_omega_heatmaps = true,   # Show Ω percentile heatmaps in smoke diagnostics

    # --- DGP parameters ---
    # --- Panel sizes ---
    N1 = 10,                       # i size (fixed across experiments)
    start_N2 = 4, N2_increment = 4,
    start_T  = 2, T_increment  = 2,
    num_sample_sizes = 10,         # how many (N2,T) pairs to generate

    # --- Monte Carlo ---
    num_reps = 300,                # reps per sample size
    seed = 42,                     # global seed for reproducibility

    # --- Covariance structure toggles (per dimension) ---
    # true = full SPD covariance, false = homoskedastic diagonal
    i_block = true,
    j_block = true,
    t_block = false,

    # --- Draw modes ---
    # :draw_once | :mixed | :full_redraw
    i_draw_mode = :mixed,
    j_draw_mode = :mixed,
    t_draw_mode = :mixed,

    # --- Means (E[FE]) ---
    E_i = 3.0,
    E_j = 3.0,
    E_t = 3.0,

    # --- Scale parameters used to set variances ---
    # For homoskedastic diagonal: Var = sigma_*^2
    # For full SPD: average diagonal is scaled to sigma_*^2
    sigma_i = 1.0,
    sigma_j = 1.0,
    sigma_t = 1.0,

    # --- X and U ---
    mu_x = 3.0,  sigma_x = 1.0,
    mu_u = 0.0,  sigma_u = 3.0,

    # --- Plotting controls ---
    # If you comment this line out, plotting will use the default Plots.jl theme.
    plot_theme = :ggplot2,

    # Show plots in the Julia session (in addition to saving if requested)
    plot_show = true,

    # Whether to create the per-estimator distribution plots
    make_dist_plots = false,

    # Which *size index* to use for the distribution plots (1-based).
    # If you omit this param, we'll fall back to `smoke_test_size`.
    plot_dist_size_index = 6,

    # Which estimators to use in plots (strings matched case-insensitively)
    plot_dist_estimators = ["FGLS"],
    asym_estimators      = ["OLS","OLS FE","FGLS","GLS"],

    # For the 2×2 multi-variance panel, enable log scale if you want
    plot_log_variance = false,


    # bounds by *n* for asymptotic plots (optional)
     asym_min_n = 90,
     asym_max_n = 10000,

    # distribution styling (optional)
    dist_bins = 30,
    dist_kde  = true,
    dist_hist = false,
)

end # module
