# params.jl

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
    # --- Estimation-side Ω block choices ---
    i_block_est = true,   # true ⇒ estimate full SPD Ωα; false ⇒ diagonal I * σ²_α 
    j_block_est = false,
    t_block_est = false,
    # --- Repeat patterns for estimation-side Ω ---
    # For FGLS1:
    repeat_alpha_fgls = false,
    repeat_gamma_fgls = false,
    repeat_lambda_fgls = false,
    subtract_sigma_u2_fgls1 = true,  # whether to subtract σ²_u from diagonals of Ω estimates
    # For FGLS2:
    repeat_alpha_fgls2 = false,
    repeat_gamma_fgls2 = false,
    repeat_lambda_fgls2 = false,
    subtract_sigma_u2_fgls2 = true,  # whether to subtract σ²_u from diagonals of Ω estimates
    iterate_fgls2 = false,            # whether to do multiple FGLS2 iterations (not just one-shot)
    # --- FGLS shrinkage controls ---
    fgls_shrinkage   = 1.0,     # off-diag shrink (1.0 = none)
    fgls_project_spd = false,   # clip eigvals ≥ fgls_spd_floor
    fgls_spd_floor   = 1e-8,

    # --- GLS (oracle) controls ---
    repeat_alpha_gls = false,
    repeat_gamma_gls = false,
    repeat_lambda_gls = false,
    gls_shrinkage    = 1.0,    # off-diag shrink; 1.0 = none
    gls_project_spd  = false,    # clip eigvals ≥ gls_spd_floor
    gls_spd_floor    = 1e-8,

    # --- Smoke Test Sample Size ---
    smoke_test_size = 2,                    # which sample size to use for smoke tests
    smoke_plot_omega_heatmaps = true,       # Show Ω percentile heatmaps in smoke diagnostics
    save_heatmap_plots = true,              # Save heatmap plots to output
    print_omegas_post_dgp = false,           # print true Ω after DGP in smoke diagnostics
    print_generated_data_head = false,      # how many rows of generated data to print in smoke test
    generated_data_rows_to_check = 20,      # how many rows of generated data to check for invariance
    smoke_test_debug_print = false,         # print extra debug info in smoke diagnostics

    # --- DGP parameters ---
    # --- Panel sizes ---
    N1 = 4,                       # i size (fixed across experiments)
    start_N2 = 5, N2_increment = 2,
    start_T  = 6, T_increment  = 6,
    num_sample_sizes = 10,         # how many (N2,T) pairs to generate

    # --- Monte Carlo ---
    num_reps =1000,                # reps per sample size
    seed = 42,                     # global seed for reproducibility

    # --- Covariance structure toggles (per dimension) ---
    # true = full SPD covariance, false = homoskedastic diagonal
    i_block = true,
    j_block = false,
    t_block = false,

    # --- Draw modes ---
    # :draw_once | :mixed | :full_redraw
    i_draw_mode = :draw_once,
    j_draw_mode = :draw_once,
    t_draw_mode = :draw_once,

    # --- Means (E[FE]) ---
    E_i = 0.0,
    E_j = 0.0,
    E_t = 0.0,

    # --- Scale parameters used to set variances ---
    # For homoskedastic diagonal: Var = sigma_*^2
    # For full SPD: average diagonal is scaled to sigma_*^2
    sigma_i = 1.0,
    sigma_j = 1.0,
    sigma_t = 1.0,

    # --- X and U ---
    mu_x = 1.0,  sigma_x = 1.0,
    mu_u = 0.0,  sigma_u = 1.0,

    # --- Plotting controls ---
    # If you comment this line out, plotting will use the default Plots.jl theme.
    plot_theme = :ggplot2,

    # Show plots in the Julia session (in addition to saving if requested)
    plot_show = true,

    # Whether to create the per-estimator distribution plots
    make_dist_plots = false,

    # Which *size index* to use for the distribution plots (1-based).
    # If you omit this param, we'll fall back to `smoke_test_size`.
    plot_dist_size_index = 3,

    # Which estimators to use in plots (strings matched case-insensitively)
    plot_dist_estimators = ["FGLS1","FGLS2"],
    asym_estimators      = ["OLS","OLS FE","FGLS1","FGLS2","GLS"],
    var_ratio_estimators = ["OLS FE","FGLS1","FGLS2","GLS"],


    # For the 2×2 multi-variance panel, enable log scale if you want
    plot_log_variance = false,


    # bounds by *n* for asymptotic plots (optional)
     asym_min_n = 600,
     asym_max_n = 10000,

    # distribution styling (optional)
    dist_bins = 30,
    dist_kde  = true,
    dist_hist = false,
)

end # module
