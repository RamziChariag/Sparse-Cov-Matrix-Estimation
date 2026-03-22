# params.jl

module RCParams

export PARAMS

const PARAMS = (; 
  
    # --- Estimation parameters ---
    beta_true = 2.0,          # true β used to form y
    c_true    = 0.0,          # true constant in y-generation
    vcov_ols  = :HC1,         # :none | :HC0 | :HC1 | :cluster
    vcov_fe   = :HC1,         # same choices
    # If you later want cluster-robust, set vcov_*=:cluster and add these:
    cluster_col_ols = nothing,  # e.g., :i or nothing
    cluster_col_fe  = nothing,  # e.g., :i or nothing

    # --- OLSFE controls ---
    # Default FE is three-way (:i, :j, :t).
    # If toggled, replace the corresponding FE *and* :t with an interaction FE:
    #   fe_alphaT = true  => use :alphaT = (i,t), and drop :i and :t
    #   fe_gammaT = true  => use :gammaT = (j,t), and drop :j and :t
    fe_alphaT = false,
    fe_gammaT = false,

    # --- FGLS controls ---
    # --- Estimation-side Ω block choices ---
    i_block_est = false,   # true ⇒ estimate full SPD Ωα; false ⇒ diagonal I * σ²_α 
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
    smoke_test_size = 4,                    # which sample size to use for smoke tests
    smoke_plot_omega_heatmaps = true,       # Show Ω percentile heatmaps in smoke diagnostics
    save_heatmap_plots = true,              # Save heatmap plots to output
    print_omegas_post_dgp = false,           # print true Ω after DGP in smoke diagnostics
    print_generated_data_head = false,      # how many rows of generated data to print in smoke test
    generated_data_rows_to_check = 20,      # how many rows of generated data to check for invariance
    smoke_test_debug_print = true,         # print extra debug info in smoke diagnostics

    # --- DGP parameters ---
    # --- Panel sizes ---
    N1 = 4,                       # i size (fixed across experiments)
    start_N2 = 5, N2_increment = 1,
    start_T  = 6, T_increment  = 4,
    num_sample_sizes = 10,         # how many (N2,T) pairs to generate

    # --- Monte Carlo ---
    num_reps =300,                # reps per sample size
    seed = 42,                     # global seed for reproducibility

    # --- Covariance structure toggles (per dimension) ---
    # true = full SPD covariance, false = homoskedastic diagonal
    i_block = false,
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
    mu_x2 = 0.0, sigma_x2 = 1.0,       # x2 distribution parameters
    mu_u = 0.0,  sigma_u = 1.0,

    # --- Second regressor (x2) controls ---
    beta2_true = 2.0,                    # true β₂ (only matters if use_x2_dgp=true)
    use_x2_dgp = false,                   # include x2 in y-generation (DGP)
    use_x2_est = true,                  # include x2 as a regressor in estimation
    correlate_x = false,                 # draw (x1,x2) jointly from bivariate normal
    rho_x = 0.5,                         # correlation between x1 and x2 (if correlate_x)
    correlate_x_alpha = false,           # correlate x with alpha (i fixed effect)
    rho_x_alpha = 0.5,                   # correlation strength between x and alpha

    # --- Plotting controls ---
    # If you comment this line out, plotting will use the default Plots.jl theme.
    plot_theme = :ggplot2,

    # Show plots in the Julia session (in addition to saving if requested)
    plot_show = true,

    # Which estimators to use in plots (strings matched case-insensitively)
    plot_dist_estimators = ["FGLS1","FGLS2"],
    asym_estimators      = ["OLS FE","FGLS1","FGLS2","GLS"],
    var_ratio_estimators = ["OLS FE","FGLS1","FGLS2","GLS"],


    # For the 2×2 multi-variance panel, enable log scale if you want
    plot_log_variance = false,


    # bounds by *n* for asymptotic plots (optional)
     asym_min_n = 200,
     asym_max_n = 10000,

    # --- Beta density & t-statistic distribution plots: this uses the same sample size used for the smoke test ---
    plot_dist_size_index = 5,           # which sample size index to use for the distribution plots (1-based)
    make_beta_density_plots = true,     # Overlayed: plot β̂ density across reps for each estimator
    make_dist_plots = false,           # Whether to create the per-estimator distribution plots
    make_tstat_plots = false,            # plot t-statistic distribution across reps
    beta_density_estimators = ["OLS FE","FGLS2"],
    tstat_estimators        = ["OLS FE","FGLS1","FGLS2"],

    # --- t-statistic vs sample size plots (size/power) ---
    make_tstat_vs_n_plots = false,        # plot avg t-stat vs n for β₁ (size) and β₂ (power)
    tstat_vs_n_estimators = ["OLS FE","FGLS2"],
    make_rejection_rate_plots = true,     # plot rejection rates vs n for size and power
    rejection_rate_estimators = ["OLS FE","FGLS2"],
    beta2_null = 0.0,                    # null hypothesis value for β₂ (power test: H0: β₂ = 0)

    # distribution styling (optional)
    dist_bins = 30,
    dist_kde  = true,
    dist_hist = false,
)

end # module