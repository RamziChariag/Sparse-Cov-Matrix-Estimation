# io.jl
module RCIO

using JLD2, DataFrames, LinearAlgebra, Statistics

import ..RCParams

export build_output_basename, output_dir, output_path,
       save_dataset!, load_dataset, dataset_exists,
       save_mc_bundle!, load_mc_bundle, block_est_dimcode,
       estimation_results_path, save_estimation_results!,
       load_estimation_results,
       plots_dir_for_results, ensure_plots_dir!, results_basename,
       plot_path_estimator_dist, plot_path_asymptotic, plot_path_multi_variance, plot_path_variance_ratio

"Build base filename according to your rules."
function build_output_basename(p::NamedTuple)
    base = "output_data"

    # block suffixes
    block_parts = String[]
    p.i_block && push!(block_parts, "i_block")
    p.j_block && push!(block_parts, "j_block")
    p.t_block && push!(block_parts, "t_block")
    if !isempty(block_parts)
        base *= "_" * join(block_parts, "_")
    end

    # draw-mode suffix
    not_once = String[]
    if p.i_draw_mode != :draw_once; push!(not_once, "i"); end
    if p.j_draw_mode != :draw_once; push!(not_once, "j"); end
    if p.t_draw_mode != :draw_once; push!(not_once, "t"); end

    if isempty(not_once)
        base *= "_full_fe"
    elseif length(not_once) == 3
        base *= "_full_re"
    else
        base *= "_re_" * join(not_once, "_")
    end

    return base
end

"Sibling to the code folder: ../generated data"
function output_dir()
    return normpath(joinpath(@__DIR__, "..", "generated data"))
end

"Full path with .jld2 extension (optional filename suffix)."
function output_path(p::NamedTuple; suffix::AbstractString = "")
    return joinpath(output_dir(), build_output_basename(p) * suffix * ".jld2")
end

"Check if dataset exists for these params (optionally with suffix)."
dataset_exists(p::NamedTuple; suffix::AbstractString = "") = isfile(output_path(p; suffix=suffix))

"""
Save everything into ONE file:
- df (DataFrame)
- Ωi, Ωj, Ωt
- sizes (NamedTuple with N1,N2,T)
- params (the NamedTuple from RCParams)
Optional: `suffix` to alter filename (e.g., \"_ST\" for smoke test).
"""
function save_dataset!(df::DataFrame, meta::NamedTuple, sizes::NamedTuple;
                       params::NamedTuple, suffix::AbstractString = "")
    dir = output_dir()
    isdir(dir) || mkpath(dir)
    path = output_path(params; suffix=suffix)
    JLD2.jldopen(path, isfile(path) ? "r+" : "w") do f
        # overwrite keys if they already exist
        for k in ("df","Ωi","Ωj","Ωt","sizes","params")
            haskey(f, k) && delete!(f, k)
        end
        f["df"]     = df
        f["Ωi"]     = meta.Ωi
        f["Ωj"]     = meta.Ωj
        f["Ωt"]     = meta.Ωt
        f["sizes"]  = sizes
        f["params"] = params
    end
    return path
end

"Load the saved dataset for these params. Returns a NamedTuple. Optional `suffix`."
function load_dataset(p::NamedTuple; suffix::AbstractString = "")
    path = output_path(p; suffix=suffix)
    if !isfile(path)
        error("Dataset does not exist at: $path")
    end
    d = JLD2.load(path)
    # Return as a NamedTuple for convenience
    return (; df = d["df"], Ωi = d["Ωi"], Ωj = d["Ωj"], Ωt = d["Ωt"],
             sizes = d["sizes"], params = d["params"], path = path)
end

"""
Save Monte Carlo datasets bundle to a single JLD2 file.

Accepted shapes for `bundle`:
- Vector{NamedTuple}                         (one dataset per size)
- Vector{<:AbstractVector{<:NamedTuple}}    (many reps per size)

On disk we store a vector-of-vectors for uniformity. If you pass a
Vector{NamedTuple}, it is wrapped as [[b] for b in bundle].
"""
function save_mc_bundle!(bundle; params::NamedTuple)
    @assert bundle isa AbstractVector "bundle must be an AbstractVector; got $(typeof(bundle))"

    # Normalize to vector-of-vectors
    tosave = begin
        isempty(bundle) && (return output_path(params))  # nothing to do
        first = bundle[1]
        if first isa NamedTuple
            [ [b] for b in bundle ]                        # wrap singletons
        elseif first isa AbstractVector
            # Light sanity check: inner elements are NamedTuple
            for (i, vec) in pairs(bundle)
                @assert vec isa AbstractVector "bundle[$i] must be a vector"
                for (j, b) in pairs(vec)
                    @assert b isa NamedTuple "bundle[$i][$j] must be a NamedTuple; got $(typeof(b))"
                end
            end
            bundle
        else
            error("Unsupported bundle element type $(typeof(first)). Expect NamedTuple or Vector{NamedTuple}.")
        end
    end

    dir = output_dir()
    isdir(dir) || mkpath(dir)
    path = output_path(params)

    JLD2.jldopen(path, isfile(path) ? "r+" : "w") do f
        haskey(f, "mc_bundle") && delete!(f, "mc_bundle")
        haskey(f, "params")    && delete!(f, "params")
        f["mc_bundle"] = tosave
        f["params"]    = params
    end
    return path
end

"Load the MC datasets bundle. Always returns a vector-of-vectors of NamedTuples."
function load_mc_bundle(p::NamedTuple)
    path = output_path(p)
    isfile(path) || error("Dataset does not exist at: $path")
    d = JLD2.load(path)
    haskey(d, "mc_bundle") || error("No `mc_bundle` found in: $path. Generate via run_mc!(save=true).")

    mb = d["mc_bundle"]
    isempty(mb) && (return mb)

    # Normalize to vector-of-vectors for consistency
    first = mb[1]
    if first isa NamedTuple
        return [ [b] for b in mb ]
    elseif first isa AbstractVector
        return mb
    else
        error("Unsupported stored bundle element type $(typeof(first)).")
    end
end

export estimation_results_path, save_estimation_results!

"Return a compact dimcode like \"i\", \"ij\", \"it\", \"ijt\" or \"none\" from params."
function block_est_dimcode(params::NamedTuple)
    dims = String[]
    if params.i_block_est; push!(dims, "i"); end
    if params.j_block_est; push!(dims, "j"); end
    if params.t_block_est; push!(dims, "t"); end
    return isempty(dims) ? "none" : join(dims, "")
end

"""
    estimation_results_path(params)

Build the output file path for estimation results by taking the base MC bundle
path (`output_path(params)`), stripping its extension, and appending
`_block_est_<dimcode>.jld2`, where `<dimcode>` is derived from params.
"""
function estimation_results_path(params::NamedTuple)
    base = output_path(params)
    root, _ = splitext(base)
    dimcode = block_est_dimcode(params)
    return string(root, "_block_est_", dimcode, ".jld2")
end

"Save `est_res` and `params` to the computed path; return that path."
function save_estimation_results!(est_res, params::NamedTuple)
    path = estimation_results_path(params)
    @save path est_res params
    return path
end

"""
    load_estimation_results(params) -> (; est_res, path, dimcode, params_saved)

Load estimation results saved by `save_estimation_results!`. The file name is
`<output_path(params) without extension>_block_est_<dimcode>.jld2`, where
`dimcode` comes from the estimation-side block toggles in `params`.

Returns:
  - est_res      :: Vector (compact results)
  - path         :: String (full path loaded)
  - dimcode      :: String ("i","ij","ijt","none") — inferred if not stored
  - params_saved :: NamedTuple or `nothing` (whatever was saved alongside)
"""
function load_estimation_results(params::NamedTuple)
    path = estimation_results_path(params)
    isfile(path) || error("Estimation results file not found at:\n  $path")

    d = JLD2.load(path)
    haskey(d, "est_res") || error("File at $path does not contain `est_res`.")

    est_res      = d["est_res"]
    # We saved `params` (not `p`) in save_estimation_results!
    params_saved = get(d, "params", nothing)

    # Prefer dimcode saved in file (if you ever add it); otherwise infer from params
    dimcode = if haskey(d, "dimcode")
        d["dimcode"]
    else
        block_est_dimcode(params_saved === nothing ? params : params_saved)
    end

    return (; est_res, path, dimcode, params_saved)
end

"Return '<project root>/output/plots/<estimation results base name>'."
function plots_dir_for_results(params::NamedTuple)
    # results file lives under ../generated data/<basename>_block_est_*.jld2
    respath   = estimation_results_path(params)
    gen_dir   = dirname(respath)                 # .../<project>/generated data
    proj_root = dirname(gen_dir)                 # .../<project>
    stem      = splitext(basename(respath))[1]   # results base name without extension
    return joinpath(proj_root, "output", "plots", stem)
end

"Ensure plots directory exists; return it."
function ensure_plots_dir!(params::NamedTuple)
    dir = plots_dir_for_results(params)
    isdir(dir) || mkpath(dir)
    return dir
end

"Base name (no extension) of the estimation results file."
results_basename(params::NamedTuple) = splitext(basename(estimation_results_path(params)))[1]

# --- Plot file paths (PNG) ---

# make a tidy, filesystem-safe slug (e.g., 'OLS FE' -> 'ols_fe')
_slug(s::AbstractString) = lowercase(replace(s, r"[^A-Za-z0-9]+" => "_"))

"…/<results base>/ <base>_<estimator>_<n>_beta_hat_dist.png"
function plot_path_estimator_dist(params::NamedTuple; estimator::AbstractString, sample_n::Integer)
    dir  = ensure_plots_dir!(params)
    base = results_basename(params)
    est  = _slug(estimator)
    return joinpath(dir, string(base, "_", est, "_", sample_n, "_beta_hat_dist.png"))
end

"…/<results base>/ <base>_<bias|variance>_asym_analysis.png"
function plot_path_asymptotic(params::NamedTuple; choice::Symbol)
    dir  = ensure_plots_dir!(params)
    base = results_basename(params)
    tag  = choice === :bias ? "bias" : "variance"
    return joinpath(dir, string(base, "_", tag, "_asym_analysis.png"))
end

"…/<results base>/ <base>_mult_asym_var.png"
function plot_path_multi_variance(params::NamedTuple)
    dir  = ensure_plots_dir!(params)
    base = results_basename(params)
    return joinpath(dir, string(base, "_mult_asym_var.png"))
end

"…/<results base>/ <base>_var_ratios.png"
function plot_path_variance_ratio(params::NamedTuple)
    dir  = ensure_plots_dir!(params)
    base = results_basename(params)
    return joinpath(dir, string(base, "_var_ratios.png"))
end

end # module
