# MODULE: phi4_mu_scan.jl
# USAGE: julia +1.10.4 --project=ekrgilttrnr ekrgilttrnr/scripts/phi4_mu_scan.jl <config>
# INPUTS: ekrgilttrnr/configs/phi4_mu_scan/<config>.toml
# OUTPUTS: ekrgilttrnr/data/phi4_mu_scan/<config>.csv
# DESCRIPTION: Scan μ² values to find critical point for φ⁴ at χ=30.
#              Reports RG steps before collapse and fixed point residual.
# LATEX_DOC: ekrgilttrnr/docs/newton_method_guide.md
#
# BASED ON: EKR GILT-TNR (https://github.com/ebelnikola/GILT_TNR_R, arXiv:2408.10312),
#           ekrgilttrnr/scripts/phi4_scan_mu.jl (simpler version)
# NEW IN THIS SCRIPT: Config-driven μ² scan with SLURM array job support for
#                     parallel parameter exploration on cluster.
#
# RUNTIME: ~30-60 min per μ² point at χ=30. Use cluster for multi-point scans.

using Serialization
using LinearAlgebra
using PyCall

# Load ProjectUtils for config handling
include("../../src/julia/ProjectUtils.jl")
using .ProjectUtils

# Load tools
include("../src/Tools.jl")
include("../src/Phi4Tools.jl")
include("../src/Phi4GaugeFixing.jl")
include("../src/GaugeFixing.jl")

# Setup with config
config_name, config, paths = setup_with_config("phi4_mu_scan", ARGS; base_dir="ekrgilttrnr")

# Extract parameters
chi = get(config, "chi", 30)
gilt_eps = get(config, "gilt_eps", 6e-6)
cg_eps = get(config, "cg_eps", 1e-10)
rotate = get(config, "rotate", false)
lam = get(config, "lam", 1.0)
kappa = get(config, "kappa", 0.3)
K = get(config, "K", 32)
D = get(config, "D", 16)
mu_sq = get(config, "mu_sq", -1.3)
max_steps = get(config, "max_steps", 40)
verbosity = get(config, "verbosity", 1)

gilt_pars = Dict(
    "gilt_eps" => gilt_eps,
    "cg_chis" => collect(1:chi),
    "cg_eps" => cg_eps,
    "verbosity" => 0,
    "rotate" => rotate,
)

phi4_pars = Dict(
    "mu_sq" => mu_sq,
    "lam" => lam,
    "kappa" => kappa,
    "K" => K,
    "D" => D,
    "symmetry_tensors" => true
)

# Log start
log_msg = """
==================================================
φ⁴ μ² Scan - Single Point
Config: $config_name
==================================================
Parameters:
  μ² = $mu_sq, λ = $lam, κ = $kappa
  χ = $chi, gilt_eps = $gilt_eps
  K = $K, D = $D
  max_steps = $max_steps
"""
log_info(log_msg, paths["log"])
if verbosity > 0
    println(log_msg)
end

# Run RG and track residuals
getitem = pyimport("operator").getitem

# Create initial tensor
A0 = initial_tensor_phi4(phi4_pars)

function run_scan(A_init, max_steps, chi, gilt_pars, paths, verbosity)
    results = Dict{Int, Any}()
    collapse_step = -1
    A = A_init
    init_dim = sum(A_init.shape[1, :])
    prev_dim = init_dim

    for step in 1:max_steps
        # Run one RG step
        result = pycall(py"gilttnr_step", PyObject, A, 0.0, gilt_pars)
        A_next = pycall(getitem, PyObject, result, 0)

        # Check for collapse: dimension DECREASING from previous (not just below chi)
        shape_sum = sum(A_next.shape[1, :])

        if verbosity > 0
            println("Step $step: dim $prev_dim -> $shape_sum")
        end

        # True collapse: dimension decreased significantly from previous step
        if shape_sum < prev_dim * 0.8 && shape_sum < chi
            collapse_step = step
            log_info("Step $step: Tensor collapsed: dim $prev_dim -> $shape_sum", paths["log"])
            if verbosity > 0
                println("Step $step: COLLAPSED from $prev_dim to $shape_sum")
            end
            break
        end

        prev_dim = shape_sum

        # Compute residual every 5 steps
        if step % 5 == 0
            # Gauge fix current and next
            A_gf, _ = fix_gauge_phi4(A; method=:environment)
            A_next_gf, _ = fix_gauge_phi4(A_next; method=:environment)

            if A_gf.shape == A_next_gf.shape
                residual = (A_gf - A_next_gf).norm()
                results[step] = residual
                log_info("Step $step: ||RG(A) - A|| = $residual", paths["log"])
                if verbosity > 0
                    println("Step $step: ||RG(A) - A|| = $residual")
                end
            else
                results[step] = NaN
                log_info("Step $step: Shape mismatch", paths["log"])
            end
        end

        A = A_next
    end

    return results, collapse_step
end

results, collapse_step = run_scan(A0, max_steps, chi, gilt_pars, paths, verbosity)

# Final summary
summary = """

==================================================
SUMMARY
==================================================
μ² = $mu_sq
Collapse step: $(collapse_step > 0 ? collapse_step : "none (survived $max_steps steps)")
Final residuals: $results

INTERPRETATION:
- Collapse < 20 steps: Too far from critical
- Collapse 20-30 steps: Close to critical
- No collapse + residual decreasing: At or near critical point
==================================================
"""
log_info(summary, paths["log"])
if verbosity > 0
    println(summary)
end

# Save results
result_data = Dict(
    "mu_sq" => mu_sq,
    "chi" => chi,
    "collapse_step" => collapse_step,
    "residuals" => results,
    "config" => config,
)
serialize(paths["data"], result_data)

println("Results saved to: $(paths["data"])")
