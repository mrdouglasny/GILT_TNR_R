# MODULE: scan_mu_quick.jl
# USAGE: julia +1.10.4 --project=ekrgilttrnr ekrgilttrnr/scripts/scan_mu_quick.jl
# INPUTS: None (parameters hardcoded)
# OUTPUTS: stdout (quick μ² scan results)
# DESCRIPTION: Quick μ² scan to find φ⁴ critical point region.
# LATEX_DOC: ekrgilttrnr/docs/newton_method_guide.md
#
# BASED ON: EKR GILT-TNR (https://github.com/ebelnikola/GILT_TNR_R, arXiv:2408.10312)
# NEW IN THIS SCRIPT: Low-precision quick scan to identify critical μ² region
#                     before running expensive high-precision binary search.
#
# RUNTIME: ~5-10 min at χ=12. Quick exploratory scan.

using PyCall
using LinearAlgebra

include("../src/Tools.jl")
include("../src/Phi4Tools.jl")

const chi = 12
const gilt_pars = Dict(
    "gilt_eps" => 6e-6,
    "cg_chis" => collect(1:chi),
    "cg_eps" => 1e-10,
    "verbosity" => 0,
    "rotate" => false,
)

# Scan μ² to find where tensor dimension stabilizes at chi (critical region)
println("Scanning μ² to find critical region...")
println("Looking for μ² where bond dimension stays at chi=$chi after RG steps")
println()

# Fine scan around -1.3 region
for mu_sq in [-1.20, -1.25, -1.27, -1.28, -1.29, -1.30, -1.31, -1.32, -1.33, -1.35, -1.40, -1.45]
    phi4_pars = Dict(
        "mu_sq" => mu_sq,
        "lam" => 1.0,
        "kappa" => 0.3,
        "K" => 32,
        "D" => 16,
        "symmetry_tensors" => true
    )

    # Run RG for several steps
    traj = phi4_trajectory(phi4_pars, 10, gilt_pars)
    A = traj["A"][end]
    shape = A.shape

    # Check max bond dimension
    max_chi = maximum(sum(shape, dims=2))

    if max_chi == chi
        println("μ² = $mu_sq: shape=$shape ✓ CRITICAL CANDIDATE")
    elseif max_chi == 1
        println("μ² = $mu_sq: shape=$shape (collapsed to trivial)")
    else
        println("μ² = $mu_sq: shape=$shape (max_chi=$max_chi)")
    end
end
