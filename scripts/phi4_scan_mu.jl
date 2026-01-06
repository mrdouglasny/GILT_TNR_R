# MODULE: phi4_scan_mu.jl
# USAGE: julia +1.10.4 --project=ekrgilttrnr ekrgilttrnr/scripts/phi4_scan_mu.jl
# INPUTS: None (parameters hardcoded)
# OUTPUTS: stdout (μ² scan results with eigenvalues)
# DESCRIPTION: Scan μ² values to find phi4 critical point via eigenvalue structure.
#              The critical point should show eigenvalues matching 2D Ising CFT:
#              λ_σ ≈ 3.668, λ_ε ≈ 2.0, λ_T ≈ 1.0
# LATEX_DOC: ekrgilttrnr/docs/newton_method_guide.md
#
# BASED ON: EKR GILT-TNR (https://github.com/ebelnikola/GILT_TNR_R, arXiv:2408.10312),
#           ekrgilttrnr/scripts/critical_temperature.jl adapted for φ⁴
# NEW IN THIS SCRIPT: Applies GILT-TNR to φ⁴ scalar field theory to locate critical
#                     μ² by scanning and checking eigenvalue spectrum.
#
# RUNTIME: ~30 min scanning 10-20 μ² points at χ=15. Scales as O(χ⁶ × n_points).

using Printf

# Load tools
include("../src/Tools.jl")
include("../src/Phi4Tools.jl")
include("../src/GaugeFixing.jl")
include("../src/KrylovTechnical.jl")

using LinearAlgebra
using KrylovKit

# Parameters
chi = 15
gilt_eps = 6e-6
cg_eps = 1e-10
lam = 1.0
kappa = 0.3
K = 32
D = 16
number_of_initial_steps = 8
N = 5
krylovdim = 15
stp = 1e-4

gilt_pars = Dict(
    "gilt_eps" => gilt_eps,
    "cg_chis" => collect(1:chi),
    "cg_eps" => cg_eps,
    "verbosity" => 0,
    "rotate" => false,
)

# μ² values to scan
mu_sq_values = [-1.35, -1.30, -1.25, -1.20, -1.15, -1.10, -1.05]

println("="^70)
println("φ⁴ Critical Point Scan")
println("="^70)
println("Parameters: λ=$lam, κ=$kappa, χ=$chi")
println("Expected at critical point: λ_σ≈3.668, λ_ε≈2.0, λ_T≈1.0")
println("="^70)
println()

results = []

for mu_sq in mu_sq_values
    print("μ² = $mu_sq: ")
    flush(stdout)

    try
        phi4_pars = Dict(
            "mu_sq" => mu_sq,
            "lam" => lam,
            "kappa" => kappa,
            "K" => K,
            "D" => D,
            "symmetry_tensors" => true
        )

        # Run RG
        trajectory_result = phi4_trajectory(phi4_pars, number_of_initial_steps, gilt_pars)
        A_crit = trajectory_result["A"][end]

        # Gauge fix
        A_crit, _ = fix_continuous_gauge(A_crit)
        A_crit, _, _ = fix_discrete_gauge(A_crit; tol=1e-7)
        A_crit /= A_crit.norm()
        A_crit_JU = py_to_ju(A_crit)

        # Linearized RG
        function linearized_RG(dA)
            dA_py = ju_to_py(dA)
            result = gilttnr_step(A_crit + stp * dA_py, gilt_pars) - gilttnr_step(A_crit - stp * dA_py, gilt_pars)
            result_fixed, _ = fix_continuous_gauge(result)
            result_fixed, _, _ = fix_discrete_gauge(result_fixed; tol=1e-7)
            result_fixed /= result_fixed.norm()
            return py_to_ju(result_fixed) / (2 * stp)
        end

        # Eigenvalues
        vals, _, _ = eigsolve(linearized_RG, A_crit_JU, N, :LM; krylovdim=krylovdim, verbosity=0)

        # Extract top eigenvalues by magnitude
        sorted_vals = sort(vals, by=abs, rev=true)
        top3 = sorted_vals[1:min(3, length(sorted_vals))]

        @printf("λ = [%.3f, %.3f, %.3f]\n", abs(top3[1]), abs(top3[2]), abs(top3[3]))

        push!(results, (mu_sq=mu_sq, eigenvalues=sorted_vals))

    catch e
        println("ERROR: $e")
        push!(results, (mu_sq=mu_sq, eigenvalues=nothing))
    end
end

println()
println("="^70)
println("Summary")
println("="^70)
println()
println("μ²        |λ₁|     |λ₂|     |λ₃|     Quality")
println("-"^60)

for r in results
    if r.eigenvalues !== nothing
        v = r.eigenvalues
        l1, l2, l3 = abs(v[1]), abs(v[2]), abs(v[3])

        # Quality metric: how close to CFT values (3.668, 2.0, 1.0)
        # Simple metric: distance from expected
        quality = sqrt((l1 - 3.668)^2 + (l2 - 2.0)^2 + (l3 - 1.0)^2)

        @printf("%.2f    %7.3f  %7.3f  %7.3f   %.3f\n", r.mu_sq, l1, l2, l3, quality)
    else
        @printf("%.2f    FAILED\n", r.mu_sq)
    end
end

println()
println("Lower quality = closer to CFT values (3.668, 2.0, 1.0)")
