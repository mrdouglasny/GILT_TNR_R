# MODULE: debug_phi4_linearized.jl
# USAGE: julia +1.10.4 --project=ekrgilttrnr ekrgilttrnr/scripts/debug_phi4_linearized.jl
# INPUTS: None (parameters hardcoded)
# OUTPUTS: stdout (debug output for linearized RG)
# DESCRIPTION: Debug script for φ⁴ linearized RG Jacobian computation.
# LATEX_DOC: ekrgilttrnr/docs/newton_method_guide.md
#
# BASED ON: EKR GILT-TNR (https://github.com/ebelnikola/GILT_TNR_R, arXiv:2408.10312),
#           ekrgilttrnr/scripts/differentiability_test.jl
# NEW IN THIS SCRIPT: Debug script for φ⁴ tensor Jacobian computation via
#                     finite differences to validate eigenvalue extraction.
#
# RUNTIME: ~10 min at χ=15. Debug script for linearized RG.

using PyCall
using LinearAlgebra

include("../src/Tools.jl")
include("../src/Phi4Tools.jl")
include("../src/Phi4GaugeFixing.jl")
include("../src/KrylovTechnical.jl")
include("../src/NumDifferentiation.jl")
using .NumDifferentiation

const chi = 15
const gilt_pars = Dict(
    "gilt_eps" => 6e-6,
    "cg_chis" => collect(1:chi),
    "cg_eps" => 1e-10,
    "verbosity" => 0,
    "rotate" => false,
)

const phi4_pars = Dict(
    "mu_sq" => -1.275,  # From binary search
    "lam" => 1.0,
    "kappa" => 0.3,
    "K" => 32,
    "D" => 16,
    "symmetry_tensors" => true
)

println("Running trajectory with increasing steps to find stable bond dimension...")
for n_steps in [5, 10, 15, 20]
    traj = phi4_trajectory(phi4_pars, n_steps, gilt_pars)
    A_crit = traj["A"][end]
    A_py = A_crit / A_crit.norm()
    shape = A_py.shape
    println("n_steps=$n_steps: shape = $shape")
end

println("\nUsing 10 steps for fixed point approximation (stable at this μ²)...")
traj = phi4_trajectory(phi4_pars, 10, gilt_pars)
A_crit = traj["A"][end]
A_crit, _ = fix_gauge_phi4(A_crit; method=:svd)
global A_crit_JU = py_to_ju(A_crit)

function gilt_phi4(A_ju)
    A_py = ju_to_py(A_ju)
    getitem = pyimport("operator").getitem
    result = pycall(py"gilttnr_step", PyObject, A_py, 0.0, gilt_pars)
    A_out = pycall(getitem, PyObject, result, 0)
    A_out, _ = fix_gauge_phi4(A_out; method=:svd)
    return py_to_ju(A_out)
end

function gilt_phi4_no_gauge(A_ju)
    A_py = ju_to_py(A_ju)
    getitem = pyimport("operator").getitem
    result = pycall(py"gilttnr_step", PyObject, A_py, 0.0, gilt_pars)
    A_out = pycall(getitem, PyObject, result, 0)
    A_out = A_out / A_out.norm()
    return py_to_ju(A_out)
end

println("\n=== Test 1: Does RG step change tensor? ===")
B = gilt_phi4(A_crit_JU)
println("A_crit_JU shape: ", A_crit_JU.shape)
println("B shape: ", B.shape)
if A_crit_JU.shape == B.shape
    diff1 = norm(B - A_crit_JU)
    println("||gilt(A) - A|| = ", diff1)
    println("If ~0: tensor is at fixed point")
else
    println("ERROR: Shape mismatch! RG changes tensor dimensions.")
    println("This breaks the linearization. Need to ensure fixed bond dimension.")
end

# Skip remaining tests if shapes don't match
if A_crit_JU.shape != B.shape
    println("\nSkipping remaining tests due to shape mismatch")
    println("SOLUTION: Run more RG steps until bond dimension stabilizes at chi")
    exit(0)
end

println("\n=== Test 2: Finite difference in random direction ===")
# Create random direction using same method as Ising (random_Z2tens)
global dA = py_to_ju(random_Z2tens(ju_to_py(A_crit_JU)))
dA = dA / norm(dA)

for stp in [1e-2, 1e-3, 1e-4, 1e-5]
    global A_plus = A_crit_JU + stp * dA
    global A_minus = A_crit_JU - stp * dA

    global B_plus = gilt_phi4(A_plus)
    global B_minus = gilt_phi4(A_minus)

    df_approx = (B_plus - B_minus) / (2*stp)
    println("stp=$stp: ||df|| = ", norm(df_approx))
end

println("\n=== Test 3: Check if gauge fixing is the issue ===")
stp = 1e-4
global A_plus = A_crit_JU + stp * dA
global A_minus = A_crit_JU - stp * dA

global B_plus_no_gf = gilt_phi4_no_gauge(A_plus)
global B_minus_no_gf = gilt_phi4_no_gauge(A_minus)
df_no_gauge = (B_plus_no_gf - B_minus_no_gf) / (2*stp)
println("Without gauge fix: ||df|| = ", norm(df_no_gauge))

global B_plus_gf = gilt_phi4(A_plus)
global B_minus_gf = gilt_phi4(A_minus)
df_with_gauge = (B_plus_gf - B_minus_gf) / (2*stp)
println("With gauge fix: ||df|| = ", norm(df_with_gauge))

println("\n=== Test 4: Check B_plus vs B_minus difference ===")
println("||B_plus - B_minus|| (no gauge) = ", norm(B_plus_no_gf - B_minus_no_gf))
println("||B_plus - B_minus|| (with gauge) = ", norm(B_plus_gf - B_minus_gf))
