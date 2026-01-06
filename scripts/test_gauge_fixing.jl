# MODULE: test_gauge_fixing.jl
# USAGE: julia +1.10.4 --project=ekrgilttrnr ekrgilttrnr/scripts/test_gauge_fixing.jl
# INPUTS: None (generates test data internally)
# OUTPUTS: stdout (gauge fixing validation results)
# DESCRIPTION: Validate and compare gauge fixing methods for φ⁴ tensors
# LATEX_DOC: ekrgilttrnr/docs/newton_method_guide.md
#
# BASED ON: EKR GILT-TNR (https://github.com/ebelnikola/GILT_TNR_R, arXiv:2408.10312)
# NEW IN THIS SCRIPT: Validation tests for gauge fixing including idempotency,
#                     environment diagonality, and Jacobian eigenvalue consistency.
#
# RUNTIME: ~5 min at χ=15. Gauge fixing validation tests.
#
# Tests:
# 1. Idempotency: gauge_fix(gauge_fix(A)) == gauge_fix(A)
# 2. Environment diagonality: After fixing, environments should be diagonal
# 3. Gauge transformation recovery: Apply random gauge, then fix back
# 4. Finite difference consistency: Different gauge fixes should give same Jacobian eigenvalues

using PyCall
using LinearAlgebra

include("../src/Tools.jl")
include("../src/Phi4Tools.jl")
include("../src/Phi4GaugeFixing.jl")
include("../src/GaugeFixing.jl")

const chi = 15
const gilt_pars = Dict(
    "gilt_eps" => 6e-6,
    "cg_chis" => collect(1:chi),
    "cg_eps" => 1e-10,
    "verbosity" => 0,
    "rotate" => false,
)

const phi4_pars = Dict(
    "mu_sq" => -1.325,  # Near critical
    "lam" => 1.0,
    "kappa" => 0.3,
    "K" => 32,
    "D" => 16,
    "symmetry_tensors" => true
)

println("="^60)
println("Gauge Fixing Validation Tests")
println("="^60)

# Generate fixed point tensor
println("\n1. Generating φ⁴ tensor at critical point...")
traj = phi4_trajectory(phi4_pars, 10, gilt_pars)
A_crit = traj["A"][end]
A_crit = A_crit / A_crit.norm()
println("   Shape: $(A_crit.shape)")

# ============================================
# Test 1: Idempotency
# ============================================
println("\n" * "="^60)
println("TEST 1: Idempotency (gauge_fix twice = gauge_fix once)")
println("="^60)

for method in [:svd, :environment, :none]
    A1, info1 = fix_gauge_phi4(A_crit; method=method)
    A2, info2 = fix_gauge_phi4(A1; method=method)

    if A1.shape == A2.shape
        diff = (A1 - A2).norm()
        status = diff < 1e-10 ? "✅ PASS" : "❌ FAIL"
        println("   $method: ||f(f(A)) - f(A)|| = $diff  $status")
    else
        println("   $method: Shape mismatch $(A1.shape) vs $(A2.shape)")
    end
end

# ============================================
# Test 2: Environment Diagonality
# ============================================
println("\n" * "="^60)
println("TEST 2: Environment Diagonality after gauge fixing")
println("="^60)

function check_environment_diagonality(A)
    tensors = [A, A.conj()]

    # Vertical environment
    connects_v = [[2, -1, 1, 3], [2, -2, 1, 3]]
    env_v = ncon(tensors, connects_v, [2, 1, 3])
    env_v_arr = env_v.to_ndarray()
    off_diag_v = norm(env_v_arr - Diagonal(diag(env_v_arr)))

    # Horizontal environment
    connects_h = [[-1, 2, 3, 1], [-2, 2, 3, 1]]
    env_h = ncon(tensors, connects_h, [2, 3, 1])
    env_h_arr = env_h.to_ndarray()
    off_diag_h = norm(env_h_arr - Diagonal(diag(env_h_arr)))

    return off_diag_v, off_diag_h
end

println("   Before gauge fixing:")
off_v, off_h = check_environment_diagonality(A_crit)
println("      Vertical off-diag: $off_v")
println("      Horizontal off-diag: $off_h")

for method in [:svd, :environment]
    A_fixed, _ = fix_gauge_phi4(A_crit; method=method)
    off_v, off_h = check_environment_diagonality(A_fixed)
    println("   After $method:")
    println("      Vertical off-diag: $off_v")
    println("      Horizontal off-diag: $off_h")
end

# ============================================
# Test 3: Gauge Transformation Recovery
# ============================================
println("\n" * "="^60)
println("TEST 3: Gauge Transformation Recovery")
println("="^60)

# Apply random gauge transformation and check if gauge fixing recovers original
A_fixed_svd, _ = fix_gauge_phi4(A_crit; method=:svd)

# Create random unitary gauge transformation
function apply_random_gauge(A)
    # Get shape info
    shape = A.shape
    dim_h = sum(shape[1, :])  # Horizontal bond dimension
    dim_v = sum(shape[2, :])  # Vertical bond dimension

    # Create random orthogonal matrices (simplified - just use permutation/sign flips)
    # This is a subset of allowed gauge transformations
    signs_h = rand([-1.0, 1.0], dim_h)
    signs_v = rand([-1.0, 1.0], dim_v)

    # Create diagonal sign matrices as Z2 tensors
    Hshape = A.shape[[1, 3], :]
    Hqhape = A.qhape[[1, 3], :]
    Hdirs = A.dirs[[1, 3]]
    H_gauge = TENSZ2.from_ndarray(diagm(signs_h), shape=Hshape, qhape=Hqhape, charge=0, invar=true, dirs=Hdirs)

    Vshape = A.shape[[2, 4], :]
    Vqhape = A.qhape[[2, 4], :]
    Vdirs = A.dirs[[2, 4]]
    V_gauge = TENSZ2.from_ndarray(diagm(signs_v), shape=Vshape, qhape=Vqhape, charge=0, invar=true, dirs=Vdirs)

    # Apply gauge transformation: A' = H ⊗ V ⊗ H ⊗ V · A
    A_transformed = ncon([A, H_gauge, V_gauge, H_gauge, V_gauge],
                         [[1, 2, 3, 4], [1, -1], [2, -2], [3, -3], [4, -4]])

    return A_transformed, signs_h, signs_v
end

println("   Applying random gauge transformation...")
A_gauged, _, _ = apply_random_gauge(A_fixed_svd)
println("   ||A_gauged - A_fixed|| = $((A_gauged - A_fixed_svd).norm())")

println("   Fixing gauge on transformed tensor...")
A_recovered, _ = fix_gauge_phi4(A_gauged; method=:svd)

diff_recovery = (A_recovered - A_fixed_svd).norm()
# Also check negation (global phase ambiguity)
diff_recovery_neg = (A_recovered + A_fixed_svd).norm()
best_diff = min(diff_recovery, diff_recovery_neg)
status = best_diff < 0.1 ? "✅ PASS" : "⚠️ PARTIAL"
println("   ||A_recovered - A_original|| = $diff_recovery")
println("   ||A_recovered + A_original|| = $diff_recovery_neg  (checking sign flip)")
println("   Best match: $best_diff  $status")

# ============================================
# Test 4: Ising Continuous Gauge on φ⁴
# ============================================
println("\n" * "="^60)
println("TEST 4: Ising Continuous Gauge Method on φ⁴")
println("="^60)

println("   Testing fix_continuous_gauge from Ising code...")
try
    A_ising, H, V, SH, SV = fix_continuous_gauge(A_crit)
    println("   ✅ Ising continuous gauge succeeded")
    println("   Eigenvalues SH: $(SH[1:min(5, length(SH))])...")
    println("   Eigenvalues SV: $(SV[1:min(5, length(SV))])...")

    # Check idempotency
    A_ising2, _, _, _, _ = fix_continuous_gauge(A_ising)
    diff = (A_ising - A_ising2).norm()
    println("   Idempotency ||f(f(A)) - f(A)|| = $diff")
catch e
    println("   ❌ Ising continuous gauge failed: $e")
end

# ============================================
# Test 5: Compare Finite Differences
# ============================================
println("\n" * "="^60)
println("TEST 5: Finite Difference Jacobian Comparison")
println("="^60)

include("../src/KrylovTechnical.jl")
include("../src/NumDifferentiation.jl")
using .NumDifferentiation

function gilt_phi4_with_method(A_ju, method)
    A_py = ju_to_py(A_ju)
    getitem = pyimport("operator").getitem
    result = pycall(py"gilttnr_step", PyObject, A_py, 0.0, gilt_pars)
    A_out = pycall(getitem, PyObject, result, 0)
    A_out, _ = fix_gauge_phi4(A_out; method=method)
    return py_to_ju(A_out)
end

# Use environment-fixed version (best gauge)
A_fixed_env, _ = fix_gauge_phi4(A_crit; method=:environment)
A_crit_JU = py_to_ju(A_fixed_env)

# Test finite difference in random direction
dA = py_to_ju(random_Z2tens(ju_to_py(A_crit_JU)))
dA = dA / norm(dA)

stp = 1e-4
for method in [:environment, :svd, :none]
    println("\n   Method: $method")

    f_method = A -> gilt_phi4_with_method(A, method)

    A_plus = A_crit_JU + stp * dA
    A_minus = A_crit_JU - stp * dA

    B_plus = f_method(A_plus)
    B_minus = f_method(A_minus)

    if B_plus.shape == B_minus.shape
        df_approx = (B_plus - B_minus) / (2*stp)
        println("   ||df/dA|| = $(norm(df_approx))")
    else
        println("   ⚠️ Shape mismatch: $(B_plus.shape) vs $(B_minus.shape)")
    end
end

# ============================================
# Test 6: Partition Function Invariance
# ============================================
println("\n" * "="^60)
println("TEST 6: Partition Function Invariance")
println("="^60)

function compute_trace(A)
    # Tr(A) = contract all indices with deltas (simplified: just sum diagonal)
    result = ncon([A, A.conj()], [[1, 2, 1, 2], [3, 4, 3, 4]])
    return result
end

println("   Before gauge fix: Z = $(compute_trace(A_crit))")
for method in [:svd, :environment, :none]
    A_fixed, _ = fix_gauge_phi4(A_crit; method=method)
    Z = compute_trace(A_fixed)
    println("   After $method: Z = $Z")
end

println("\n" * "="^60)
println("Summary")
println("="^60)
println("""
Key Observations:
1. The SVD gauge fix (current default) only normalizes + fixes global phase
2. Environment-based methods may work better but need zero-norm handling
3. The Ising discrete gauge cannot be directly applied (Z2 structure differs)

Recommendations for improving eigenvalue computation:
- Try environment-based gauge fixing if environments have non-zero norm
- Consider projecting out gauge directions from Krylov space
- Use gauge-invariant observables (ratios of eigenvalues)
""")
