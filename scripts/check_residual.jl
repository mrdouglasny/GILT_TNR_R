# MODULE: check_residual.jl
# USAGE: julia +1.10.4 --project=ekrgilttrnr ekrgilttrnr/scripts/check_residual.jl
# INPUTS: None (generates tensors internally)
# OUTPUTS: stdout (residual ||RG(A) - A|| for Ising and φ⁴)
# DESCRIPTION: Check fixed point quality for both Ising and φ⁴ models.
# LATEX_DOC: ekrgilttrnr/docs/newton_method_guide.md
#
# BASED ON: EKR GILT-TNR (https://github.com/ebelnikola/GILT_TNR_R, arXiv:2408.10312),
#           ekrgilttrnr/scripts/check_ising_residual.jl
# NEW IN THIS SCRIPT: Compares fixed point residuals between Ising and φ⁴
#                     to validate that both models have converged.
#
# RUNTIME: ~10 min at χ=30. Diagnostic script.

using PyCall
using LinearAlgebra

include("../src/Tools.jl")
include("../src/GaugeFixing.jl")
include("../src/Phi4Tools.jl")
include("../src/Phi4GaugeFixing.jl")

const chi = 30  # Need chi=30 for Ising stability (chi=15 collapses)
const gilt_pars = Dict("gilt_eps" => 6e-6, "cg_chis" => collect(1:chi), "cg_eps" => 1e-10, "verbosity" => 0, "rotate" => false)

# Helper to compute residual
function compute_residual(A, gilt_pars; gauge_method=:ising)
    getitem = pyimport("operator").getitem
    result = pycall(py"gilttnr_step", PyObject, A, 0.0, gilt_pars)
    A_next = pycall(getitem, PyObject, result, 0)

    if gauge_method == :ising
        A_gf, _, _, _, _ = fix_continuous_gauge(A)
        A_gf, _, _, _ = fix_discrete_gauge(A_gf)
        A_gf = A_gf / A_gf.norm()

        A_next_gf, _, _, _, _ = fix_continuous_gauge(A_next)
        A_next_gf, _, _, _ = fix_discrete_gauge(A_next_gf)
        A_next_gf = A_next_gf / A_next_gf.norm()
    else
        A_gf, _ = fix_gauge_phi4(A; method=:environment)
        A_next_gf, _ = fix_gauge_phi4(A_next; method=:environment)
    end

    if A_gf.shape == A_next_gf.shape
        return (A_gf - A_next_gf).norm()
    else
        return NaN  # Shape changed
    end
end

println("="^60)
println("Fixed Point Residual ||RG(A) - A|| Comparison")
println("="^60)

# =============================================
# Part 1: Ising - create tensor directly in Python to avoid type issues
# =============================================
println("\n--- ISING (χ=$chi) ---")

py"""
import numpy as np
from numpy import sinh, exp
from ncon import ncon

def create_ising_tensor_isotropic(beta):
    '''Create Ising tensor at inverse temperature beta'''
    hamiltonian = np.array([[-1, 1], [1, -1]])
    boltz = np.exp(-beta * hamiltonian)
    A = np.einsum('ab,bc,cd,da->abcd', boltz, boltz, boltz, boltz)

    # Gauge transform to Z2 basis
    u = np.array([[1, 1], [1, -1]]) / np.sqrt(2)
    u_dg = u.T.conjugate()
    A = ncon((A, u, u, u_dg, u_dg), ([1,2,3,4], [-1,1], [-2,2], [3,-3], [4,-4]))
    A[abs(A) < 1e-10] = 0
    return A

# Critical beta for isotropic Ising: sinh(2*beta_c)^2 = 1 => beta_c = 0.4406868...
beta_c = 0.5 * np.arcsinh(1.0)
A_ising_init = create_ising_tensor_isotropic(beta_c)
"""

# Convert to Z2 tensor
TensorZ2 = pyimport("tensors").TensorZ2
A_ising_py = py"A_ising_init"
dim, qim = [1, 1], [0, 1]
A_ising = TensorZ2.from_ndarray(A_ising_py, shape=[dim, dim, dim, dim],
                                 qhape=[qim, qim, qim, qim], dirs=[1, 1, -1, -1])

println("Initial tensor shape: $(A_ising.shape)")

# Run RG and check residuals
for n_steps in [5, 10, 15, 20, 23, 25]
    # Run trajectory
    A = A_ising
    for i in 1:n_steps
        getitem = pyimport("operator").getitem
        result = pycall(py"gilttnr_step", PyObject, A, 0.0, gilt_pars)
        A = pycall(getitem, PyObject, result, 0)
    end

    shape_sum = sum(A.shape[1, :])
    if shape_sum < chi
        println("n=$n_steps: Tensor collapsed to dim $shape_sum")
        continue
    end

    residual = compute_residual(A, gilt_pars; gauge_method=:ising)
    println("n=$n_steps: ||RG(A) - A|| = $(round(residual, digits=6))")
end

# =============================================
# Part 2: φ⁴ (uses chi=15)
# =============================================
chi_phi4 = 15
gilt_pars_phi4 = Dict("gilt_eps" => 6e-6, "cg_chis" => collect(1:chi_phi4), "cg_eps" => 1e-10, "verbosity" => 0, "rotate" => false)

println("\n--- φ⁴ (χ=$chi_phi4, μ²=-1.325) ---")

phi4_pars = Dict("mu_sq" => -1.325, "lam" => 1.0, "kappa" => 0.3,
                 "K" => 32, "D" => 16, "symmetry_tensors" => true)

for n_steps in [5, 10, 12, 15]
    traj = phi4_trajectory(phi4_pars, n_steps, gilt_pars_phi4)
    A = traj["A"][end]

    shape_sum = sum(A.shape[1, :])
    if shape_sum < chi_phi4
        println("n=$n_steps: Tensor collapsed to dim $shape_sum")
        continue
    end

    residual = compute_residual(A, gilt_pars_phi4; gauge_method=:phi4)
    println("n=$n_steps: ||RG(A) - A|| = $(round(residual, digits=6))")
end

println("\n" * "="^60)
println("INTERPRETATION:")
println("  Ising residual should decrease → converging to fixed point")
println("  φ⁴ residual ~1 means NOT at fixed point (explains bad eigenvalues)")
println("="^60)
