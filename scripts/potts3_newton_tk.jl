# MODULE: potts3_newton_tk.jl
# USAGE: julia --project=ekrgilttrnr/TensorKitPotts ekrgilttrnr/scripts/potts3_newton_tk.jl [options]
# INPUTS: Critical relT from bisection (~1.00291)
# OUTPUTS: ekrgilttrnr/newton/potts3_*.data (serialized fixed point tensors)
# DESCRIPTION: Newton method for 3-state Potts fixed point using TensorKit Z₃ tensors.
#
# BASED ON: ekrgilttrnr/scripts/newton.jl (Ising Z₂ version)
#           ekrgilttrnr/scripts/potts3_newton.py (Python prototype)
# NEW IN THIS SCRIPT: Uses TensorKit.jl Z₃ symmetric tensors with proper gauge fixing.
#
# Method:
#   1. Load approximate fixed point from Python Gilt-TNR warmup
#   2. Convert to TensorKit Z₃ tensor
#   3. Build Jacobian approximation via Krylov eigensolver
#   4. Newton iteration with Z₃ gauge fixing
#   5. Extract scaling dimensions from converged fixed point
#
# RUNTIME: ~30-60 min at χ=16, scales as O(χ⁶)

using ArgParse
using Serialization
using LinearAlgebra
using Printf
using Dates

# Add paths
push!(LOAD_PATH, joinpath(@__DIR__, "../TensorKitPotts/src"))

using TensorKitPotts
using TensorKit
using KrylovKit
using PyCall

# =============================================================================
# Argument Parsing
# =============================================================================

settings = ArgParseSettings()
@add_arg_table! settings begin
    "--chi"
        help = "Bond dimension per Z₃ sector"
        arg_type = Int64
        default = 12
    "--relT"
        help = "Relative temperature (1.0 = critical, from bisection use ~1.00291)"
        arg_type = Float64
        default = 1.00291
    "--warmup"
        help = "Number of Gilt-TNR warmup steps"
        arg_type = Int64
        default = 6
    "--newton_iter"
        help = "Number of Newton iterations"
        arg_type = Int64
        default = 20
    "--eigensystem_size"
        help = "Number of eigenvalues for Jacobian approximation"
        arg_type = Int64
        default = 9
    "--gilt_eps"
        help = "Gilt threshold parameter"
        arg_type = Float64
        default = 1e-6
    "--diff_step"
        help = "Step size for numerical differentiation"
        arg_type = Float64
        default = 1e-4
    "--diff_order"
        help = "Order of finite difference (2, 3, or 4)"
        arg_type = Int64
        default = 2
    "--verbosity"
        help = "Verbosity level for eigensolver"
        arg_type = Int64
        default = 0
    "--simple_flow"
        help = "Use simple RG flow instead of Newton (skip Jacobian)"
        action = :store_true
end

pars = parse_args(settings; as_symbols=true)

chi = pars[:chi]
relT = pars[:relT]
n_warmup = pars[:warmup]
N_newton = pars[:newton_iter]
eigensystem_size = pars[:eigensystem_size]
gilt_eps = pars[:gilt_eps]
diff_step = pars[:diff_step]
diff_order = pars[:diff_order]
verbosity = pars[:verbosity]
simple_flow = pars[:simple_flow]

# =============================================================================
# Python Setup (for Gilt-TNR coarse-graining)
# =============================================================================

pushfirst!(PyVector(pyimport("sys")."path"), joinpath(@__DIR__, "../GiltTNR"))

py"""
import numpy as np
from GiltTNR2D_Potts import get_initial_tensor_potts_relT, get_scaldims_potts
from GiltTNR2D import gilttnr_step

def run_warmup(relT, chi, n_steps, gilt_eps):
    '''Run Gilt-TNR warmup to get approximate fixed point.'''
    potts_pars = {'q': 3, 'relT': relT, 'symmetry_tensors': False}
    gilt_pars = {
        'gilt_eps': gilt_eps,
        'cg_chis': list(range(1, chi + 1)),
        'cg_eps': 1e-10,
        'verbosity': 0,
        'rotate': False
    }

    A = get_initial_tensor_potts_relT(potts_pars)
    log_fact = 0.0

    for step in range(n_steps):
        result = gilttnr_step(A, log_fact, gilt_pars)
        A = result[0]
        log_fact = result[1]

    # Normalize
    A_arr = A.to_ndarray()
    A_arr = A_arr / np.linalg.norm(A_arr)

    return A_arr, gilt_pars

def gilt_step_array(A_arr, gilt_pars):
    '''Apply one Gilt-TNR step to numpy array, return normalized result.'''
    from tensors import Tensor
    A = Tensor.from_ndarray(A_arr)
    result = gilttnr_step(A, 0.0, gilt_pars)
    A_new = result[0].to_ndarray()
    A_new = A_new / np.linalg.norm(A_new)
    return A_new

def extract_scaldims(A_arr):
    '''Extract scaling dimensions from tensor.'''
    from tensors import Tensor
    A = Tensor.from_ndarray(A_arr)
    return get_scaldims_potts(A)
"""

# =============================================================================
# Numerical Differentiation
# =============================================================================

const df_coefficients = [
    Float64[],
    [-0.5, 0.5],
    [1/12, -2/3, 2/3, -1/12],
    [-1/60, 3/20, -3/4, 3/4, -3/20, 1/60],
]

const df_offsets = [
    Int[],
    [-1, 1],
    [-2, -1, 1, 2],
    [-3, -2, -1, 1, 2, 3],
]

"""
    df(f, x, v; stp, order)

Compute directional derivative of f at x in direction v.
"""
function df(f::Function, x, v; stp::Float64=1e-4, order::Int=2)
    coeffs = df_coefficients[order]
    offsets = df_offsets[order]
    result = zero(x)
    for (c, o) in zip(coeffs, offsets)
        result = result + c * f(x + o * stp * v)
    end
    return result / stp
end

# =============================================================================
# Gram-Schmidt and Jacobian Approximation
# =============================================================================

"""
    build_jacobian_approximation(vectors, values)

Build Jacobian approximation from eigenvectors/eigenvalues.
Handles complex conjugate pairs by constructing real 2×2 blocks.
"""
function build_jacobian_approximation(vectors::Vector{T}, values::Vector) where T
    rank = length(values)
    jac = zeros(ComplexF64, rank, rank)
    basis = Vector{T}(undef, rank)

    i = 1
    while i <= rank
        if imag(values[i]) ≈ 0
            # Real eigenvalue
            jac[i, i] = real(values[i])
            basis[i] = real(vectors[i])
            i += 1
        else
            # Complex conjugate pair
            if i < rank && conj(values[i]) ≈ values[i+1]
                λ_re = real(values[i])
                λ_im = imag(values[i])
                v1 = real(vectors[i])
                v2 = imag(vectors[i])
                n1, n2 = norm(v1), norm(v2)
                v1 = v1 / n1
                v2 = v2 / n2

                jac[i, i] = λ_re
                jac[i+1, i+1] = λ_re
                jac[i+1, i] = -λ_im * n2 / n1
                jac[i, i+1] = λ_im * n1 / n2

                basis[i] = v1
                basis[i+1] = v2
                i += 2
            else
                # Unmatched complex eigenvalue (shouldn't happen)
                @warn "Unmatched complex eigenvalue at index $i"
                jac[i, i] = values[i]
                basis[i] = vectors[i]
                i += 1
            end
        end
    end
    return jac, basis
end

"""
    gram_schmidt(basis)

Orthonormalize basis vectors using Gram-Schmidt.
Returns (transformation matrix, orthonormal basis).
"""
function gram_schmidt(basis::Vector{T}) where T
    n = length(basis)
    G = zeros(ComplexF64, n, n)
    ortho = Vector{T}(undef, n)

    ortho[1] = basis[1] / norm(basis[1])
    G[1, 1] = 1.0 / norm(basis[1])

    for k in 2:n
        v = basis[k]
        for j in 1:k-1
            v = v - dot(ortho[j], v) * ortho[j]
        end
        nv = norm(v)
        ortho[k] = v / nv

        # Build transformation matrix row
        G[k, k] = 1.0 / nv
        for j in 1:k-1
            G[k, j] = -dot(ortho[j], basis[k]) / nv
        end
    end

    return G, ortho
end

# =============================================================================
# Main Newton Method
# =============================================================================

function main()
    println("=" ^ 70)
    println("3-State Potts: Newton Method with TensorKit Z₃ Symmetry")
    println("=" ^ 70)
    println()
    println("Parameters:")
    println("  χ = $chi")
    println("  relT = $relT")
    println("  warmup steps = $n_warmup")
    println("  Newton iterations = $N_newton")
    println("  eigensystem size = $eigensystem_size")
    println("  gilt_eps = $gilt_eps")
    println()
    println("Timestamp: $(now())")
    println()

    # =========================================================================
    # Step 1: Get approximate fixed point from Python Gilt-TNR
    # =========================================================================
    println("Step 1: Running Gilt-TNR warmup...")
    @time begin
        A_init, gilt_pars_py = py"run_warmup"(relT, chi, n_warmup, gilt_eps)
    end
    println("  Initial tensor shape: $(size(A_init))")
    println()

    # =========================================================================
    # Step 2: Define RG map with gauge fixing
    # =========================================================================

    """
    One RG step with gauge fixing (working with arrays for simplicity).
    """
    function gilt_gauged(A_arr)
        # Apply Gilt-TNR step via Python
        A_new = py"gilt_step_array"(A_arr, gilt_pars_py)

        # Convert to TensorKit for gauge fixing
        V = ℂ^size(A_new, 1)
        T = TensorMap(A_new, V ⊗ V ← V ⊗ V)
        A_tk = Z3TensorTK(T, chi)

        # Apply continuous gauge fixing
        A_fixed, _, _, _, _ = fix_continuous_gauge_z3(A_tk)

        # NOTE: Discrete gauge fixing disabled for now (needs debugging)
        # A_fixed, _ = fix_discrete_gauge_z3(A_fixed)

        # Extract array and normalize
        result = convert(Array, A_fixed.tensor)
        result = result / norm(result)

        return result
    end

    # =========================================================================
    # Mode-dependent: Jacobian or Simple Flow
    # =========================================================================

    eigenvalues = ComplexF64[]
    eigenvectors = Vector{Array{ComplexF64,4}}()
    approx_rank = 0

    # Variables for Jacobian mode (will be set if not simple_flow)
    ImJ_inv_mat = nothing
    ortho_basis = nothing

    if simple_flow
        # Simple flow mode: just iterate R(A) → A with gauge fixing
        println("Step 2: Simple RG flow mode (skipping Jacobian computation)...")
        println()
    else
        # Full Newton mode with Jacobian approximation
        println("Step 2: Computing Jacobian eigensystem...")

        # Linearized RG map
        function dgilt(δA)
            return df(gilt_gauged, A_init, δA; stp=diff_step, order=diff_order)
        end

        # Random initial vector
        initial_vec = randn(ComplexF64, size(A_init)...)
        initial_vec = initial_vec / norm(initial_vec)

        # Krylov eigensolver
        println("  Running Arnoldi iteration...")
        @time begin
            eigresult = eigsolve(dgilt, initial_vec, eigensystem_size, :LM;
                                 verbosity=verbosity,
                                 issymmetric=false,
                                 ishermitian=false,
                                 krylovdim=eigensystem_size + 20)
        end

        eigenvalues = eigresult[1]
        eigenvectors = eigresult[2]

        println()
        println("  Eigenvalues of linearized RG map:")
        for (i, λ) in enumerate(eigenvalues)
            Δ = log(abs(λ)) / (2π)  # Scaling dimension
            println(@sprintf("    λ_%d = %+.6f %+.6fi  →  Δ ≈ %.4f", i, real(λ), imag(λ), Δ))
        end
        println()

        # Handle complex conjugate pairs at boundary
        approx_rank = eigensystem_size
        if length(eigenvalues) > eigensystem_size
            if conj(eigenvalues[eigensystem_size]) ≈ eigenvalues[eigensystem_size + 1]
                approx_rank = eigensystem_size + 1
            end
        end

        eigenvalues = eigenvalues[1:approx_rank]
        eigenvectors = eigenvectors[1:approx_rank]

        # =========================================================================
        # Step 3: Build Jacobian approximation
        # =========================================================================
        println("Step 3: Building Jacobian approximation (rank $approx_rank)...")

        jac_nonortho, basis_nonortho = build_jacobian_approximation(eigenvectors, eigenvalues)
        G, ortho_basis = gram_schmidt(basis_nonortho)
        jac_approx = inv(G) * jac_nonortho * G

        ImJ_inv_mat = inv(I - jac_approx)
    end

    # =========================================================================
    # Define iteration step function
    # =========================================================================
    function iterate_step(A, gilt_gauged_fn, use_jacobian, ImJ_inv_mat_local, ortho_basis_local, rank)
        R_A = gilt_gauged_fn(A)
        if use_jacobian && ImJ_inv_mat_local !== nothing
            residual = A - R_A
            # Project to subspace
            δA_in = zero(residual)
            for i in 1:rank
                δA_in = δA_in + ortho_basis_local[i] * dot(ortho_basis_local[i], residual)
            end
            δA_out = residual - δA_in
            # Apply (I-J)^{-1}
            result_in = zero(residual)
            for i in 1:rank, j in 1:rank
                result_in = result_in + ImJ_inv_mat_local[i, j] * ortho_basis_local[i] * dot(ortho_basis_local[j], residual)
            end
            correction = result_in + δA_out
            return A - correction
        else
            # Simple flow: just return R(A)
            return R_A
        end
    end

    # =========================================================================
    # Step 4: RG/Newton iteration
    # =========================================================================
    println()
    mode_name = simple_flow ? "Simple RG flow" : "Newton iteration"
    println("Step 4: $mode_name...")
    println("-" ^ 70)
    println(@sprintf("%-6s  %-12s  %-10s  %-10s  %-10s", "Iter", "Step Size", "x_σ", "x_ε", "Δε err"))
    println("-" ^ 70)

    # Ensure complex type for gauge-fixed arrays
    A_hist = [ComplexF64.(A_init)]

    use_jacobian = !simple_flow

    @time begin
        for iter in 1:N_newton
            A_new = iterate_step(A_hist[end], gilt_gauged, use_jacobian, ImJ_inv_mat, ortho_basis, approx_rank)
            step_size = norm(A_new - A_hist[end])
            push!(A_hist, A_new)

            # Extract scaling dimensions
            scaldims = py"extract_scaldims"(A_new)
            x_sigma = length(scaldims) > 1 ? scaldims[2] : NaN
            x_eps = length(scaldims) > 3 ? scaldims[4] : NaN
            eps_err = abs(x_eps - 0.8) / 0.8 * 100

            println(@sprintf("%-6d  %-12.2e  %-10.6f  %-10.6f  %-10.2f%%",
                            iter, step_size, x_sigma, x_eps, eps_err))

            # Check convergence
            if step_size < 1e-10
                println("  Converged!")
                break
            end
        end
    end

    # =========================================================================
    # Step 6: Final results
    # =========================================================================
    println()
    println("=" ^ 70)
    println("FINAL RESULTS")
    println("=" ^ 70)

    A_final = A_hist[end]
    scaldims = py"extract_scaldims"(A_final)

    println()
    println("Scaling dimensions at fixed point:")
    println(@sprintf("  x_0 (I)    = %.6f  (exact: 0)", scaldims[1]))
    if length(scaldims) > 1
        println(@sprintf("  x_1 (σ)    = %.6f  (exact: %.6f, error: %.2f%%)",
                        scaldims[2], 2/15, abs(scaldims[2] - 2/15)/(2/15)*100))
    end
    if length(scaldims) > 2
        println(@sprintf("  x_2 (σ̄)   = %.6f  (exact: %.6f)", scaldims[3], 2/15))
    end
    if length(scaldims) > 3
        println(@sprintf("  x_3 (ε)    = %.6f  (exact: %.6f, error: %.2f%%)",
                        scaldims[4], 4/5, abs(scaldims[4] - 4/5)/(4/5)*100))
    end
    if length(scaldims) > 4
        println(@sprintf("  x_4        = %.6f", scaldims[5]))
    end

    # =========================================================================
    # Step 7: Save results
    # =========================================================================
    output_dir = joinpath(@__DIR__, "../newton")
    mkpath(output_dir)

    filename = "potts3_chi$(chi)_relT$(relT)_rank$(approx_rank).data"
    filepath = joinpath(output_dir, filename)

    result = Dict(
        "A_newton" => A_hist,
        "eigenvalues" => eigenvalues,
        "eigenvectors" => eigenvectors,
        "scaldims_final" => scaldims,
        "chi" => chi,
        "relT" => relT,
        "gilt_eps" => gilt_eps,
        "approx_rank" => approx_rank,
        "timestamp" => string(now()),
    )

    serialize(filepath, result)
    println()
    println("Results saved to: $filepath")

    println()
    println("=" ^ 70)
    println("COMPLETED")
    println("=" ^ 70)
end

# Run
main()
