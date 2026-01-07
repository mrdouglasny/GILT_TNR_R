# MODULE: ContinuousGauge.jl
# USAGE: include("src/ContinuousGauge.jl") or using EKRNewton
# INPUTS: 4-leg tensors (Julia arrays or Python TensorZ2/TensorZN)
# OUTPUTS: Gauge-fixed tensors with diagonal environments
# DESCRIPTION: Continuous gauge fixing via environment diagonalization
#
# BASED ON: GaugeFixing.jl lines 86-221 (continuous gauge section)
# NEW IN THIS MODULE: Extracted as standalone module, works with ZNTensor{N}

"""
Continuous gauge fixing for tensor network fixed points.

This module fixes the continuous (unitary rotation) gauge freedom in
4-leg tensors by diagonalizing the horizontal and vertical environments.

The environment E_v for vertical gauge is:
    E_v[i,j] = sum_{a,b,c} A[a,i,b,c] * conj(A[a,j,b,c])

The environment E_h for horizontal gauge is:
    E_h[i,j] = sum_{b,c,d} A[i,b,c,d] * conj(A[j,b,c,d])

Diagonalizing these environments fixes the unitary gauge up to phases.
Works for any Z_N symmetric tensor (model-independent).
"""

using LinearAlgebra
using PyCall

################################################
# Python Interface (for TensorZ2 compatibility)
################################################

# Lazy initialization of Python modules
const _py_ncon = Ref{Any}(nothing)

function get_ncon()
    if _py_ncon[] === nothing
        pushfirst!(pyimport("sys")."path", "GiltTNR")
        _py_ncon[] = pyimport("ncon").ncon
    end
    return _py_ncon[]
end

################################################
# Environment Computation
################################################

"""
    environment_for_vertical_gauge(tensors)

Compute environment for vertical gauge fixing.

The environment contracts indices H_in (0), H_out (2), V_out (3)
leaving V_in (1) as free indices for both tensors.

ncon pattern: [[2, -1, 1, 3], [2, -2, 1, 3]]
Result: E[i,j] = sum_{a,b,c} A[a,i,b,c] * conj(A[a,j,b,c])
"""
function environment_for_vertical_gauge(tensors::Vector{PyObject})
    ncon = get_ncon()
    connects = [[2, -1, 1, 3], [2, -2, 1, 3]]
    con_order = [2, 1, 3]
    environment = ncon(tensors, connects, con_order)
    environment /= environment.norm()
    return environment
end

function environment_for_vertical_gauge(tensors::Vector{Array{T, 4}}) where T
    # E[i,j] = sum_{a,b,c} A[a,i,b,c] * conj(A[a,j,b,c])
    A, Aconj = tensors
    # Using einsum notation: "aibc,ajbc->ij"
    env = zeros(T, size(A, 2), size(Aconj, 2))
    for a in axes(A, 1), b in axes(A, 3), c in axes(A, 4)
        for i in axes(A, 2), j in axes(Aconj, 2)
            env[i, j] += A[a, i, b, c] * conj(Aconj[a, j, b, c])
        end
    end
    normalize!(env)
    return env
end

# Optimized version using tensor contraction
function environment_for_vertical_gauge_fast(A::Array{T, 4}) where T
    # Reshape A[a,i,b,c] -> A[i, (a,b,c)]
    dim_i = size(A, 2)
    A_flat = reshape(permutedims(A, (2, 1, 3, 4)), dim_i, :)
    # E = A_flat * A_flat' gives E[i,j] = sum_k A_flat[i,k] * conj(A_flat[j,k])
    env = A_flat * A_flat'
    normalize!(env)
    return env
end

"""
    environment_for_horizontal_gauge(tensors)

Compute environment for horizontal gauge fixing.

The environment contracts indices V_in (1), H_out (2), V_out (3)
leaving H_in (0) as free indices for both tensors.

ncon pattern: [[-1, 2, 3, 1], [-2, 2, 3, 1]]
Result: E[i,j] = sum_{b,c,d} A[i,b,c,d] * conj(A[j,b,c,d])
"""
function environment_for_horizontal_gauge(tensors::Vector{PyObject})
    ncon = get_ncon()
    connects = [[-1, 2, 3, 1], [-2, 2, 3, 1]]
    con_order = [2, 3, 1]
    environment = ncon(tensors, connects, con_order)
    environment /= environment.norm()
    return environment
end

function environment_for_horizontal_gauge(tensors::Vector{Array{T, 4}}) where T
    # E[i,j] = sum_{b,c,d} A[i,b,c,d] * conj(A[j,b,c,d])
    A, Aconj = tensors
    env = zeros(T, size(A, 1), size(Aconj, 1))
    for b in axes(A, 2), c in axes(A, 3), d in axes(A, 4)
        for i in axes(A, 1), j in axes(Aconj, 1)
            env[i, j] += A[i, b, c, d] * conj(Aconj[j, b, c, d])
        end
    end
    normalize!(env)
    return env
end

# Optimized version using tensor contraction
function environment_for_horizontal_gauge_fast(A::Array{T, 4}) where T
    # Reshape A[i,b,c,d] -> A[i, (b,c,d)]
    dim_i = size(A, 1)
    A_flat = reshape(A, dim_i, :)
    # E = A_flat * A_flat'
    env = A_flat * A_flat'
    normalize!(env)
    return env
end

################################################
# Phase Correction Helpers
################################################

"""
    correct_phase_of_a_vec!(v)

Multiply vector by sign to make sum positive.
This fixes the overall phase ambiguity of eigenvectors.
"""
function correct_phase_of_a_vec!(v)
    s = sum(v)
    if real(s) < 0
        v .*= -1
    elseif real(s) == 0 && imag(s) < 0
        v .*= -1
    end
    return v
end

"""
    correct_phase_of_each_column!(V)

Apply phase correction to each column of matrix V.
"""
function correct_phase_of_each_column!(V)
    for i in axes(V, 2)
        correct_phase_of_a_vec!(@view V[:, i])
    end
    return V
end

################################################
# Plain Tensor Detection
################################################

"""
    is_plain_tensor(A::PyObject) -> Bool

Check if A is a plain tensor (no Z_N symmetry structure).

Plain tensors from `Tensor.from_ndarray()` have trivial charges:
qhape = [[0], [0], [0], [0]] (single charge 0 per leg).

Symmetric tensors (TensorZ2, TensorZ3) have multiple charge sectors.
"""
function is_plain_tensor(A::PyObject)
    try
        qhape = A.qhape
        # Check if all charges are trivial (single charge 0 per leg)
        for q in qhape
            if length(q) != 1 || q[1] != 0
                return false
            end
        end
        return true
    catch
        # No qhape attribute = plain array (shouldn't happen for Tensor objects)
        return true
    end
end

################################################
# Main Gauge Fixing Functions
################################################

"""
    fix_continuous_gauge(A::PyObject) -> (A_fixed, H, V, SH, SV)

Fix continuous gauge for Python TensorZ2/TensorZN tensors.

Returns:
- A_fixed: Gauge-fixed tensor
- H, V: Gauge transformation matrices (horizontal, vertical)
- SH, SV: Eigenvalues of environments (for diagnostics)

For plain tensors (from Tensor.from_ndarray), dispatches to array-based implementation.
"""
function fix_continuous_gauge(A::PyObject)
    # Check if this is a plain tensor (no Z_N symmetry structure)
    if is_plain_tensor(A)
        return fix_continuous_gauge_plain(A)
    end

    # Original path for TensorZ2/TensorZ3 (symmetric tensors)
    ncon = get_ncon()
    tensors = [A, A.conj()]

    # Vertical gauge
    environment = environment_for_vertical_gauge(tensors)
    SV, V = environment.eig(1, 0, hermitian = true)
    A = ncon([A, V, V.conj()], [[-1, 2, -3, 4], [2, -2], [4, -4]])

    # Horizontal gauge (use original tensors for second environment)
    tensors = [A, A.conj()]
    environment = environment_for_horizontal_gauge(tensors)
    SH, H = environment.eig(1, 0, hermitian = true)
    A = ncon([A, H, H.conj()], [[1, -2, 3, -4], [1, -1], [3, -3]])

    return A, H, V, SH, SV
end

"""
    fix_continuous_gauge_plain(A::PyObject) -> (A_fixed, H, V, SH, SV)

Fix continuous gauge for plain Python tensors (Tensor.from_ndarray).

Extracts the numpy array, uses Julia's eigen decomposition,
and wraps result back in a Python Tensor.
"""
function fix_continuous_gauge_plain(A::PyObject)
    # Extract numpy array
    arr = A.to_ndarray()

    # Use pure Julia implementation
    arr_fixed, H, V, SH, SV = fix_continuous_gauge_array(arr)

    # Wrap back in Python Tensor
    # IMPORTANT: Use pycall with PyObject return type to prevent auto-conversion
    tensors_mod = pyimport("tensors")
    A_fixed = pycall(tensors_mod.Tensor.from_ndarray, PyObject, arr_fixed)

    return A_fixed, H, V, SH, SV
end

"""
    fix_continuous_gauge(A::Array) -> (A_fixed, H, V, SH, SV)

Fix continuous gauge for Julia Array tensors.

Uses eigen decomposition with eigenvalues sorted by decreasing magnitude.
This version uses Python ncon for tensor contractions.
"""
function fix_continuous_gauge(A::Array{T, 4}) where T
    ncon = get_ncon()
    tensors = [A, conj(A)]

    # Vertical gauge
    environment = Hermitian(environment_for_vertical_gauge(tensors))
    decomposition = eigen(environment; sortby = x -> -abs(x))
    SV, V = decomposition.values, decomposition.vectors
    correct_phase_of_each_column!(V)
    A = ncon([A, V, conj(V)], [[-1, 2, -3, 4], [2, -2], [4, -4]])

    # Horizontal gauge
    tensors = [A, conj(A)]
    environment = Hermitian(environment_for_horizontal_gauge(tensors))
    decomposition = eigen(environment; sortby = x -> -abs(x))
    SH, H = decomposition.values, decomposition.vectors
    correct_phase_of_each_column!(H)
    A = ncon([A, H, conj(H)], [[1, -2, 3, -4], [1, -1], [3, -3]])

    return A, H, V, SH, SV
end

"""
    fix_continuous_gauge_array(A::Array) -> (A_fixed, H, V, SH, SV)

Fix continuous gauge for Julia Array tensors using pure Julia operations.

This is the preferred implementation for plain tensors as it avoids
Python-Julia type conversion issues.

Index convention: A[H_in, V_in, H_out, V_out] = A[1, 2, 3, 4]
- Vertical gauge transforms indices 2 and 4 (V_in, V_out)
- Horizontal gauge transforms indices 1 and 3 (H_in, H_out)
"""
function fix_continuous_gauge_array(A::Array{T, 4}) where T
    # --- Vertical gauge ---
    # Environment: E_v[i,j] = sum_{a,b,c} A[a,i,b,c] * conj(A[a,j,b,c])
    # Use fast version
    env_v = environment_for_vertical_gauge_fast(A)
    env_v = (env_v + env_v') / 2  # Hermitianize for numerical stability

    decomposition = eigen(Hermitian(env_v); sortby = x -> -abs(x))
    SV, V = decomposition.values, decomposition.vectors
    correct_phase_of_each_column!(V)

    # Apply V gauge to indices 2 and 4
    # A'[i,j,k,l] = sum_{m,n} V[m,j] * A[i,m,k,n] * conj(V[n,l])
    # Using permutedims and matrix multiplication:
    # Contract index 2: A[i,m,k,n] * V[m,j] -> A'[i,j,k,n]
    # Contract index 4: A'[i,j,k,n] * V*[n,l] -> A''[i,j,k,l]
    A = apply_gauge_to_vertical_indices(A, V)

    # --- Horizontal gauge ---
    # Environment: E_h[i,j] = sum_{b,c,d} A[i,b,c,d] * conj(A[j,b,c,d])
    env_h = environment_for_horizontal_gauge_fast(A)
    env_h = (env_h + env_h') / 2

    decomposition = eigen(Hermitian(env_h); sortby = x -> -abs(x))
    SH, H = decomposition.values, decomposition.vectors
    correct_phase_of_each_column!(H)

    # Apply H gauge to indices 1 and 3
    # A'[i,j,k,l] = sum_{m,n} H[m,i] * A[m,j,n,l] * conj(H[n,k])
    A = apply_gauge_to_horizontal_indices(A, H)

    return A, H, V, SH, SV
end

"""
    apply_gauge_to_vertical_indices(A, V) -> A_transformed

Apply gauge transformation V to vertical indices (2 and 4) of tensor A.

Computes: A'[i,j,k,l] = sum_{m,n} V[m,j] * A[i,m,k,n] * conj(V[n,l])
"""
function apply_gauge_to_vertical_indices(A::Array{T, 4}, V::AbstractMatrix) where T
    d1, d2, d3, d4 = size(A)
    # Reshape A[i,m,k,n] -> A[(i,k), (m,n)]
    A_reshaped = reshape(permutedims(A, (1, 3, 2, 4)), d1 * d3, d2 * d4)

    # V ⊗ V* applied to (m,n) indices
    # Result shape: (d2, d4) -> (d2, d4) with new V
    # We want: (V ⊗ conj(V)) acting on the vectorized (m,n) space
    # But more efficiently: do contractions sequentially

    # First contract index 2: sum_m V[m,j] * A[i,m,k,n] -> B[i,j,k,n]
    # A is (d1, d2, d3, d4), V is (d2, d2')
    A_perm = permutedims(A, (1, 3, 4, 2))  # A[i,k,n,m]
    A_flat = reshape(A_perm, d1 * d3 * d4, d2)  # A[(i,k,n), m]
    B_flat = A_flat * V  # B[(i,k,n), j]
    B = reshape(B_flat, d1, d3, d4, size(V, 2))  # B[i,k,n,j]
    B = permutedims(B, (1, 4, 2, 3))  # B[i,j,k,n]

    # Now contract index 4: sum_n B[i,j,k,n] * conj(V[n,l]) -> C[i,j,k,l]
    d2_new = size(B, 2)
    B_perm = permutedims(B, (1, 2, 3, 4))  # B[i,j,k,n]
    B_flat = reshape(B_perm, d1 * d2_new * d3, d4)  # B[(i,j,k), n]
    C_flat = B_flat * conj(V)  # C[(i,j,k), l]
    C = reshape(C_flat, d1, d2_new, d3, size(V, 2))  # C[i,j,k,l]

    return C
end

"""
    apply_gauge_to_horizontal_indices(A, H) -> A_transformed

Apply gauge transformation H to horizontal indices (1 and 3) of tensor A.

Computes: A'[i,j,k,l] = sum_{m,n} H[m,i] * A[m,j,n,l] * conj(H[n,k])
"""
function apply_gauge_to_horizontal_indices(A::Array{T, 4}, H::AbstractMatrix) where T
    d1, d2, d3, d4 = size(A)

    # First contract index 1: sum_m H[m,i] * A[m,j,n,l] -> B[i,j,n,l]
    # A is (d1, d2, d3, d4), H is (d1, d1')
    A_flat = reshape(A, d1, d2 * d3 * d4)  # A[m, (j,n,l)]
    B_flat = H' * A_flat  # B[i, (j,n,l)] (H' because H[m,i] means sum over first index)
    B = reshape(B_flat, size(H, 2), d2, d3, d4)  # B[i,j,n,l]

    # Now contract index 3: sum_n B[i,j,n,l] * conj(H[n,k]) -> C[i,j,k,l]
    d1_new = size(B, 1)
    B_perm = permutedims(B, (1, 2, 4, 3))  # B[i,j,l,n]
    B_flat = reshape(B_perm, d1_new * d2 * d4, d3)  # B[(i,j,l), n]
    C_flat = B_flat * conj(H)  # C[(i,j,l), k]
    C = reshape(C_flat, d1_new, d2, d4, size(H, 2))  # C[i,j,l,k]
    C = permutedims(C, (1, 2, 4, 3))  # C[i,j,k,l]

    return C
end

################################################
# ZNTensor Support
################################################

# Import ZNTensor if available
# This will be filled in when the module is loaded as part of EKRNewton

"""
    fix_continuous_gauge_zn(...)

Fix continuous gauge for ZNTensor{N}.

NOTE: This is a placeholder. The actual implementation requires either:
1. Converting ZNTensor to PyObject, applying gauge, converting back
2. Implementing block-structure-preserving eigendecomposition

For now, use the PyObject interface via zn_to_py() and py_to_zn().
"""
function fix_continuous_gauge_zn(args...)
    error("Direct ZNTensor gauge fixing not yet implemented. " *
          "Convert to PyObject first using zn_to_py(), apply fix_continuous_gauge(), " *
          "then convert back with py_to_zn().")
end

################################################
# Utility Functions
################################################

"""
    is_gauge_fixed(A::Array; tol=1e-10) -> Bool

Check if tensor A has diagonal environments (gauge is fixed).
"""
function is_gauge_fixed(A::Array{T, 4}; tol::Real = 1e-10) where T
    # Check vertical environment
    env_v = environment_for_vertical_gauge_fast(A)
    off_diag_v = norm(env_v - diagm(diag(env_v)))

    # Check horizontal environment
    env_h = environment_for_horizontal_gauge_fast(A)
    off_diag_h = norm(env_h - diagm(diag(env_h)))

    return off_diag_v < tol && off_diag_h < tol
end

"""
    gauge_fixing_residual(A::Array) -> (res_h, res_v)

Compute off-diagonal norm of environments (measure of gauge fixing quality).
"""
function gauge_fixing_residual(A::Array{T, 4}) where T
    env_v = environment_for_vertical_gauge_fast(A)
    res_v = norm(env_v - diagm(diag(env_v)))

    env_h = environment_for_horizontal_gauge_fast(A)
    res_h = norm(env_h - diagm(diag(env_h)))

    return res_h, res_v
end

################################################
# Export
################################################

export environment_for_vertical_gauge, environment_for_horizontal_gauge
export fix_continuous_gauge, fix_continuous_gauge_array, fix_continuous_gauge_plain
export is_plain_tensor
export apply_gauge_to_vertical_indices, apply_gauge_to_horizontal_indices
export correct_phase_of_a_vec!, correct_phase_of_each_column!
export is_gauge_fixed, gauge_fixing_residual
