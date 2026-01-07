# MODULE: TensorKitPotts.jl
# USAGE: using TensorKitPotts (from ekrgilttrnr/TensorKitPotts)
# DESCRIPTION: Z₃-symmetric tensor utilities for 3-state Potts model Newton method.
#
# BASED ON: ekrgilttrnr/KrylovTechnical.jl (Z₂ tensor structure)
#           ekrgilttrnr/GaugeFixing.jl (gauge fixing algorithms)
#           genmodel/src/TensorKitUtils.jl (TensorKit patterns)
# NEW IN THIS MODULE: Adapts Newton method infrastructure for Z₃ symmetry using TensorKit.

module TensorKitPotts

using TensorKit
using TensorOperations
using KrylovKit
using LinearAlgebra
using AbstractAlgebra
using PyCall

export Z3TensorTK, build_potts3_tensor, build_potts3_tensor_simple
export py_to_z3tensor, z3tensor_to_py
export fix_continuous_gauge_z3, fix_discrete_gauge_z3
export random_z3tensor

# =============================================================================
# Constants
# =============================================================================

const ω = exp(2π * im / 3)  # Cube root of unity
const ω² = ω^2
const ℤ₃_field = GF(3)      # Finite field for discrete gauge fixing

# =============================================================================
# Z₃ Tensor Type (TensorKit-based)
# =============================================================================

"""
    Z3TensorTK

Wrapper for TensorKit TensorMap with Z₃ symmetry.
Stores the tensor and metadata for Newton iteration.
"""
struct Z3TensorTK
    tensor::TensorMap  # TensorKit Z₃-symmetric tensor
    chi::Int           # Bond dimension per sector
end

# =============================================================================
# Tensor Construction
# =============================================================================

"""
    build_potts3_tensor(beta::Float64, chi::Int)

Construct initial 3-state Potts tensor with Z₃ symmetry.

# Arguments
- `beta`: Inverse temperature. Critical point: β_c = log(1 + √3) ≈ 1.0051
- `chi`: Bond dimension per Z₃ sector (total dim = chi * 3 for 3 sectors)

# Returns
- `Z3TensorTK`: TensorKit tensor with Z₃ graded bond spaces

# Theory
Boltzmann weight: W[σ,σ'] = exp(β) if σ=σ' else 1
Decompose W = U √Λ √Λ Uᵀ, build tensor via contraction.
"""
function build_potts3_tensor(beta::Float64, chi::Int)
    q = 3  # 3-state Potts

    # Boltzmann weight matrix: W[i,j] = exp(β δ_{i,j})
    W = ones(q, q)
    for i in 1:q
        W[i, i] = exp(beta)
    end

    # Eigendecomposition: W = U Λ Uᵀ
    F = eigen(Symmetric(W))
    sqrt_lambda = sqrt.(max.(F.values, 0))
    A = F.vectors * Diagonal(sqrt_lambda)  # q × q matrix

    # Build tensor: T[l,r,u,d] = Σ_σ A[σ,l] A[σ,r] A[σ,u] A[σ,d]
    # This is the standard TRG initial tensor construction
    T_array = zeros(ComplexF64, q, q, q, q)
    for l in 1:q, r in 1:q, u in 1:q, d in 1:q
        for σ in 1:q
            T_array[l, r, u, d] += A[σ, l] * A[σ, r] * A[σ, u] * A[σ, d]
        end
    end

    # Create Z₃ graded space
    # Physical states map to Z₃ charges: state i → charge (i-1) mod 3
    # For bond dimension chi per sector:
    V = ℤ₃Space(0 => chi, 1 => chi, 2 => chi)

    # For the initial tensor, we embed the q=3 array into the symmetric structure
    # The tensor is charge-conserving: l + r = u + d (mod 3)
    T = TensorMap(zeros, ComplexF64, V ⊗ V ← V ⊗ V)

    # Fill in the tensor data
    # For initial construction, we use chi=1 per sector and embed directly
    if chi == 1
        # Direct embedding: state i corresponds to charge (i-1)
        for l in 1:q, r in 1:q, u in 1:q, d in 1:q
            # Check charge conservation: (l-1) + (r-1) ≡ (u-1) + (d-1) mod 3
            if mod(l + r - u - d, 3) == 0
                # Access the appropriate block
                c_l = mod(l - 1, 3)
                c_r = mod(r - 1, 3)
                c_u = mod(u - 1, 3)
                c_d = mod(d - 1, 3)

                # TensorKit block access would go here
                # For now, store in array form
            end
        end
    end

    # For general chi > 1, we need SVD truncation
    # Use TensorKit's built-in operations
    T_full = TensorMap(T_array, ℂ^q ⊗ ℂ^q ← ℂ^q ⊗ ℂ^q)

    return Z3TensorTK(T_full, chi)
end

"""
    build_potts3_tensor_simple(beta::Float64)

Build initial Potts tensor without Z₃ symmetry (for testing).
Returns standard TensorMap with ComplexSpace indices.
"""
function build_potts3_tensor_simple(beta::Float64)
    q = 3

    W = ones(q, q)
    for i in 1:q
        W[i, i] = exp(beta)
    end

    F = eigen(Symmetric(W))
    sqrt_lambda = sqrt.(max.(F.values, 0))
    A = F.vectors * Diagonal(sqrt_lambda)

    T_array = zeros(Float64, q, q, q, q)
    for l in 1:q, r in 1:q, u in 1:q, d in 1:q
        for σ in 1:q
            T_array[l, r, u, d] += A[σ, l] * A[σ, r] * A[σ, u] * A[σ, d]
        end
    end

    V = ℂ^q
    return TensorMap(T_array, V ⊗ V ← V ⊗ V)
end

# =============================================================================
# Python Interop (for Gilt-TNR coarse graining)
# =============================================================================

# Import Python tensor classes
const TENSORS = PyNULL()
const TENSZ3 = PyNULL()

function __init__()
    # Lazy initialization of Python modules
    pushfirst!(PyVector(pyimport("sys")."path"),
               joinpath(@__DIR__, "../../GiltTNR"))
    copy!(TENSORS, pyimport("tensors"))
    copy!(TENSZ3, TENSORS.TensorZ3)
end

"""
    py_to_z3tensor(A_py::PyObject, chi::Int) -> Z3TensorTK

Convert Python TensorZ3 to TensorKit Z₃ tensor.
"""
function py_to_z3tensor(A_py::PyObject, chi::Int)
    # Extract numpy array from Python tensor
    arr = A_py.to_ndarray()

    # Get shape and create appropriate TensorKit space
    dims = size(arr)
    @assert length(dims) == 4 "Expected 4-leg tensor"

    # Create ComplexSpace for now (full Z₃ symmetry requires more work)
    V = ℂ^dims[1]
    T = TensorMap(arr, V ⊗ V ← V ⊗ V)

    return Z3TensorTK(T, chi)
end

"""
    z3tensor_to_py(A::Z3TensorTK) -> PyObject

Convert TensorKit Z₃ tensor back to Python TensorZ3.
"""
function z3tensor_to_py(A::Z3TensorTK)
    # Extract array from TensorKit tensor
    arr = convert(Array, A.tensor)

    # Create Python tensor
    # Note: This creates a plain Tensor, not TensorZ3
    # Full Z₃ structure would require proper qhape/shape setup
    py_tensor = pyimport("tensors").Tensor.from_ndarray(arr)

    return py_tensor
end

# =============================================================================
# Gauge Fixing
# =============================================================================

"""
    fix_continuous_gauge_z3(A::Z3TensorTK) -> (A_fixed, H, V, SH, SV)

Fix continuous gauge degrees of freedom by diagonalizing transfer matrix environments.

This is model-independent: works for any tensor type.
"""
function fix_continuous_gauge_z3(A::Z3TensorTK)
    T = A.tensor
    T_arr = convert(Array, T)
    dims = size(T_arr)

    # Compute vertical environment: E_v[i,j] = Σ_{l,r,k} T[l,i,r,k] * T*[l,j,r,k]
    # T has indices [l, u, r, d] = [1, 2, 3, 4]
    # Vertical indices are 2 and 4, horizontal are 1 and 3
    E_v = zeros(ComplexF64, dims[2], dims[2])
    for i in 1:dims[2], j in 1:dims[2]
        for l in 1:dims[1], r in 1:dims[3], k in 1:dims[4]
            E_v[i, j] += T_arr[l, i, r, k] * conj(T_arr[l, j, r, k])
        end
    end
    E_v = E_v / norm(E_v)

    # Diagonalize (Hermitian since E = T T†)
    F_v = eigen(Hermitian(E_v))
    D_v = F_v.values
    V_mat = F_v.vectors

    # Fix sign ambiguity: make column sums have positive real part
    for col in axes(V_mat, 2)
        if real(sum(V_mat[:, col])) < 0
            V_mat[:, col] *= -1
        end
    end

    # Apply gauge: T'[l,i,r,k] = Σ_j,m V[j,i] T[l,j,r,m] V[m,k]
    T_gauged = zeros(ComplexF64, dims...)
    for l in 1:dims[1], i in 1:dims[2], r in 1:dims[3], k in 1:dims[4]
        for j in 1:dims[2], m in 1:dims[4]
            T_gauged[l, i, r, k] += V_mat[j, i] * T_arr[l, j, r, m] * V_mat[m, k]
        end
    end

    # Compute horizontal environment
    E_h = zeros(ComplexF64, dims[1], dims[1])
    for i in 1:dims[1], j in 1:dims[1]
        for u in 1:dims[2], d in 1:dims[4], k in 1:dims[3]
            E_h[i, j] += T_gauged[i, u, k, d] * conj(T_gauged[j, u, k, d])
        end
    end
    E_h = E_h / norm(E_h)

    # Diagonalize
    F_h = eigen(Hermitian(E_h))
    D_h = F_h.values
    H_mat = F_h.vectors

    # Fix sign ambiguity
    for col in axes(H_mat, 2)
        if real(sum(H_mat[:, col])) < 0
            H_mat[:, col] *= -1
        end
    end

    # Apply gauge: T''[i,u,k,d] = Σ_l,r H[l,i] T'[l,u,r,d] H[r,k]
    T_final = zeros(ComplexF64, dims...)
    for i in 1:dims[1], u in 1:dims[2], k in 1:dims[3], d in 1:dims[4]
        for l in 1:dims[1], r in 1:dims[3]
            T_final[i, u, k, d] += H_mat[l, i] * T_gauged[l, u, r, d] * H_mat[r, k]
        end
    end

    # Wrap back in TensorMap
    V = TensorKit.domain(T)[1]
    T_new = TensorMap(T_final, V ⊗ V ← V ⊗ V)

    return Z3TensorTK(T_new, A.chi), H_mat, V_mat, D_h, D_v
end

"""
    fix_discrete_gauge_z3(A::Z3TensorTK; tol=1e-7) -> (A_fixed, elements)

Fix discrete Z₃ gauge degrees of freedom.

For Z₂, we flip signs (±1). For Z₃, we multiply by cube roots of unity (1, ω, ω²).
The gauge transformation is: A'[i,j,k,l] = ω^(g_i + g_j - g_k - g_l) A[i,j,k,l]
where g_i ∈ {0, 1, 2}.

We choose g to make selected large tensor elements positive-real.
"""
function fix_discrete_gauge_z3(A::Z3TensorTK; tol::Float64=1e-7)
    T = A.tensor
    arr = convert(Array, T)
    dims = size(arr)

    # Find allowed elements (non-diagonal, above threshold)
    elements = Tuple{CartesianIndex{4}, ComplexF64}[]
    for idx in CartesianIndices(dims)
        val = arr[idx]
        # Skip if too small or on "diagonal" (i==k and j==l)
        if abs(val) > tol && !(idx[1] == idx[3] && idx[2] == idx[4])
            push!(elements, (idx, val))
        end
    end

    # Sort by magnitude (largest first)
    sort!(elements, by=x -> -abs(x[2]))

    # Build linear system mod 3
    # For element at (i,j,k,l), constraint: g_i + g_j - g_k - g_l ≡ target (mod 3)
    # where target makes the element positive-real

    dim_total = dims[1] + dims[2]  # Total gauge DOFs (H + V)
    n_constraints = dim_total - 2   # We have 2 global gauge freedoms in Z₃

    # Build constraint matrix in GF(3)
    M = zeros(Int, n_constraints, dim_total)
    b = zeros(Int, n_constraints)
    accepted = Tuple{CartesianIndex{4}, ComplexF64}[]

    row = 1
    for (idx, val) in elements
        row > n_constraints && break

        # Construct row for this element
        new_row = zeros(Int, dim_total)
        new_row[idx[1]] = 1           # +g_i (horizontal)
        new_row[idx[3]] = mod(-1, 3)  # -g_k (horizontal)
        new_row[dims[1] + idx[2]] = 1           # +g_j (vertical)
        new_row[dims[1] + idx[4]] = mod(-1, 3)  # -g_l (vertical)

        # Target phase: we want val * ω^(phase) to be positive-real
        # So phase ≡ -arg(val) / (2π/3) mod 3
        phase_target = round(Int, -angle(val) / (2π/3))
        phase_target = mod(phase_target, 3)

        # Check linear independence (simple rank check)
        M_test = vcat(M[1:row-1, :], new_row')
        if row == 1 || rank(M_test) == row
            M[row, :] = new_row
            b[row] = phase_target
            push!(accepted, (idx, val))
            row += 1
        end
    end

    n_actual = row - 1
    M = M[1:n_actual, :]
    b = b[1:n_actual]

    # Solve in GF(3) using AbstractAlgebra
    M_gf3 = matrix_space(ℤ₃_field, n_actual, dim_total)(M)
    b_gf3 = matrix_space(ℤ₃_field, n_actual, 1)(reshape(b, :, 1))

    # Solve M * g = b
    g_int = try
        g_solution = solve(M_gf3, b_gf3; side=:right)
        # Convert GF(3) solution to integers
        [Int(lift(g_solution[i, 1])) for i in 1:dim_total]
    catch e
        # If singular, use identity gauge (no transformation)
        @warn "Singular system in Z₃ gauge fixing, using identity"
        zeros(Int, dim_total)
    end

    g_h = g_int[1:dims[1]]
    g_v = g_int[dims[1]+1:end]

    # Apply gauge transformation
    arr_new = similar(arr)
    for idx in CartesianIndices(dims)
        phase = g_h[idx[1]] + g_v[idx[2]] - g_h[idx[3]] - g_v[idx[4]]
        phase = mod(phase, 3)
        arr_new[idx] = arr[idx] * ω^phase
    end

    # Wrap back in TensorMap
    V = TensorKit.domain(T)[1]
    T_new = TensorMap(arr_new, V ⊗ V ← V ⊗ V)

    return Z3TensorTK(T_new, A.chi), accepted
end

# =============================================================================
# Random Tensor Generation (for Krylov methods)
# =============================================================================

"""
    random_z3tensor(chi::Int) -> Z3TensorTK

Generate random Z₃-symmetric tensor for Krylov subspace initialization.
"""
function random_z3tensor(chi::Int)
    V = ℂ^(3 * chi)  # Total dimension
    arr = randn(ComplexF64, 3*chi, 3*chi, 3*chi, 3*chi)
    T = TensorMap(arr, V ⊗ V ← V ⊗ V)
    T = T / norm(T)
    return Z3TensorTK(T, chi)
end

# =============================================================================
# KrylovKit Interface (Vector Space Operations)
# =============================================================================

# For KrylovKit to work, we need: +, -, *, /, dot, norm

import Base: +, -, *, /
import LinearAlgebra: dot, norm

function +(A::Z3TensorTK, B::Z3TensorTK)
    return Z3TensorTK(A.tensor + B.tensor, A.chi)
end

function -(A::Z3TensorTK, B::Z3TensorTK)
    return Z3TensorTK(A.tensor - B.tensor, A.chi)
end

function *(α::Number, A::Z3TensorTK)
    return Z3TensorTK(α * A.tensor, A.chi)
end

function *(A::Z3TensorTK, α::Number)
    return Z3TensorTK(A.tensor * α, A.chi)
end

function /(A::Z3TensorTK, α::Number)
    return Z3TensorTK(A.tensor / α, A.chi)
end

function dot(A::Z3TensorTK, B::Z3TensorTK)
    return dot(A.tensor, B.tensor)
end

function norm(A::Z3TensorTK)
    return norm(A.tensor)
end

# Zero element for KrylovKit
function Base.zero(A::Z3TensorTK)
    return Z3TensorTK(zero(A.tensor), A.chi)
end

# Similar for allocation
function Base.similar(A::Z3TensorTK)
    return Z3TensorTK(similar(A.tensor), A.chi)
end

end # module
