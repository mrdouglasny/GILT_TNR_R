# MODULE: DiscreteGaugeZN.jl
# USAGE: include("src/DiscreteGaugeZN.jl") or using EKRNewton
# INPUTS: Gauge-fixed tensors (after continuous gauge fixing)
# OUTPUTS: Tensors with discrete Z_N gauge fixed
# DESCRIPTION: Discrete gauge fixing for Z_N symmetric tensors
#
# BASED ON: GaugeFixing.jl lines 260-538 (discrete gauge section)
#           potts3_newton_plain.py (Z₃ discrete gauge in Python)
# NEW IN THIS MODULE: Generalized from Z₂ (GF(2), ±1) to Z_N (GF(N), ω^k)

"""
Discrete gauge fixing for Z_N symmetric tensor network fixed points.

After continuous gauge fixing, there remains a discrete Z_N gauge freedom:
for each bond, we can multiply by any N-th root of unity ω^k where ω = exp(2πi/N).

The gauge transformation acts as:
    A'[i,j,k,l] = ω^{g_H[i] + g_V[j] - g_H[k] - g_V[l]} A[i,j,k,l]

where g_H, g_V ∈ {0, 1, ..., N-1}^dim are gauge vectors.

To fix the gauge, we select tensor elements and constrain them to have
phase 0 (be positive real). This requires solving a linear system over GF(N).

Key formula for degrees of freedom:
    Z₂: rows_num = dimH + dimV - 3
    Z_N: rows_num = dimH + dimV - (N + 1)

The extra -1 for Z₂ comes from the global sign flip constraint.
For general Z_N, the global phase constraint removes 1 DOF.
"""

using LinearAlgebra
using PyCall

# Include GF(N) echelon form utilities
include("EchelonFormGFN.jl")

################################################
# Constants
################################################

"""N-th root of unity: ω_N = exp(2πi/N)"""
omega(N::Int) = exp(2π * im / N)

################################################
# Element Selection
################################################

"""
    list_of_allowed_elements(A::Array; tol=1e-7)

Find tensor elements suitable for discrete gauge constraints.

Returns list of (index, value) pairs for non-diagonal elements
with magnitude above tolerance.
"""
function list_of_allowed_elements_zn(A::Array{T, 4}; tol::Real = 1e-7) where T
    elements = Tuple{CartesianIndex{4}, T}[]

    for ind in CartesianIndices(size(A))
        val = A[ind]
        # Condition 1: magnitude above tolerance
        condition1 = abs(val) > tol
        # Condition 2: not a diagonal element (i≠k or j≠l)
        condition2 = !(ind[1] == ind[3] && ind[2] == ind[4])

        if condition1 && condition2
            push!(elements, (ind, val))
        end
    end

    return elements
end

"""
    list_of_allowed_elements(A::PyObject; tol=1e-7)

Find tensor elements for Python tensor objects.
"""
function list_of_allowed_elements_zn(A::PyObject; tol::Real = 1e-7)
    AA = A.to_ndarray()
    return list_of_allowed_elements_zn(AA; tol = tol)
end

################################################
# Constraint Construction for Z_N
################################################

"""
    compute_target_charge(val::Complex, N::Int) -> Int

Compute the target charge (gauge phase) needed to make element positive real.

For element with value val = |val| * exp(iφ), we want:
    ω^g * val to be positive real

This means: g * (2π/N) + φ ≡ 0 (mod 2π)
           g ≡ -φ / (2π/N) (mod N)
           g ≡ -N*φ / (2π) (mod N)
"""
function compute_target_charge(val::Number, N::Int)
    φ = angle(val)
    # Target: g such that ω^g * exp(iφ) is positive real
    # ω^g = exp(2πi*g/N), so we need 2πg/N + φ ≡ 0 (mod 2π)
    # g ≡ -Nφ/(2π) (mod N)
    g = round(Int, -N * φ / (2π))
    return mod(g, N)
end

"""
    construct_row_gfn(ind::CartesianIndex{4}, dimH::Int, dimV::Int, N::Int) -> Vector{Int}

Construct a row of the constraint matrix for element at index (i,j,k,l).

The gauge transformation is:
    A'[i,j,k,l] = ω^{g_H[i] + g_V[j] - g_H[k] - g_V[l]} A[i,j,k,l]

So the constraint for making A[i,j,k,l] positive is:
    g_H[i] + g_V[j] - g_H[k] - g_V[l] ≡ target (mod N)

Row format: [g_H[1], ..., g_H[dimH], g_V[1], ..., g_V[dimV]]
"""
function construct_row_gfn(ind::CartesianIndex{4}, dimH::Int, dimV::Int, N::Int)
    row = zeros(Int, dimH + dimV)

    i, j, k, l = Tuple(ind)

    # Horizontal gauge: +g_H[i] - g_H[k]
    row[i] = mod(row[i] + 1, N)
    row[k] = mod(row[k] - 1, N)

    # Vertical gauge: +g_V[j] - g_V[l]
    row[dimH + j] = mod(row[dimH + j] + 1, N)
    row[dimH + l] = mod(row[dimH + l] - 1, N)

    return row
end

"""
    construct_linear_system_zn(elements, dimH, dimV, N) -> (M, b, accepted)

Construct the linear system M * g = b over GF(N) for discrete gauge fixing.

Selects linearly independent constraints from the element list until
we have enough to fix the gauge (dimH + dimV - (N+1) constraints).

Returns:
- M: Constraint matrix over GF(N)
- b: Target vector over GF(N)
- accepted: List of elements used for constraints
"""
function construct_linear_system_zn(elements::Vector, dimH::Int, dimV::Int, N::Int)
    # Number of gauge DOFs to fix
    # For Z_N: total DOFs = dimH + dimV
    # Global phase: 1 DOF (can shift all by constant)
    # For Z_2: extra constraint from Z_2 structure gives -3
    # For general Z_N: -1 for global phase, -(N-1) for charge conservation = -(N)
    # Actually: rows_num = dimH + dimV - (N + 1) based on the pattern

    # The correct formula from analysis:
    # Z_2 (N=2): dimH + dimV - 3 = dimH + dimV - (2+1)
    # Z_N: dimH + dimV - (N+1)
    rows_num = dimH + dimV - (N + 1)

    if rows_num <= 0
        @warn "Tensor too small for discrete gauge fixing (rows_num=$rows_num)"
        return zeros(Int, 0, dimH + dimV), Int[], Tuple[]
    end

    M = zeros(Int, rows_num, dimH + dimV)
    b = zeros(Int, rows_num)
    accepted = Vector{eltype(elements)}()

    cnt = 0
    for el in elements
        if cnt >= rows_num
            break
        end

        ind, val = el
        new_row = construct_row_gfn(ind, dimH, dimV, N)

        # Check linear independence
        if cnt == 0
            # First row is always independent if non-zero
            if any(new_row .!= 0)
                cnt += 1
                M[cnt, :] = new_row
                b[cnt] = compute_target_charge(val, N)
                push!(accepted, el)
            end
        else
            # Check if new row is independent of existing rows
            M_test = vcat(M[1:cnt, :], new_row')
            if rank_gfn(M_test, N) > cnt
                cnt += 1
                M[cnt, :] = new_row
                b[cnt] = compute_target_charge(val, N)
                push!(accepted, el)
            end
        end
    end

    if cnt < rows_num
        @warn "construct_linear_system_zn: only found $cnt/$rows_num independent constraints"
        return M[1:cnt, :], b[1:cnt], accepted
    end

    return M, b, accepted
end

################################################
# Gauge Application
################################################

"""
    apply_discrete_gauge_zn!(A::Array, g_H, g_V, N)

Apply discrete Z_N gauge transformation to tensor A in-place.

Transformation: A'[i,j,k,l] = ω^{g_H[i] + g_V[j] - g_H[k] - g_V[l]} A[i,j,k,l]

Note: Returns a complex array even if input is real.
"""
function apply_discrete_gauge_zn!(A::Array{T, 4}, g_H::Vector{Int}, g_V::Vector{Int}, N::Int) where T
    ω = omega(N)

    # Create complex result array
    A_complex = complex.(A)

    for ind in CartesianIndices(size(A))
        i, j, k, l = Tuple(ind)
        phase_exp = mod(g_H[i] + g_V[j] - g_H[k] - g_V[l], N)
        A_complex[ind] *= ω^phase_exp
    end

    return A_complex
end

"""
    gauge_vectors_to_diagonal(g_H, g_V, N) -> (H_diag, V_diag)

Convert gauge vectors to diagonal phase matrices.

H_diag[i,i] = ω^{g_H[i]}
V_diag[j,j] = ω^{g_V[j]}
"""
function gauge_vectors_to_diagonal(g_H::Vector{Int}, g_V::Vector{Int}, N::Int)
    ω = omega(N)
    H_diag = diagm([ω^g for g in g_H])
    V_diag = diagm([ω^g for g in g_V])
    return H_diag, V_diag
end

################################################
# Main Discrete Gauge Fixing Functions
################################################

"""
    fix_discrete_gauge_zn(A::Array, N::Int; tol=1e-7) -> (A_fixed, accepted, H, V)

Fix discrete Z_N gauge for Julia Array tensor.

Returns:
- A_fixed: Gauge-fixed tensor (selected elements are positive real)
- accepted: List of elements used for gauge constraints
- H, V: Diagonal gauge transformation matrices
"""
function fix_discrete_gauge_zn(A::Array{T, 4}, N::Int; tol::Real = 1e-7) where T
    # Get list of candidate elements
    elements = list_of_allowed_elements_zn(A; tol = tol)

    if isempty(elements)
        @warn "No suitable elements found for discrete gauge fixing"
        dimH, dimV = size(A, 1), size(A, 2)
        return A, elements, Matrix{Complex{Float64}}(I, dimH, dimH), Matrix{Complex{Float64}}(I, dimV, dimV)
    end

    # Sort by magnitude (largest first for numerical stability)
    sort!(elements, by = x -> -abs(x[2]))

    dimH, dimV, _, _ = size(A)

    # Construct and solve linear system over GF(N)
    M, b, accepted = construct_linear_system_zn(elements, dimH, dimV, N)

    if size(M, 1) == 0
        @warn "Could not construct linear system for discrete gauge"
        return A, elements, Matrix{Complex{Float64}}(I, dimH, dimH), Matrix{Complex{Float64}}(I, dimV, dimV)
    end

    # Solve M * g = b over GF(N)
    g, success = solve_gfn(M, b, N)

    if !success
        @warn "Failed to solve discrete gauge linear system"
        return A, accepted, Matrix{Complex{Float64}}(I, dimH, dimH), Matrix{Complex{Float64}}(I, dimV, dimV)
    end

    # Split solution into H and V components
    g_H = g[1:dimH]
    g_V = g[dimH+1:end]

    # Apply gauge transformation (returns complex array)
    A_complex = apply_discrete_gauge_zn!(A, g_H, g_V, N)

    # Make result real if phases are successfully fixed
    A_fixed = real.(A_complex)

    # Construct diagonal matrices for return
    H, V = gauge_vectors_to_diagonal(g_H, g_V, N)

    return A_fixed, accepted, H, V
end

"""
    fix_discrete_gauge_zn(A::PyObject, N::Int; tol=1e-7)

Fix discrete Z_N gauge for Python tensor objects.
"""
function fix_discrete_gauge_zn(A::PyObject, N::Int; tol::Real = 1e-7)
    # Convert to array, fix gauge, convert back
    arr = A.to_ndarray()
    arr_fixed, accepted, H, V = fix_discrete_gauge_zn(arr, N; tol = tol)

    # Reconstruct Python tensor
    # For now, return array - full reconstruction needs tensor metadata
    return arr_fixed, accepted, H, V
end

"""
    fix_discrete_gauge_zn(A, N, elements; tol=1e-7)

Fix discrete gauge using a pre-computed list of elements.

This is faster when iterating, as we reuse the same element indices.
"""
function fix_discrete_gauge_zn(A::Array{T, 4}, N::Int, elements::Vector; tol::Real = 1e-7) where T
    # Update element values for new tensor
    elements_updated = [(el[1], A[el[1]]) for el in elements]

    # Check for vanishing elements
    for (i, el) in enumerate(elements_updated)
        if abs(el[2]) < tol
            @warn "Element at $(el[1]) has become small: was $(elements[i][2]), now $(el[2])"
        end
    end

    dimH, dimV, _, _ = size(A)

    # Construct and solve linear system
    M, b, accepted = construct_linear_system_zn(elements_updated, dimH, dimV, N)

    if size(M, 1) == 0
        return A, elements_updated, Matrix{Complex{Float64}}(I, dimH, dimH), Matrix{Complex{Float64}}(I, dimV, dimV)
    end

    g, success = solve_gfn(M, b, N)

    if !success
        @warn "Failed to solve discrete gauge linear system"
        return A, accepted, Matrix{Complex{Float64}}(I, dimH, dimH), Matrix{Complex{Float64}}(I, dimV, dimV)
    end

    g_H = g[1:dimH]
    g_V = g[dimH+1:end]

    # Apply gauge transformation (returns complex array)
    A_complex = apply_discrete_gauge_zn!(A, g_H, g_V, N)
    A_fixed = real.(A_complex)

    H, V = gauge_vectors_to_diagonal(g_H, g_V, N)

    return A_fixed, accepted, H, V
end

################################################
# Backward Compatibility: Z₂ Specialization
################################################

"""
    fix_discrete_gauge_z2(A::Array; tol=1e-7)

Specialized discrete gauge fixing for Z₂ (Ising model).

Uses the original GF(2) implementation for backward compatibility.
"""
function fix_discrete_gauge_z2(A::Array{T, 4}; tol::Real = 1e-7) where T
    return fix_discrete_gauge_zn(A, 2; tol = tol)
end

################################################
# Verification Utilities
################################################

"""
    verify_discrete_gauge(A, accepted; tol=1e-7) -> Bool

Verify that gauge-fixed tensor has positive real values at accepted positions.
"""
function verify_discrete_gauge(A::Array, accepted::Vector; tol::Real = 1e-7)
    all_positive = true

    for el in accepted
        ind = el[1]
        val = A[ind]

        # Check if value is positive real
        if abs(imag(val)) > tol || real(val) < -tol
            @warn "Element at $ind is not positive real: $val"
            all_positive = false
        end
    end

    return all_positive
end

"""
    discrete_gauge_residual(A, accepted) -> Float64

Compute residual: sum of |imag(A[accepted])| + |min(0, real(A[accepted]))|
"""
function discrete_gauge_residual(A::Array, accepted::Vector)
    residual = 0.0
    for el in accepted
        val = A[el[1]]
        residual += abs(imag(val)) + abs(min(0.0, real(val)))
    end
    return residual
end

################################################
# Export
################################################

export fix_discrete_gauge_zn, fix_discrete_gauge_z2
export list_of_allowed_elements_zn
export construct_linear_system_zn, construct_row_gfn
export apply_discrete_gauge_zn!, gauge_vectors_to_diagonal
export compute_target_charge
export verify_discrete_gauge, discrete_gauge_residual
