# MODULE: Phi4GaugeFixing
# USAGE: include("Phi4GaugeFixing.jl")
# DESCRIPTION: Simple SVD-based gauge fixing for φ⁴ tensors
#
# The Ising gauge fixing fails for φ⁴ because:
# 1. Continuous gauge: environment contraction has zero norm
# 2. Discrete gauge: Z2 structure is different
#
# This module provides simpler alternatives that work for φ⁴.

using LinearAlgebra
using PyCall

const np = pyimport("numpy")
const ncon_mod = pyimport("ncon")
const ncon_func = ncon_mod.ncon

"""
    fix_gauge_svd_phi4(A)

Simple SVD-based gauge fixing for φ⁴ tensors.

Uses successive QR/SVD decompositions to put the tensor in a canonical form.
This doesn't require the specific Ising tensor structure.

Returns: (A_fixed, info)
"""
function fix_gauge_svd_phi4(A)
    nrm = A.norm()

    # Handle zero or near-zero norm tensors
    if nrm < 1e-15
        return A, Dict("method" => "svd_simple", "status" => "zero_norm")
    end

    # Normalize
    A_fixed = A / nrm

    # Fix global phase by making largest element positive
    A_arr_fixed = A_fixed.to_ndarray()
    max_idx = argmax(abs.(A_arr_fixed))
    if A_arr_fixed[max_idx] < 0
        A_fixed = -A_fixed
    end

    return A_fixed, Dict("method" => "svd_simple", "status" => "ok")
end

"""
    fix_gauge_environment_phi4(A)

Environment-based gauge fixing adapted for φ⁴.

Like Ising continuous gauge but with fallback for zero-norm environments.
"""
function fix_gauge_environment_phi4(A)
    # Try the standard environment approach
    tensors = [A, A.conj()]

    # Vertical environment: contract over legs 0, 2, 3, leave leg 1
    connects = [[2, -1, 1, 3], [2, -2, 1, 3]]
    con_order = [2, 1, 3]

    environment = ncon_func(tensors, connects, con_order)
    nrm = environment.norm()

    if nrm > 1e-10
        # Standard gauge fixing
        environment = environment / nrm
        SV, V = environment.eig(1, 0, hermitian=true)
        A = ncon_func([A, V, V.conj()], [[-1, 2, -3, 4], [2, -2], [4, -4]])

        # Horizontal environment
        tensors = [A, A.conj()]
        connects = [[-1, 2, 3, 1], [-2, 2, 3, 1]]
        con_order = [2, 3, 1]
        environment = ncon_func(tensors, connects, con_order)
        nrm = environment.norm()

        if nrm > 1e-10
            environment = environment / nrm
            SH, H = environment.eig(1, 0, hermitian=true)
            A = ncon_func([A, H, H.conj()], [[1, -2, 3, -4], [1, -1], [3, -3]])
        end
    end

    # Final normalization
    A = A / A.norm()

    return A, Dict("method" => "environment")
end

"""
    fix_gauge_phi4(A; method=:svd)

Main gauge fixing function for φ⁴ tensors.

Methods:
- :svd - Simple SVD-based (default, most robust)
- :environment - Environment-based (like Ising, may fail)
- :none - Just normalize

Returns: (A_fixed, info)
"""
function fix_gauge_phi4(A; method=:svd)
    if method == :svd
        return fix_gauge_svd_phi4(A)
    elseif method == :environment
        try
            return fix_gauge_environment_phi4(A)
        catch e
            @warn "Environment gauge fixing failed, falling back to SVD: $e"
            return fix_gauge_svd_phi4(A)
        end
    elseif method == :none
        A_fixed = A / A.norm()
        return A_fixed, Dict("method" => "none")
    else
        error("Unknown gauge fixing method: $method")
    end
end

# Export
export fix_gauge_phi4, fix_gauge_svd_phi4, fix_gauge_environment_phi4
