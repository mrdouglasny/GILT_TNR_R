# MODULE: Phi4Tools
# USAGE: include("Phi4Tools.jl")
# DESCRIPTION: Julia interface for φ⁴ tensors in Gilt-TNR
#
# This module provides tools for working with the 2D φ⁴ lattice field theory
# using the Gilt-TNR algorithm. The φ⁴ model has Z₂ symmetry and is in the
# same universality class as the 2D Ising model.

########################################################
# Load dependencies
########################################################

using PyCall

# Add GiltTNR to Python path
pushfirst!(pyimport("sys")."path", joinpath(@__DIR__, "GiltTNR"))

# Import the φ⁴ module
@pyinclude(joinpath(@__DIR__, "GiltTNR/GiltTNR2D_Phi4.py"))

# Import Tensor class for wrapping plain arrays
const tensors = pyimport("tensors")
const np = pyimport("numpy")

########################################################
# Initial Tensor Construction
########################################################

"""
    initial_tensor_phi4(pars)

Construct the initial φ⁴ tensor for Gilt-TNR.

Parameters (in pars dict):
- "mu_sq": Mass squared parameter (critical ≈ +2.73 for λ=1, κ=1)
- "lam": Quartic coupling (default: 1.0)
- "kappa": Kinetic coupling (default: 1.0)
- "K": Quadrature order (default: 32)
- "D": Initial bond dimension (default: 16)
- "symmetry_tensors": Whether to use Z2 tensors (default: true)

Returns:
- A_0: Initial tensor (TensorZ2 or Tensor)
"""
function initial_tensor_phi4(pars)
    A_0 = py"get_initial_tensor_phi4"(pars)
    return A_0
end

"""
    initial_tensor_phi4_simple(pars)

Construct a plain (non-Z2) φ⁴ tensor.
"""
function initial_tensor_phi4_simple(pars)
    A_0 = py"get_initial_tensor_phi4_simple"(pars)
    return A_0
end

########################################################
# Parameter Utilities
########################################################

"""
    phi4_pars_identifier(pars)

Create a unique identifier string for φ⁴ parameters.
"""
function phi4_pars_identifier(pars)
    mu_sq = pars["mu_sq"]
    lam = pars["lam"]
    kappa = pars["kappa"]
    return "phi4_mu_sq=$(mu_sq)_lam=$(lam)_kappa=$(kappa)"
end

"""
    form_phi4_file_name(phi4_pars, gilt_pars, len)

Create a filename for saving φ⁴ trajectories.
"""
function form_phi4_file_name(phi4_pars, gilt_pars, len)
    phi4_id = phi4_pars_identifier(phi4_pars)
    gilt_id = gilt_pars_identifier(gilt_pars)  # From Tools.jl
    return "$(gilt_id)_$(phi4_id)_len=$(len)"
end

########################################################
# Critical Point Search
########################################################

@enum Phi4Phase begin
    PHI4_SYMMETRIC     # High-T / disordered phase
    PHI4_BROKEN        # Low-T / ordered phase
    PHI4_UNDETERMINED
end

"""
    to_python_tensor(A)

Convert Julia array to Python Tensor if needed.
"""
function to_python_tensor(A)
    if A isa PyObject
        return A
    else
        # Convert Julia array to Python Tensor, keeping as PyObject
        py_array = pycall(np.array, PyObject, A)
        return pycall(tensors.Tensor.from_ndarray, PyObject, py_array)
    end
end

"""
    phi4_phase(A, gilt_pars; max_steps=30, tol=1e-4)

Determine the phase of a φ⁴ tensor by running RG and checking eigenvalues.

Returns: (phase, num_steps)
- phase: PHI4_SYMMETRIC, PHI4_BROKEN, or PHI4_UNDETERMINED
- num_steps: Number of RG steps taken
"""
function phi4_phase(A, gilt_pars; max_steps=30, tol=1e-4)
    # Ensure A is a Python tensor
    A = to_python_tensor(A)

    # Helper to extract tuple element as PyObject
    getitem = pyimport("operator").getitem

    for i in 1:max_steps
        result = pycall(py"gilttnr_step", PyObject, A, 0.0, gilt_pars)
        # Get first element, keep as PyObject
        A = pycall(getitem, PyObject, result, 0)
        spectrum = py"get_A_spectrum_phi4"(A)

        if length(spectrum) < 2
            return PHI4_UNDETERMINED, i
        end

        second_ev = spectrum[2]

        if abs(second_ev - 1) < tol
            return PHI4_BROKEN, i
        elseif abs(second_ev) < tol
            return PHI4_SYMMETRIC, i
        end
    end
    return PHI4_UNDETERMINED, max_steps
end

"""
    lookup_critical_mu_sq(lam, kappa, K, D, gilt_pars, search_tol)

Look up saved critical μ² value. Returns (low, high, critical) or nothing if not found.
"""
function lookup_critical_mu_sq(lam, kappa, K, D, gilt_pars, search_tol)
    filename_pattern = "phi4_" * gilt_pars_identifier(gilt_pars) *
                       "_lam=$(lam)_kappa=$(kappa)_K=$(K)_D=$(D)_tol=$(search_tol)"

    crit_dir = "critical_temperatures"
    if !isdir(crit_dir)
        return nothing
    end

    saved_files = filter(f -> startswith(f, filename_pattern), readdir(crit_dir))
    if !isempty(saved_files)
        saved_file = joinpath(crit_dir, saved_files[1])
        data = deserialize(saved_file)
        @info "Found saved critical μ²: $(data["mu_sq_critical"]) from $saved_file"
        return (data["mu_sq_low"], data["mu_sq_high"], data["mu_sq_critical"])
    end
    return nothing
end

"""
    find_critical_mu_sq(lam, kappa, K, D, gilt_pars;
                        mu_sq_low=nothing, mu_sq_high=nothing,
                        search_tol=1e-6, verbose=true)

Binary search to find the critical μ² for the φ⁴ model.

Arguments:
- lam, kappa, K, D: φ⁴ model parameters
- gilt_pars: Gilt-TNR algorithm parameters
- mu_sq_low: Starting point in broken phase (default: auto based on λ)
- mu_sq_high: Starting point in symmetric phase (default: auto based on λ)
- search_tol: Tolerance for binary search (default: 1e-6)
- verbose: Print progress (default: true)

Returns: (mu_sq_low, mu_sq_high, mu_sq_critical)

Note: The phase structure is:
- Broken phase (Z₂ symmetry broken, ordered): small μ² (e.g., μ² < 2.7)
- Symmetric phase (Z₂ preserved, disordered): large positive μ² (e.g., μ² > 2.8)
- Critical point for λ=1, κ=1: μ²_c ≈ +2.73

The critical μ² is POSITIVE and scales roughly as ~2.7λ for κ=1.
"""
function find_critical_mu_sq(lam, kappa, K, D, gilt_pars;
                              mu_sq_low=nothing, mu_sq_high=nothing,
                              search_tol=1e-6, verbose=true)

    # Auto-set bounds based on λ if not provided
    # Critical μ² is POSITIVE and scales roughly as ~2.7λ for κ=1
    if mu_sq_low === nothing
        mu_sq_low = 2.0 * lam  # Well into broken phase (below critical)
    end
    if mu_sq_high === nothing
        mu_sq_high = 3.5 * lam  # Well into symmetric phase (above critical)
    end

    phi4_pars_low = Dict(
        "mu_sq" => mu_sq_low,
        "lam" => lam,
        "kappa" => kappa,
        "K" => K,
        "D" => D,
        "symmetry_tensors" => true
    )

    phi4_pars_high = Dict(
        "mu_sq" => mu_sq_high,
        "lam" => lam,
        "kappa" => kappa,
        "K" => K,
        "D" => D,
        "symmetry_tensors" => true
    )

    # Sanity check
    A_low = initial_tensor_phi4(phi4_pars_low)
    phase_low, _ = phi4_phase(A_low, gilt_pars)

    A_high = initial_tensor_phi4(phi4_pars_high)
    phase_high, _ = phi4_phase(A_high, gilt_pars)

    if phase_low != PHI4_BROKEN
        @warn "Low μ² point is not in broken phase"
    end
    if phase_high != PHI4_SYMMETRIC
        @warn "High μ² point is not in symmetric phase"
    end

    if verbose
        @info "Starting binary search for critical μ²"
        @info "Initial range: [$mu_sq_low, $mu_sq_high]"
    end

    gap = mu_sq_high - mu_sq_low

    while gap > search_tol
        mu_sq_mid = (mu_sq_low + mu_sq_high) / 2

        phi4_pars_mid = Dict(
            "mu_sq" => mu_sq_mid,
            "lam" => lam,
            "kappa" => kappa,
            "K" => K,
            "D" => D,
            "symmetry_tensors" => true
        )

        A_mid = initial_tensor_phi4(phi4_pars_mid)
        phase_mid, steps = phi4_phase(A_mid, gilt_pars)

        if verbose
            @info "μ² = $mu_sq_mid → $phase_mid (steps=$steps)"
        end

        if phase_mid == PHI4_BROKEN
            mu_sq_low = mu_sq_mid
        elseif phase_mid == PHI4_SYMMETRIC
            mu_sq_high = mu_sq_mid
        else
            @warn "Undetermined phase at μ² = $mu_sq_mid"
            break
        end

        gap = mu_sq_high - mu_sq_low
    end

    mu_sq_critical = (mu_sq_low + mu_sq_high) / 2

    if verbose
        @info "Critical μ² ≈ $mu_sq_critical"
        @info "Final range: [$mu_sq_low, $mu_sq_high]"
    end

    return mu_sq_low, mu_sq_high, mu_sq_critical
end

########################################################
# Trajectory Handling
########################################################

"""
    phi4_trajectory(phi4_pars, len, gilt_pars)

Compute an RG trajectory for φ⁴ starting from given parameters.

Returns a Dict with:
- "A": Vector of tensors along trajectory
- "log_fact": Vector of log normalization factors
- "errs": Vector of error arrays
"""
function phi4_trajectory(phi4_pars, len::Int64, gilt_pars)
    A_init = initial_tensor_phi4(phi4_pars)
    # Ensure it's a Python tensor
    A_init = to_python_tensor(A_init)

    A_hist = [A_init]
    log_fact_hist = Float64[0.0]
    errs_hist = Vector{Float64}[[0, 0, 0, 0, 0]]

    # Helper to extract tuple elements
    getitem = pyimport("operator").getitem

    for i in 2:len
        result = pycall(py"gilttnr_step", PyObject, A_hist[i-1], log_fact_hist[i-1], gilt_pars)
        A = pycall(getitem, PyObject, result, 0)
        log_fact = convert(Float64, pycall(getitem, PyObject, result, 1))
        errs = convert(Vector{Float64}, pycall(getitem, PyObject, result, 2))
        push!(A_hist, A)
        push!(log_fact_hist, log_fact)
        push!(errs_hist, errs)
    end

    return Dict(
        "A" => A_hist,
        "log_fact" => log_fact_hist,
        "errs" => errs_hist,
    )
end

########################################################
# Exports
########################################################

export initial_tensor_phi4, initial_tensor_phi4_simple
export phi4_pars_identifier, form_phi4_file_name
export Phi4Phase, PHI4_SYMMETRIC, PHI4_BROKEN, PHI4_UNDETERMINED
export phi4_phase, find_critical_mu_sq, lookup_critical_mu_sq
export phi4_trajectory
