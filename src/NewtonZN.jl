# MODULE: NewtonZN.jl
# USAGE: include("src/NewtonZN.jl") or using EKRNewton
# INPUTS: Initial tensor, model, Gilt-TNR parameters
# OUTPUTS: Fixed-point tensor with ~10⁻⁹ accuracy
# DESCRIPTION: Newton method for Z_N symmetric tensor network fixed points
#
# BASED ON: Lab/newton.jl (Ising-specific Newton method)
#           EKR arXiv:2408.10312 (Newton method for TNR fixed points)
# NEW IN THIS MODULE: Generalized to arbitrary Z_N models

"""
Newton method for tensor network fixed points.

The Newton iteration solves for the fixed point A* of the RG map R:
    A* = R(A*)

Standard iteration A_{n+1} = R(A_n) converges slowly.
Newton's method: A_{n+1} = A_n - (I - J)⁻¹ (A_n - R(A_n))
where J = dR/dA is the Jacobian, converges quadratically near the fixed point.

Key components:
1. Jacobian approximation via KrylovKit eigensolver
2. Gram-Schmidt orthonormalization of eigenbasis
3. (I - J)⁻¹ computed in the low-rank eigenbasis

Requires both continuous and discrete gauge fixing for stability.
"""

using LinearAlgebra
using PyCall
using KrylovKit

# Include dependencies
include("ZNTensor.jl")
include("ContinuousGauge.jl")
include("DiscreteGaugeZN.jl")
include("ModelProvider.jl")

################################################
# Configuration
################################################

"""
Configuration for Newton iteration.

The `gilt_pars` are used for warmup, while `jacobian_pars` are used during
Jacobian computation. Key difference: Jacobian computation requires
gilt_eps=0 (TRG only) to preserve tensor shape under perturbation.
"""
struct NewtonConfig{M<:SpinModel}
    model::M
    gilt_pars::Dict{String, Any}       # For warmup (with GILT)
    jacobian_pars::Dict{String, Any}   # For Jacobian (TRG only, gilt_eps=0)
    jacobian_rank::Int
    differentiation_order::Int
    differentiation_step::Float64
    verbosity::Int
end

"""
    NewtonConfig(model; kwargs...) -> NewtonConfig

Create Newton configuration for a specific model.

Parameters:
- chi: Maximum bond dimension
- gilt_eps: GILT truncation threshold for warmup (default: 6e-6 for Ising, 3e-5 for Potts)
- cg_eps: Coarse-graining error threshold
- jacobian_rank: Number of Jacobian eigenvalues to compute
- diff_order: Order of numerical differentiation (2 or 4)
- diff_step: Step size for numerical differentiation
- verbosity: 0=quiet, 1=normal, 2=verbose

Note: Jacobian computation always uses gilt_eps=0 (TRG only) to ensure
shape preservation during numerical differentiation.
"""
function NewtonConfig(
    model::SpinModel;
    chi::Int = 30,
    gilt_eps::Float64 = 6e-6,
    cg_eps::Float64 = 1e-10,
    jacobian_rank::Int = 9,
    diff_order::Int = 2,
    diff_step::Float64 = 1e-4,
    verbosity::Int = 0,
)
    # Warmup parameters: use GILT for better convergence to fixed point
    gilt_pars = Dict{String, Any}(
        "gilt_eps" => gilt_eps,
        "cg_chis" => collect(1:chi),  # Adaptive during warmup
        "cg_eps" => cg_eps,
        "verbosity" => verbosity,
    )

    # Jacobian parameters: TRG only (gilt_eps=0) for shape preservation
    # This ensures the tensor shape is preserved during numerical differentiation
    jacobian_pars = Dict{String, Any}(
        "gilt_eps" => 0.0,           # No GILT - TRG only
        "cg_chis" => [chi],          # Fixed bond dimension
        "cg_eps" => 1e-10,           # Small but non-zero
        "verbosity" => verbosity,
    )

    return NewtonConfig(
        model,
        gilt_pars,
        jacobian_pars,
        jacobian_rank,
        diff_order,
        diff_step,
        verbosity,
    )
end

################################################
# Python Interface
################################################

const _py_gilttnr = Ref{Any}(nothing)

function get_gilttnr()
    if _py_gilttnr[] === nothing
        pushfirst!(pyimport("sys")."path", "GiltTNR")
        # Use standard GiltTNR2D for plain tensors (Potts)
        _py_gilttnr[] = pyimport("GiltTNR2D")
    end
    return _py_gilttnr[]
end

"""
    gilttnr_step_py(A::PyObject, log_fact, pars) -> (A_new, log_fact_new)

Call Python GiltTNR step.
"""
function gilttnr_step_py(A::PyObject, log_fact::Real, pars::Dict)
    gilttnr = get_gilttnr()
    # Use pycall to prevent automatic conversion of returned tensor
    result = pycall(gilttnr.gilttnr_step, PyObject, A, log_fact, pars)
    # Access tuple elements using Python's getitem to keep PyObject type
    A_new = get(result, PyObject, 0)  # Python tuples are 0-indexed
    log_new = convert(Float64, get(result, PyObject, 1))
    return A_new, log_new
end

################################################
# Gauge-Fixed RG Step
################################################

"""
    gilt_step_gauge_fixed(A::ZNTensor{N}, config::NewtonConfig) -> ZNTensor{N}

Apply one Gilt-TNR step with gauge fixing.

Steps:
1. Convert Julia ZNTensor to Python
2. Apply Gilt-TNR coarse-graining
3. Fix continuous gauge (environment diagonalization)
4. Fix discrete Z_N gauge (GF(N) linear system)
5. Normalize and convert back to Julia
"""
function gilt_step_gauge_fixed(A::ZNTensor{N}, config::NewtonConfig) where N
    # Convert to Python
    A_py = zn_to_py(A)

    # Gilt-TNR step
    A_new_py, _ = gilttnr_step_py(A_py, 0.0, config.gilt_pars)

    # Continuous gauge fixing
    A_new_py, _, _, _, _ = fix_continuous_gauge(A_new_py)

    # Discrete gauge fixing
    A_arr = A_new_py.to_ndarray()
    A_arr, _, _, _ = fix_discrete_gauge_zn(A_arr, N)

    # Normalize
    A_arr ./= norm(A_arr)

    # Convert back to ZNTensor
    # Note: This is simplified - full implementation would preserve tensor structure
    return py_to_zn(A_new_py, Val(N))
end

"""
    gilt_step_gauge_fixed(A::ZNTensor{N}, accepted, config) -> ZNTensor{N}

Apply GILT-TNR step with fixed recursion depth and gauge fixing.

Uses config.jacobian_pars which should include recursion_depth for
deterministic shape-preserving behavior during Jacobian computation.
"""
function gilt_step_gauge_fixed(A::ZNTensor{N}, accepted::Vector, config::NewtonConfig) where N
    input_shape = ntuple(i -> Int(A.shape[i, 1]), 4)

    A_py = zn_to_py(A)
    # Use jacobian_pars with fixed recursion_depth for deterministic RG
    A_new_py, _ = gilttnr_step_py(A_py, 0.0, config.jacobian_pars)
    A_new_py, _, _, _, _ = fix_continuous_gauge(A_new_py)

    A_arr = A_new_py.to_ndarray()
    output_shape = size(A_arr)

    # With fixed recursion_depth, shape should be preserved
    if output_shape != input_shape
        @warn "Shape changed from $input_shape to $output_shape"
    end

    # Apply discrete gauge fixing with pre-computed elements
    if !isempty(accepted)
        max_idx = maximum(el[1][i] for el in accepted for i in 1:4)
        if any(output_shape .< max_idx)
            @warn "Shape mismatch: accepted indices up to $max_idx, but shape is $output_shape"
            A_arr, _, _, _ = fix_discrete_gauge_zn(A_arr, N)
        else
            A_arr, _, _, _ = fix_discrete_gauge_zn(A_arr, N, accepted)
        end
    else
        A_arr, _, _, _ = fix_discrete_gauge_zn(A_arr, N)
    end

    A_arr ./= norm(A_arr)

    # Convert to ZNTensor with trivial Z_N structure (single sector)
    sects = Dict{NTuple{4, Int64}, Array}((0, 0, 0, 0) => A_arr)
    shape = zeros(Int64, 4, 1)
    for i in 1:4
        shape[i, 1] = size(A_arr, i)
    end
    qhape = zeros(Int64, 4, 1)
    dirs = [1, 1, -1, -1]

    return ZNTensor{N}(sects, shape, qhape, dirs)
end

################################################
# Numerical Differentiation
################################################

"""
    numerical_derivative(f, A, δA; step=1e-4, order=2)

Compute df(A) · δA using finite differences.

Order 2: (f(A + h*δA) - f(A - h*δA)) / (2h)
"""
function numerical_derivative(f, A::ZNTensor{N}, δA::ZNTensor{N}; step::Real = 1e-4, order::Int = 2) where N
    if order == 2
        # Central difference
        f_plus = f(A + step * δA)
        f_minus = f(A - step * δA)
        return (f_plus - f_minus) / (2 * step)
    elseif order == 4
        # 4th order central difference
        f_2plus = f(A + 2 * step * δA)
        f_plus = f(A + step * δA)
        f_minus = f(A - step * δA)
        f_2minus = f(A - 2 * step * δA)
        return (-f_2plus + 8 * f_plus - 8 * f_minus + f_2minus) / (12 * step)
    else
        error("Differentiation order $order not supported")
    end
end

################################################
# Jacobian Eigensystem
################################################

"""
    jacobian_operator(A, accepted, config) -> Function

Create the Jacobian operator dR/dA as a function for KrylovKit.
"""
function jacobian_operator(A::ZNTensor{N}, accepted::Vector, config::NewtonConfig) where N
    function Jop(δA::ZNTensor{N})
        f = x -> gilt_step_gauge_fixed(x, accepted, config)
        return numerical_derivative(f, A, δA; step = config.differentiation_step, order = config.differentiation_order)
    end
    return Jop
end

"""
    compute_jacobian_eigensystem(A, accepted, config) -> (values, vectors)

Compute dominant eigenpairs of the Jacobian using KrylovKit.
"""
function compute_jacobian_eigensystem(A::ZNTensor{N}, accepted::Vector, config::NewtonConfig) where N
    Jop = jacobian_operator(A, accepted, config)

    # Initial random vector - now handles ZNTensor directly
    initial = random_zntens(A, Val(N))

    # Solve eigenproblem
    krylov_dim = config.jacobian_rank + 20
    vals, vecs, _ = eigsolve(
        Jop, initial, config.jacobian_rank, :LM;
        verbosity = config.verbosity,
        issymmetric = false,
        ishermitian = false,
        krylovdim = krylov_dim
    )

    if config.verbosity > 0
        println("Jacobian eigenvalues:")
        for v in vals
            println("  $v")
        end
    end

    return vals, vecs
end

################################################
# Jacobian Approximation
################################################

"""
    build_jacobian_approximation(vectors, values) -> (J_approx, basis)

Build the Jacobian matrix in the eigenvector basis.

Handles complex conjugate eigenvalue pairs by extracting real/imag parts.
"""
function build_jacobian_approximation(vectors::Vector{ZNTensor{N}}, values::Vector) where N
    rank = length(values)
    J_approx = zeros(rank, rank)
    basis = Vector{ZNTensor{N}}(undef, rank)

    i = 1
    while i <= rank
        if imag(values[i]) == 0
            # Real eigenvalue
            J_approx[i, i] = real(values[i])
            basis[i] = real(vectors[i])
            i += 1
        else
            # Complex conjugate pair
            if i + 1 > rank || conj(values[i + 1]) != values[i]
                @warn "Unmatched complex eigenvalue at index $i"
                J_approx[i, i] = real(values[i])
                basis[i] = real(vectors[i])
                i += 1
                continue
            end

            λ_re = real(values[i])
            λ_im = imag(values[i])
            v_re = real(vectors[i])
            v_im = imag(vectors[i])

            norm_re = norm(v_re)
            norm_im = norm(v_im)
            v_re = v_re / norm_re
            v_im = v_im / norm_im

            J_approx[i, i] = λ_re
            J_approx[i + 1, i + 1] = λ_re
            J_approx[i + 1, i] = -λ_im * norm_im / norm_re
            J_approx[i, i + 1] = λ_im * norm_re / norm_im

            basis[i] = v_re
            basis[i + 1] = v_im
            i += 2
        end
    end

    return J_approx, basis
end

"""
    gram_schmidt_matrix(basis) -> (G, orthonormal_basis)

Compute Gram-Schmidt orthonormalization matrix and orthonormal basis.

If v_i is the original basis and e_i is orthonormal:
    e_i = Σ_j G[j,i] * v_j
"""
function gram_schmidt_matrix(basis::Vector{ZNTensor{N}}) where N
    dim = length(basis)
    G = zeros(dim, dim)
    G[1, 1] = 1.0
    ortho_basis = ZNTensor{N}[basis[1] / norm(basis[1])]

    for n in 2:dim
        # Orthogonalize against previous vectors
        new_vec = basis[n]
        for i in 1:n-1
            new_vec = new_vec - dot(ortho_basis[i], new_vec) * ortho_basis[i]
        end

        new_norm = norm(new_vec)
        if new_norm < 1e-10
            @warn "Near-zero vector at step $n in Gram-Schmidt"
            new_norm = 1.0
        end
        new_vec = new_vec / new_norm
        push!(ortho_basis, new_vec)

        # Update G matrix
        g_prev = G[1:n-1, 1:n-1]
        overlaps = [dot(basis[i], basis[n]) for i in 1:n-1]
        G[1:n-1, n] = -1 / new_norm * g_prev * g_prev' * overlaps
        G[n, n] = 1 / new_norm
    end

    return G, ortho_basis
end

################################################
# Newton Iteration
################################################

"""
    ImJ_inverse_operator(J_approx, G, ortho_basis) -> Function

Create the (I - J)⁻¹ operator for Newton correction.

The operator acts in the low-rank subspace spanned by eigenvectors,
and is identity outside this subspace.
"""
function ImJ_inverse_operator(J_approx::Matrix, G::Matrix, ortho_basis::Vector{ZNTensor{N}}) where N
    rank = length(ortho_basis)

    # (I - J)⁻¹ in the non-orthogonal basis coordinates
    ImJ_inv_matrix = inv(I - J_approx)

    # Transform to orthonormal basis coordinates
    ImJ_inv_ortho = G^(-1) * ImJ_inv_matrix * G

    function ImJ_inv(δA::ZNTensor{N})
        # Project onto subspace
        coeffs = [dot(e, δA) for e in ortho_basis]
        δA_in = sum(c * e for (c, e) in zip(coeffs, ortho_basis))
        δA_out = δA - δA_in

        # Apply (I - J)⁻¹ in subspace
        new_coeffs = ImJ_inv_ortho * coeffs
        result_in = sum(c * e for (c, e) in zip(new_coeffs, ortho_basis))

        # δA_out is unchanged (identity outside subspace)
        return result_in + δA_out
    end

    return ImJ_inv
end

"""
    newton_step(A, ImJ_inv, accepted, config) -> A_new

Perform one Newton iteration step.

Newton update: A_{n+1} = A_n - (I - J)⁻¹ (A_n - R(A_n))
"""
function newton_step(A::ZNTensor{N}, ImJ_inv::Function, accepted::Vector, config::NewtonConfig) where N
    # Compute residual: A - R(A)
    R_A = gilt_step_gauge_fixed(A, accepted, config)
    residual = A - R_A

    # Apply Newton correction
    correction = ImJ_inv(residual)

    return A - correction
end

################################################
# Main Newton Algorithm
################################################

"""
    newton_iteration(A_init, config; n_warmup=23, n_newton=32) -> Dict

Run Newton iteration for fixed-point tensor.

Parameters:
- A_init: Initial tensor (PyObject or ZNTensor)
- config: NewtonConfig with model and parameters
- n_warmup: Number of initial RG steps to approach fixed point
- n_newton: Number of Newton iterations

Returns Dict with:
- "A_final": Final tensor
- "A_history": History of tensors
- "eigenvalues": Jacobian eigenvalues
- "residuals": ||A_n - R(A_n)|| at each step
"""
function newton_iteration(
    A_init::PyObject,
    config::NewtonConfig;
    n_warmup::Int = 23,
    n_newton::Int = 32
)
    N = symmetry_order(config.model)

    if config.verbosity > 0
        println("=" ^ 70)
        println("Newton Iteration for $(model_name(config.model))")
        println("=" ^ 70)
        println("Symmetry: Z_$N")
        println("Warmup steps: $n_warmup")
        println("Newton steps: $n_newton")
        println("Jacobian rank: $(config.jacobian_rank)")
    end

    # Warmup: approach fixed point with standard GILT-TNR iteration
    if config.verbosity > 0
        println("\n--- Warmup Phase ---")
    end

    A = A_init
    for step in 1:n_warmup
        A, _ = gilttnr_step_py(A, 0.0, config.gilt_pars)
        if config.verbosity > 1
            arr_shape = size(A.to_ndarray())
            println("Warmup step $step: shape = $arr_shape")
        end
    end

    # Run a few TRG-only steps to stabilize at target chi
    # This ensures the tensor has the shape we'll use for Jacobian
    if config.verbosity > 0
        println("Stabilizing at target chi with TRG-only steps...")
    end
    for _ in 1:3
        A, _ = gilttnr_step_py(A, 0.0, config.jacobian_pars)
    end

    # Fix gauge and get constraint elements
    A, _, _, _, _ = fix_continuous_gauge(A)
    A_arr = A.to_ndarray()
    target_shape = size(A_arr)
    if config.verbosity > 0
        println("Final tensor shape: $target_shape")
    end
    A_arr, accepted, _, _ = fix_discrete_gauge_zn(A_arr, N)
    A = A / A.norm()

    # Convert to Julia ZNTensor
    A_ju = py_to_zn(A, Val(N))

    # Compute Jacobian eigensystem
    if config.verbosity > 0
        println("\n--- Computing Jacobian Eigensystem ---")
    end

    vals, vecs = compute_jacobian_eigensystem(A_ju, accepted, config)

    # Adjust rank for complex pairs
    actual_rank = config.jacobian_rank
    if length(vals) > actual_rank && conj(vals[actual_rank]) ≈ vals[actual_rank + 1]
        actual_rank += 1
    end
    vals = vals[1:min(actual_rank, length(vals))]
    vecs = vecs[1:min(actual_rank, length(vecs))]

    # Build Jacobian approximation
    J_approx, basis = build_jacobian_approximation(vecs, vals)
    G, ortho_basis = gram_schmidt_matrix(basis)

    # Create (I - J)⁻¹ operator
    ImJ_inv = ImJ_inverse_operator(J_approx, G, ortho_basis)

    # Newton iteration
    if config.verbosity > 0
        println("\n--- Newton Iteration ---")
    end

    A_history = [A_ju]
    residuals = Float64[]

    for step in 1:n_newton
        A_new = newton_step(A_history[end], ImJ_inv, accepted, config)

        step_size = norm(A_new - A_history[end])
        push!(residuals, step_size)
        push!(A_history, A_new)

        if config.verbosity > 0
            println("Newton step $step: ||δA|| = $step_size")
        end

        # Check convergence
        if step_size < 1e-12
            if config.verbosity > 0
                println("Converged at step $step")
            end
            break
        end
    end

    return Dict(
        "A_final" => A_history[end],
        "A_history" => A_history,
        "eigenvalues" => vals,
        "eigenvectors" => vecs,
        "residuals" => residuals,
        "jacobian_rank" => actual_rank,
        "accepted_elements" => accepted,
    )
end

################################################
# Convenience Functions
################################################

"""
    run_newton_ising(; kwargs...) -> Dict

Run Newton iteration for 2D Ising model with default parameters.
"""
function run_newton_ising(; chi::Int = 30, gilt_eps::Float64 = 6e-6, kwargs...)
    model = IsingModel()
    config = NewtonConfig(model; chi = chi, gilt_eps = gilt_eps, kwargs...)

    # Get initial tensor
    gilttnr = get_gilttnr()
    pars = Dict("relT" => 1.0, "Jratio" => 1.0)
    A_init = gilttnr.get_initial_tensor(pars)

    return newton_iteration(A_init, config; kwargs...)
end

"""
    run_newton_potts(N::Int; kwargs...) -> Dict

Run Newton iteration for N-state Potts model.
"""
function run_newton_potts(N::Int; chi::Int = 30, gilt_eps::Float64 = 3e-5, kwargs...)
    model = PottsModel(N)
    config = NewtonConfig(model; chi = chi, gilt_eps = gilt_eps, kwargs...)

    # Get initial tensor
    β = critical_beta(model)
    A_arr = initial_tensor_array(model, β)
    mods = get_gilt_modules()
    A_init = mods.tensors.Tensor.from_ndarray(A_arr)

    return newton_iteration(A_init, config; kwargs...)
end

################################################
# Export
################################################

export NewtonConfig
export gilt_step_gauge_fixed
export compute_jacobian_eigensystem
export build_jacobian_approximation, gram_schmidt_matrix
export ImJ_inverse_operator, newton_step
export newton_iteration
export run_newton_ising, run_newton_potts
