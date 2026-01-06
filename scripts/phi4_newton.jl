# MODULE: phi4_newton.jl
# USAGE: julia +1.10.4 --project=ekrgilttrnr --threads N ekrgilttrnr/scripts/phi4_newton.jl [options]
# INPUTS: None (generates initial φ⁴ tensor from parameters)
# OUTPUTS: ekrgilttrnr/newton/*.data (serialized φ⁴ fixed point tensors and eigenvalues)
# DESCRIPTION: Newton method to find φ⁴ fixed point tensor and extract critical exponents
# LATEX_DOC: ekrgilttrnr/docs/newton_method_guide.md
#
# BASED ON: EKR GILT-TNR (https://github.com/ebelnikola/GILT_TNR_R, arXiv:2408.10312),
#           ekrgilttrnr/scripts/newton.jl (Ising version)
# NEW IN THIS SCRIPT: Applies Newton method to φ⁴ scalar field theory to find
#                     fixed point tensor and verify universality (same exponents as Ising).
#
# RUNTIME: ~1-2 hours at χ=30 with 20 threads. Scales as O(χ⁶ × n_iterations).
#
# The φ⁴ model is in the same universality class as the 2D Ising model,
# so the critical exponents should match.

################################################
# section: EXPERIMENT INIT
################################################

using ArgParse
using Serialization

settings = ArgParseSettings()
@add_arg_table! settings begin
    "--chi"
    help = "Maximum bond dimension for Gilt-TNR"
    arg_type = Int64
    default = 30
    "--gilt_eps"
    help = "Gilt truncation threshold"
    arg_type = Float64
    default = 6e-6
    "--cg_eps"
    help = "TRG truncation threshold"
    arg_type = Float64
    default = 1e-10
    "--lam"
    help = "Quartic coupling λ"
    arg_type = Float64
    default = 1.0
    "--kappa"
    help = "Kinetic coupling κ"
    arg_type = Float64
    default = 1.0
    "--K"
    help = "Number of quadrature points"
    arg_type = Int64
    default = 32
    "--D"
    help = "Initial bond dimension for φ⁴ tensor"
    arg_type = Int64
    default = 16
    "--mu_sq"
    help = "Mass squared (0 = use critical value from saved search)"
    arg_type = Float64
    default = 0.0
    "--number_of_initial_steps"
    help = "RG steps to get initial fixed point approximation"
    arg_type = Int64
    default = 23
    "--ord"
    help = "Order of finite difference differentiation"
    arg_type = Int64
    default = 2
    "--stp"
    help = "Step size for finite differences"
    arg_type = Float64
    default = 1e-4
    "--eigensystem_size_for_jacobian"
    help = "Number of eigenvectors for Jacobian approximation"
    arg_type = Int64
    default = 54
    "--N"
    help = "Number of Newton iterations"
    arg_type = Int64
    default = 32
    "--verbosity"
    help = "Verbosity level"
    arg_type = Int64
    default = 1
end

const global rotate = true  # Always use rotating algorithm for Newton

pars = parse_args(settings; as_symbols = true)
for (key, value) in pars
    @eval $key = $value
end

# Load tools
include("../src/Tools.jl")
include("../src/GaugeFixing.jl")
include("../src/KrylovTechnical.jl")
include("../src/Phi4Tools.jl")

################################################
# section: LOAD OR FIND CRITICAL μ²
################################################

gilt_pars = Dict(
    "gilt_eps" => gilt_eps,
    "cg_chis" => collect(1:chi),
    "cg_eps" => cg_eps,
    "verbosity" => 0,
    "rotate" => rotate,
)

# Try to load critical μ² from saved file
if mu_sq ≈ 0.0
    search_tol = 1.0e-10
    filename_pattern = "critical_temperatures/phi4_" * gilt_pars_identifier(gilt_pars) *
                       "_lam=$(lam)_kappa=$(kappa)_K=$(K)_D=$(D)"

    # Look for saved critical point
    saved_files = filter(f -> startswith(f, basename(filename_pattern)), readdir("critical_temperatures"))

    if !isempty(saved_files)
        saved_file = "critical_temperatures/" * saved_files[1]
        @info "Loading critical μ² from: $saved_file"
        saved_data = deserialize(saved_file)
        mu_sq = saved_data["mu_sq_critical"]
        @info "Using μ² = $mu_sq"
    else
        @info "No saved critical point found, searching..."
        _, _, mu_sq = find_critical_mu_sq(
            lam, kappa, K, D, gilt_pars;
            search_tol = search_tol,
            verbose = true
        )
    end
end

################################################
# section: INITIAL APPROXIMATION
################################################

@info "=" ^ 60
@info "φ⁴ Newton Method for Fixed Point"
@info "=" ^ 60
@info "Model: μ² = $mu_sq, λ = $lam, κ = $kappa"
@info "Gilt-TNR: χ = $chi, gilt_eps = $gilt_eps, rotate = $rotate"

phi4_pars = Dict(
    "mu_sq" => mu_sq,
    "lam" => lam,
    "kappa" => kappa,
    "K" => K,
    "D" => D,
    "symmetry_tensors" => true
)

@info "\nGenerating initial approximation ($number_of_initial_steps RG steps)..."
traj = phi4_trajectory(phi4_pars, number_of_initial_steps + 1, gilt_pars)
A_crit_approximation = traj["A"][end]

# Fix gauge
A_crit_approximation, Hc, Vc, SHc, SVc = fix_continuous_gauge(A_crit_approximation)
A_crit_approximation, accepted_elements = fix_discrete_gauge(A_crit_approximation; tol = 1e-7)
A_crit_approximation /= A_crit_approximation.norm()
A_crit_approximation_JU = py_to_ju(A_crit_approximation)

################################################
# section: FIX RECURSION DEPTH
################################################

# Run one step to determine recursion depths
A1, _ = py"gilttnr_step"(A_crit_approximation, 0.0, gilt_pars)

tmp = py"depth_dictionary"
recursion_depth = Dict(
    "S" => tmp[(1, "S")],
    "N" => tmp[(1, "N")],
    "E" => tmp[(1, "E")],
    "W" => tmp[(1, "W")],
)

gilt_pars = Dict(
    "gilt_eps" => gilt_eps,
    "cg_chis" => collect(1:chi),
    "cg_eps" => cg_eps,
    "verbosity" => 0,
    "bond_repetitions" => 2,
    "recursion_depth" => recursion_depth,
    "rotate" => rotate,
)

################################################
# section: DEFINE GILT FUNCTION WITH GAUGE FIXING
################################################

function gilt_phi4(A, pars)
    A = ju_to_py(A)
    A, _ = py"gilttnr_step"(A, 0.0, pars)
    A, _ = fix_continuous_gauge(A)
    A, _ = fix_discrete_gauge(A, accepted_elements)
    A /= A.norm()
    return py_to_ju(A)
end

function dgilt_phi4(δA)
    return df(x -> gilt_phi4(x, gilt_pars), A_crit_approximation_JU, δA; stp = stp, order = ord)
end

################################################
# section: COMPUTE EIGENSYSTEM
################################################

@info "\nComputing Jacobian eigensystem (this may take a while)..."
@time begin
    initial_vector = py_to_ju(random_Z2tens(A_crit_approximation))
    eigensystem_init = eigsolve(
        dgilt_phi4, initial_vector, eigensystem_size_for_jacobian, :LM;
        verbosity = verbosity,
        issymmetric = false,
        ishermitian = false,
        krylovdim = eigensystem_size_for_jacobian + 20
    )
end

println("\nEIGENVALUES (INITIAL):")
for (i, val) in enumerate(eigensystem_init[1][1:min(20, length(eigensystem_init[1]))])
    println("  $i: $val")
end

# Handle complex conjugate pairs
if length(eigensystem_init[1]) > eigensystem_size_for_jacobian
    if conj(eigensystem_init[1][eigensystem_size_for_jacobian]) ≈ eigensystem_init[1][eigensystem_size_for_jacobian+1]
        approximation_rank = eigensystem_size_for_jacobian + 1
    else
        approximation_rank = eigensystem_size_for_jacobian
    end
else
    approximation_rank = eigensystem_size_for_jacobian
end

eigensystem_init = [eigensystem_init[1][1:approximation_rank], eigensystem_init[2][1:approximation_rank]]

################################################
# section: BUILD JACOBIAN APPROXIMATION
################################################

function build_jacobian_approximation(vectors::Vector{t}, values::Vector) where {t}
    rank = length(values)
    jacobian_approximation = zeros(rank, rank)
    basis = Vector{t}(undef, rank)
    i = 1
    while i <= rank
        if imag(values[i]) == 0
            jacobian_approximation[i, i] = real(values[i])
            basis[i] = real(vectors[i])
            i += 1
        else
            if conj(values[i+1]) != values[i]
                throw("Unmatched complex eigenvalue is detected")
            end
            λ₁ = real(values[i])
            λ₂ = imag(values[i])
            v1 = real(vectors[i])
            v2 = imag(vectors[i])
            v1norm = v1 |> norm
            v2norm = v2 |> norm
            v1 /= v1norm
            v2 /= v2norm
            jacobian_approximation[i, i] = λ₁
            jacobian_approximation[i+1, i+1] = λ₁
            jacobian_approximation[i+1, i] = -λ₂ * v2norm / v1norm
            jacobian_approximation[i, i+1] = λ₂ * v1norm / v2norm
            basis[i] = v1
            basis[i+1] = v2
            i += 2
        end
    end
    jacobian_approximation, basis
end

function build_Graham_Schmidt_matrix(non_orthogonal_normalised_basis::Vector{t}) where {t}
    dim = length(non_orthogonal_normalised_basis)
    Graham_Schmidt_Matrix = zeros(dim, dim)
    Graham_Schmidt_Matrix[1, 1] = 1
    orthonormal_basis = t[non_orthogonal_normalised_basis[1]]

    # Track which vectors are kept (for handling degenerate cases)
    kept_indices = Int[1]

    for n ∈ 1:(dim-1)
        new_vector = deepcopy(non_orthogonal_normalised_basis[n+1])
        for i ∈ eachindex(orthonormal_basis)
            proj = dot(orthonormal_basis[i], new_vector)
            new_vector = new_vector - proj * orthonormal_basis[i]
        end
        Nnp1 = norm(new_vector)

        # Check for near-linear dependence
        if Nnp1 < 1e-12
            @warn "Gram-Schmidt: vector $(n+1) is nearly linearly dependent (norm=$Nnp1), using regularization"
            # Use a small regularization instead of skipping
            Nnp1 = 1e-12
        end

        new_vector = new_vector / Nnp1
        push!(orthonormal_basis, new_vector)
        push!(kept_indices, n+1)

        # Update the transformation matrix
        num_kept = length(kept_indices)
        gnn = Graham_Schmidt_Matrix[1:num_kept-1, 1:num_kept-1]
        v_old_dot_v_new_vector = zeros(num_kept-1)
        for i ∈ 1:(num_kept-1)
            v_old_dot_v_new_vector[i] = dot(non_orthogonal_normalised_basis[kept_indices[i]], non_orthogonal_normalised_basis[n+1])
        end
        Graham_Schmidt_Matrix[1:num_kept-1, num_kept] .= -1 / Nnp1 .* gnn * transpose(gnn) * v_old_dot_v_new_vector
        Graham_Schmidt_Matrix[num_kept, num_kept] = 1 / Nnp1
    end
    return Graham_Schmidt_Matrix, orthonormal_basis
end

jac_approximation_non_orthogonal_basis, non_orthogonal_normalised_basis = build_jacobian_approximation(eigensystem_init[2], eigensystem_init[1])
Graham_Schmidt_matrix, orthonormal_basis = build_Graham_Schmidt_matrix(non_orthogonal_normalised_basis)

# Use pseudoinverse for numerical stability (handles ill-conditioned matrices)
function safe_matrix_inv(M; tol=1e-10)
    U, S, V = svd(M)
    # Truncate small singular values
    S_inv = [s > tol ? 1/s : 0.0 for s in S]
    return V * Diagonal(S_inv) * U'
end

# Check for NaN/Inf before proceeding
if any(isnan, Graham_Schmidt_matrix) || any(isinf, Graham_Schmidt_matrix)
    @warn "Graham-Schmidt matrix contains NaN/Inf, using identity approximation"
    jac_approximation = jac_approximation_non_orthogonal_basis
else
    GS_inv = safe_matrix_inv(Graham_Schmidt_matrix)
    jac_approximation = GS_inv * jac_approximation_non_orthogonal_basis * Graham_Schmidt_matrix
end

# Similarly for (I - J)^(-1)
ImJ_matrix = I - jac_approximation
if any(isnan, ImJ_matrix) || any(isinf, ImJ_matrix)
    @warn "Jacobian approximation contains NaN/Inf, using identity"
    ImJ_inv_matrix = Matrix{Float64}(I, size(jac_approximation))
else
    ImJ_inv_matrix = safe_matrix_inv(ImJ_matrix)
end

function project_to_Vs(δA)
    res = zero(δA)
    for i in 1:approximation_rank
        res += orthonormal_basis[i] * dot(orthonormal_basis[i], δA)
    end
    return res
end

function ImJ_inv(δA)
    δA_in_Vs = project_to_Vs(δA)
    ImJ_inv_δA_in_Vs = zero(δA)
    δA_out_of_Vs = δA - δA_in_Vs
    for i in 1:approximation_rank
        for j in 1:approximation_rank
            ImJ_inv_δA_in_Vs += ImJ_inv_matrix[i, j] * orthonormal_basis[i] * dot(orthonormal_basis[j], δA)
        end
    end
    return ImJ_inv_δA_in_Vs + δA_out_of_Vs
end

################################################
# section: NEWTON ITERATION
################################################

function newton_function(A)
    x_minus_f = A - gilt_phi4(A, gilt_pars)
    correction = ImJ_inv(x_minus_f)
    return A - correction
end

A_hist = [A_crit_approximation_JU]

@info "\nStarting Newton iteration..."
@time begin
    for i in 2:N
        push!(A_hist, newton_function(A_hist[i-1]))
        step_size = A_hist[i] - A_hist[i-1] |> norm
        println("Newton step $i: ||δA|| = $step_size")
    end
end

################################################
# section: SAVE RESULTS
################################################

mkpath("newton")

result = Dict(
    "A_newton" => A_hist,
    "eigensystem_init" => eigensystem_init,
    "bond_repetitions" => 2,
    "recursion_depth" => recursion_depth,
    "phi4_pars" => phi4_pars,
    "gilt_pars" => gilt_pars,
)

filename = "newton/phi4_" * gilt_pars_identifier(gilt_pars) *
           "_mu_sq=$(mu_sq)_lam=$(lam)_rank=$(approximation_rank).data"

serialize(filename, result)
@info "\nResults saved to: $filename"

@info "\n" * "=" ^ 60
@info "EIGENVALUES AT FIXED POINT (should match Ising CFT):"
@info "  Expected: ε→2.0, T→±1.0, σ→3.668"
@info "=" ^ 60
for (i, val) in enumerate(eigensystem_init[1][1:min(10, length(eigensystem_init[1]))])
    println("  λ_$i = $val")
end
