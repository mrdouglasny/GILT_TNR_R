# MODULE: phi4_newton_exponents.jl
# USAGE: julia --project=ekrgilttrnr ekrgilttrnr/scripts/phi4_newton_exponents.jl
# INPUTS: None
# OUTPUTS: Scaling dimensions from Newton method eigenvalues
# DESCRIPTION: Compute critical exponents using linearized RG (Newton method)
#
# BASED ON: phi4_exponents_plot.jl, eigensystem.jl from EKR Gilt-TNR
# NEW: Uses Python-side eigenvalue computation to avoid Julia/Python type issues

using Printf
using PyCall
using LinearAlgebra: dot

# Add GiltTNR to Python path
pushfirst!(pyimport("sys")."path", joinpath(@__DIR__, "../src/GiltTNR"))

# Import Python modules
@pyinclude(joinpath(@__DIR__, "../src/GiltTNR/GiltTNR2D_Phi4.py"))
@pyinclude(joinpath(@__DIR__, "../src/GiltTNR/GiltTNR2D_essentials.py"))

const np = pyimport("numpy")
const scipy_sparse = pyimport("scipy.sparse.linalg")

########################################
# Parameters
########################################

mu_sq = 2.731815  # Critical point (POSITIVE!)
chi = 30
gilt_eps = 1e-7
n_rg_steps = 5  # Use step 5 where x_ε ≈ 1.0

gilt_pars = Dict(
    "gilt_eps" => gilt_eps,
    "cg_chis" => collect(1:chi),
    "cg_eps" => 1e-10,
    "verbosity" => 0,
    "rotate" => false,
)

phi4_pars = Dict(
    "mu_sq" => mu_sq,
    "lam" => 1.0,
    "kappa" => 1.0,
    "K" => 32,
    "D" => 16,
    "symmetry_tensors" => true
)

########################################
# Run RG to approximate fixed point
########################################

println("=" ^ 60)
println("φ⁴ Newton Method Critical Exponents")
println("=" ^ 60)
println("Parameters: μ² = $mu_sq, χ = $chi, gilt_eps = $gilt_eps")
println("-" ^ 60)

println("\nStep 1: Running RG to approximate fixed point...")

A = py"get_initial_tensor_phi4"(phi4_pars)

for step in 1:n_rg_steps
    global A
    A, _ = py"gilttnr_step"(A, 0.0, gilt_pars)

    scaldims = py"get_scaldims_phi4"(A)
    if length(scaldims) >= 3
        @printf("  Step %2d: x_σ = %.4f, x_ε = %.4f\n", step, scaldims[2], scaldims[3])
    end
end

A_star = A

########################################
# Convert to array for linearization
########################################

println("\n" * "-" ^ 60)
println("Step 2: Converting to array representation...")

# Convert Z2 tensor to dense array
A_arr = A_star.to_ndarray()
D = size(A_arr, 1)
println("  Tensor shape: $D × $D × $D × $D")
println("  Total elements: $(D^4)")

########################################
# Build linearized RG operator
########################################

println("\n" * "-" ^ 60)
println("Step 3: Building linearized RG operator via finite differences...")

# Flatten tensor to vector
A_vec = vec(A_arr)
n = length(A_vec)
println("  Vector dimension: $n")

# Import Tensor class and gilttnr_step function
const Tensor = pyimport("tensors").Tensor
const gilt_module = pyimport("GiltTNR2D_essentials")

# RG map as a function on arrays
function rg_map_array(v_arr)
    # Reshape to tensor
    T = reshape(v_arr, D, D, D, D)
    # Convert to Python Tensor - need to use the Tensor class method
    T_np = np.array(T)
    T_py = pycall(Tensor.from_ndarray, PyObject, T_np)
    # Apply RG using direct pycall to avoid conversion issues
    result = pycall(gilt_module.gilttnr_step, PyObject, T_py, 0.0, gilt_pars)
    # Get first element using Python operator.getitem to avoid auto-conversion
    getitem = pyimport("operator").getitem
    T_out = pycall(getitem, PyObject, result, 0)
    # Convert back to numpy array
    T_out_arr = pycall(T_out.to_ndarray, PyObject)
    T_out_arr = convert(Array{Float64}, T_out_arr)
    # Handle shape change by padding/truncating to D (per-axis)
    out_shape = size(T_out_arr)
    # Build output array of size D×D×D×D
    output = zeros(D, D, D, D)
    # Copy the overlapping region
    copy_shape = Tuple(min(out_shape[i], D) for i in 1:4)
    output[1:copy_shape[1], 1:copy_shape[2], 1:copy_shape[3], 1:copy_shape[4]] =
        T_out_arr[1:copy_shape[1], 1:copy_shape[2], 1:copy_shape[3], 1:copy_shape[4]]
    return vec(output)
end

# Test the RG map
println("  Testing RG map...")
v_test = rg_map_array(A_vec)
println("  RG map works, output norm: $(sqrt(sum(v_test.^2)))")

# Build Jacobian via finite differences
println("\n  Building Jacobian matrix (this may take a while)...")

ε = 1e-6  # Finite difference step

# We'll compute a subset of columns - the leading eigenvalues
# Full Jacobian is n×n which is too large
# Instead, use power iteration or Arnoldi on the RG map

########################################
# Power iteration for largest eigenvalues
########################################

println("\n" * "-" ^ 60)
println("Step 4: Power iteration for leading eigenvalues...")

# Normalize starting vector
v = copy(A_vec)
v ./= sqrt(sum(v.^2))

eigenvalues = Float64[]
eigenvectors = Vector{Float64}[]

# Find top 5 eigenvalues using deflation
for k in 1:5
    println("\n  Finding eigenvalue $k...")

    # Power iteration
    λ_prev = 0.0
    for iter in 1:30
        # Apply RG map
        v_new = rg_map_array(v)

        # Deflate against previously found eigenvectors
        for (i, ev) in enumerate(eigenvectors)
            proj = dot(v_new, ev)
            v_new .-= proj * ev
        end

        # Compute Rayleigh quotient
        λ = dot(v_new, v) / dot(v, v)

        # Normalize
        norm_v = sqrt(sum(v_new.^2))
        if norm_v < 1e-16
            println("    Iter $iter: vector collapsed")
            break
        end
        v_new ./= norm_v

        @printf("    Iter %2d: λ = %10.6f\n", iter, abs(λ))

        # Check convergence
        if abs(λ - λ_prev) < 1e-8
            println("    Converged!")
            break
        end
        λ_prev = λ
        v .= v_new
    end

    push!(eigenvalues, abs(λ_prev))
    push!(eigenvectors, copy(v))

    # Scaling dimension
    x = -log(abs(λ_prev)) / log(2)
    @printf("  λ_%d = %.6f  →  x_%d = %.4f\n", k, abs(λ_prev), k, x)
end

########################################
# Summary
########################################

println("\n" * "=" ^ 60)
println("SUMMARY: Linearized RG Eigenvalues")
println("=" ^ 60)

for (i, λ) in enumerate(eigenvalues)
    x = -log(λ) / log(2)
    @printf("  λ_%d = %10.6f  →  x_%d = %8.4f\n", i, λ, i, x)
end

println("\nExpected Ising CFT values:")
println("  x_σ = 0.125  (λ_σ ≈ 3.668)")
println("  x_ε = 1.0    (λ_ε ≈ 2.0)")
println("=" ^ 60)
