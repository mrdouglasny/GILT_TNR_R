#!/usr/bin/env julia
# MODULE: compare_jacobian_vs_transfer.jl
# USAGE: julia --project=ekrgilttrnr scripts/compare_jacobian_vs_transfer.jl
# INPUTS: None
# OUTPUTS: Comparison of Jacobian vs transfer matrix eigenvalues
# DESCRIPTION: Debug Jacobian computation by comparing two methods
#
# BASED ON: run_newton_potts3.jl, run_potts_transfer_matrix.jl
# NEW IN THIS MODULE: Side-by-side comparison for debugging

using Dates
using LinearAlgebra

println("="^70)
println("Jacobian vs Transfer Matrix Comparison")
println("="^70)
println("Started: $(now())")

const PROJECT_ROOT = dirname(@__DIR__)

# CFT predictions
const X_SIGMA = 2/15
const X_EPSILON = 4/5

# Load modules
println("\nLoading modules...")
include(joinpath(PROJECT_ROOT, "src", "EchelonFormGFN.jl"))
include(joinpath(PROJECT_ROOT, "src", "ZNTensor.jl"))
include(joinpath(PROJECT_ROOT, "src", "ContinuousGauge.jl"))
include(joinpath(PROJECT_ROOT, "src", "DiscreteGaugeZN.jl"))
include(joinpath(PROJECT_ROOT, "src", "ModelProvider.jl"))
include(joinpath(PROJECT_ROOT, "src", "NewtonZN.jl"))

using PyCall

# Setup Python path
pushfirst!(pyimport("sys")."path", joinpath(PROJECT_ROOT, "GiltTNR"))
gilttnr = pyimport("GiltTNR2D")
gilttnr_potts = pyimport("GiltTNR2D_Potts")
tensors_mod = pyimport("tensors")

# Parameters - small chi for quick debugging
chi = 16
gilt_eps = 3e-5
n_warmup = 10

println("\n--- Parameters ---")
println("chi = $chi")
println("gilt_eps = $gilt_eps")
println("n_warmup = $n_warmup")

# Build initial tensor
β_c = log(1 + sqrt(3))
println("\nCritical β = $(round(β_c, digits=6))")

# Build Potts tensor
q = 3
W = ones(q, q)
for i in 1:q
    W[i,i] = exp(β_c)
end
T_spin = zeros(q, q, q, q)
for a in 1:q, b in 1:q, c in 1:q, d in 1:q
    T_spin[a,b,c,d] = W[a,b] * W[b,c] * W[c,d] * W[d,a]
end

A = pycall(tensors_mod.Tensor.from_ndarray, PyObject, T_spin)

# Warmup parameters
pars = Dict{String, Any}(
    "gilt_eps" => gilt_eps,
    "cg_chis" => collect(1:chi),
    "cg_eps" => 1e-10,
    "verbosity" => 0,
)

# Run warmup
println("\n--- Warmup Phase ---")
for step in 1:n_warmup
    result = pycall(gilttnr.gilttnr_step, PyObject, A, 0.0, pars)
    global A = get(result, PyObject, 0)

    # Get scaling dimensions
    scaldims = pycall(gilttnr_potts.get_scaldims_potts, PyObject, A)
    scaldims_arr = convert(Array{Float64}, scaldims)
    x_sigma = length(scaldims_arr) >= 2 ? scaldims_arr[2] : NaN
    x_epsilon = length(scaldims_arr) >= 4 ? scaldims_arr[4] : NaN

    shape = size(A.to_ndarray())
    println("Step $step: shape=$shape, x_σ=$(round(x_sigma, digits=4)), x_ε=$(round(x_epsilon, digits=4))")
end

println("\n--- Transfer Matrix Eigenvalues ---")
# Build row transfer matrix
arr = convert(Array{Float64}, A.to_ndarray())
d1, d2, d3, d4 = size(arr)

# T[l1,l2; r1,r2] = sum_{u,v} A[l1,u,r1,v] * A[l2,v,r2,u]
# Using einsum: T_ijkl = A_aibc * A_jcka
T_transfer = zeros(d1*d2, d3*d4)
for l1 in 1:d1, l2 in 1:d2, r1 in 1:d3, r2 in 1:d4
    row = (l1-1)*d2 + l2
    col = (r1-1)*d4 + r2
    for u in 1:d2, v in 1:d4
        if u <= d2 && v <= d4 && u <= d3 && v <= d1
            # Note: need to be careful with index mapping
        end
    end
end

# Simpler: use Python ncon
ncon_py = pyimport("ncon").ncon
transmat = ncon_py([A, A], [[-1, 3, -3, 4], [-2, 4, -4, 3]])
es = transmat.eig([0,1], [2,3], hermitian=false)[1]
es_arr = convert(Array, es.to_ndarray())
es_abs = sort(abs.(es_arr), rev=true)

println("Transfer matrix eigenvalues (largest 8):")
for (i, λ) in enumerate(es_abs[1:min(8, length(es_abs))])
    x = -log(λ / es_abs[1]) / π
    println("  λ_$i = $(round(λ, digits=6)), x = $(round(x, digits=4))")
end

# Compute expected Jacobian eigenvalues from scaling dimensions
println("\n--- Expected Jacobian Eigenvalues ---")
println("From scaling dimensions x, Jacobian eigenvalue λ_J = 2^(2x)")
println("  x_σ = $X_SIGMA → λ_J = $(round(2^(2*X_SIGMA), digits=4))")
println("  x_ε = $X_EPSILON → λ_J = $(round(2^(2*X_EPSILON), digits=4))")

# Simple test: apply RG step and measure change
println("\n--- RG Step Test ---")
println("Testing if RG step is approximately identity near fixed point...")

# Store original tensor
A_arr_orig = copy(convert(Array{Float64}, A.to_ndarray()))
norm_orig = norm(A_arr_orig)
A_arr_orig ./= norm_orig

# Apply one TRG-only step (gilt_eps=0)
pars_trg = Dict{String, Any}(
    "gilt_eps" => 0.0,
    "cg_chis" => [chi],
    "cg_eps" => 1e-10,
    "verbosity" => 0,
)

result = pycall(gilttnr.gilttnr_step, PyObject, A, 0.0, pars_trg)
A_new = get(result, PyObject, 0)

# Fix gauge
A_new, _, _, _, _ = fix_continuous_gauge(A_new)
A_new_arr = convert(Array{Float64}, A_new.to_ndarray())

println("Original shape: $(size(A_arr_orig))")
println("After RG shape: $(size(A_new_arr))")

if size(A_arr_orig) == size(A_new_arr)
    A_new_arr ./= norm(A_new_arr)
    residual = norm(A_arr_orig - A_new_arr)
    dot_product = abs(dot(vec(A_arr_orig), vec(A_new_arr)))
    println("||A - R(A)|| = $residual")
    println("|⟨A, R(A)⟩| = $dot_product")
    println("If close to fixed point: residual should be small, dot product ≈ 1")
else
    println("⚠️ Shape mismatch - cannot compare directly")
end

println("\n--- Analysis ---")
println("""
Key insight:
- Transfer matrix eigenvalues give scaling dimensions via x = -log(λ_T / λ_0) / π
- Jacobian eigenvalues should give: λ_J = 2^(2x)

For 3-state Potts:
- x_σ = 2/15 → λ_J = $(round(2^(2*X_SIGMA), digits=4))
- x_ε = 4/5  → λ_J = $(round(2^(2*X_EPSILON), digits=4))

The spurious large eigenvalues (|λ| > 100) in current Jacobian computation
likely come from:
1. Non-fixed-point tensor (warmup not converged)
2. Gauge fixing instabilities for plain tensors
3. Numerical differentiation errors

The fact that transfer matrix gives correct x_σ, x_ε but Jacobian doesn't
suggests the issue is in the Jacobian computation method, not the tensor.
""")

println("\nCompleted: $(now())")
