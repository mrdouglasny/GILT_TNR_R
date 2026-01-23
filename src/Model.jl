# MODULE: ekrgilttrnr/src/Model.jl
# USAGE: include("Model.jl") or using .PottsModel
# DESCRIPTION: Unified q-state Potts model definition for both MC and TRG.
#              Single source of truth for model physics.
#
# PROVENANCE: Based on GiltTNR2D_Potts.py tensor construction.
# What's new: Pure Julia implementation with MC support.
#
# q-STATE POTTS MODEL:
#   Spins: σ ∈ {0, 1, ..., q-1}
#   Interaction: -β δ(σ_i, σ_j)
#   Critical point: β_c = ln(1 + √q)
#
# 3-STATE POTTS CFT (c=4/5):
#   x_identity = 0
#   x_spin (σ) = 2/15 ≈ 0.1333
#   x_energy (ε) = 4/5 = 0.8

module PottsModel

using LinearAlgebra

export Potts, critical_beta
export build_tensor, boltzmann_weight
export random_config, total_action, delta_action, delta_action_with_proposal, bond_action
export partition_function_exact

# --- Model Definition ---

"""
    Potts(q, beta)

q-state Potts model at inverse temperature β.

Action: S = -β Σ_{⟨i,j⟩} δ(σ_i, σ_j)
where δ is Kronecker delta.

Critical point: β_c = ln(1 + √q)
"""
struct Potts
    q::Int
    beta::Float64
end

# Convenience constructor at critical point
Potts(q::Int) = Potts(q, critical_beta(q))

"""
    critical_beta(q)

Critical inverse temperature for q-state Potts model.
β_c = ln(1 + √q)
"""
critical_beta(q::Int) = log(1 + sqrt(q))

# --- Bond Action ---

"""
    bond_action(m::Potts, σ1, σ2)

Bond action: -β δ(σ1, σ2)
Returns -β if spins are equal, 0 otherwise.
"""
bond_action(m::Potts, σ1::Integer, σ2::Integer) = σ1 == σ2 ? -m.beta : 0.0

"""
    boltzmann_weight(m::Potts, σ1, σ2)

Boltzmann weight: exp(-bond_action) = exp(β) if σ1==σ2, else 1
"""
boltzmann_weight(m::Potts, σ1::Integer, σ2::Integer) = σ1 == σ2 ? exp(m.beta) : 1.0

# --- For TRG: Tensor Construction ---

"""
    build_tensor(m::Potts)

Build the rank-4 TRG tensor T[l,r,u,d] for square lattice.

Uses SITE construction (same as 2dising/src/Model.jl):
    T[l,r,u,d] = Σ_σ A[σ,l] A[σ,r] A[σ,u] A[σ,d]

where A is the half-weight matrix from eigendecomposition of W.

Boltzmann weight: W[i,j] = exp(β δ_{i,j})
"""
function build_tensor(m::Potts)
    q = m.q
    β = m.beta

    # Boltzmann weight matrix W[i,j] = exp(β) if i==j else 1
    W = ones(q, q)
    for i in 1:q
        W[i, i] = exp(β)
    end

    # Eigendecomposition: W = U Λ Uᵀ
    # W is symmetric positive definite
    evals, evecs = eigen(Symmetric(W))

    # Half-weight matrix: A = U √Λ
    # A[σ, α] where σ ∈ 1:q is spin state, α ∈ 1:q is bond index
    A = evecs * Diagonal(sqrt.(evals))

    # Site tensor: T[l,r,u,d] = Σ_σ A[σ,l] A[σ,r] A[σ,u] A[σ,d]
    T = zeros(q, q, q, q)
    for l in 1:q, r in 1:q, u in 1:q, d in 1:q
        for σ in 1:q
            T[l, r, u, d] += A[σ, l] * A[σ, r] * A[σ, u] * A[σ, d]
        end
    end

    return T
end

# --- For MC: Configuration and Energy ---

"""
    random_config(m::Potts, M::Int, N::Int)

Generate random spin configuration on M×N lattice.
Spins are integers in 1:q (Julia 1-indexed).
"""
function random_config(m::Potts, M::Int, N::Int)
    return rand(1:m.q, M, N)
end

"""
    total_action(m::Potts, config)

Compute total action S = -β Σ_{⟨i,j⟩} δ(σ_i, σ_j).
Uses periodic boundary conditions.
"""
function total_action(m::Potts, config::Matrix{<:Integer})
    M, N = size(config)
    S = 0.0

    for i in 1:M, j in 1:N
        σ = config[i, j]

        # Only count each bond once: right and up neighbors
        σ_r = config[mod1(i+1, M), j]
        σ_u = config[i, mod1(j+1, N)]

        S += bond_action(m, σ, σ_r) + bond_action(m, σ, σ_u)
    end

    return S
end

"""
    delta_action(m::Potts, config, i, j, σ_new)

Compute action change if spin at (i,j) changes to σ_new.
ΔS = S_new - S_old

Note: Self-loops (when M=1 or N=1) are skipped since changing a spin
changes both ends of a self-loop bond, giving zero net change.
"""
function delta_action(m::Potts, config::Matrix{<:Integer},
                      i::Int, j::Int, σ_new::Integer)
    M, N = size(config)
    σ_old = config[i, j]

    if σ_old == σ_new
        return 0.0
    end

    # Sum over 4 neighbors (skip self-loops)
    neighbor_coords = [
        (mod1(i+1, M), j),  # right
        (mod1(i-1, M), j),  # left
        (i, mod1(j+1, N)),  # up
        (i, mod1(j-1, N))   # down
    ]

    ΔS = 0.0
    for (ni, nj) in neighbor_coords
        if (ni, nj) != (i, j)  # Skip self-loops
            σ_n = config[ni, nj]
            ΔS += bond_action(m, σ_new, σ_n) - bond_action(m, σ_old, σ_n)
        end
    end

    return ΔS
end

"""
    delta_action(m::Potts, config, i, j)

Compute action change for Metropolis proposal: pick random new spin.
Returns (ΔS, σ_new).
"""
function delta_action_with_proposal(m::Potts, config::Matrix{<:Integer}, i::Int, j::Int)
    σ_old = config[i, j]
    # Pick a different spin uniformly
    σ_new = rand(1:m.q)
    while σ_new == σ_old && m.q > 1
        σ_new = rand(1:m.q)
    end
    return delta_action(m, config, i, j, σ_new), σ_new
end

# --- For Exact Enumeration ---

"""
    partition_function_exact(m::Potts, M::Int, N::Int)

Compute exact partition function by summing over all q^(M*N) configurations.
Only feasible for small lattices (M*N ≲ 12 for q=3).
"""
function partition_function_exact(m::Potts, M::Int, N::Int)
    q = m.q
    V = M * N

    if V > 16
        error("Exact enumeration too expensive for $M×$N lattice with q=$q states")
    end

    Z = 0.0

    # Iterate over all q^V configurations
    for config_idx in 0:(q^V - 1)
        # Decode configuration
        config = zeros(Int, M, N)
        idx = config_idx
        for j in 1:N, i in 1:M
            config[i, j] = (idx % q) + 1  # 1-indexed
            idx ÷= q
        end

        S = total_action(m, config)
        Z += exp(-S)
    end

    return Z
end

end # module
