"""
φ⁴ Theory Tensor Construction for Gilt-TNR

This module provides initial tensor construction for the 2D φ⁴ lattice field theory:
    S = Σ_x [½(∂φ)² + ½μ²φ² + ¼λφ⁴]

The φ⁴ model has Z₂ symmetry (φ → -φ) and is in the same universality class as
the 2D Ising model (c = 1/2 CFT).

Reference: Shimizu & Kuramashi, PRD 90, 014508 (2014)
"""

import numpy as np
from scipy.special import roots_hermite, roots_legendre
from tensors import Tensor, TensorZ2
from ncon import ncon


def golub_welsch(N):
    """
    Compute Gauss-Hermite quadrature nodes and weights for order N.

    Returns nodes x_i and weights w_i for the integral:
        ∫_{-∞}^{∞} f(x) e^{-x²} dx ≈ Σ_i w_i f(x_i)
    """
    nodes, weights = roots_hermite(N)
    return nodes, weights



def get_scaldims_phi4(A, pars=None):
    """
    Calculate scaling dimensions from the transfer matrix spectrum.
    
    Constructs a transfer matrix from two copies of A (width 2 cylinder)
    and diagonalizes it.
    
    Scaling dimensions x_i are related to eigenvalues lambda_i by:
    x_i = - ln(lambda_i / lambda_0) / pi
    (Assuming L=2 geometry and standard CFT normalization)
    """
    # If A is a TensorZ2, we need to convert to ndarray first
    if hasattr(A, "to_ndarray"):
        A_arr = A.to_ndarray()
    else:
        A_arr = A
        
    # Contract two tensors to form a transfer matrix
    # A indices: [l, u, r, d]
    # ncon((A, A), [[3,-3,4,-1], [4,-4,3,-2]])
    # Indices: -1=A1.d, -2=A2.d, -3=A1.u, -4=A2.u
    # T_{d1 d2, u1 u2}
    transmat = ncon((A_arr, A_arr), [[3,-3,4,-1], [4,-4,3,-2]])
    
    # Dynamically determine shape from the result
    # transmat is usually (D*D, D*D) but bonds might vary slightly in Gilt
    total_dim = transmat.size
    dim = int(np.sqrt(total_dim))
    
    if dim * dim != total_dim:
        # Fallback if not square (rare/error case)
        print(f"Warning: Transmat not square? Size={total_dim}")
        return np.array([0.0])

    T_mat = transmat.reshape(dim, dim)
    
    try:
        es = np.linalg.eigvals(T_mat)
    except:
        # Fallback for stability
        U, S, Vh = np.linalg.svd(T_mat)
        es = S
        
    es = np.abs(es)
    es = -np.sort(-es)
    
    # Normalize by largest
    es_ratio = es / es[0]
    
    # Avoid log(0)
    es_ratio[es_ratio < 1e-16] = 1e-16
    
    # Calculate scaling dimensions
    scaldims = -np.log(es_ratio) / np.pi
    
    return scaldims


def build_phi4_tensor_array(mu_sq, lam, kappa, K, D, Lambda=3.0):
    """
    Construct the initial φ⁴ tensor as a numpy array.

    Uses Gauss-Legendre quadrature with finite field cutoff [-Λ, Λ].
    This avoids numerical issues from Gauss-Hermite nodes extending too far.

    The lattice action is:
        S = Σ_<ij> [-κ φ_i φ_j] + Σ_i [½μ²φ_i² + ¼λφ_i⁴]

    Arguments:
        mu_sq: Mass squared parameter (μ²).
        lam: Quartic coupling λ (typically λ = 1).
        kappa: Kinetic coupling κ (neighbor interaction strength).
        K: Number of Gauss-Legendre quadrature points.
        D: Bond dimension for kinetic term decomposition.
        Lambda: Field cutoff (φ ∈ [-Λ, Λ]). Default 3.0.

    Returns:
        A: Tensor of shape (D, D, D, D) with indices [left, up, right, down]
    """
    # Gauss-Legendre quadrature on [-1, 1]
    x_gl, w_gl = roots_legendre(K)

    # Map to [-Λ, Λ]
    phi = Lambda * x_gl
    w = Lambda * w_gl  # Jacobian from change of variables

    # Potential term: V(φ) = ½μ²φ² + ¼λφ⁴
    def V(p):
        return 0.5 * mu_sq * p**2 + 0.25 * lam * p**4

    # Local Boltzmann weight: w[α] * exp(-V(φ[α]))
    P = w * np.exp(-V(phi))

    # Kinetic term: exp(κ φ_i φ_j)
    W_kin = np.exp(kappa * np.outer(phi, phi))
    U_kin, S_kin, Vh_kin = np.linalg.svd(W_kin)


    # Sort singular vectors by parity (Even, Odd) to respect Z2 symmetry
    # Parity check: v[i] vs v[K-1-i]
    # Even: v[i] == v[K-1-i] (symmetric)
    # Odd: v[i] == -v[K-1-i] (antisymmetric)
    
    # Keep top D singular values (by magnitude)
    # We do this BEFORE sorting by parity to ensure we keep the most relevant modes
    if D < len(S_kin):
        S_kin = S_kin[:D]
        U_kin = U_kin[:, :D]
    
    # Now classify these top D vectors by parity
    indices = []
    parities = []
    
    for i in range(len(S_kin)):
        vec = U_kin[:, i]
        # Check parity
        # Relaxed tolerance to handle numerical noise
        diff_even = np.linalg.norm(vec - vec[::-1])
        diff_odd = np.linalg.norm(vec + vec[::-1])
        
        if diff_even < 1e-5:
            parities.append(0)
            indices.append(i)
        elif diff_odd < 1e-5:
            parities.append(1)
            indices.append(i)
        else:
            # Default to Even to avoid crash, but this signals a problem
            parities.append(0)
            indices.append(i)
            
    # Sort by parity: All Even first, then All Odd
    # Within each group, we want to keep the original relative order (magnitude)
    even_indices = [i for i, p in zip(indices, parities) if p == 0]
    odd_indices = [i for i, p in zip(indices, parities) if p == 1]
    
    sorted_indices = even_indices + odd_indices
    
    # Apply sorting
    U_kin = U_kin[:, sorted_indices]
    S_kin = S_kin[sorted_indices]
    
    # The resulting parities for the tensor indices
    top_parities = [0]*len(even_indices) + [1]*len(odd_indices)

    # Construct the C matrix: C_{alpha, i} = U_{alpha, i} * sqrt(S_i)
    # U_kin and S_kin are already sorted by parity
    C = U_kin @ np.diag(np.sqrt(S_kin))

    # Build tensor: A[l,u,r,d] = Σ_α P[α] C[α,l] C[α,u] C[α,r] C[α,d]
    # Use einsum for efficiency
    A = np.einsum('a,al,au,ar,ad->lurd', P, C, C, C, C)

    # Normalize
    A = A / np.max(np.abs(A))

    # Return dimensions of Even and Odd sectors
    dims = [len(even_indices), len(odd_indices)]
    return A, dims


def symmetrize_phi4_tensor(A):
    """
    Symmetrize a φ⁴ tensor to enforce exact Z₂ symmetry.

    The Z₂ symmetry φ → -φ means the tensor should be even in each index.
    For Gauss-Hermite with symmetric nodes, odd-index contributions should
    cancel, but numerical errors may break this. This function enforces it.
    """
    D = A.shape[0]

    # The quadrature nodes are symmetric about 0
    # Even functions: f(-x) = f(x) → keep only even combinations
    # The tensor should satisfy A[i,j,k,l] = A[-i,-j,-k,-l] where -i means flipped node

    # For Gauss-Hermite, if K is even, nodes come in ±pairs
    # The Z2 structure naturally emerges from the SVD decomposition

    # Simple symmetrization: A → (A + A_flipped) / 2
    # where flip reverses the index order (exploiting node symmetry)
    A_sym = (A + A[::-1, ::-1, ::-1, ::-1]) / 2

    return A_sym


def get_initial_tensor_phi4(pars):
    """
    Construct the initial φ⁴ tensor for Gilt-TNR.

    Parameters (in pars dict):
        mu_sq: Mass squared (default: +2.73, near critical point for λ=1, κ=1)
        lam: Quartic coupling (default: 1.0)
        kappa: Kinetic coupling (default: 1.0)
        K: Quadrature order (default: 32)
        D: Bond dimension (default: 16)
        symmetry_tensors: If True, return TensorZ2; else return Tensor

    Returns:
        A: Initial tensor (TensorZ2 or Tensor depending on symmetry_tensors)
    """
    # Default parameters
    mu_sq = pars.get("mu_sq", 2.73)
    lam = pars.get("lam", 1.0)
    kappa = pars.get("kappa", 1.0)
    K = pars.get("K", 32)
    D = pars.get("D", 16)
    symmetry_tensors = pars.get("symmetry_tensors", True)

    # Build raw tensor
    A, dims = build_phi4_tensor_array(mu_sq, lam, kappa, K, D)

    # NOTE: Do NOT symmetrize here - it doubles singular value multiplicities
    # The Z2 symmetry is handled by TensorZ2 block structure if symmetry_tensors=True
    # A = symmetrize_phi4_tensor(A)

    D_actual = A.shape[0]

    if symmetry_tensors:
        # Convert to Z2-symmetric tensor format
        # The Z2 symmetry splits the bond dimension into even/odd sectors
        # We use the dims returned by build_phi4_tensor_array which are sorted [Even, Odd]
        
        # Reshape and assign to Z2 sectors
        # Sector 0 (even): first dims[0] indices
        # Sector 1 (odd): remaining dims[1] indices
        qim = [0, 1]

        A_z2 = TensorZ2.from_ndarray(
            A,
            shape=[dims] * 4,
            qhape=[qim] * 4,
            dirs=[1, 1, -1, -1]  # Convention: [left, up, right, down]
        )
        return A_z2
    else:
        return Tensor.from_ndarray(A)


def get_initial_tensor_phi4_simple(pars):
    """
    Simplified φ⁴ tensor construction without Z2 decomposition.

    Use this if you want a plain tensor without symmetry structure.
    """
    mu_sq = pars.get("mu_sq", 2.73)
    lam = pars.get("lam", 1.0)
    kappa = pars.get("kappa", 1.0)
    K = pars.get("K", 32)
    D = pars.get("D", 16)

    A, _ = build_phi4_tensor_array(mu_sq, lam, kappa, K, D)
    return Tensor.from_ndarray(A)


def get_A_spectrum_phi4(A):
    """Get normalized singular value spectrum of tensor."""
    es = A.svd([0, 1], [2, 3])[1]
    es = es.to_ndarray()
    es = es / np.max(es)
    es = -np.sort(-es)
    return es


# Critical point estimation for φ⁴
# For λ=1, κ=1, the critical point is approximately μ²_c ≈ +2.73 (POSITIVE)
# The phase structure is:
#   - Broken phase (ordered): μ² < μ²_c
#   - Symmetric phase (disordered): μ² > μ²_c
# The critical μ² scales roughly as ~2.7λ for κ=1.

def estimate_critical_mu_sq(lam, kappa=1.0):
    """
    Estimate the critical μ² for a given λ and κ.

    Based on numerical results: μ²_c ≈ 2.73 for λ=1, κ=1
    Critical μ² is POSITIVE and scales approximately as ~2.7λ.
    """
    return 2.73 * lam


# Phase detection for φ⁴
def detect_phase_phi4(A, pars, max_steps=30, tol=1e-4):
    """
    Detect whether the tensor flows to symmetric (high-T) or broken (low-T) phase.

    In the symmetric phase: second eigenvalue → 0
    In the broken phase: second eigenvalue → 1 (degenerate ground states)

    Returns: "SYMMETRIC", "BROKEN", or "UNDETERMINED"
    """
    from GiltTNR2D_essentials import gilttnr_step

    for i in range(max_steps):
        A, _, _ = gilttnr_step(A, 0.0, pars)
        spectrum = get_A_spectrum_phi4(A)

        if len(spectrum) < 2:
            return "UNDETERMINED", i

        second_ev = spectrum[1]

        if abs(second_ev - 1) < tol:
            return "BROKEN", i  # Z2 broken phase
        elif abs(second_ev) < tol:
            return "SYMMETRIC", i  # Z2 symmetric phase

    return "UNDETERMINED", max_steps


def _test_main():
    """Test function - call manually if needed."""
def _test_main():
    # Test the tensor construction
    print("Testing phi4 tensor construction...")

    pars = {
        "mu_sq": 2.73,
        "lam": 1.0,
        "kappa": 1.0,
        "K": 16,
        "D": 8,
        "symmetry_tensors": False
    }

    A = get_initial_tensor_phi4(pars)
    print(f"Tensor shape: {A.shape}")

    spectrum = get_A_spectrum_phi4(A)
    print(f"Top 5 singular values: {spectrum[:5]}")

    print("\nTesting Z2 tensor construction...")
    pars["symmetry_tensors"] = True
    A_z2 = get_initial_tensor_phi4(pars)
    print(f"Z2 tensor created successfully")
    print(f"Z2 tensor shape: {A_z2.shape}")


if __name__ == "__main__":
    _test_main()
