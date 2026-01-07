# MODULE: GiltTNR2D_Potts.py
# USAGE: from GiltTNR2D_Potts import get_initial_tensor_potts, get_scaldims_potts
# DESCRIPTION: Initial tensor and scaling dimension extraction for q-state Potts model.
#
# BASED ON: GiltTNR2D_Ising_benchmarks.py (same repo)
# NEW IN THIS SCRIPT: Adapts tensor construction for q-state Potts model.
#                     Uses plain Tensor (no Z_q symmetry implementation yet).
#
# 3-STATE POTTS CFT (c=4/5):
#   Primary operators and scaling dimensions:
#   - Identity: x = 0
#   - Spin (σ): x = 2/15 ≈ 0.1333
#   - Energy (ε): x = 4/5 = 0.8
#   - Subleading: x = 4/3 ≈ 1.333, x = 7/5 = 1.4, ...
#
# Critical point: β_c = ln(1 + √3) ≈ 1.0051

import numpy as np
from ncon import ncon
from tensors import Tensor, TensorZ3
import logging

def get_initial_tensor_potts(pars):
    """
    Construct initial TRG tensor for q-state Potts model.

    Parameters:
    -----------
    pars : dict
        Must contain:
        - 'q': number of states (default 3)
        - 'beta': inverse temperature (default: critical point)
        - 'symmetry_tensors': if True, would use Z_q tensors (not implemented, falls back to plain)

    Returns:
    --------
    A_0 : Tensor
        Initial 4-leg tensor for TRG, shape (q, q, q, q)

    Theory:
    -------
    Boltzmann weight: W[σ,σ'] = exp(β) if σ=σ' else 1

    VERTEX construction (same as Ising):
        T[a,b,c,d] = W[a,b] W[b,c] W[c,d] W[d,a]

    This vertex construction has exact Z_q symmetry under global rotation,
    which is required for the charge basis transformation to work correctly.
    """
    q = pars.get('q', 3)

    # Default to critical point
    if 'beta' not in pars:
        if q == 3:
            beta = np.log(1 + np.sqrt(3))  # 3-state Potts critical point
        elif q == 2:
            beta = np.log(1 + np.sqrt(2)) / 2  # Ising critical point
        else:
            # General q-state Potts critical point: exp(β_c) = 1 + √q
            beta = np.log(1 + np.sqrt(q))
    else:
        beta = pars['beta']

    # Construct Boltzmann weight matrix
    # W[i,j] = exp(β δ_{i,j}) = exp(β) if i==j else 1
    W = np.ones((q, q))
    np.fill_diagonal(W, np.exp(beta))

    # VERTEX construction: T[a,b,c,d] = W[a,b] W[b,c] W[c,d] W[d,a]
    # This has exact Z_q symmetry (invariant under σ → σ+1 mod q)
    T = np.einsum('ab,bc,cd,da->abcd', W, W, W, W)

    # Convert to Tensor object
    try:
        A_0 = Tensor.from_ndarray(T)
    except Exception as e:
        # If Tensor import failed, return raw array
        print(f"Warning: Could not create Tensor object: {e}")
        A_0 = T

    return A_0


def get_initial_tensor_potts_relT(pars):
    """
    Construct initial tensor with relative temperature parameterization.

    Parameters:
    -----------
    pars : dict
        - 'q': number of states (default 3)
        - 'relT': relative temperature, where relT=1 is critical point
        - 'symmetry_tensors': if True, use TensorZ3 for q=3

    Returns:
    --------
    A_0 : Tensor or TensorZ3
    """
    q = pars.get('q', 3)
    relT = pars.get('relT', 1.0)
    use_symmetry = pars.get('symmetry_tensors', False)

    # Critical point
    beta_c = np.log(1 + np.sqrt(q))

    # Actual beta (relT > 1 means higher T, lower beta)
    beta = beta_c / relT

    pars_with_beta = dict(pars)
    pars_with_beta['beta'] = beta
    pars_with_beta['symmetry_tensors'] = use_symmetry

    if use_symmetry and q == 3:
        return get_initial_tensor_potts_z3(pars_with_beta)
    else:
        return get_initial_tensor_potts(pars_with_beta)


def get_initial_tensor_potts_z3(pars):
    """
    Construct Z₃-symmetric initial tensor for 3-state Potts model.

    Uses the VERTEX construction (same as Ising) which has exact Z₃ symmetry,
    then transforms to the Z₃ charge basis using the discrete Fourier transform.

    Parameters:
    -----------
    pars : dict
        - 'beta': inverse temperature

    Returns:
    --------
    A_0 : TensorZ3
        Z₃-symmetric tensor with charge=0

    Theory:
    -------
    1. Build vertex tensor: T_spin[a,b,c,d] = W[a,b] W[b,c] W[c,d] W[d,a]
    2. Transform to charge basis using Z₃ DFT: T_charge = F⊗F⊗F†⊗F† T_spin
       where F[k,σ] = ω^{kσ}/√3 with ω = exp(2πi/3)
    3. The resulting tensor is block-diagonal in charge sectors
    """
    beta = pars.get('beta', np.log(1 + np.sqrt(3)))
    q = 3

    # Construct Boltzmann weight matrix
    # W[i,j] = exp(β δ_{i,j}) = exp(β) if i==j else 1
    W = np.ones((q, q))
    np.fill_diagonal(W, np.exp(beta))

    # VERTEX construction: T[a,b,c,d] = W[a,b] W[b,c] W[c,d] W[d,a]
    # This has exact Z₃ symmetry (invariant under σ → σ+1 mod 3)
    T_spin = np.einsum('ab,bc,cd,da->abcd', W, W, W, W)

    # Z₃ DFT matrix (analogous to Hadamard for Z₂)
    # F[k,σ] = ω^{kσ}/√3 where ω = exp(2πi/3)
    omega = np.exp(2j * np.pi / 3)
    F = np.array([
        [1, 1, 1],
        [1, omega, omega**2],
        [1, omega**2, omega]
    ]) / np.sqrt(3)
    F_dg = F.T.conj()

    # Transform to charge basis: T' = F⊗F⊗F†⊗F† T
    # For indices with dir=+1 (incoming): apply F
    # For indices with dir=-1 (outgoing): apply F†
    T_charge = ncon(
        (T_spin, F, F, F_dg, F_dg),
        ([1, 2, 3, 4], [-1, 1], [-2, 2], [3, -3], [4, -4])
    )

    # Clean up small imaginary parts (should be real due to symmetry)
    T_charge[np.abs(T_charge) < 1e-10] = 0

    # Create TensorZ3 with shape and qhape for 3-state system
    # Each index has 3 sectors of dimension 1 (one for each charge)
    dim = [1, 1, 1]  # Each charge sector has dimension 1
    qim = [0, 1, 2]  # Charges 0, 1, 2

    # dirs: +1 for incoming (domain), -1 for outgoing (codomain)
    # Standard convention: [left, up, right, down] = [1, 1, -1, -1]
    dirs = [1, 1, -1, -1]

    try:
        A_0 = TensorZ3.from_ndarray(
            T_charge,
            shape=[dim]*4,
            qhape=[qim]*4,
            dirs=dirs,
            charge=0,  # Total charge is 0 (Z₃ invariant)
            invar=True
        )
    except Exception as e:
        logging.warning(f"TensorZ3 creation failed: {e}. Falling back to plain Tensor.")
        A_0 = Tensor.from_ndarray(T_charge)

    return A_0


def get_scaldims_potts(A, pars=None):
    """
    Extract scaling dimensions from transfer matrix eigenvalues.

    Parameters:
    -----------
    A : Tensor
        Coarse-grained tensor after TRG steps
    pars : dict, optional
        Not used, kept for API compatibility

    Returns:
    --------
    scaldims : ndarray
        Scaling dimensions extracted from transfer matrix

    Theory:
    -------
    Transfer matrix: T_row[l,r] = Σ_u A[l,r,u,u]
    Eigenvalues λ_n give scaling dimensions via:
        x_n = -log(|λ_n/λ_0|) / π
    where λ_0 is the largest eigenvalue.
    """
    logging.info("Diagonalizing the transfer matrix for Potts.")

    # Build row-to-row transfer matrix
    # T[l1,l2; r1,r2] = A[l1,r1,u,v] * A[l2,r2,v,u]
    transmat = ncon((A, A), [[3,-101,4,-1], [4,-102,3,-2]])

    # Diagonalize
    es = transmat.eig([0,1], [2,3], hermitian=False)[0]
    es = es.to_ndarray()
    es = np.abs(es)
    es = -np.sort(-es)  # Sort descending

    # Avoid log(0)
    es[es == 0] += 1e-16

    # Extract scaling dimensions
    log_es = np.log(es)
    log_es -= np.max(log_es)  # Normalize by largest
    log_es /= -np.pi

    return log_es


def get_exact_free_energy_potts(pars):
    """
    Exact free energy for q-state Potts model (only known analytically at critical point).

    For 3-state Potts at criticality:
    f_c = -β_c - (1/β_c) * [...]  (complex integral expression)

    This is a placeholder - exact expressions are complicated.
    """
    q = pars.get('q', 3)
    beta = pars.get('beta', np.log(1 + np.sqrt(q)))

    # At criticality, use known result
    # For general temperature, this would need numerical integration
    if q == 3:
        # Placeholder: return approximate value at criticality
        # The exact value involves complex integrals
        return None
    else:
        return None


# Expected CFT scaling dimensions for reference
POTTS3_CFT_DIMENSIONS = {
    'identity': 0.0,
    'spin': 2/15,           # ≈ 0.1333
    'energy': 4/5,          # = 0.8
    'spin2': 4/3,           # ≈ 1.333 (two spin operators)
    'T_stress': 2.0,        # stress tensor
}

ISING_CFT_DIMENSIONS = {
    'identity': 0.0,
    'spin': 1/8,            # = 0.125
    'energy': 1.0,
    'T_stress': 2.0,
}
