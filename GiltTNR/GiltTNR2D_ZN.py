#!/usr/bin/env python3
"""
MODULE: GiltTNR2D_ZN.py
USAGE: from GiltTNR2D_ZN import get_initial_tensor_zn, get_scaldims_zn
INPUTS: Model parameters (N, beta, model_type)
OUTPUTS: Initial Z_N symmetric tensors for TRG
DESCRIPTION: Generalized Z_N tensor construction for Potts and Clock models

BASED ON: GiltTNR2D_Potts.py (3-state Potts specific)
          GiltTNR2D.py (Ising model)
NEW IN THIS MODULE: Unified Z_N construction for arbitrary N
"""

import numpy as np
from ncon import ncon
from tensors import Tensor

################################################
# Z_N DFT Matrix
################################################

def get_dft_matrix(N):
    """
    Compute the N×N discrete Fourier transform matrix.

    F[k, s] = ω^{ks} / √N, where ω = exp(2πi/N)

    Used to transform from spin basis to charge basis.
    """
    omega = np.exp(2j * np.pi / N)
    F = np.array([[omega**(k * s) for s in range(N)] for k in range(N)]) / np.sqrt(N)
    return F


def get_dft_matrix_inverse(N):
    """
    Compute the inverse DFT matrix (F†).
    """
    return get_dft_matrix(N).conj().T


################################################
# Boltzmann Weight Matrices
################################################

def get_potts_boltzmann(N, beta):
    """
    Boltzmann weight matrix for N-state Potts model.

    W[σ, σ'] = exp(β δ_{σ,σ'}) = exp(β) if σ=σ', else 1

    Interaction: S = -β Σ_{<ij>} δ_{σ_i, σ_j}
    """
    W = np.ones((N, N))
    np.fill_diagonal(W, np.exp(beta))
    return W


def get_clock_boltzmann(N, beta):
    """
    Boltzmann weight matrix for N-state clock model.

    W[σ, σ'] = exp(β cos(2π(σ-σ')/N))

    Interaction: S = -β Σ_{<ij>} cos(2π(σ_i - σ_j)/N)
    """
    W = np.zeros((N, N))
    for i in range(N):
        for j in range(N):
            angle = 2 * np.pi * (i - j) / N
            W[i, j] = np.exp(beta * np.cos(angle))
    return W


def get_boltzmann_matrix(N, beta, model_type='potts'):
    """
    Get Boltzmann weight matrix for specified model.

    Parameters:
    - N: Number of states
    - beta: Inverse temperature
    - model_type: 'potts' or 'clock'
    """
    if model_type == 'potts':
        return get_potts_boltzmann(N, beta)
    elif model_type == 'clock':
        return get_clock_boltzmann(N, beta)
    else:
        raise ValueError(f"Unknown model type: {model_type}")


################################################
# Critical Temperature
################################################

def critical_beta(N, model_type='potts'):
    """
    Return the critical inverse temperature β_c.

    For Potts on square lattice: β_c = ln(1 + √N)
    For clock model: approximate (exact only for N ≤ 4)
    """
    if model_type == 'potts':
        return np.log(1 + np.sqrt(N))
    elif model_type == 'clock':
        if N <= 4:
            return np.log(1 + np.sqrt(N))  # Same as Potts for small N
        else:
            # BKT transition - approximate lower transition
            return 1.0  # Placeholder
    else:
        raise ValueError(f"Unknown model type: {model_type}")


################################################
# Tensor Construction
################################################

def get_initial_tensor_zn_array(pars):
    """
    Construct initial 4-leg tensor as numpy array in charge basis.

    Uses vertex construction (like Ising):
        T_spin[a,b,c,d] = W[a,b] W[b,c] W[c,d] W[d,a]

    Then transforms to charge basis via Z_N DFT:
        T_charge = F ⊗ F ⊗ F† ⊗ F† · T_spin

    Parameters dict should contain:
    - N: Number of states (default: 2 for Ising)
    - beta: Inverse temperature (default: critical)
    - model_type: 'potts' or 'clock' (default: 'potts')
    """
    N = pars.get('N', 2)
    model_type = pars.get('model_type', 'potts')
    beta = pars.get('beta', critical_beta(N, model_type))

    # Get Boltzmann weight matrix
    W = get_boltzmann_matrix(N, beta, model_type)

    # Vertex construction: T[a,b,c,d] = W[a,b] W[b,c] W[c,d] W[d,a]
    T_spin = np.einsum('ab,bc,cd,da->abcd', W, W, W, W)

    # Transform to charge basis
    F = get_dft_matrix(N)
    F_dg = F.conj().T

    # T_charge[i,j,k,l] = Σ F[i,a] F[j,b] F†[c,k] F†[d,l] T_spin[a,b,c,d]
    T_charge = ncon(
        (T_spin, F, F, F_dg, F_dg),
        ([1, 2, 3, 4], [-1, 1], [-2, 2], [3, -3], [4, -4])
    )

    # Clean up small imaginary parts (should be real for real beta)
    T_charge[np.abs(T_charge) < 1e-12] = 0
    T_charge = np.real(T_charge)

    return T_charge


def get_initial_tensor_zn(pars):
    """
    Construct initial tensor as Python Tensor object.

    For use with GiltTNR coarse-graining.
    """
    T_arr = get_initial_tensor_zn_array(pars)
    return Tensor.from_ndarray(T_arr)


################################################
# Scaling Dimension Extraction
################################################

def transfer_matrix_zn(A, direction='horizontal'):
    """
    Contract tensor to form transfer matrix.

    Horizontal: T[i,j; k,l] = Σ_m A[i,m,k,*] A[j,m,l,*]  (contract vertical)
    Vertical:   T[i,j; k,l] = Σ_m A[*,i,*,k] A[*,j,*,l]  (contract horizontal)
    """
    if isinstance(A, Tensor):
        A = A.to_ndarray()

    if direction == 'horizontal':
        # Contract index 1 (vertical in), sum over indices 3 (vertical out)
        # Result: T[i,j,k,l] where (i,j) are left, (k,l) are right
        T = np.einsum('imkn,jmln->ijkl', A, A)
    else:  # vertical
        T = np.einsum('mink,mjnl->ijkl', A, A)

    # Reshape to matrix
    dim = T.shape[0] * T.shape[1]
    T_mat = T.reshape(dim, dim)

    return T_mat


def get_scaldims_zn(A, n_dims=8):
    """
    Extract scaling dimensions from transfer matrix eigenvalues.

    x_i = -log(λ_i / λ_0) / (2π)

    where λ_0 is the largest eigenvalue (identity operator).
    """
    if isinstance(A, Tensor):
        A = A.to_ndarray()

    # Get transfer matrix
    T = transfer_matrix_zn(A, direction='horizontal')

    # Compute eigenvalues
    eigenvalues = np.linalg.eigvals(T)
    eigenvalues = np.sort(np.abs(eigenvalues))[::-1]  # Sort by magnitude

    # Convert to scaling dimensions
    lambda_0 = eigenvalues[0]
    scaldims = []

    for i in range(min(n_dims, len(eigenvalues))):
        if eigenvalues[i] > 1e-15 * lambda_0:
            x = -np.log(eigenvalues[i] / lambda_0) / (2 * np.pi)
            scaldims.append(x)
        else:
            scaldims.append(np.inf)

    return np.array(scaldims)


################################################
# CFT Reference Values
################################################

CFT_DATA = {
    # Ising (N=2 Potts)
    2: {
        'central_charge': 0.5,
        'scaling_dims': {
            'identity': 0.0,
            'spin': 1/8,
            'energy': 1.0,
        }
    },
    # 3-state Potts
    3: {
        'central_charge': 4/5,
        'scaling_dims': {
            'identity': 0.0,
            'spin': 2/15,
            'spin_bar': 2/15,
            'energy': 4/5,
        }
    },
    # 4-state Potts (Ashkin-Teller point)
    4: {
        'central_charge': 1.0,
        'scaling_dims': {
            'identity': 0.0,
            'spin': 1/8,
            'energy': 1/2,
        }
    },
}


def get_cft_data(N):
    """
    Get CFT reference data for N-state Potts model.

    Returns dict with central_charge and scaling_dims.
    """
    if N in CFT_DATA:
        return CFT_DATA[N]
    else:
        return {
            'central_charge': None,
            'scaling_dims': {'identity': 0.0}
        }


################################################
# Convenience Functions
################################################

def verify_scaling_dimensions(A, N, tol=0.1, verbose=True):
    """
    Compare measured scaling dimensions to CFT predictions.

    Returns dict with measured values and errors.
    """
    cft = get_cft_data(N)
    measured = get_scaldims_zn(A)

    results = {
        'measured': measured,
        'cft': cft['scaling_dims'],
        'errors': {},
    }

    if verbose:
        print(f"Scaling dimensions for {N}-state Potts:")
        print(f"  CFT central charge: {cft['central_charge']}")
        print(f"  Index | Measured | CFT | Error")
        print(f"  " + "-" * 40)

    # Match measured to CFT values
    cft_vals = list(cft['scaling_dims'].values())
    for i, x_meas in enumerate(measured[:len(cft_vals)]):
        x_cft = cft_vals[i] if i < len(cft_vals) else None

        if x_cft is not None and x_cft > 0:
            error = abs(x_meas - x_cft) / x_cft * 100
            results['errors'][i] = error

            if verbose:
                print(f"  {i:5} | {x_meas:8.4f} | {x_cft:8.4f} | {error:5.1f}%")
        elif verbose:
            print(f"  {i:5} | {x_meas:8.4f} | {'N/A':>8} | {'N/A':>5}")

    return results


################################################
# Main test
################################################

if __name__ == "__main__":
    print("=" * 60)
    print("Testing GiltTNR2D_ZN.py")
    print("=" * 60)

    for N in [2, 3, 4]:
        print(f"\n--- {N}-state Potts model ---")

        pars = {'N': N, 'model_type': 'potts'}
        T = get_initial_tensor_zn(pars)

        print(f"Tensor shape: {T.to_ndarray().shape}")
        print(f"Critical β: {critical_beta(N):.4f}")

        # Get scaling dimensions
        scaldims = get_scaldims_zn(T)
        print(f"Initial scaling dims: {scaldims[:4]}")

        # Verify against CFT
        verify_scaling_dimensions(T, N, verbose=True)
