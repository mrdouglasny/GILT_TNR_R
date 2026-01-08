#!/usr/bin/env python3
"""
Fix GILT for Z3 equivariant tensors.

The key insight: GILT's trace-based mode identification works correctly in
spin basis but not in charge basis for Z3. The fix is to:
1. Transform tensor to spin basis
2. Apply GILT
3. Transform back to charge basis

For a tensor with sector dimensions [n_0, n_1, n_2]:
- Total dimension χ = n_0 + n_1 + n_2
- Build transformation matrix U that maps charge basis to spin basis
- U is unitary and χ×χ

The transformation structure:
- For balanced sectors (n_0 = n_1 = n_2 = n): U = F† ⊗ I_n
- For unbalanced sectors: more complex construction

The "direct sum of tensored sectors" interpretation:
- Each charge sector q with dimension n_q contributes n_q basis states
- In spin basis, these states mix according to the DFT
- The spin basis still has dimension χ, not 3χ
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'GiltTNR'))

import numpy as np
from ncon import ncon
from GiltTNR2D import gilttnr_step, get_envspec, optimize_Rp
from GiltTNR2D_Potts import get_initial_tensor_potts_relT
from tensors import Tensor, TensorZ3


def get_dft_matrix(N=3):
    """DFT matrix: F[s,q] = omega^(sq) / sqrt(N), where omega = e^{2πi/N}."""
    omega = np.exp(2j * np.pi / N)
    F = np.array([[omega**(s*q) for q in range(N)] for s in range(N)]) / np.sqrt(N)
    return F


def get_sector_dims_from_shape(shape, leg=0, N=3):
    """Extract sector dimensions from TensorZ3 shape."""
    if isinstance(shape[leg], dict):
        return [shape[leg].get(q, 0) for q in range(N)]
    elif isinstance(shape[leg], (list, tuple)):
        return list(shape[leg])
    else:
        # Assume equal sectors
        chi = shape[leg]
        n = chi // N
        return [n, n, chi - 2*n]


def build_transform_matrix(sector_dims, N=3, debug=False):
    """
    Build unitary transformation from charge to spin basis.

    For sector dimensions [n_0, n_1, n_2], the transformation applies
    the DFT to mix charge values while preserving total dimension.

    The key insight: within each "layer" of multiplicities, charges mix
    via DFT. We have min(n_q) complete layers where all charges are present,
    plus partial layers for larger sectors.

    For the complete layers: U = F† ⊗ I_{n_min}
    For the extra multiplicities: map to themselves (no mixing possible)
    """
    ns = sector_dims
    chi = sum(ns)
    n_min = min(ns)
    n_max = max(ns)

    F = get_dft_matrix(N)
    Fdag = F.conj().T

    # Build charge index offsets
    charge_start = [0]
    for q in range(N - 1):
        charge_start.append(charge_start[-1] + ns[q])

    # Build transformation matrix U (chi × chi)
    U = np.zeros((chi, chi), dtype=complex)

    # Complete layers: For k < n_min, all N charges have a state at offset k
    # These can be fully mixed by DFT
    for k in range(n_min):
        for s in range(N):
            # Output: spin s, multiplicity k → index in the output organization
            # We use same organization as input: [sector 0, sector 1, sector 2]
            out_idx = charge_start[s] + k  # Place in "sector s" position
            for q in range(N):
                # Input: charge q, multiplicity k
                in_idx = charge_start[q] + k
                U[out_idx, in_idx] = Fdag[s, q]

    # Extra multiplicities (k >= n_min for some sectors)
    # These can't be fully mixed, so map them to themselves
    for q in range(N):
        for k in range(n_min, ns[q]):
            idx = charge_start[q] + k
            U[idx, idx] = 1.0

    # Check unitarity
    if debug:
        UUdag = U @ U.conj().T
        UdagU = U.conj().T @ U
        err_UUdag = np.linalg.norm(UUdag - np.eye(chi))
        err_UdagU = np.linalg.norm(UdagU - np.eye(chi))
        print(f"    Transform matrix for sectors {ns}: ||UU†-I||={err_UUdag:.2e}, ||U†U-I||={err_UdagU:.2e}")

    return U


def transform_tensor_to_spin(T_z3, N=3):
    """
    Transform TensorZ3 from charge basis to spin basis.

    Returns (T_spin, transforms) where transforms is the list of U matrices.
    """
    arr = T_z3.to_ndarray()

    # Get sector dims for each leg
    if hasattr(T_z3, 'shape'):
        sector_dims = [get_sector_dims_from_shape(T_z3.shape, leg, N) for leg in range(4)]
    else:
        chi = arr.shape[0]
        n = chi // N
        sector_dims = [[n, n, chi - 2*n] for _ in range(4)]

    # Build transformation matrices
    debug_transform = (T_z3.to_ndarray().shape[0] > 3)  # Debug for non-initial tensors
    Us = [build_transform_matrix(dims, N, debug=debug_transform) for dims in sector_dims]

    # Apply transformation: T_spin = U_0 ⊗ U_1 ⊗ T_charge ⊗ U_2† ⊗ U_3†
    # In einsum notation: T_spin[a,b,c,d] = U0[a,i] U1[b,j] T[i,j,k,l] U2*[c,k] U3*[d,l]
    result = np.einsum('ai,bj,ijkl,ck,dl->abcd',
                       Us[0], Us[1], arr, Us[2].conj(), Us[3].conj())

    return Tensor.from_ndarray(result), Us


def transform_tensor_to_charge(T_spin, transforms, original_z3=None, N=3):
    """
    Transform tensor from spin basis back to charge basis.

    Forward was: T_spin[a,b,c,d] = U0[a,i] U1[b,j] T[i,j,k,l] U2*[c,k] U3*[d,l]
    This is: T_spin = (U0 ⊗ U1) T (U2* ⊗ U3*)

    Inverse: T_charge = (U0† ⊗ U1†) T_spin ((U2*)† ⊗ (U3*)†)
           = (U0† ⊗ U1†) T_spin (U2^T ⊗ U3^T)

    In einsum notation:
    T_charge[i,j,k,l] = U0†[i,a] U1†[j,b] T_spin[a,b,c,d] U2^T[k,c] U3^T[l,d]
    """
    arr = T_spin.to_ndarray()
    Us = transforms

    # U† = U*.T (conjugate transpose)
    U0dag = Us[0].conj().T  # [i,a]
    U1dag = Us[1].conj().T  # [j,b]
    # (U*)† = U^T (just transpose)
    U2T = Us[2].T  # [k,c]
    U3T = Us[3].T  # [l,d]

    result = np.einsum('ia,jb,abcd,kc,ld->ijkl', U0dag, U1dag, arr, U2T, U3T)

    # Check imaginary part
    imag_norm = np.linalg.norm(result.imag)
    real_norm = np.linalg.norm(result.real)
    total_norm = np.linalg.norm(result)
    if imag_norm > 1e-10 * real_norm:
        print(f"  Warning: Non-negligible imaginary part: |imag|={imag_norm:.2e}, |real|={real_norm:.2e}")

    # Try to reconstruct TensorZ3
    if original_z3 is not None and hasattr(original_z3, 'shape'):
        try:
            shape = original_z3.shape
            qhape = original_z3.qhape if hasattr(original_z3, 'qhape') else [[0,1,2]]*4
            dirs = original_z3.dirs if hasattr(original_z3, 'dirs') else [1,1,-1,-1]

            # Convert shape to list format
            if isinstance(shape[0], dict):
                shape_list = [[d for d in leg.values()] for leg in shape]
                qhape_list = [[q for q in leg.keys()] for leg in shape]
            else:
                shape_list = [list(leg) for leg in shape]
                qhape_list = [[0,1,2]]*4

            # Use real part (Z3 tensor should be real)
            return TensorZ3.from_ndarray(result.real, shape=shape_list,
                                         qhape=qhape_list, dirs=dirs)
        except Exception as e:
            print(f"  Warning: Cannot reconstruct TensorZ3: {e}")

    return Tensor.from_ndarray(result)


def gilt_in_spin_basis(A_z3, pars, N=3):
    """
    Apply GILT to Z3 tensor by working in spin basis.
    """
    # Transform to spin basis
    A_spin, transforms = transform_tensor_to_spin(A_z3, N)

    # Use plain tensor parameters for GILT
    pars_plain = dict(pars)
    pars_plain['symmetry_tensors'] = False

    # Apply GILT on all 4 edges
    for where in ["S", "E", "N", "W"]:
        U, S = get_envspec(A_spin, A_spin, pars_plain, where=where)
        Rp, _ = optimize_Rp(U, S, pars_plain)

        if where == "S":
            A_spin = ncon([Rp, A_spin], [[-2, 1], [-1, 1, -3, -4]])
        elif where == "E":
            A_spin = ncon([Rp, A_spin], [[-3, 1], [-1, -2, 1, -4]])
        elif where == "N":
            A_spin = ncon([Rp, A_spin], [[-4, 1], [-1, -2, -3, 1]])
        elif where == "W":
            A_spin = ncon([Rp, A_spin], [[-1, 1], [1, -2, -3, -4]])

    # Transform back to charge basis
    A_charge = transform_tensor_to_charge(A_spin, transforms, A_z3, N)

    return A_charge


def gilttnr_step_fixed(A_z3, log_fact, pars, N=3):
    """
    One GILT-TNR step with GILT applied in spin basis.

    The key insight: we transform to spin basis, apply GILT, and transform
    back BEFORE doing TRG. This ensures dimensions stay consistent.
    """
    # Transform to spin basis (as plain tensor)
    A_spin, transforms = transform_tensor_to_spin(A_z3, N)

    # Parameters for plain tensor operations
    pars_plain = dict(pars)
    pars_plain['symmetry_tensors'] = False

    # Run full GILT-TNR step in spin basis (as plain tensor)
    # This includes both GILT filtering AND TRG coarse-graining
    A_spin_new, log_new = gilttnr_step(A_spin, log_fact, pars_plain)

    # Now we need to transform back to charge basis
    # But the tensor dimensions have changed!
    # The new tensor is a plain tensor - we need to impose Z3 structure on it

    # Option 1: Keep it as plain tensor (lose explicit Z3 symmetry)
    # Option 2: Project onto Z3-equivariant subspace (may lose accuracy)
    # Option 3: Use Z3 TRG but with spin-basis GILT

    # For now, let's try option 3: do GILT in spin basis, then TRG in charge basis
    # This requires transforming back after GILT but before TRG

    return A_spin_new, log_new


def gilttnr_step_fixed_v2(A_z3, log_fact, pars, N=3, debug=False):
    """
    Alternative approach: GILT in spin basis, TRG in charge basis.

    1. Transform Z3 → spin
    2. Apply GILT (in spin basis)
    3. Transform spin → Z3
    4. Apply TRG coarse-graining (in charge basis, with Z3 tensors)
    """
    if debug:
        print(f"  Input shape: {A_z3.to_ndarray().shape}")
        if hasattr(A_z3, 'shape'):
            print(f"  Sector structure: {A_z3.shape}")

    # Transform to spin basis
    A_spin, transforms = transform_tensor_to_spin(A_z3, N)

    if debug:
        print(f"  Spin shape: {A_spin.to_ndarray().shape}")

    # Apply GILT in spin basis
    pars_plain = dict(pars)
    pars_plain['symmetry_tensors'] = False

    for where in ["S", "E", "N", "W"]:
        U, S = get_envspec(A_spin, A_spin, pars_plain, where=where)
        Rp, _ = optimize_Rp(U, S, pars_plain)

        Rp_shape = Rp.to_ndarray().shape
        A_shape = A_spin.to_ndarray().shape

        if debug:
            print(f"  GILT {where}: Rp shape={Rp_shape}, A shape={A_shape}")

        # Apply Rp following GiltTNR2D.apply_Rp convention:
        # "S" applies to leg 2 (E leg in tensor [W,S,E,N])
        # "E" applies to leg 1 (S leg)
        # "N" applies to leg 0 (W leg)
        # "W" applies to leg 3 (N leg)
        # Note: we use A_spin for both A1 and A2, so Rp1=Rp2=Rp
        if where == "S":
            # A1 contracts leg 2 with Rp: ([-1,-2,3,-4], [3,-3])
            A_spin = ncon([A_spin, Rp], [[-1,-2,1,-4], [1,-3]])
        elif where == "E":
            # A2 contracts leg 1 with Rp: ([-1,2,-3,-4], [2,-2])
            A_spin = ncon([A_spin, Rp], [[-1,1,-3,-4], [1,-2]])
        elif where == "N":
            # A1 contracts leg 0 with Rp: ([1,-2,-3,-4], [1,-1])
            A_spin = ncon([A_spin, Rp], [[1,-2,-3,-4], [1,-1]])
        elif where == "W":
            # A2 contracts leg 3 with Rp: ([-1,-2,-3,4], [4,-4])
            A_spin = ncon([A_spin, Rp], [[-1,-2,-3,1], [1,-4]])

    if debug:
        print(f"  After GILT shape: {A_spin.to_ndarray().shape}")
        print(f"  After GILT norm: {np.linalg.norm(A_spin.to_ndarray()):.6e}")

    # TRG coarse-graining in spin basis (plain tensor)
    pars_no_gilt = dict(pars)
    pars_no_gilt['gilt_eps'] = 0
    pars_no_gilt['symmetry_tensors'] = False

    try:
        A_new, log_new = gilttnr_step(A_spin, log_fact, pars_no_gilt)
    except Exception as e:
        if debug:
            print(f"  TRG error: {e}")
            import traceback
            traceback.print_exc()
        raise

    if debug:
        print(f"  After TRG shape: {A_new.to_ndarray().shape}")

    return A_new, log_new


def gilttnr_step_fixed_v3(A_z3, log_fact, pars, N=3, debug=False):
    """
    Full spin-basis approach: transform once, then GILT+TRG all in spin basis.
    Don't try to convert back to Z3 at each step.

    IMPORTANT: The DFT produces complex128 tensors, but for a real Z3 tensor
    the result is effectively real (imaginary part is machine epsilon).
    We MUST convert to float64 before TRG, because SVD of complex matrices
    gives different results than SVD of real matrices even when the imaginary
    part is negligible.

    NOTE: This version does everything in spin basis. It works correctly but
    doesn't leverage the efficiency of charge-basis TRG. Use gilttnr_step_fixed_v4
    for the efficient version that does GILT per-link in spin basis but TRG
    in charge basis.
    """
    # If input is already a plain tensor (from previous steps), use it directly
    if hasattr(A_z3, 'sects'):
        # It's a TensorZ3 - transform to spin
        A_spin, _ = transform_tensor_to_spin(A_z3, N)
        arr_spin = A_spin.to_ndarray()

        # Check imaginary part
        imag_norm = np.linalg.norm(arr_spin.imag)
        real_norm = np.linalg.norm(arr_spin.real)

        if debug:
            arr_z3 = A_z3.to_ndarray()
            print(f"  Transformed Z3→spin: ||Z3||={np.linalg.norm(arr_z3):.4e}, ||spin||={np.linalg.norm(arr_spin):.4e}")
            print(f"  Imaginary check: ||imag||={imag_norm:.2e}, ||real||={real_norm:.2e}")

        # Convert to real (the DFT of a real Z3 tensor is real in spin basis)
        if imag_norm < 1e-10 * real_norm:
            A_spin = Tensor.from_ndarray(arr_spin.real)
        else:
            print(f"  WARNING: Non-negligible imaginary part! ||imag||/||real||={imag_norm/real_norm:.2e}")
            # Still convert to real, but warn
            A_spin = Tensor.from_ndarray(arr_spin.real)
    else:
        # Already plain tensor - ensure it's real
        arr = A_z3.to_ndarray()
        if arr.dtype == np.complex128 or arr.dtype == np.complex64:
            imag_norm = np.linalg.norm(arr.imag)
            real_norm = np.linalg.norm(arr.real)
            if imag_norm > 1e-10 * real_norm:
                print(f"  WARNING: Non-negligible imaginary part! ||imag||/||real||={imag_norm/real_norm:.2e}")
            A_spin = Tensor.from_ndarray(arr.real)
        else:
            A_spin = A_z3

    if debug:
        print(f"  Spin shape: {A_spin.to_ndarray().shape}, dtype: {A_spin.to_ndarray().dtype}")

    # Full GILT-TNR in spin basis
    pars_plain = dict(pars)
    pars_plain['symmetry_tensors'] = False

    A_new, log_new = gilttnr_step(A_spin, log_fact, pars_plain)

    if debug:
        print(f"  After step shape: {A_new.to_ndarray().shape}")

    return A_new, log_new


def transform_single_leg_to_spin(A_z3, leg, N=3):
    """
    Transform a single leg of a TensorZ3 from charge basis to spin basis.

    This is the key for efficient GILT: only transform the leg being processed,
    keeping other legs in charge basis.

    Args:
        A_z3: TensorZ3 input tensor
        leg: Which leg to transform (0=W, 1=S, 2=E, 3=N)
        N: Group order (default 3 for Z3)

    Returns:
        (A_mixed, U): Tensor with one leg in spin basis, transformation matrix
    """
    arr = A_z3.to_ndarray()

    # Get sector dims for the specified leg
    if hasattr(A_z3, 'shape'):
        sector_dims = get_sector_dims_from_shape(A_z3.shape, leg, N)
    else:
        chi = arr.shape[leg]
        n = chi // N
        sector_dims = [n, n, chi - 2*n]

    # Build transformation matrix for this leg
    U = build_transform_matrix(sector_dims, N, debug=False)

    # Apply transformation to just this leg
    # Move the target leg to position 0, apply U, move back
    arr_t = np.moveaxis(arr, leg, 0)
    shape_orig = arr_t.shape
    arr_flat = arr_t.reshape(shape_orig[0], -1)
    arr_transformed = U @ arr_flat
    arr_t = arr_transformed.reshape(shape_orig)
    result = np.moveaxis(arr_t, 0, leg)

    return result, U


def transform_single_leg_to_charge(arr, leg, U):
    """
    Transform a single leg back from spin basis to charge basis.

    Args:
        arr: Array with one leg in spin basis
        leg: Which leg to transform back
        U: Transformation matrix from transform_single_leg_to_spin

    Returns:
        Array with all legs in charge basis
    """
    # U† transforms spin back to charge
    Udag = U.conj().T

    arr_t = np.moveaxis(arr, leg, 0)
    shape_orig = arr_t.shape
    arr_flat = arr_t.reshape(shape_orig[0], -1)
    arr_transformed = Udag @ arr_flat
    arr_t = arr_transformed.reshape(shape_orig)
    result = np.moveaxis(arr_t, 0, leg)

    return result


def optimize_Rp_spin_basis_v1(U, S, pars, sector_dims, N=3):
    """
    DEPRECATED: This approach doesn't work because eigendecomposition
    doesn't commute with basis change in the expected way.

    The transformed U has large imaginary parts (~89% of real norm).
    """
    from GiltTNR2D import build_Rp

    chi = sum(sector_dims)
    F = build_transform_matrix(sector_dims, N, debug=False)

    U_arr = U.to_ndarray()
    U_spin = np.einsum('sq,tp,qpi->sti', F, F, U_arr)

    imag_norm = np.linalg.norm(U_spin.imag)
    real_norm = np.linalg.norm(U_spin.real)
    if imag_norm > 1e-10 * real_norm:
        print(f"  WARNING in optimize_Rp_spin_basis: Non-negligible imag ||imag||/||real||={imag_norm/real_norm:.2e}")
    U_spin = U_spin.real

    t = np.einsum('ssi->i', U_spin)
    S_arr = S.to_ndarray()
    S_flipped = S_arr[::-1]

    C_err_constterm = np.linalg.norm(t * S_flipped)

    gilt_eps = pars["gilt_eps"]
    ratio = S_flipped / gilt_eps
    weight = ratio**2 / (1 + ratio**2)
    tp = t * weight

    Rp_spin = np.einsum('sti,i->st', U_spin, tp)

    Fdag = F.conj().T
    Rp_charge = np.einsum('qs,pt,st->qp', Fdag, Fdag, Rp_spin)

    imag_norm = np.linalg.norm(Rp_charge.imag)
    real_norm = np.linalg.norm(Rp_charge.real)
    if imag_norm > 1e-10 * real_norm:
        print(f"  WARNING: Rp has non-negligible imaginary part")
    Rp_charge = Rp_charge.real

    err = np.linalg.norm((t - tp) * S_flipped) / C_err_constterm if C_err_constterm > 0 else 0
    return Tensor.from_ndarray(Rp_charge), err


def get_envspec_spin_basis(A_charge, pars, where, sector_dims, N=3):
    """
    Compute environment spectrum with eigendecomposition in spin basis.

    The correct approach:
    1. Build environment E·E† in charge basis (exploits block structure)
    2. Transform E·E† to spin basis: (E·E†)_spin = F (E·E†)_charge F†
    3. Eigendecompose in spin basis to get U, S
    4. Return U, S in spin basis

    This is different from transforming U after eigendecomposition!
    Eigendecomposition doesn't commute with basis change.

    Args:
        A_charge: Tensor in charge basis (plain tensor, not TensorZ3)
        pars: Parameters
        where: Edge direction ("S", "E", "N", "W")
        sector_dims: Sector dimensions for the bond
        N: Group order

    Returns:
        U, S: Environment eigenvectors and spectrum in spin basis
    """
    # Build corner tensors (traces over pairs of legs)
    A = A_charge
    Aconj = A.conjugate()

    SW = ncon((A, Aconj), ([1,-1,-11,2], [1,-2,-12,2]))
    NW = ncon((A, Aconj), ([1,2,-1,-11], [1,2,-2,-12]))
    NE = ncon((A, Aconj), ([-11,1,2,-1], [-12,1,2,-2]))
    SE = ncon((A, Aconj), ([-1,-11,1,2], [-2,-12,1,2]))

    # Build E depending on edge direction
    if where == "S":
        nconlist = (SW, NW, NE, SE)
    elif where == "W":
        nconlist = (NW, NE, SE, SW)
    elif where == "N":
        nconlist = (NE, SE, SW, NW)
    elif where == "E":
        nconlist = (SE, SW, NW, NE)
    else:
        raise ValueError(f"Unknown direction: {where}")

    E = ncon(nconlist, ([1,2,-1,-11], [3,4,1,2], [5,6,3,4], [-2,-12,5,6]))

    # E has shape (chi, chi², chi, chi²) - actually this is wrong
    # Let me check the actual structure...
    # E has indices [-1, -11] from first tensor, [-2, -12] from last
    # After contraction: E[a, a', b, b'] where (a,a') and (b,b') are the two ends of the bond

    # Reshape E to matrix: E_matrix[(a,a'), (b,b')] = E[a,a',b,b']
    E_arr = E.to_ndarray()
    chi = E_arr.shape[0]  # Bond dimension
    chi_sq = chi * chi

    # Reshape to (chi², chi²) matrix
    E_matrix = E_arr.reshape(chi_sq, chi_sq)

    # Compute E·E† in charge basis
    EEdagger_charge = E_matrix @ E_matrix.conj().T

    # Transform to spin basis
    # Each index of EEdagger corresponds to a pair (a, a')
    # The DFT acts on each index independently:
    # (E·E†)_spin = (F ⊗ F) (E·E†)_charge (F† ⊗ F†)

    F = build_transform_matrix(sector_dims, N, debug=False)

    # Build F ⊗ F for the (chi²) × (chi²) matrix
    F_kron = np.kron(F, F)  # (chi²) × (chi²)
    Fdag_kron = F_kron.conj().T

    # Transform: (E·E†)_spin = F⊗F · (E·E†)_charge · (F⊗F)†
    EEdagger_spin = F_kron @ EEdagger_charge @ Fdag_kron

    # Check imaginary part
    imag_norm = np.linalg.norm(EEdagger_spin.imag)
    real_norm = np.linalg.norm(EEdagger_spin.real)
    if imag_norm > 1e-10 * real_norm:
        print(f"  WARNING in get_envspec_spin_basis: E·E† has non-negligible imag ||imag||/||real||={imag_norm/real_norm:.2e}")

    # Convert to real for eigendecomposition
    EEdagger_spin = EEdagger_spin.real

    # Eigendecompose in spin basis
    eigenvalues, eigenvectors = np.linalg.eigh(EEdagger_spin)

    # Sort by eigenvalue (largest first)
    idx = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]

    # Singular values are sqrt of eigenvalues (all non-negative for E·E†)
    S = np.sqrt(np.maximum(eigenvalues, 0))

    # Reshape eigenvectors: each column is (chi², ) -> (chi, chi)
    # U[a, a', i] = eigenvectors[(a,a'), i].reshape(chi, chi, n_modes)
    n_modes = eigenvectors.shape[1]
    U = eigenvectors.T.reshape(n_modes, chi, chi).transpose(1, 2, 0)  # (chi, chi, n_modes)

    return Tensor.from_ndarray(U), Tensor.from_ndarray(S)


def optimize_Rp_in_spin_basis(U_spin, S, pars):
    """
    Optimize Rp given U and S already in spin basis.

    Standard GILT algorithm:
    1. Compute t = Tr(U) for each mode
    2. Optimize t' = t · w(S, ε)
    3. Build Rp = Σ_i U[:,:,i] · t'_i

    Args:
        U_spin: Environment eigenvectors in spin basis, shape (chi, chi, n_modes)
        S: Environment spectrum (singular values)
        pars: Parameters including gilt_eps

    Returns:
        Rp_spin, err: Filter matrix in spin basis, truncation error
    """
    U_arr = U_spin.to_ndarray()
    S_arr = S.to_ndarray()

    # Compute traces: t[i] = Σ_a U[a,a,i]
    t = np.einsum('aai->i', U_arr)

    # Flip S (EKR convention for symmetry tensors)
    S_flipped = S_arr[::-1]

    # Compute normalization constant
    C_err_constterm = np.linalg.norm(t * S_flipped)
    if C_err_constterm == 0:
        C_err_constterm = 1.0  # Avoid division by zero

    # Optimize t': minimize (t'-t)²S² + ε²t'²
    gilt_eps = pars["gilt_eps"]
    ratio = S_flipped / gilt_eps
    weight = ratio**2 / (1 + ratio**2)
    tp = t * weight

    # Build Rp: Rp[a,b] = Σ_i U[a,b,i] · t'[i]
    Rp_spin = np.einsum('abi,i->ab', U_arr, tp)

    # Compute truncation error
    err = np.linalg.norm((t - tp) * S_flipped) / C_err_constterm

    return Tensor.from_ndarray(Rp_spin), err


def gilttnr_step_fixed_v4(A_z3, log_fact, pars, N=3, debug=False):
    """
    EXPERIMENTAL: Eigendecomposition in spin basis (after transforming E·E†).

    Approach:
    1. Build E·E† in charge basis (efficient, exploits block structure)
    2. Transform E·E† to spin basis: F⊗F · (E·E†) · (F⊗F)†
    3. Eigendecompose in spin basis to get U, S
    4. Compute t = Tr(U), optimize t', build Rp in spin basis
    5. Transform Rp back to charge basis: F† · Rp · F
    6. Apply Rp in charge basis

    RESULT: This approach FAILS because E·E† in spin basis has large imaginary parts.

    The fundamental issue: A is real in charge basis. But (E·E†)_spin = F⊗F · (E·E†)_charge · (F⊗F)†
    is NOT the same as computing E·E† from A_spin = F·A·F†.

    The DFT transformation doesn't commute with the contraction operations that build E.

    Use v3 (full spin basis) instead - it works correctly by transforming A to spin basis
    BEFORE any operations.
    """
    from GiltTNR2D import apply_Rp

    if debug:
        print(f"  Input: shape={A_z3.to_ndarray().shape}")
        if hasattr(A_z3, 'shape'):
            print(f"  Sector structure: {A_z3.shape}")

    A = A_z3

    # Get sector dimensions
    if hasattr(A, 'shape'):
        sector_dims_list = [get_sector_dims_from_shape(A.shape, leg, N) for leg in range(4)]
    else:
        chi = A.to_ndarray().shape[0]
        n = chi // N
        sector_dims_list = [[n, n, chi - 2*n]] * 4

    pars_plain = dict(pars)
    pars_plain['symmetry_tensors'] = False

    # Convert to plain tensor for processing
    A_plain = Tensor.from_ndarray(A.to_ndarray()) if hasattr(A, 'sects') else A

    # Apply GILT on each edge with spin-basis eigendecomposition
    for where in ["S", "E", "N", "W"]:
        leg_map = {"S": 1, "E": 2, "N": 3, "W": 0}
        leg = leg_map[where]
        sector_dims = sector_dims_list[leg]

        # Get environment spectrum in spin basis
        U_spin, S = get_envspec_spin_basis(A_plain, pars_plain, where, sector_dims, N)

        # Optimize Rp in spin basis
        Rp_spin, err = optimize_Rp_in_spin_basis(U_spin, S, pars_plain)

        if debug:
            Rp_arr = Rp_spin.to_ndarray()
            print(f"  GILT {where}: Rp_spin shape={Rp_arr.shape}, err={err:.4e}")

        # Transform Rp back to charge basis
        F = build_transform_matrix(sector_dims, N, debug=False)
        Fdag = F.conj().T
        Rp_spin_arr = Rp_spin.to_ndarray()

        # Rp_charge = F† · Rp_spin · F
        Rp_charge = Fdag @ Rp_spin_arr @ F

        # Check imaginary part
        imag_norm = np.linalg.norm(Rp_charge.imag)
        real_norm = np.linalg.norm(Rp_charge.real)
        if imag_norm > 1e-10 * real_norm:
            print(f"  WARNING: Rp_charge has non-negligible imag after {where}")
        Rp_charge = Rp_charge.real

        Rp = Tensor.from_ndarray(Rp_charge)

        if debug:
            print(f"  GILT {where}: Rp_charge shape={Rp_charge.shape}")

        # Apply Rp
        A_plain, _ = apply_Rp(A_plain, A_plain, Rp, Rp, where=where)

    if debug:
        print(f"  After GILT: shape={A_plain.to_ndarray().shape}")

    # TRG coarse-graining
    pars_no_gilt = dict(pars)
    pars_no_gilt['gilt_eps'] = 0
    pars_no_gilt['symmetry_tensors'] = False

    from GiltTNR2D import gilttnr_step
    A_new, log_new = gilttnr_step(A_plain, log_fact, pars_no_gilt)

    if debug:
        print(f"  After TRG: shape={A_new.to_ndarray().shape}")

    return A_new, log_new


def gilttnr_step_fixed_v5(A_z3, log_fact, pars, N=3, debug=False):
    """
    DEPRECATED: Hybrid approach that transforms bond legs only.
    This doesn't work because partial basis transformations produce complex tensors.
    Use v6 instead.
    """
    from GiltTNR2D import apply_Rp

    if debug:
        print(f"  Input: shape={A_z3.to_ndarray().shape}")
        if hasattr(A_z3, 'shape'):
            print(f"  Sector structure: {A_z3.shape}")

    A = A_z3

    # Get sector dimensions
    if hasattr(A, 'shape'):
        sector_dims_list = [get_sector_dims_from_shape(A.shape, leg, N) for leg in range(4)]
    else:
        chi = A.to_ndarray().shape[0]
        n = chi // N
        sector_dims_list = [[n, n, chi - 2*n]] * 4

    pars_plain = dict(pars)
    pars_plain['symmetry_tensors'] = False

    # Get tensor as array
    A_arr = A.to_ndarray() if hasattr(A, 'to_ndarray') else A

    # Apply GILT on each edge
    # For each edge, transform the two legs at that bond to spin basis
    for where in ["S", "E", "N", "W"]:
        # Determine which legs form this bond
        # Tensor layout: [W=0, S=1, E=2, N=3]
        # Edge "S" connects leg 1 of one tensor to leg 0 of neighbor -> legs 1, 0
        # Edge "E" connects leg 2 of one tensor to leg 3 of neighbor -> legs 2, 3
        # Edge "N" connects leg 3 of one tensor to leg 2 of neighbor -> legs 3, 2
        # Edge "W" connects leg 0 of one tensor to leg 1 of neighbor -> legs 0, 1
        bond_legs = {"S": (1, 0), "E": (2, 3), "N": (3, 2), "W": (0, 1)}
        leg1, leg2 = bond_legs[where]

        # For homogeneous case (A1 = A2), we only need one tensor
        # Transform the two bond legs to spin basis
        sector_dims_1 = sector_dims_list[leg1]
        sector_dims_2 = sector_dims_list[leg2]

        F1 = build_transform_matrix(sector_dims_1, N, debug=False)
        F2 = build_transform_matrix(sector_dims_2, N, debug=False)

        # Transform legs to spin basis
        # A_hybrid[..., leg1_spin, ..., leg2_spin, ...] = F1 @ A[..., leg1_charge, ...] and similar for leg2
        A_hybrid = A_arr.copy()

        # Transform leg1
        A_hybrid = np.moveaxis(A_hybrid, leg1, 0)
        shape_temp = A_hybrid.shape
        A_hybrid = A_hybrid.reshape(shape_temp[0], -1)
        A_hybrid = F1 @ A_hybrid
        A_hybrid = A_hybrid.reshape(shape_temp)
        A_hybrid = np.moveaxis(A_hybrid, 0, leg1)

        # Transform leg2
        A_hybrid = np.moveaxis(A_hybrid, leg2, 0)
        shape_temp = A_hybrid.shape
        A_hybrid = A_hybrid.reshape(shape_temp[0], -1)
        A_hybrid = F2 @ A_hybrid
        A_hybrid = A_hybrid.reshape(shape_temp)
        A_hybrid = np.moveaxis(A_hybrid, 0, leg2)

        # Check imaginary part - should be small for Potts
        imag_norm = np.linalg.norm(A_hybrid.imag)
        real_norm = np.linalg.norm(A_hybrid.real)
        if debug:
            print(f"  {where}: After bond transform ||imag||/||real|| = {imag_norm/real_norm:.2e}")

        if imag_norm > 1e-10 * real_norm:
            print(f"  WARNING at {where}: Hybrid tensor has significant imag part")

        # Use real part
        A_hybrid = A_hybrid.real
        A_plain = Tensor.from_ndarray(A_hybrid)

        # Now build environment and do GILT in this hybrid basis
        # The bond legs are in spin basis, other legs in charge basis
        U_env, S_env = get_envspec(A_plain, A_plain, pars_plain, where=where)

        # The environment eigenvectors U are now in hybrid basis
        # But the trace is over the bond legs, which are in spin basis
        # So the trace is correct!

        Rp, err = optimize_Rp(U_env, S_env, pars_plain)

        if debug:
            Rp_arr = Rp.to_ndarray()
            print(f"  GILT {where}: Rp shape={Rp_arr.shape}, err={err:.4e}")

        # Transform Rp back to charge basis
        # Rp acts on the bond, which is in spin basis
        # Rp_charge = F† @ Rp_spin @ F
        F_bond = F1  # Both legs of the bond have same dimension
        Fdag = F_bond.conj().T
        Rp_arr = Rp.to_ndarray()
        Rp_charge = Fdag @ Rp_arr @ F_bond

        # Check imaginary part
        imag_norm = np.linalg.norm(Rp_charge.imag)
        real_norm = np.linalg.norm(Rp_charge.real)
        if imag_norm > 1e-10 * real_norm:
            print(f"  WARNING: Rp_charge has significant imag after {where}")
        Rp_charge = Rp_charge.real

        Rp = Tensor.from_ndarray(Rp_charge)

        # Apply Rp to original tensor (in charge basis)
        A_plain_orig = Tensor.from_ndarray(A_arr)
        A_new, _ = apply_Rp(A_plain_orig, A_plain_orig, Rp, Rp, where=where)
        A_arr = A_new.to_ndarray()

    if debug:
        print(f"  After GILT: shape={A_arr.shape}")

    # TRG coarse-graining in charge basis
    pars_no_gilt = dict(pars)
    pars_no_gilt['gilt_eps'] = 0
    pars_no_gilt['symmetry_tensors'] = False

    from GiltTNR2D import gilttnr_step
    A_new, log_new = gilttnr_step(Tensor.from_ndarray(A_arr), log_fact, pars_no_gilt)

    if debug:
        print(f"  After TRG: shape={A_new.to_ndarray().shape}")

    return A_new, log_new


# === Tests ===

def gilttnr_step_fixed_v6(A_z3, log_fact, pars, N=3, debug=False):
    """
    Analysis: Z3 GILT-TNR failure modes.

    KEY FINDINGS (Jan 2026):

    1. BASIS TRANSFORMATION IS CORRECT:
       - E_spin = T @ E_charge @ T†  where T = Fdag⊗F  (verified to ~1e-15)
       - E·E†_spin = T @ E·E†_charge @ T†  (verified to ~1e-15)
       - The trace is basis-independent for unitary transformations

    2. EIGENVALUE ORDERING:
       - TensorZ3.eig orders by charge sector (block structure)
       - Plain Tensor.eig orders by eigenvalue magnitude
       - After reordering, traces match to ~1e-15
       - This is NOT the cause of the problem!

    3. ROOT CAUSE: TRUNCATION DIMENSIONS
       After step 1:
       - Z3: shape (10, 10, 10, 10) with sectors [[3,3,4], [3,4,3], ...]
       - Plain: shape (12, 10, 12, 10)

       The TensorZ3 truncation uses per-sector greedy allocation which gives
       different bond dimensions than plain tensor truncation. This is the bug
       documented in truncation_bug.tex.

    4. WHY V3 WORKS:
       V3 converts to spin basis (plain tensor) at the start, so all operations
       use plain tensor truncation. This gives consistent dimensions throughout.

    CONCLUSION:
    The issue is not GILT mode identification (which is basis-independent) but
    the TensorZ3 truncation algorithm in SVD/eigendecomposition. Fix options:
    a) Use v3 (full spin basis) - works but no Z3 efficiency gains
    b) Fix TensorZ3 truncation to match plain behavior - requires modifying EKR code
    c) Accept different dimensions - may cause divergence due to truncation imbalance
    """
    # Just use standard GILT-TNR with Z3 tensors
    from GiltTNR2D import gilttnr_step

    pars_z3 = dict(pars)
    pars_z3['symmetry_tensors'] = True

    if debug:
        print(f"  v6 Input: shape={A_z3.to_ndarray().shape}")
        if hasattr(A_z3, 'shape'):
            print(f"  Sector structure: {A_z3.shape}")

    A_new, log_new = gilttnr_step(A_z3, log_fact, pars_z3)

    if debug:
        print(f"  After step: shape={A_new.to_ndarray().shape}")

    return A_new, log_new


def test_transform_initial():
    """Test transformation on initial tensor."""
    print("=" * 80)
    print("Test: Initial Tensor Transformation")
    print("=" * 80)

    pars_plain = {
        'q': 3, 'relT': 1.0, 'gilt_eps': 1e-6,
        'cg_chis': [16], 'cg_eps': 1e-10, 'verbosity': 0,
        'symmetry_tensors': False,
    }
    pars_z3 = dict(pars_plain)
    pars_z3['symmetry_tensors'] = True

    A_plain = get_initial_tensor_potts_relT(pars_plain)
    A_z3 = get_initial_tensor_potts_relT(pars_z3)

    # Transform to spin basis
    A_spin, transforms = transform_tensor_to_spin(A_z3)

    arr_plain = A_plain.to_ndarray()
    arr_spin = A_spin.to_ndarray()

    diff = np.linalg.norm(arr_plain - arr_spin) / np.linalg.norm(arr_plain)
    print(f"\n||plain - Z3→spin|| / ||plain|| = {diff:.2e}")
    print("✓ Transform correct!" if diff < 1e-10 else "✗ Transform error")

    # Round-trip
    A_back = transform_tensor_to_charge(A_spin, transforms, A_z3)
    arr_back = A_back.to_ndarray()
    arr_orig = A_z3.to_ndarray()

    diff_rt = np.linalg.norm(arr_orig - arr_back) / np.linalg.norm(arr_orig)
    print(f"Round-trip error: {diff_rt:.2e}")


def test_gilt_modes():
    """Compare GILT modes in charge vs spin basis."""
    print("\n" + "=" * 80)
    print("Test: GILT Mode Comparison")
    print("=" * 80)

    pars = {
        'q': 3, 'relT': 1.0, 'gilt_eps': 1e-6,
        'cg_chis': [16], 'cg_eps': 1e-10, 'verbosity': 0,
        'symmetry_tensors': True,
    }

    A_z3 = get_initial_tensor_potts_relT(pars)
    A_spin, _ = transform_tensor_to_spin(A_z3)

    pars_plain = dict(pars)
    pars_plain['symmetry_tensors'] = False

    print(f"\n{'Edge':<6} {'Charge modes':<15} {'Spin modes':<15} {'Ratio':<10}")
    print("-" * 50)

    for where in ["S", "E", "N", "W"]:
        # Charge basis
        U_ch, _ = get_envspec(A_z3, A_z3, pars, where=where)
        t_ch = ncon(U_ch, [1, 1, -1]).to_ndarray()
        n_ch = np.sum(np.abs(t_ch) > 0.1)

        # Spin basis
        U_sp, _ = get_envspec(A_spin, A_spin, pars_plain, where=where)
        t_sp = ncon(U_sp, [1, 1, -1]).to_ndarray()
        n_sp = np.sum(np.abs(t_sp) > 0.1)

        ratio = n_ch / n_sp if n_sp > 0 else float('inf')
        print(f"{where:<6} {n_ch:<15} {n_sp:<15} {ratio:<10.2f}")


def test_flow():
    """Test GILT-TNR flow with v3 (spin basis) approach."""
    print("\n" + "=" * 80)
    print("Test: GILT-TNR Flow (Plain vs v3 spin-basis)")
    print("=" * 80)

    chi = 16
    gilt_eps = 1e-6
    n_steps = 8

    pars_plain = {
        'q': 3, 'relT': 1.0, 'gilt_eps': gilt_eps,
        'cg_chis': [chi], 'cg_eps': 1e-10, 'verbosity': 0,
        'symmetry_tensors': False,
    }
    pars_z3 = dict(pars_plain)
    pars_z3['symmetry_tensors'] = True

    A_plain = get_initial_tensor_potts_relT(pars_plain)
    A_z3_v3 = get_initial_tensor_potts_relT(pars_z3)

    log_p, log_v3 = 0.0, 0.0

    print(f"\nPlain: Standard GILT-TNR with plain tensors")
    print(f"v3: Transform Z3→spin, do GILT-TNR in spin basis")
    print()
    print("Step   Plain          v3 (spin)      v3/Plain")
    print("-" * 55)

    for step in range(1, n_steps + 1):
        # Plain flow
        A_plain, log_p = gilttnr_step(A_plain, log_p, pars_plain)
        norm_p = np.linalg.norm(A_plain.to_ndarray())

        # v3: Full spin basis
        try:
            A_z3_v3, log_v3 = gilttnr_step_fixed_v3(A_z3_v3, log_v3, pars_z3, debug=False)
            norm_v3 = np.linalg.norm(A_z3_v3.to_ndarray())
        except Exception as e:
            print(f"  v3 error at step {step}: {e}")
            norm_v3 = float('nan')

        r_v3 = norm_v3 / norm_p if not np.isnan(norm_v3) else float('nan')

        print(f"{step:<6} {norm_p:<14.4e} {norm_v3:<14.4e} {r_v3:<10.6f}")


if __name__ == "__main__":
    test_transform_initial()
    test_gilt_modes()
    test_flow()
