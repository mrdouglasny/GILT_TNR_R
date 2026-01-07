# MODULE: EchelonFormGFN.jl
# USAGE: include("src/EchelonFormGFN.jl") or using EKRNewton
# INPUTS: Integer matrices representing linear systems over GF(N)
# OUTPUTS: Row echelon form, solutions to linear systems
# DESCRIPTION: Gaussian elimination over finite fields GF(N) for N prime
#
# BASED ON: EchelonForm.jl (GF(2) implementation with XOR)
# NEW IN THIS MODULE: Generalized to GF(N) using modular arithmetic instead of XOR

"""
Row echelon form and linear algebra over finite fields GF(N).

For Z_N symmetric tensors, discrete gauge fixing requires solving linear systems
over GF(N) where N is the symmetry order. For prime N, GF(N) = Z/NZ.

Key differences from GF(2) (Boolean) implementation:
- GF(2): row reduction uses XOR
- GF(N): row reduction uses modular subtraction with modular inverse

Note: This implementation assumes N is prime. For composite N, would need
polynomial arithmetic in GF(p^k).
"""

using LinearAlgebra

################################################
# GF(N) Arithmetic Helpers
################################################

"""
    mod_inverse(a, N) -> Int

Compute the modular multiplicative inverse of a (mod N).
Returns x such that a*x ≡ 1 (mod N).

Uses extended Euclidean algorithm. Requires gcd(a, N) = 1.
For prime N, all non-zero elements have inverses.
"""
function mod_inverse(a::Int, N::Int)
    a = mod(a, N)
    if a == 0
        error("Cannot compute inverse of 0 in GF($N)")
    end

    # Extended Euclidean algorithm
    t, new_t = 0, 1
    r, new_r = N, a

    while new_r != 0
        quotient = div(r, new_r)
        t, new_t = new_t, t - quotient * new_t
        r, new_r = new_r, r - quotient * new_r
    end

    if r > 1
        error("$a is not invertible mod $N (gcd = $r)")
    end

    return mod(t, N)
end

"""
    mod_sub(a, b, N) -> Int

Modular subtraction: (a - b) mod N.
"""
mod_sub(a::Int, b::Int, N::Int) = mod(a - b, N)

"""
    mod_mul(a, b, N) -> Int

Modular multiplication: (a * b) mod N.
"""
mod_mul(a::Int, b::Int, N::Int) = mod(a * b, N)

################################################
# Matrix Row Operations for GF(N)
################################################

"""
    flip_rows_gfn!(A, rows)

Swap two rows in matrix A. Works for any field.
"""
function flip_rows_gfn!(A::Matrix{Int}, rows::Vector{Int})
    for j in 1:size(A, 2)
        A[rows[1], j], A[rows[2], j] = A[rows[2], j], A[rows[1], j]
    end
    return A
end

"""
    flip_columns_gfn!(A, cols)

Swap two columns in matrix A. Works for any field.
"""
function flip_columns_gfn!(A::Matrix{Int}, cols::Vector{Int})
    for i in 1:size(A, 1)
        A[i, cols[1]], A[i, cols[2]] = A[i, cols[2]], A[i, cols[1]]
    end
    return A
end

"""
    reduce_rows_below_gfn!(A, row, col, N)

Eliminate entries below pivot at (row, col) using GF(N) arithmetic.

For each row i below the pivot row:
    A[i, :] = A[i, :] - factor * A[row, :]  (mod N)
where factor = A[i, col] * inv(A[row, col]) (mod N)
"""
function reduce_rows_below_gfn!(A::Matrix{Int}, row::Int, col::Int, N::Int)
    pivot = A[row, col]
    if pivot == 0
        error("Cannot reduce with zero pivot")
    end

    pivot_inv = mod_inverse(pivot, N)

    for i in (row + 1):size(A, 1)
        if A[i, col] != 0
            # factor = A[i, col] * pivot_inv (mod N)
            factor = mod_mul(A[i, col], pivot_inv, N)

            # A[i, :] = A[i, :] - factor * A[row, :] (mod N)
            for j in 1:size(A, 2)
                A[i, j] = mod_sub(A[i, j], mod_mul(factor, A[row, j], N), N)
            end
        end
    end

    return A
end

"""
    reduce_rows_above_gfn!(A, row, col, N)

Eliminate entries above pivot at (row, col) using GF(N) arithmetic.
Used for back-substitution to get reduced row echelon form.
"""
function reduce_rows_above_gfn!(A::Matrix{Int}, row::Int, col::Int, N::Int)
    pivot = A[row, col]
    if pivot == 0
        error("Cannot reduce with zero pivot")
    end

    pivot_inv = mod_inverse(pivot, N)

    for i in 1:(row - 1)
        if A[i, col] != 0
            factor = mod_mul(A[i, col], pivot_inv, N)
            for j in 1:size(A, 2)
                A[i, j] = mod_sub(A[i, j], mod_mul(factor, A[row, j], N), N)
            end
        end
    end

    return A
end

################################################
# Echelon Form
################################################

"""
    echelon_form_gfn!(A, N) -> (A, rank)

Compute row echelon form of matrix A over GF(N).
Modifies A in place and returns the rank.

Example for GF(3):
    A = [1 2 0; 2 1 1; 0 1 2]
    echelon_form_gfn!(A, 3)
    # A is now in row echelon form with all entries in {0, 1, 2}
"""
function echelon_form_gfn!(A::Matrix{Int}, N::Int)
    # Ensure all entries are in GF(N)
    A .= mod.(A, N)

    n_rows, n_cols = size(A)
    rank = 0
    pivot_col = 1

    for pivot_row in 1:n_rows
        if pivot_col > n_cols
            break
        end

        # Find non-zero entry in current column (at or below pivot row)
        found = false
        for i in pivot_row:n_rows
            if A[i, pivot_col] != 0
                if i != pivot_row
                    flip_rows_gfn!(A, [pivot_row, i])
                end
                found = true
                break
            end
        end

        if !found
            # No pivot in this column, try next column
            pivot_col += 1
            continue
        end

        # Reduce rows below pivot
        reduce_rows_below_gfn!(A, pivot_row, pivot_col, N)

        rank += 1
        pivot_col += 1
    end

    return A, rank
end

"""
    reduced_echelon_form_gfn!(A, N) -> (A, rank, pivot_cols)

Compute reduced row echelon form (RREF) of matrix A over GF(N).
Each pivot is normalized to 1, and entries above pivots are eliminated.

Returns the matrix, rank, and indices of pivot columns.
"""
function reduced_echelon_form_gfn!(A::Matrix{Int}, N::Int)
    # First get row echelon form
    A, rank = echelon_form_gfn!(A, N)

    if rank == 0
        return A, 0, Int[]
    end

    n_rows, n_cols = size(A)
    pivot_cols = Int[]

    # Find pivot columns and normalize pivots to 1
    pivot_row = 1
    for col in 1:n_cols
        if pivot_row > n_rows
            break
        end

        if A[pivot_row, col] != 0
            # Normalize pivot to 1
            pivot = A[pivot_row, col]
            if pivot != 1
                pivot_inv = mod_inverse(pivot, N)
                for j in 1:n_cols
                    A[pivot_row, j] = mod_mul(A[pivot_row, j], pivot_inv, N)
                end
            end

            # Eliminate above
            reduce_rows_above_gfn!(A, pivot_row, col, N)

            push!(pivot_cols, col)
            pivot_row += 1
        end
    end

    return A, rank, pivot_cols
end

################################################
# Linear System Solver
################################################

"""
    solve_gfn(A, b, N) -> (solution, success)

Solve the linear system A * x = b over GF(N).

Returns (x, true) if a solution exists, or (nothing, false) if inconsistent.
If multiple solutions exist (underdetermined), returns one particular solution.

Example:
    A = [1 1; 0 2]
    b = [2, 1]
    x, success = solve_gfn(A, b, 3)
    # x = [1, 1] (since 1+1=2, 2*1=2≡2 in GF(3))
"""
function solve_gfn(A::Matrix{Int}, b::Vector{Int}, N::Int)
    n_rows, n_cols = size(A)
    @assert length(b) == n_rows "Dimension mismatch"

    # Augmented matrix [A | b]
    Ab = hcat(A, reshape(b, :, 1))
    Ab = mod.(Ab, N)

    # Get RREF
    Ab, rank, pivot_cols = reduced_echelon_form_gfn!(Ab, N)

    # Check for inconsistency: if any row has all zeros in A but non-zero in b
    for i in 1:n_rows
        if all(Ab[i, 1:n_cols] .== 0) && Ab[i, n_cols + 1] != 0
            return nothing, false
        end
    end

    # Extract solution (free variables set to 0)
    x = zeros(Int, n_cols)
    for (i, col) in enumerate(pivot_cols)
        x[col] = Ab[i, n_cols + 1]
    end

    return x, true
end

"""
    solve_gfn_full(A, b, N) -> (particular, null_basis, success)

Solve A * x = b over GF(N) and return the full solution space.

Returns:
- particular: One particular solution (or nothing if inconsistent)
- null_basis: Basis vectors for the null space of A
- success: Whether a solution exists

The general solution is: x = particular + sum(c_i * null_basis[i]) for c_i ∈ GF(N)
"""
function solve_gfn_full(A::Matrix{Int}, b::Vector{Int}, N::Int)
    # First get a particular solution
    x_part, success = solve_gfn(A, b, N)
    if !success
        return nothing, Matrix{Int}(undef, 0, 0), false
    end

    # Find null space basis
    null_basis = nullspace_gfn(A, N)

    return x_part, null_basis, true
end

"""
    nullspace_gfn(A, N) -> Matrix{Int}

Compute a basis for the null space of A over GF(N).
Returns a matrix where each column is a basis vector.
"""
function nullspace_gfn(A::Matrix{Int}, N::Int)
    _, n_cols = size(A)

    # Get RREF
    A_rref = copy(A)
    A_rref, _, pivot_cols = reduced_echelon_form_gfn!(A_rref, N)

    # Free variables are non-pivot columns
    free_cols = setdiff(1:n_cols, pivot_cols)
    n_free = length(free_cols)

    if n_free == 0
        return Matrix{Int}(undef, n_cols, 0)
    end

    # Build null space basis
    null_basis = zeros(Int, n_cols, n_free)

    for (k, free_col) in enumerate(free_cols)
        # Set free variable to 1
        null_basis[free_col, k] = 1

        # Solve for pivot variables
        for (i, pivot_col) in enumerate(pivot_cols)
            # From RREF: A_rref[i, pivot_col] * x[pivot_col] + sum(...) = 0
            # x[pivot_col] = -sum(A_rref[i, j] * x[j] for j in free_cols)
            null_basis[pivot_col, k] = mod(-A_rref[i, free_col], N)
        end
    end

    return null_basis
end

################################################
# Utility Functions
################################################

"""
    rank_gfn(A, N) -> Int

Compute the rank of matrix A over GF(N).
"""
function rank_gfn(A::Matrix{Int}, N::Int)
    A_copy = copy(A)
    _, r = echelon_form_gfn!(A_copy, N)
    return r
end

"""
    is_linearly_independent_gfn(rows::Vector{Vector{Int}}, N::Int) -> Bool

Check if a set of row vectors are linearly independent over GF(N).
"""
function is_linearly_independent_gfn(rows::Vector{Vector{Int}}, N::Int)
    if isempty(rows)
        return true
    end

    n_rows = length(rows)
    A = reduce(vcat, [r' for r in rows])
    return rank_gfn(A, N) == n_rows
end

################################################
# Export
################################################

export mod_inverse, mod_sub, mod_mul
export echelon_form_gfn!, reduced_echelon_form_gfn!
export solve_gfn, solve_gfn_full, nullspace_gfn
export rank_gfn, is_linearly_independent_gfn
