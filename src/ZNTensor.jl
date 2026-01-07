# MODULE: ZNTensor.jl
# USAGE: include("src/ZNTensor.jl") or using EKRNewton
# INPUTS: Python TensorZN objects from GiltTNR
# OUTPUTS: Julia ZNTensor{N} objects compatible with KrylovKit
# DESCRIPTION: Generalized Z_N symmetric tensor wrapper for Newton method
#
# BASED ON: KrylovTechnical.jl (Z2Tensor implementation)
# NEW IN THIS MODULE: Parametric ZNTensor{N} supporting arbitrary cyclic symmetry groups

"""
Generalized Z_N symmetric tensor for KrylovKit operations.

The tensor has block structure where each block corresponds to a charge sector.
For Z_N, charges are 0, 1, ..., N-1, and charge conservation requires:
    charge(leg1) + charge(leg2) ≡ charge(leg3) + charge(leg4) (mod N)

Fields:
- sects: Dictionary mapping charge tuples (q1,q2,q3,q4) to array blocks
- shape: Matrix of dimensions, shape[leg, charge_index] = dimension
- qhape: Matrix of charges, qhape[leg, :] = [0, 1, ..., N-1] (or subset)
- dirs: Vector of directions, dirs[leg] ∈ {-1, +1}
- qodulus: The N in Z_N (modulus for charge arithmetic)
"""
struct ZNTensor{N}
    sects::Dict{NTuple{4, Int64}, Array}
    shape::Matrix{Int64}
    qhape::Matrix{Int64}
    dirs::Vector{Int64}
end

# Accessor for the symmetry order
symmetry_order(::ZNTensor{N}) where N = N

# Type aliases for common cases
const Z2Tensor = ZNTensor{2}
const Z3Tensor = ZNTensor{3}
const Z4Tensor = ZNTensor{4}

################################################
# Python ↔ Julia Conversion
################################################

using PyCall

# Lazily initialized Python modules
const _py_modules = Ref{Any}(nothing)

function get_py_modules()
    if _py_modules[] === nothing
        pushfirst!(pyimport("sys")."path", "GiltTNR")
        tensors = pyimport("tensors")
        _py_modules[] = (
            tensors = tensors,
            TensorZ2 = tensors.TensorZ2,
            TensorZ3 = hasproperty(tensors, :TensorZ3) ? tensors.TensorZ3 : nothing,
            Tensor = tensors.Tensor,
        )
    end
    return _py_modules[]
end

"""
    py_to_zn(A::PyObject, ::Val{N}) -> ZNTensor{N}

Convert a Python TensorZN or plain Tensor to Julia ZNTensor{N}.

For plain tensors (from Tensor.from_ndarray), creates a trivial ZNTensor
with a single charge sector (0,0,0,0) containing the full array.
"""
function py_to_zn(A::PyObject, ::Val{N}) where N
    # Check if this is a plain tensor (no symmetry structure)
    if is_plain_tensor_py(A)
        return plain_to_zn(A, Val(N))
    end

    # Standard conversion for symmetric tensors
    return ZNTensor{N}(
        deepcopy(A.sects),
        deepcopy(A.shape),
        deepcopy(A.qhape),
        deepcopy(A.dirs)
    )
end

"""
    is_plain_tensor_py(A::PyObject) -> Bool

Check if A is a plain tensor (trivial charges).
"""
function is_plain_tensor_py(A::PyObject)
    try
        qhape = A.qhape
        # Plain tensors have qhape = [[0], [0], [0], [0]]
        for q in qhape
            if length(q) != 1 || q[1] != 0
                return false
            end
        end
        return true
    catch
        return true  # No qhape = plain
    end
end

"""
    plain_to_zn(A::PyObject, ::Val{N}) -> ZNTensor{N}

Convert a plain Python tensor to ZNTensor{N} with trivial charge structure.

The full tensor is stored in the (0,0,0,0) sector.
"""
function plain_to_zn(A::PyObject, ::Val{N}) where N
    arr = A.to_ndarray()
    d1, d2, d3, d4 = size(arr)

    # Create trivial sector structure
    sects = Dict{NTuple{4, Int64}, Array}()
    sects[(0, 0, 0, 0)] = arr

    # Shape: each leg has a single dimension for charge 0
    shape = zeros(Int64, 4, 1)
    shape[1, 1] = d1
    shape[2, 1] = d2
    shape[3, 1] = d3
    shape[4, 1] = d4

    # Qhape: each leg has only charge 0
    qhape = zeros(Int64, 4, 1)  # Already all zeros

    # Dirs: default outgoing (1)
    dirs = [1, 1, -1, -1]  # Standard TRG convention

    return ZNTensor{N}(sects, shape, qhape, dirs)
end

# Convenience methods
py_to_zn(A::PyObject, N::Int) = py_to_zn(A, Val(N))

# Backward compatible alias
py_to_ju(A::PyObject) = py_to_zn(A, Val(2))

"""
    zn_to_py(A::ZNTensor{N}) -> PyObject

Convert a Julia ZNTensor{N} back to Python TensorZN.

For trivial ZNTensors (single (0,0,0,0) sector), returns a plain Tensor.
"""
function zn_to_py(A::ZNTensor{N}) where N
    mods = get_py_modules()

    # Check if this is a trivial tensor (plain tensor stored as ZNTensor)
    if is_trivial_zn(A)
        return zn_to_plain_py(A)
    end

    # Choose appropriate Python tensor class
    TensorClass = if N == 2
        mods.TensorZ2
    elseif N == 3 && mods.TensorZ3 !== nothing
        mods.TensorZ3
    else
        # For generic N or if TensorZ3 not available, use plain Tensor
        # and reconstruct as needed
        mods.Tensor
    end

    if N == 2
        out = TensorClass(
            A.shape,
            qhape = A.qhape,
            dirs = A.dirs,
            invar = true,
            charge = 0,
        )
    else
        # For N > 2, need qodulus parameter
        out = TensorClass(
            A.shape,
            qhape = A.qhape,
            dirs = A.dirs,
            invar = true,
            charge = 0,
            qodulus = N,
        )
    end

    # Copy sector data
    py"""
    import copy
    a = $out
    for k, v in $(A.sects).items():
        a[k] = copy.deepcopy(v)
    """

    return out
end

"""
    is_trivial_zn(A::ZNTensor) -> Bool

Check if A has trivial charge structure (single (0,0,0,0) sector).
"""
function is_trivial_zn(A::ZNTensor)
    # Check if qhape is trivial (all zeros, single charge per leg)
    if size(A.qhape, 2) != 1
        return false
    end
    if any(A.qhape .!= 0)
        return false
    end
    # Check if we have only the (0,0,0,0) sector
    return length(A.sects) == 1 && haskey(A.sects, (0, 0, 0, 0))
end

"""
    zn_to_plain_py(A::ZNTensor) -> PyObject

Convert a trivial ZNTensor back to a plain Python Tensor.
"""
function zn_to_plain_py(A::ZNTensor)
    mods = get_py_modules()

    # Extract the array from the (0,0,0,0) sector
    arr = A.sects[(0, 0, 0, 0)]

    # Create plain Python Tensor
    # Use pycall to prevent auto-conversion
    return pycall(mods.Tensor.from_ndarray, PyObject, arr)
end

# Backward compatible alias
ju_to_py(A::Z2Tensor) = zn_to_py(A)

################################################
# Helper Functions
################################################

"""
Check if a charge tuple satisfies Z_N charge conservation.
"""
function is_valid_charge(charges::NTuple{4, Int64}, dirs::Vector{Int64}, N::Int)
    # Charge conservation: Σ_i dirs[i] * charges[i] ≡ 0 (mod N)
    total = sum(dirs[i] * charges[i] for i in 1:4)
    return mod(total, N) == 0
end

"""
Get block dimension for a given charge on a given leg.
"""
function key_to_shape(shape::Matrix{Int64}, qhape::Matrix{Int64}, charge::Int, leg::Int)
    index = findfirst(x -> x == charge, qhape[leg, :])
    return index === nothing ? 0 : shape[leg, index]
end

"""
Extend tensor blocks with zeros to a new shape.
Used when adding tensors with different bond dimensions.
"""
function extend_blocks_by_zeros(A::ZNTensor{N}, new_shape::Matrix{Int64}) where N
    extended_sects = Dict{NTuple{4, Int64}, Array}()

    for (key, block) in A.sects
        # Compute new dimensions for this sector
        new_dims = ntuple(4) do leg
            key_to_shape(new_shape, A.qhape, key[leg], leg)
        end

        # Skip if any dimension is zero (charge not present)
        if any(d == 0 for d in new_dims)
            continue
        end

        # Create extended block
        tmp = zeros(eltype(block), new_dims...)
        old_dims = size(block)
        tmp[1:old_dims[1], 1:old_dims[2], 1:old_dims[3], 1:old_dims[4]] .= block
        extended_sects[key] = tmp
    end

    return ZNTensor{N}(extended_sects, new_shape, A.qhape, A.dirs)
end

"""
Compute element-wise maximum shape for two tensors.
"""
function common_shape(v::ZNTensor{N}, w::ZNTensor{N}) where N
    return max.(v.shape, w.shape)
end

################################################
# Random Tensor Generation
################################################

py"""
import numpy as np

def random_with_sign(*args, **kwargs):
    return 2*np.random.random_sample(*args, **kwargs) - 1
"""

"""
    random_zntens(A::PyObject, ::Val{N}) -> PyObject

Generate a random ZNTensor with same structure as A.
Returns a Python object (for compatibility with existing code).
"""
function random_zntens(A::PyObject, ::Val{N}) where N
    mods = get_py_modules()
    dims = A.shape
    qhape = A.qhape
    dirs = A.dirs

    if N == 2
        t = mods.TensorZ2.initialize_with(
            py"random_with_sign", dims,
            qhape = qhape, charge = 0, invar = true, dirs = dirs
        )
    else
        # For N > 2, initialize manually
        t = mods.Tensor.initialize_with(
            py"random_with_sign", dims,
            qhape = qhape, charge = 0, invar = true, dirs = dirs,
            qodulus = N
        )
    end
    t = t / t.norm()
    return t
end

"""
    random_zntens(chi::Int, ::Val{N}) -> PyObject

Generate a random ZNTensor with bond dimension chi per sector.
"""
function random_zntens(chi::Int, ::Val{N}) where N
    mods = get_py_modules()

    # Shape: chi per sector, N sectors
    dims = fill(chi, 4, N)
    # Charges: 0, 1, ..., N-1
    qhape = hcat([fill(i, 4) for i in 0:(N-1)]...)
    # Directions: standard TRG convention
    dirs = [-1, 1, 1, -1]

    if N == 2
        t = mods.TensorZ2.initialize_with(
            py"random_with_sign", dims,
            qhape = qhape, charge = 0, invar = true, dirs = dirs
        )
    else
        t = mods.Tensor.initialize_with(
            py"random_with_sign", dims,
            qhape = qhape, charge = 0, invar = true, dirs = dirs,
            qodulus = N
        )
    end
    t = t / t.norm()
    return t
end

"""
    random_zntens(A::ZNTensor{N}, ::Val{N}) -> ZNTensor{N}

Generate a random ZNTensor with same structure as A (Julia ZNTensor).
For trivial tensors, creates a random array with matching dimensions.
"""
function random_zntens(A::ZNTensor{N}, ::Val{M}) where {N, M}
    @assert N == M "Symmetry order mismatch: tensor is Z_$N but Val($M) specified"

    # Check if trivial (plain tensor)
    if is_trivial_zn(A)
        # For trivial tensors, create random array matching dimensions
        arr = A.sects[(0, 0, 0, 0)]
        dims = size(arr)
        rand_arr = randn(dims)
        rand_arr ./= sqrt(sum(rand_arr .^ 2))  # Normalize

        return ZNTensor{N}(
            Dict{NTuple{4, Int64}, Array}((0, 0, 0, 0) => rand_arr),
            copy(A.shape),
            copy(A.qhape),
            copy(A.dirs)
        )
    end

    # For symmetric tensors, create random tensor with same structure
    new_sects = Dict{NTuple{4, Int64}, Array}()
    total_norm_sq = 0.0

    for (k, v) in A.sects
        rand_block = randn(size(v))
        new_sects[k] = rand_block
        total_norm_sq += sum(rand_block .^ 2)
    end

    # Normalize
    norm_factor = sqrt(total_norm_sq)
    for k in keys(new_sects)
        new_sects[k] ./= norm_factor
    end

    return ZNTensor{N}(new_sects, copy(A.shape), copy(A.qhape), copy(A.dirs))
end

# Convenience aliases
random_Z2tens(A::PyObject) = random_zntens(A, Val(2))
random_Z2tens(chi::Int) = random_zntens(chi, Val(2))
random_Z3tens(A::PyObject) = random_zntens(A, Val(3))
random_Z3tens(chi::Int) = random_zntens(chi, Val(3))
random_Z2tens(A::ZNTensor{2}) = random_zntens(A, Val(2))
random_Z3tens(A::ZNTensor{3}) = random_zntens(A, Val(3))

################################################
# Base Operations
################################################

import Base: getindex, +, -, *, /, similar, zero, real, imag

function Base.getindex(A::ZNTensor{N}, ind...) where N
    return A.sects[ind]
end

function Base.getindex(A::ZNTensor{N}, ind::NTuple{4, Int64}) where N
    return A.sects[ind]
end

function Base.:+(v::ZNTensor{N}, w::ZNTensor{N}) where N
    if v.shape == w.shape
        # Fast path: shapes match
        result = deepcopy(v)
        for (k, block) in result.sects
            block .+= w[k]
        end
        return result
    end

    # Extend both to common shape
    new_shape = common_shape(v, w)
    v_ext = extend_blocks_by_zeros(v, new_shape)
    w_ext = extend_blocks_by_zeros(w, new_shape)

    result = deepcopy(v_ext)
    for (k, block) in result.sects
        if haskey(w_ext.sects, k)
            block .+= w_ext[k]
        end
    end
    return result
end

function Base.:-(v::ZNTensor{N}, w::ZNTensor{N}) where N
    if v.shape == w.shape
        result = deepcopy(v)
        for (k, block) in result.sects
            block .-= w[k]
        end
        return result
    end

    new_shape = common_shape(v, w)
    v_ext = extend_blocks_by_zeros(v, new_shape)
    w_ext = extend_blocks_by_zeros(w, new_shape)

    result = deepcopy(v_ext)
    for (k, block) in result.sects
        if haskey(w_ext.sects, k)
            block .-= w_ext[k]
        end
    end
    return result
end

function Base.:/(x::ZNTensor{N}, num) where N
    sects = Dict{NTuple{4, Int64}, Array}()
    for (k, block) in x.sects
        sects[k] = block / num
    end
    return ZNTensor{N}(sects, x.shape, x.qhape, x.dirs)
end

function Base.:*(α, x::ZNTensor{N}) where N
    sects = Dict{NTuple{4, Int64}, Array}()
    for (k, block) in x.sects
        sects[k] = block * α
    end
    return ZNTensor{N}(sects, x.shape, x.qhape, x.dirs)
end

function Base.:*(v::ZNTensor{N}, α) where N
    return α * v
end

function Base.similar(x::ZNTensor{N}) where N
    sects = Dict{NTuple{4, Int64}, Array}()
    for (k, block) in x.sects
        sects[k] = similar(block)
    end
    return ZNTensor{N}(sects, copy(x.shape), copy(x.qhape), copy(x.dirs))
end

function Base.zero(x::ZNTensor{N}) where N
    sects = Dict{NTuple{4, Int64}, Array}()
    for (k, block) in x.sects
        sects[k] = zero(block)
    end
    return ZNTensor{N}(sects, copy(x.shape), copy(x.qhape), copy(x.dirs))
end

function Base.real(x::ZNTensor{N}) where N
    sects = Dict{NTuple{4, Int64}, Array}()
    for (k, block) in x.sects
        sects[k] = real.(block)
    end
    return ZNTensor{N}(sects, x.shape, x.qhape, x.dirs)
end

function Base.imag(x::ZNTensor{N}) where N
    sects = Dict{NTuple{4, Int64}, Array}()
    for (k, block) in x.sects
        sects[k] = imag.(block)
    end
    return ZNTensor{N}(sects, x.shape, x.qhape, x.dirs)
end

################################################
# KrylovKit Required Operations
################################################

import LinearAlgebra: mul!, rmul!, axpy!, axpby!, dot, norm

function LinearAlgebra.mul!(w::ZNTensor{N}, v::ZNTensor{N}, α) where N
    # Handle shape mismatch
    if w.shape != v.shape
        new_shape = v.shape
        for (k, block) in w.sects
            expected_size = ntuple(4) do i
                key_to_shape(new_shape, w.qhape, k[i], i)
            end
            if size(block) != expected_size && all(s > 0 for s in expected_size)
                w.sects[k] = zeros(eltype(block), expected_size)
            end
        end
        w.shape .= new_shape
    end

    for (k, block) in w.sects
        if haskey(v.sects, k)
            v_block = v.sects[k]
            if size(block) == size(v_block)
                block .= v_block * α
            else
                w.sects[k] = v_block * α
            end
        else
            block .= 0
        end
    end
    return w
end

function LinearAlgebra.rmul!(v::ZNTensor{N}, α) where N
    for (_, block) in v.sects
        block .*= α
    end
    return v
end

function LinearAlgebra.axpy!(α, v::ZNTensor{N}, w::ZNTensor{N}) where N
    if w.shape != v.shape
        new_shape = max.(w.shape, v.shape)
        for (k, block) in w.sects
            expected_size = ntuple(4) do i
                key_to_shape(new_shape, w.qhape, k[i], i)
            end
            if size(block) != expected_size && all(s > 0 for s in expected_size)
                new_block = zeros(eltype(block), expected_size)
                old_size = size(block)
                new_block[1:old_size[1], 1:old_size[2], 1:old_size[3], 1:old_size[4]] .= block
                w.sects[k] = new_block
            end
        end
        w.shape .= new_shape
    end

    for (k, block) in w.sects
        if haskey(v.sects, k)
            v_block = v.sects[k]
            if size(block) == size(v_block)
                block .+= α * v_block
            else
                v_size = size(v_block)
                block[1:v_size[1], 1:v_size[2], 1:v_size[3], 1:v_size[4]] .+= α * v_block
            end
        end
    end
    return w
end

function LinearAlgebra.axpby!(α, v::ZNTensor{N}, β, w::ZNTensor{N}) where N
    for (k, block) in w.sects
        if haskey(v.sects, k)
            block .= α * v.sects[k] + β * block
        else
            block .*= β
        end
    end
    return w
end

function LinearAlgebra.dot(v::ZNTensor{N}, w::ZNTensor{N}) where N
    if v.shape == w.shape
        res = 0.0
        for (k, v_block) in v.sects
            if haskey(w.sects, k)
                res += dot(v_block, w[k])
            end
        end
        return res
    end

    # Extend to common shape for correct inner product
    new_shape = common_shape(v, w)
    v_ext = extend_blocks_by_zeros(v, new_shape)
    w_ext = extend_blocks_by_zeros(w, new_shape)

    res = 0.0
    for (k, v_block) in v_ext.sects
        if haskey(w_ext.sects, k)
            res += dot(v_block, w_ext[k])
        end
    end
    return res
end

function LinearAlgebra.norm(v::ZNTensor{N}) where N
    nrmsq = 0.0
    for (_, block) in v.sects
        nrmsq += norm(block)^2
    end
    return sqrt(nrmsq)
end

################################################
# Export
################################################

export ZNTensor, Z2Tensor, Z3Tensor, Z4Tensor
export symmetry_order
export py_to_zn, zn_to_py, py_to_ju, ju_to_py
export random_zntens, random_Z2tens, random_Z3tens
export extend_blocks_by_zeros, common_shape, key_to_shape
export is_valid_charge
