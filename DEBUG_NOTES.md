# EKR Gilt-TNR Debugging Notes

## Environment

- **Target:** Julia 1.10.4 + NumPy 1.x (original development environment)
- **Current:** Julia 1.12.3 + NumPy 2.2.6
- **Status:** ⚠️ Partial - significant NumPy 2.x compatibility issues

## Issues Found & Fixed

### 1. NumPy `np.float_` Removal ✅ FIXED
- **Error:** `AttributeError: np.float_ was removed in NumPy 2.0`
- **Files:** `tensors/abeliantensor.py`, `tensors/symmetrytensors.py`, `plots/plotFError.py`
- **Fix:** Global replacement `np.float_` → `np.float64`
- **Status:** ✅ Complete

### 2. CairoMakie Julia 1.12 Incompatibility ✅ WORKED AROUND
- **Error:** `FieldError: type Core.TypeName has no field mt`
- **File:** `src/Tools.jl`
- **Fix:** Commented out `using CairoMakie` and `import CairoMakie: lines, scatter`
- **Impact:** Plotting functions unavailable, but eigenvalue computation doesn't need them
- **Status:** ✅ Workaround implemented

### 3. Missing Data Files ✅ WORKED AROUND
- **Files:** `IsingExactLevels`, `IsingEvenExactLevels`
- **Fix:** Commented out `deserialize()` calls in `Tools.jl`
- **Impact:** Exact spectrum comparison functions unavailable
- **Status:** ✅ Workaround implemented

### 4. Tensor `invar` Property Loss ⚠️ PARTIAL FIX
- **Error:** `AssertionError` in `AbelianTensor.matrix_dot: assert self.invar and other.invar`
- **Root Cause:** Multiple NumPy 2.x behavioral changes affecting tensor operations:
  1. `conjugate()` method returns plain `ndarray`, losing `invar` flag
  2. GILT operations create tensors with `invar=False`
  3. Possible deeper incompatibilities in tensor contraction library

- **Attempted Fix:**
  ```python
  def conjugate(self):
      """Return the complex conjugate, preserving tensor structure."""
      new_sects = {k: np.conj(v) for k, v in self.sects.items()}
      return self.__class__(
          shape=self.shape, qhape=self.qhape, qodulus=self.qodulus,
          sects=new_sects, dtype=self.dtype, defval=np.conj(self.defval),
          invar=self.invar,  # CRITICAL: Preserve invar flag
          charge=self.charge, dirs=self.dirs
      )
  ```

- **Status:** ⚠️ Partial - `conjugate()` fix works in isolation but GILT operations still create tensors with `invar=False`

## Call Stack Where Failure Occurs

```
trajectory()
  → initial_tensor()
    → gilttnr_step()  [Python]
      → gilt_plaq()
        → apply_gilt()
          → get_envspec()
            → ncon((A1, A1.conjugate()), ...)
              → A1.dot(A1.conjugate())
                → matrix_dot()
                  → assert self.invar and other.invar  ❌ FAILS HERE
```

**Problem:** `A1` already has `invar=False` before calling `conjugate()`. This suggests GILT operations are creating non-invariant tensors.

## Hypothesis

The GiltTNR library has subtle dependencies on NumPy 1.x behavior that break in NumPy 2.x:
1. Array creation/manipulation may initialize `invar` differently
2. Tensor slicing/reshaping operations may not preserve tensor class properties
3. The `ncon` (tensor contraction) library may have NumPy 2.x issues

## Recommended Path Forward

### Option 1: Use Original Environment (RECOMMENDED)
Create a Julia 1.10.4 + NumPy 1.x environment:
```bash
# Use juliaup to install Julia 1.10.4
juliaup add 1.10.4
juliaup default 1.10.4

# Create Python venv with NumPy 1.x
python3 -m venv venv-numpy1
source venv-numpy1/bin/activate
pip install 'numpy<2.0' scipy matplotlib
```

### Option 2: Deep Debugging (EFFORT-INTENSIVE)
- Trace all GILT operations to find where `invar` is lost
- Add `invar` preservation to all tensor creation/manipulation functions
- May require dozens of fixes across the codebase

### Option 3: Alternative Implementation
- Use our `genmodel/scripts/linearized_rg.jl` instead
- Compare with EKR results from their published data
- Avoid dependency hell entirely

## Files Modified

1. `/workspaces/rg/ekrgilttrnr/src/GiltTNR/tensors/abeliantensor.py` - Added `conjugate()` override
2. `/workspaces/rg/ekrgilttrnr/src/GiltTNR/tensors/symmetrytensors.py` - `np.float_` → `np.float64`
3. `/workspaces/rg/ekrgilttrnr/src/GiltTNR/plots/plotFError.py` - `np.float_` → `np.float64`
4. `/workspaces/rg/ekrgilttrnr/src/Tools.jl` - Commented out CairoMakie, exact spectrum data
5. `/workspaces/rg/ekrgilttrnr/src/test_setup.jl` - Test script for validation

## Test Results

✅ **Working:**
- Julia dependencies load (with warnings)
- Python GiltTNR library imports via PyCall
- Critical temperature finder returns `relT = 0.0`
- `conjugate()` method preserves `invar` in isolation

❌ **Not Working:**
- Full trajectory generation (GILT operations create non-invariant tensors)
- Eigenvalue computation (depends on trajectory)

## Next Steps

1. **For production use:** Set up Julia 1.10.4 + NumPy 1.x environment
2. **For comparison:** Run EKR code in original environment and save eigenvalue results
3. **For validation:** Compare with `genmodel/scripts/linearized_rg.jl` results

---
*Last Updated: 2026-01-01*
*Debugged by: Claude (Sonnet 4.5)*
