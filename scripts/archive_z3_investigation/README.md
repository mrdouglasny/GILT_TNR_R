# Archive: Z3 + GILT-TNR Investigation Scripts

**Date:** January 7-8, 2026

**Purpose:** Debug scripts created during investigation of why TensorZ3 + GILT-TNR fails while plain tensors work.

## Investigation Summary

Two root causes were identified:
1. **Trace-based CDL identification** differs between charge and spin basis (Z3 has 3x more spurious modes)
2. **Rp.split() rotation ambiguity** - block-diagonal SVD gives different rotations than dense SVD

## Working Solution

The working fix is in the main scripts directory: `gilt_z3_fix.py` (v3 spin-basis approach)

See documentation in `ekrgilttrnr/docs/truncation_bug.tex` (Section 10-12)

## Key Scripts (for reference)

- `debug_rp_split.py` - Identified Rp.split() rotation issue
- `compare_z2_z3_gilt.py` - Showed why Z2 works but Z3 fails
- `trg_no_truncation.py` - Verified TRG step is correct
- `svd_basis_test.py` - Verified TensorZ3.svd() gives correct singular values

## Cleanup

These scripts are archived (not deleted) in case they're needed for future debugging.
