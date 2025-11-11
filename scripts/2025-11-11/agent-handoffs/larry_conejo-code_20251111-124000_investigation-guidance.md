# Investigation Guidance: Mean Rate Discrepancy

**From:** larry  
**To:** conejo-code  
**Date:** 2025-11-11 12:40:00  
**Priority:** High  
**Status:** In Progress

## Current Findings

**Initial Investigation Results:**
1. ✅ Normalization calculation mismatch identified
2. ⚠️ Period calculation issue: `nperiod=1.0` instead of expected value (960.0)
3. ⚠️ Values don't align as expected

## Investigation Priority Order

### Step 1: Fix Period Calculation (CRITICAL)

**Issue:** `nperiod=1.0` is clearly wrong - should be ~960.0

**Why this matters:**
- If `nperiod=1.0`, then normalization divides by 1 instead of 960
- This would explain the 6.5x difference (if actual nperiod should be ~960)
- Period calculation is fundamental to the rate calculation

**Investigation Tasks:**
1. **Review period calculation code:**
   - Find where `nperiod` is calculated in `adapt_matlab_turnrate_method.py`
   - Check `calculate_aggregate_turnrate_matlab_method()` function
   - Verify track overlap calculation

2. **Compare with MATLAB logic:**
   ```matlab
   nperiod = nperiod + (min(t(j).endFrame, t_stim_end(i)*frame_rate) - max(t(j).startFrame, t_stim_start(i)*frame_rate)) / frame_rate / tperiod
   ```
   - Verify Python implementation matches this exactly
   - Check if `t_stim_start` and `t_stim_end` are correct
   - Verify `frame_rate` and `tperiod` values

3. **Debug period calculation:**
   - Add print statements to show:
     - Track start/end frames
     - Stimulus start/end times
     - Frame rate
     - Period length (tperiod)
     - Track overlap calculation per track
     - Sum of overlaps
     - Final nperiod value

4. **Test with known values:**
   - Use test file: `GMR61@GMR61_T_Re_Sq_0to100PWM_30#C_Bl_7PWM_202508221246.h5`
   - Expected: ~960.0 periods
   - Verify calculation step-by-step

**Expected Fix:**
- Once `nperiod` is corrected to ~960.0, the normalization should work correctly
- Mean rate should drop from 0.26 to ~0.04 turns/min (if that's the correct expected value)

### Step 2: Verify Normalization After Period Fix

**After fixing period calculation:**
1. **Recalculate turn rate:**
   - Run `calculate_aggregate_turnrate_matlab_method()` again
   - Check if mean rate is now ~0.04 turns/min

2. **Verify normalization formula:**
   ```python
   turnrate = rate_from_time(...) ./ nperiod * 60
   ```
   - Verify division by `nperiod` happens correctly
   - Verify multiplication by 60 (conversion to turns/min)
   - Check if `rate_from_time()` output is correct

3. **Compare with MATLAB:**
   - If MATLAB output available, compare directly
   - Verify normalization matches MATLAB exactly

### Step 3: Compare with MATLAB Output

**If MATLAB code available:**
1. **Run MATLAB code:**
   - Use same test file
   - Extract exact mean rate value
   - Extract `nperiod` value
   - Extract `rate_from_time()` output before normalization

2. **Compare values:**
   - Python `nperiod` vs MATLAB `nperiod`
   - Python `rate_from_time()` vs MATLAB `rate_from_time()`
   - Python final rate vs MATLAB final rate

3. **Document differences:**
   - Any differences in calculation
   - Any differences in normalization
   - Root cause of any discrepancies

### Step 4: Create Investigation Report

**Report Structure:**
1. **Executive Summary:**
   - Discrepancy identified
   - Root cause found
   - Resolution approach

2. **Period Calculation Investigation:**
   - Issue found: `nperiod=1.0` instead of ~960.0
   - Root cause analysis
   - Fix applied
   - Verification results

3. **Normalization Verification:**
   - Normalization calculation after period fix
   - Comparison with expected values
   - Verification of formula

4. **MATLAB Comparison (if available):**
   - MATLAB output values
   - Python output values
   - Comparison and differences

5. **Resolution:**
   - Was discrepancy expected or a bug?
   - Final mean rate value
   - Recommendations for integration

## Debugging Strategy

### Add Debug Output

**In `calculate_aggregate_turnrate_matlab_method()`:**
```python
# Debug period calculation
print(f"Track {track_key}:")
print(f"  Start frame: {start_frame}")
print(f"  End frame: {end_frame}")
print(f"  Stimulus start: {t_stim_start} s")
print(f"  Stimulus end: {t_stim_end} s")
print(f"  Frame rate: {frame_rate} fps")
print(f"  Period length: {tperiod} s")
print(f"  Track overlap: {overlap} s")
print(f"  Periods contributed: {overlap / tperiod}")

print(f"\nTotal nperiod: {nperiod}")
print(f"Expected: ~960.0")
```

### Test Cases

**Create test cases:**
1. **Single track test:**
   - One track, known start/end frames
   - Calculate expected nperiod
   - Verify calculation

2. **Multiple tracks test:**
   - All tracks from test file
   - Calculate total nperiod
   - Verify equals ~960.0

3. **Edge cases:**
   - Track starts before stimulus
   - Track ends after stimulus
   - Track entirely within stimulus
   - Track entirely outside stimulus

## Immediate Next Steps

1. **conejo-code:** Focus on period calculation first
   - Find where `nperiod=1.0` is coming from
   - Fix the calculation
   - Verify `nperiod` is now ~960.0

2. **conejo-code:** After period fix, recalculate turn rate
   - Check if mean rate is now correct
   - Verify normalization works

3. **conejo-code:** Document findings
   - Create investigation report
   - Document root cause
   - Document fix
   - Document verification

4. **conejo-code:** Compare with MATLAB (if possible)
   - Run MATLAB code
   - Compare values
   - Document differences

## Questions to Answer

1. **Why is `nperiod=1.0`?**
   - Is it initialized incorrectly?
   - Is the calculation wrong?
   - Are track overlaps not being summed?

2. **What should `nperiod` be?**
   - Expected: ~960.0 periods
   - How is this calculated?
   - Does it match MATLAB?

3. **After fixing `nperiod`, what is the mean rate?**
   - Should be ~0.04 turns/min (if that's correct)
   - Or should it be 0.26 turns/min (if that's correct)?
   - Need MATLAB comparison to verify

## Success Criteria

- ✅ `nperiod` calculation fixed and verified (~960.0)
- ✅ Turn rate recalculated with correct `nperiod`
- ✅ Mean rate matches expected value (or discrepancy explained)
- ✅ Root cause identified and documented
- ✅ Investigation report complete
- ✅ Clear recommendation for integration

## Support

**Key Files:**
- `scripts/2025-11-11/adapt_matlab_turnrate_method.py` - Main implementation
- `scripts/2025-11-11/compare_turnrate_methods.py` - Comparison script
- `D:\Matlab-Track-Analysis-MirnaLab\user specific\Silverio\load_multi_data_gr21a.m` - MATLAB reference

**Test File:**
- `data/GMR61@GMR61_T_Re_Sq_0to100PWM_30#C_Bl_7PWM_202508221246.h5`

---

**larry** 🎖️  
**Date:** 2025-11-11 12:40:00  
**Priority:** High - Focus on period calculation first

