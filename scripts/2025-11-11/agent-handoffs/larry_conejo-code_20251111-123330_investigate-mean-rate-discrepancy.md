# Handoff: Investigate Mean Rate Discrepancy

**From:** larry  
**To:** conejo-code  
**Date:** 2025-11-11 12:33:30  
**Priority:** High  
**Status:** Pending

## Context

Both conejo-code and mari-test have completed validation of the MATLAB reference method adaptation. While validation passed overall, a significant discrepancy in mean turn rate has been identified that requires full investigation and complete documentation before integration proceeds.

## Task

**Task 0.4: Investigate Mean Rate Discrepancy**

Fully investigate and document the mean turn rate discrepancy between expected (~0.04 turns/min) and measured (0.26 turns/min) values. This investigation must be completed and documented accurately and completely before integration proceeds.

## Discrepancy Details

**Expected Value:** ~0.04 turns/min (from conejo-code's handoff document)  
**Measured Value:** 0.26 turns/min (from mari-test's validation)  
**Difference:** 6.5x higher than expected  
**Impact:** Medium severity - requires investigation before integration

**Both validators note:**
- conejo-code's handoff: Expected ~0.04 turns/min
- mari-test's validation: Found 0.26 turns/min
- Both recommend investigating before integration

## Investigation Requirements

### 1. Verify Expected Value Against MATLAB Output

**Objective:** Confirm what the actual MATLAB code produces

**Tasks:**
- Run original MATLAB code (`load_multi_data_gr21a.m`) on same test file
- Extract exact mean turn rate from MATLAB output
- Compare with Python adaptation output (0.26 turns/min)
- Document MATLAB output value

**Test File:**
```
data/GMR61@GMR61_T_Re_Sq_0to100PWM_30#C_Bl_7PWM_202508221246.h5
```

**MATLAB Reference:**
```
D:\Matlab-Track-Analysis-MirnaLab\user specific\Silverio\load_multi_data_gr21a.m
```

### 2. Check Normalization Factors

**Objective:** Verify all normalization steps are correct

**Key Calculations to Verify:**
- `rate_from_time(turnStart_total, tperiod, stepsize, binsize) ./ double(nperiod) * 60`
- Division by `nperiod` (960 periods)
- Multiplication by 60 (conversion to turns/min)
- Binsize (0.5s) and stepsize (0.1s) parameters

**Tasks:**
- Review `rate_from_time()` function implementation
- Verify division by `nperiod` is correct
- Verify multiplication by 60 is correct
- Check binsize and stepsize usage
- Calculate expected value manually:
  - 62 turns / 960 periods = 0.0646 turns/period
  - Convert to turns/min (verify calculation)

### 3. Verify Period Calculation

**Objective:** Ensure `nperiod` calculation matches MATLAB

**Current Value:** 960.0 periods

**Tasks:**
- Compare `nperiod` calculation with MATLAB output
- Verify track overlap calculation
- Check if period length (15s) is correct
- Verify sum of track overlaps divided by tperiod

**MATLAB Pattern:**
```matlab
nperiod = nperiod + (min(t(j).endFrame, t_stim_end(i)*frame_rate) - max(t(j).startFrame, t_stim_start(i)*frame_rate)) / frame_rate / tperiod
```

### 4. Validate LED Timing Interpolation

**Objective:** Ensure LED timing values are correct

**Tasks:**
- Verify `led12Val_toff` interpolation matches MATLAB
- Check period-relative time calculation
- Verify values are in [0, 15] range
- Check interpolation accuracy

### 5. Check `rate_from_time()` Implementation

**Objective:** Verify Python implementation matches MATLAB exactly

**Tasks:**
- Compare Python implementation with MATLAB function
- Verify periodic boundary conditions
- Verify binning logic matches exactly
- Check if normalization is applied correctly
- Review all mathematical operations

**MATLAB Reference:** Need to find `rate_from_time()` function definition

### 6. Document Findings Completely

**Objective:** Provide comprehensive documentation

**Required Documentation:**
- Detailed investigation report
- Root cause of discrepancy
- Whether discrepancy is expected or a bug
- All calculations shown step-by-step
- MATLAB comparison results
- Code review findings
- Numerical analysis
- Clear recommendations for resolution

## Deliverables

1. **Investigation Report**
   - File: `docs/logs/2025-11-11/mean-rate-discrepancy-investigation.md`
   - Must include:
     - Executive summary
     - MATLAB comparison results
     - Code review findings
     - Numerical analysis
     - Root cause identification
     - Resolution recommendations
     - Complete calculations

2. **Updated Validation Report**
   - Update `docs/logs/2025-11-11/matlab-reference-validation.md` with investigation findings
   - Document discrepancy resolution

3. **Documentation of Resolution**
   - Clear explanation of whether discrepancy is expected or requires fix
   - If expected: explain why (normalization differences, etc.)
   - If bug: provide fix and verify correction

## Success Criteria

- ✅ Root cause of discrepancy identified
- ✅ Discrepancy fully explained (expected vs bug)
- ✅ All calculations verified
- ✅ MATLAB comparison completed
- ✅ Complete documentation provided
- ✅ Clear recommendation for resolution
- ✅ Investigation report ready for integration decision

## Blocking Status

**This task blocks:**
- Task 0.2 integration into `engineer_dataset_from_h5.py`
- All downstream analysis work
- Final integration decision

**Integration cannot proceed until:**
- Investigation complete
- Discrepancy fully documented
- Root cause identified
- Resolution approach determined

## Questions to Answer

1. **What does MATLAB actually produce?**
   - Run MATLAB code and get exact mean rate value
   - Compare with Python output (0.26 turns/min)

2. **Is the discrepancy expected?**
   - Due to normalization differences?
   - Due to calculation method differences?
   - Or is it a bug?

3. **What is the correct normalization?**
   - Should it be 0.04 or 0.26?
   - What does MATLAB use?
   - What should Python use?

4. **Is the Python implementation correct?**
   - Does it match MATLAB exactly?
   - Are there any calculation errors?
   - Are normalization factors correct?

## Next Steps

1. **conejo-code**: Start investigation immediately
2. **conejo-code**: Run MATLAB code on test file
3. **conejo-code**: Compare with Python output
4. **conejo-code**: Review all calculations
5. **conejo-code**: Document findings completely
6. **conejo-code**: Create investigation report
7. **larry**: Review investigation report
8. **larry**: Make integration decision based on findings

## Support

**MATLAB Reference:**
- `D:\Matlab-Track-Analysis-MirnaLab\user specific\Silverio\load_multi_data_gr21a.m`
- Line 325: `rate_from_time(turnStart_total, tperiod, stepsize, binsize) ./ double(nperiod) * 60`

**Python Implementation:**
- `scripts/2025-11-11/adapt_matlab_turnrate_method.py`
- `scripts/2025-11-11/compare_turnrate_methods.py`

**Test File:**
- `data/GMR61@GMR61_T_Re_Sq_0to100PWM_30#C_Bl_7PWM_202508221246.h5`

**Validation Reports:**
- `docs/logs/2025-11-11/matlab-reference-validation.md` (conejo-code)
- `docs/logs/2025-11-11/mari-test_matlab-validation-report.md` (mari-test)

---

**larry** 🎖️  
**Date:** 2025-11-11 12:33:30  
**Priority:** High - Blocks integration until complete

