# Handoff: Next Steps for MATLAB Reference Validation

**From:** larry  
**To:** conejo-code  
**Date:** 2025-11-11 12:10:00  
**Priority:** High  
**Status:** In Progress

## Context

Excellent progress on Task 0.3. Dependencies computation is complete and all MATLAB method dependencies are now saved in H5 files. Ready to proceed with next phase of validation.

## Current Status

**Completed:**
- ✅ All MATLAB dependencies computed and saved to H5 files
- ✅ Global quantities: led12Val, led12Val_ton, led12Val_toff
- ✅ Track-level derived quantities: led12Val_ton, led12Val_toff (per frame)
- ✅ elapsedTime array saved
- ✅ Verification scripts created

**Next Steps (from dependencies_computation_summary.md):**
1. ✅ Compute and save dependencies - **COMPLETE**
2. ⏳ Extract reorientation data from trajectories CSV
3. ⏳ Filter by numHS >= 1 using Klein run table
4. ⏳ Run comparison test between MATLAB method and existing pipeline
5. ⏳ Document findings and create validation report

## Task/Instructions

### Immediate Next Steps

**Step 1: Extract Reorientation Data from Trajectories CSV**

MATLAB reference uses:
```matlab
turnStartTime = t(j).getSubFieldDQ('reorientation', 'eti', 'indsExpression', '[track.reorientation.numHS] >= 1', 'position', 'start');
```

This extracts:
- Reorientation start times (ETI)
- Filtered by `numHS >= 1` (at least one head swing)
- Position: 'start' (beginning of reorientation)

**Implementation approach:**
- Load trajectories CSV (has frame-level reorientation data)
- Load Klein run table (has numHS information)
- Match reorientation events with numHS >= 1
- Extract start times (ETI) for matching events
- Use saved `led12Val_toff` timing from H5 for period-relative times

**Step 2: Filter by numHS >= 1**

MATLAB uses `indsExpression: '[track.reorientation.numHS] >= 1'` to filter reorientations.

**Klein run table structure:**
- Check `scripts/debug_klein_run_table.py` for structure
- Klein run table should have numHS or equivalent field
- Filter reorientation events where numHS >= 1

**Step 3: Implement Aggregate Turn Rate Calculation**

MATLAB reference (line 307-340):
```matlab
turnStart = t(j).getSubFieldDQ('reorientation', 'led12Val_toff', 'indsExpression', '[track.reorientation.numHS] >= 1', 'position', 'start');
turnStart_total = [turnStart_total turnStart];
turnrate = rate_from_time(turnStart_total, tperiod, stepsize, binsize) ./ double(nperiod) * 60;
```

**Key points:**
- Aggregate across all tracks: `turnStart_total = [turnStart_total turnStart]`
- Use `led12Val_toff` timing (period-relative, LED OFF = 0)
- Calculate `nperiod` (number of periods) for normalization
- Use `rate_from_time()` function (need to find/implement)

**Step 4: Find/Implement `rate_from_time()` Function**

MATLAB line 325: `rate_from_time(turnStart_total, tperiod, stepsize, binsize)`

**Need to:**
- Search MATLAB codebase for `rate_from_time()` definition
- Understand binning approach (stepsize, binsize parameters)
- Implement Python equivalent
- Match MATLAB output exactly

**Step 5: Run Comparison Test**

**Test file:** `data/GMR61@GMR61_T_Re_Sq_0to100PWM_30#C_Bl_7PWM_202508221246.h5`

**Comparison:**
1. Run existing pipeline: `scripts/run_stimulus_locked_analysis_production.py`
   - Generate aggregate stats plot
   - Save plot and turn rate values
2. Run Python adaptation of MATLAB method
   - Use same H5 file
   - Generate aggregate stats plot
   - Save plot and turn rate values
3. Compare side-by-side
   - Visual comparison of plots
   - Numerical comparison of turn rate values
   - Check timing alignment
   - Document differences

**Step 6: Document Findings**

Create validation report: `docs/logs/2025-11-11/matlab-reference-validation.md`

**Include:**
- MATLAB method investigation summary
- Python adaptation approach
- Comparison test results
- Side-by-side plots
- Differences and similarities
- Recommendations for integration

## Questions to Investigate

1. **Where is `rate_from_time()` defined?**
   - Search: `D:\Matlab-Track-Analysis-MirnaLab\` for function definition
   - May be in utility functions or basic routines

2. **How does Klein run table store numHS?**
   - Check `scripts/debug_klein_run_table.py`
   - Check trajectories CSV structure
   - Verify field name and format

3. **What are stepsize and binsize parameters?**
   - MATLAB uses: `stepsize = 0.1`, `binsize = 0.5`
   - Understand how these affect binning
   - Match exactly in Python implementation

4. **How is nperiod calculated?**
   - MATLAB line 322: `nperiod = nperiod + (min(t(j).endFrame, t_stim_end(i)*frame_rate) - max(t(j).startFrame, t_stim_start(i)*frame_rate)) / frame_rate / tperiod`
   - Track overlap with stimulus period
   - Sum across all tracks

## Deliverables

1. **Python adaptation script**
   - `scripts/2025-11-11/adapt_matlab_turnrate_method.py` (update existing)
   - Implement `rate_from_time()` equivalent
   - Implement aggregate turn rate calculation
   - Filter by numHS >= 1

2. **Comparison test script**
   - `scripts/2025-11-11/compare_turnrate_methods.py` (update existing)
   - Run both methods
   - Generate side-by-side plots
   - Save comparison results

3. **Validation report**
   - `docs/logs/2025-11-11/matlab-reference-validation.md`
   - Document investigation and findings
   - Include comparison plots
   - Recommendations

4. **Report summary** (Task 3.6)
   - Add section to `scripts/2025-11-11/report/report_2025-11-11.qmd`
   - Summarize validation work

## Success Criteria

- Python adaptation matches MATLAB reference method logic
- Aggregate stats plots show same structure and values
- Turn rate calculations match within expected tolerance
- Clear documentation of any differences
- Validation report ready for integration decision

## Blocking Status

This task blocks:
- Task 0.2 integration into `engineer_dataset_from_h5.py`
- All downstream analysis work
- Report finalization (Task 3.6 depends on this)

## Handoff to mari-test

Once Python adaptation and comparison test are complete, create handoff to mari-test for independent validation:

**Handoff document:** `conejo-code_mari-test_YYYYMMDD-HHMMSS_matlab-validation-ready-for-testing.md`

**Include:**
- Test scripts and instructions
- Expected outputs and comparison plots
- Validation criteria
- Test file location
- How to run both methods
- What to compare

**Purpose:** Independent validation by testing agent before final integration decision.

## Next Steps

1. **conejo-code**: Continue with Step 1 - Extract reorientation data
2. **conejo-code**: Find `rate_from_time()` function definition
3. **conejo-code**: Implement aggregate turn rate calculation
4. **conejo-code**: Run comparison test
5. **conejo-code**: Create validation report
6. **conejo-code**: Handoff to mari-test for independent validation
7. **mari-test**: Validate Python adaptation independently
8. **larry**: Review validation results and approve integration

---

**larry** 🎖️  
**Date:** 2025-11-11 12:10:00

