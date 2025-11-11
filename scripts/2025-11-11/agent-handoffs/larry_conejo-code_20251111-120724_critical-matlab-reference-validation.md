# Handoff: Critical MATLAB Reference Validation Task

**From:** larry  
**To:** conejo-code  
**Date:** 2025-11-11 12:07:24  
**Priority:** High  
**Status:** Pending

## Context

Before integrating LED value alignment into the H5 processing pipeline, we need to validate our approach against the MATLAB reference method `load_multi_data_gr21a.m`. This method is considered identical to our stimulus-locked turn rate analysis and provides the gold standard for comparison.

## Task

**Task 0.3: Validate Against MATLAB Reference Method**

Investigate the MATLAB reference method and create a Python adaptation. Run a comparison test between the existing pipeline and the Python adaptation to ensure alignment before integration.

## Deliverables

1. **Investigation**
   - Study `load_multi_data_gr21a.m` turn rate calculation method
   - Document key patterns: `rate_from_time()`, `getSubFieldDQ()`, binning approach
   - Identify all dependencies and helper functions

2. **Python Adaptation**
   - Create `scripts/2025-11-11/adapt_matlab_turnrate_method.py`
   - Implement equivalent functions to MATLAB reference
   - Match aggregate turn rate calculation logic

3. **Comparison Test**
   - Create `scripts/2025-11-11/compare_turnrate_methods.py`
   - Run existing pipeline: `scripts/run_stimulus_locked_analysis_production.py`
   - Run Python adaptation
   - Generate side-by-side aggregate stats plots
   - Compare turn rate values and timing

4. **Validation Report**
   - Create `docs/logs/2025-11-11/matlab-reference-validation.md`
   - Document findings and differences
   - Recommend integration approach

5. **Report Summary**
   - Add summary section to `scripts/2025-11-11/report/report_2025-11-11.qmd`
   - Document investigation and test results

## Reference Files

- **MATLAB Reference:** `D:\Matlab-Track-Analysis-MirnaLab\user specific\Silverio\load_multi_data_gr21a.m`
- **Key Sections:**
  - Lines 307-340: `plot_turnrate` - aggregate turn rate calculation
  - Line 325: `rate_from_time()` function call
  - Line 318: `getSubFieldDQ('reorientation', 'led12Val_toff', ...)`
  - Line 252: `getSubFieldDQ('reorientation', 'led12Val_ton', ...)`

## Test Requirements

- Use test file: `data/GMR61@GMR61_T_Re_Sq_0to100PWM_30#C_Bl_7PWM_202508221246.h5`
- Generate aggregate stats plots from both methods
- Compare turn rate values, timing, and plot structure
- Document any differences

## Success Criteria

- Python adaptation matches MATLAB reference method logic
- Aggregate stats plots show same structure and values
- Turn rate calculations match within expected tolerance
- Clear documentation of differences
- Validation report ready for integration decision

## Blocking

This task blocks:
- Task 0.2 integration into `engineer_dataset_from_h5.py`
- All downstream analysis work
- Report finalization (Task 3.6 depends on this)

## Questions

- What is the `rate_from_time()` function implementation?
- How does `getSubFieldDQ()` extract LED-based timing?
- What are the exact binning and period calculation methods?
- Are there any differences we need to account for?

## Next Steps

1. **conejo-code**: Start investigation of MATLAB reference method
2. **conejo-code**: Create Python adaptation
3. **conejo-code**: Run comparison test
4. **conejo-code**: Create validation report
5. **conejo-code**: Add summary to report (Task 3.6)
6. **larry**: Review validation results and approve integration

---

**larry** 🎖️  
**Date:** 2025-11-11 12:07:24

