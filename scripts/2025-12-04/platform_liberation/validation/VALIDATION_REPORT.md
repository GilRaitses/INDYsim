# Platform Transfer Validation Report

**Generated**: [TIMESTAMP]
**Experiment**: GMR61@GMR61_T_Re_Sq_50to250PWM_30#C_Bl_7PWM_202506251614
**Status**: [PASS/FAIL]

---

## Executive Summary

This report documents the validation of the Python mechanosensation analysis pipeline against the reference MATLAB implementation. The validation ensures that all computational methods produce identical results, enabling a complete platform transfer from MATLAB to Python.

### Overall Result: [PASS/FAIL]

| Layer | Validation | Status | Checks |
|-------|------------|--------|--------|
| 1a | H5 Schema | [ ] | 0/0 |
| 1b | Data Integrity | [ ] | 0/0 |
| 2 | Computation | [ ] | 0/0 |
| 3 | Results | [ ] | 0/0 |

---

## Layer 1: Data Integrity Validation

### 1a. H5 Schema Validation

Verifies that the H5 file contains all required fields for analysis.

| Field | Status | Notes |
|-------|--------|-------|
| `/eti` | [ ] | Global elapsed time |
| `/metadata/lengthPerPixel` | [ ] | Camera calibration |
| `/global_quantities/led1Val` | [ ] | LED1 stimulus values |
| `/global_quantities/led2Val` | [ ] | LED2 stimulus values |
| `/tracks/{key}/derived_quantities/shead` | [ ] | Head position (2, N) |
| `/tracks/{key}/derived_quantities/smid` | [ ] | Midpoint position (2, N) |
| `/tracks/{key}/derived_quantities/eti` | [ ] | Track elapsed time |
| `/tracks/{key}/points/loc` or `/tracks/{key}/derived_quantities/sloc` | [ ] | Position data |

### 1b. Data Integrity (.mat vs H5)

Verifies that H5 file contains identical raw data to source MATLAB .mat file.

| Field | MATLAB | H5 | Match | Max Diff |
|-------|--------|-----|-------|----------|
| lengthPerPixel | [value] | [value] | [ ] | [diff] |
| led1Val | [count] pts | [count] pts | [ ] | [diff] |
| led2Val | [count] pts | [count] pts | [ ] | [diff] |
| track_count | [N] | [N] | [ ] | [diff] |
| track_1/shead | [shape] | [shape] | [ ] | [diff] |
| track_1/smid | [shape] | [shape] | [ ] | [diff] |
| track_1/eti | [count] pts | [count] pts | [ ] | [diff] |

---

## Layer 2: Computation Validation

Compares intermediate computational values between MATLAB and Python.

### Reference: Mason Klein Scripts

| Computation | Original Script | Location |
|-------------|-----------------|----------|
| HeadUnitVec | Just_ReverseCrawl_Matlab.m | scripts/2025-11-20/mason's scritps/ |
| VelocityVec | Just_ReverseCrawl_Matlab.m | scripts/2025-11-20/mason's scritps/ |
| SpeedRun | Just_ReverseCrawl_Matlab.m | scripts/2025-11-20/mason's scritps/ |
| SpeedRunVel | Just_ReverseCrawl_Matlab.m | scripts/2025-11-20/mason's scritps/ |
| Reversals | Behavior_analysis_ReverseCrawl_Stop_Continue_Turn_Matlab.m | scripts/2025-11-20/mason's scritps/ |

### Intermediate Value Comparison

| Field | Tolerance | Max Diff | Mean Diff | Status |
|-------|-----------|----------|-----------|--------|
| HeadUnitVec | 1e-10 | [value] | [value] | [ ] |
| VelocityVec | 1e-10 | [value] | [value] | [ ] |
| SpeedRun | 1e-10 | [value] | [value] | [ ] |
| CosThetaFactor | 1e-10 | [value] | [value] | [ ] |
| SpeedRunVel | 1e-10 | [value] | [value] | [ ] |

### Mathematical Method Verification

```
Step 1: HeadVec = shead - smid
        HeadUnitVec = HeadVec / ||HeadVec||

Step 2: dx = diff(xpos), dy = diff(ypos), dt = diff(times)
        distance = sqrt(dx² + dy²)
        SpeedRun = distance / dt
        VelocityVec = [dx, dy] / distance

Step 3: CosThetaFactor = VelocityVec · HeadUnitVec

Step 4: SpeedRunVel = SpeedRun × CosThetaFactor
```

---

## Layer 3: Result Validation

Compares final analysis outputs.

### Reversal Detection

| Metric | MATLAB | Python | Match | Tolerance |
|--------|--------|--------|-------|-----------|
| Count | [N] | [N] | [ ] | exact |
| Start times | [list] | [list] | [ ] | < 0.001s |
| End times | [list] | [list] | [ ] | < 0.001s |
| Durations | [list] | [list] | [ ] | < 0.001s |

### Detection Criteria

- SpeedRunVel < 0 (negative = reverse crawling)
- Duration ≥ 3.0 seconds

---

## Validation Scripts Run

### MATLAB

```
1. matlab/load_experiment_and_compute.m
   - Loaded experiment from matfiles/
   - Computed SpeedRunVel with intermediates
   - Detected reversals
   - Saved: test_data/matlab_validation_output.mat

2. matlab/validate_h5_against_mat.m
   - Compared H5 data against source .mat
   - Saved: test_data/data_integrity_results.mat
```

### Python

```
1. python/load_experiment_and_compute.py
   - Loaded experiment from h5_exports/
   - Computed SpeedRunVel with intermediates
   - Detected reversals
   - Saved: test_data/python_validation_full.npz

2. python/validate_h5_schema.py
   - Verified H5 schema completeness

3. python/validate_data_integrity.py
   - Compared raw data arrays

4. run_full_validation.py
   - Ran all validation layers
   - Generated this report
```

---

## Files Generated

| File | Description |
|------|-------------|
| `test_data/matlab_validation_output.mat` | MATLAB intermediate and final values |
| `test_data/matlab_speedrunvel.csv` | MATLAB SpeedRunVel timeseries |
| `test_data/python_validation_full.npz` | Python intermediate and final values |
| `test_data/python_speedrunvel.csv` | Python SpeedRunVel timeseries |
| `test_data/python_validation_output.json` | Python summary (human-readable) |
| `test_data/validation_report.json` | Machine-readable validation results |
| `test_data/data_integrity_results.mat` | Data integrity comparison results |

---

## Acceptance Criteria

### Tolerances Applied

| Check | Criterion | Tolerance |
|-------|-----------|-----------|
| Input arrays (shead, smid, loc, eti) | Exact match | 0 |
| lengthPerPixel | Exact match | 1e-12 |
| LED arrays | Exact match | 0 |
| HeadUnitVec | Max diff | < 1e-10 |
| VelocityVec | Max diff | < 1e-10 |
| SpeedRun | Max diff | < 1e-10 |
| CosThetaFactor | Max diff | < 1e-10 |
| SpeedRunVel | Max diff | < 1e-10 |
| Reversal indices | Exact match | 0 |
| Reversal times | Max diff | < 0.001s |

### Sign-Off

- [ ] All Layer 1 checks pass (Data Integrity)
- [ ] All Layer 2 checks pass (Computation)
- [ ] All Layer 3 checks pass (Results)
- [ ] Manual review of SpeedRunVel plots (visual verification)
- [ ] Reversal events match between pipelines

---

## Documentation References

| Document | Purpose |
|----------|---------|
| `FIELD_MAPPING.md` | MATLAB field → H5 path mapping |
| `reverse.plan.md` | Validation framework design |
| `scripts/2025-11-24/mason_scripts_documentation.qmd` | Original Mason method documentation |

---

## Troubleshooting Guide

### Common Issues

1. **H5 file not found**
   - Ensure H5 export has been run from INDYsim
   - Check path: `{eset_dir}/h5_exports/{experiment_name}.h5`

2. **Track not found in H5**
   - H5 uses `track_XXX` naming (zero-padded)
   - Check track numbering matches between .mat and H5

3. **SpeedRunVel mismatch**
   - Verify lengthPerPixel matches exactly
   - Check array orientation (2,N vs N,2)
   - Ensure same track is being compared

4. **Reversal count mismatch**
   - Check time array alignment
   - Verify duration threshold is identical (3.0s)

---

## Certification

This validation certifies that the Python implementation in `engineer_data.py` produces results identical to Mason Klein's MATLAB implementation within specified tolerances. The platform transfer is complete and the Python pipeline may be used for production analysis.

**Validated by**: [NAME]
**Date**: [DATE]
**Signature**: ________________

