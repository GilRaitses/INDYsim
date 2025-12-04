# Platform Transfer Discrepancy Report

**Project:** Mechanosensation Analysis Pipeline
**Date:** December 4, 2025
**Validation Experiment:** GMR61@GMR61_T_Re_Sq_50to250PWM_30#C_Bl_7PWM_202506251614

---

## Executive Summary

This report documents all discrepancies identified during the validation of the Python data engineering pipeline against the reference MATLAB implementation. All discrepancies were resolved, and the pipelines now produce numerically identical results.

---

## 1. Position Data Source Discrepancy

### Severity: CRITICAL

### Description
The Python pipeline initially used raw position data (`points/loc`) while MATLAB uses smoothed position data (`getDerivedQuantity('sloc')`). This caused significant errors in SpeedRunVel computation.

### Impact
| Metric | MATLAB (correct) | Python (incorrect) | Error Factor |
|--------|-----------------|-------------------|--------------|
| SpeedRunVel max | 0.0004 cm/s | 0.0013 cm/s | 3.25x |
| SpeedRunVel min | -0.0003 cm/s | -0.0022 cm/s | 7.3x |
| Negative points | 2.9% | 11.3% | 3.9x |
| Reversals detected | 2 | 0 | False negative |

### Root Cause
- MATLAB: `pos = track.getDerivedQuantity('sloc')` returns smoothed centroid
- Python: Was reading `track['points']['loc']` which is the raw (unsmoothed) centroid

### Resolution
Changed Python to read `track['derived_quantities']['sloc']` as primary source.

### Verification
After fix, SpeedRunVel values match to floating-point precision (max diff < 1e-10).

---

## 2. Camera Calibration Missing from H5

### Severity: HIGH

### Description
The H5 export pipeline did not include `lengthPerPixel` (camera calibration), which is required for converting pixel displacements to real-world velocities (cm/s).

### Impact
Without correct `lengthPerPixel`:
- All velocity values would be in pixels/second, not cm/second
- Comparison with literature values impossible
- Threshold-based detection criteria invalid

### Root Cause
The original H5 export script (`convert_matlab_to_h5.py`) did not extract camera calibration from the MATLAB experiment.

### Resolution
Created `append_camcal_to_h5.py` script that:
1. Loads MATLAB experiment with camera calibration object
2. Computes `lengthPerPixel` using calibration methods:
   ```matlab
   cc = eset.expt(1).camcalinfo;
   real_coords = cc.c2rX(pixel_coords_x, pixel_coords_y);
   lengthPerPixel = real_distance / pixel_distance;
   ```
3. Appends to H5 at `/lengthPerPixel` (root) and `/metadata@lengthPerPixel`

### Calibration Value
**lengthPerPixel = 0.01018533 cm/pixel** (verified against MATLAB computation)

---

## 3. LED Data Structure Mismatch

### Severity: MEDIUM

### Description
The Python scripts expected LED values as direct datasets, but H5 stores them as groups containing `yData`.

### Expected Structure
```
/global_quantities/led1Val  → Dataset (float64, shape M)
```

### Actual Structure
```
/global_quantities/led1Val/  → Group
    yData                    → Dataset (float64, shape M)
```

### Impact
Script crashed with `TypeError: Accessing a group is done with bytes or str, not <class 'slice'>`

### Resolution
Updated all H5 reading code to handle both structures:
```python
if isinstance(gq['led1Val'], h5py.Dataset):
    led1_ydata = gq['led1Val'][:]
elif isinstance(gq['led1Val'], h5py.Group):
    led1_ydata = gq['led1Val']['yData'][:]
```

---

## 4. lengthPerPixel Location Mismatch

### Severity: MEDIUM

### Description
Scripts expected `lengthPerPixel` in `/metadata` (as attribute or dataset), but after fix it was placed at root level.

### Expected Locations (scripts checked)
1. `/metadata.attrs['lengthPerPixel']`
2. `/metadata/lengthPerPixel`

### Actual Location (after camcal export)
1. `/lengthPerPixel` (root level, primary)
2. `/metadata.attrs['lengthPerPixel']` (backup)

### Resolution
Updated all scripts to check root level first:
```python
if 'lengthPerPixel' in f:
    length_per_pixel = float(f['lengthPerPixel'][()])
elif 'metadata' in f and 'lengthPerPixel' in f['metadata'].attrs:
    length_per_pixel = float(f['metadata'].attrs['lengthPerPixel'])
```

---

## 5. LED Time Array Location

### Severity: LOW

### Description
Scripts searched for separate LED time arrays (`led1Val_xData`, `led1Val_time`), but LED values use the global frame time.

### Expected
```
/global_quantities/led1Val_time  → LED1 timestamps
/global_quantities/led2Val_time  → LED2 timestamps
```

### Actual
LED values are sampled at each frame, so they use the global `/eti` (elapsed time index) array.

### Resolution
Updated documentation and scripts to use `/eti` for LED timing:
```python
led_xdata = f['eti'][:]  # Global time array applies to LED values
```

---

## 6. Track Identification Method

### Severity: LOW

### Description
Initial scripts used array index to select tracks, but track numbering may not match array order.

### Incorrect Approach
```python
track = f['tracks'][track_keys[0]]  # Array index
```

### Correct Approach
```python
for key in track_keys:
    if int(key.replace('track_', '')) == target_track_num:
        track = f['tracks'][key]
        break
```

### Impact
Could analyze different tracks in MATLAB vs Python if track order differs.

### Resolution
All scripts now select tracks by track NUMBER, not array index.

---

## Validation Results After All Fixes

### SpeedRunVel Comparison
| Statistic | MATLAB | Python | Difference |
|-----------|--------|--------|------------|
| Min | -0.000334 | -0.000334 | < 1e-10 |
| Max | 0.000402 | 0.000402 | < 1e-10 |
| Mean | 0.000089 | 0.000089 | < 1e-10 |
| Negative % | 2.862% | 2.862% | 0 |

### Reversal Detection Comparison
| Reversal | MATLAB Start | Python Start | MATLAB Duration | Python Duration |
|----------|--------------|--------------|-----------------|-----------------|
| 1 | 11.90 s | 11.90 s | 3.45 s | 3.45 s |
| 2 | 1011.15 s | 1011.15 s | 4.05 s | 4.05 s |

---

## Recommendations for Future Development

1. **H5 Export Enhancement:** Update `convert_matlab_to_h5.py` to include camera calibration by default.

2. **Schema Validation:** Run `validate_h5_schema.py` after every H5 export to catch missing fields early.

3. **Position Data Warning:** Add runtime check that warns if `sloc` is not available and `loc` is used as fallback.

4. **Documentation:** Maintain `FIELD_MAPPING.md` as authoritative reference for H5 schema.

---

## Files Modified

| File | Changes |
|------|---------|
| `engineer_data.py` | Use `sloc`, check root for `lengthPerPixel` |
| `load_experiment_and_compute.py` | Use `sloc`, handle LED groups, check root for calibration |
| `validate_h5_schema.py` | Updated expected field locations |
| `validate_data_integrity.py` | Handle LED groups, check root for calibration |
| `validate_h5_against_mat.m` | Check root for `lengthPerPixel` |
| `FIELD_MAPPING.md` | Documented actual H5 structure |

---

## Certification

This validation confirms that the Python mechanosensation analysis pipeline produces results numerically identical to the reference MATLAB implementation when processing the same experiment data.

**Validated by:** Automated validation suite
**Date:** December 4, 2025
**Experiment:** GMR61@GMR61_T_Re_Sq_50to250PWM_30#C_Bl_7PWM_202506251614
**Result:** PASS - All discrepancies resolved

