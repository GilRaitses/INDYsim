# MATLAB Dependencies Computation and H5 Storage Summary

**Date:** 2025-11-11  
**Agent:** conejo-code  
**Task:** Task 0.3 - Validate Against MATLAB Reference Method

## Summary

All MATLAB method dependencies have been computed and saved to H5 files. The H5 files now contain all required data for the MATLAB turn rate calculation method adaptation.

## What Was Computed

### 1. Global Quantities (Experiment-level)

**Computed:**
- `global_quantities/led12Val/yData` - Combined LED values (led1Val + led2Val + offset)
- `global_quantities/led12Val/xData` - Time axis for LED values
- `global_quantities/led12Val_ton/yData` - LED ON time values (relative to period start)
- `global_quantities/led12Val_ton/xData` - Time axis
- `global_quantities/led12Val_toff/yData` - LED OFF time values (relative to LED OFF)
- `global_quantities/led12Val_toff/xData` - Time axis
- `elapsedTime` - Experiment-level elapsed time array

**Implementation:**
- `compute_led12Val()` - Combines led1Val + led2Val, adds 60 offset to initial values
- `add_ton_toff_matlab()` - Creates period-relative time values using square wave detection

### 2. Track-Level Derived Quantities (Per Frame)

**Computed for each track:**
- `tracks/{track_key}/derived_quantities/led12Val_ton` - LED ON time per frame (interpolated)
- `tracks/{track_key}/derived_quantities/led12Val_toff` - LED OFF time per frame (interpolated)

**Implementation:**
- `interpolate_led_timing_to_track()` - Linear interpolation from global LED timing to track ETI

## Files Created

1. **`compute_and_save_matlab_dependencies.py`**
   - Main script to compute and save all dependencies
   - Creates backup before modifying H5 file
   - Verifies saved data

2. **`verify_matlab_dependencies.py`**
   - Verification script to check dependency availability
   - Creates manifest of available/missing dependencies

3. **`matlab_dependencies_complete_manifest.md`**
   - Complete documentation of all MATLAB dependencies
   - Status of each dependency in H5 files

## Usage

### Compute and Save Dependencies

```bash
python scripts/2025-11-11/compute_and_save_matlab_dependencies.py [h5_file]
```

If no file specified, processes first H5 file in `data/` directory.

### Verify Dependencies

```bash
python scripts/2025-11-11/verify_matlab_dependencies.py [h5_file]
```

### Use Saved Dependencies in Code

```python
from adapt_matlab_turnrate_method import extract_track_data_from_h5_with_dependencies

# Extract track data with saved dependencies
track_data = extract_track_data_from_h5_with_dependencies(h5_file, track_key='track_1')

# Access saved data
track_eti = track_data['eti']
track_led12Val_toff = track_data['led12Val_toff']  # Per-frame LED timing
track_led12Val_ton = track_data['led12Val_ton']
```

## Verification Results

**Before computation:**
- ✗ Missing: led12Val, led12Val_ton, led12Val_toff (global)
- ✗ Missing: led12Val_ton, led12Val_toff (track-level)
- ✗ Missing: elapsedTime

**After computation:**
- ✓ All dependencies available in H5 file
- ✓ 14 dependencies available (was 8)
- ✓ 0 dependencies missing (was 5)

## H5 File Structure

```
H5 File
├── global_quantities/
│   ├── led1Val/
│   │   └── yData
│   ├── led2Val/
│   │   └── yData
│   ├── led12Val/          ← NEW
│   │   ├── yData
│   │   └── xData
│   ├── led12Val_ton/      ← NEW
│   │   ├── yData
│   │   └── xData
│   └── led12Val_toff/     ← NEW
│       ├── yData
│       └── xData
├── tracks/
│   └── track_{N}/
│       └── derived_quantities/
│           ├── eti
│           ├── led12Val_ton    ← NEW
│           ├── led12Val_toff   ← NEW
│           ├── speed
│           ├── deltatheta
│           └── spineLength
└── elapsedTime              ← NEW
```

## Next Steps

1. ✅ Compute and save dependencies - **COMPLETE**
2. ⏳ Extract reorientation data from trajectories CSV
3. ⏳ Filter by numHS >= 1 using Klein run table
4. ⏳ Run comparison test between MATLAB method and existing pipeline
5. ⏳ Document findings and create validation report

## Notes

- Backup files are created automatically (suffix: `_backup_.h5`)
- All computed arrays are compressed with gzip
- Interpolation uses linear interpolation (np.interp)
- Period length defaults to 15 seconds (can be configured)
- Frame rate defaults to 10 fps (can be configured)

