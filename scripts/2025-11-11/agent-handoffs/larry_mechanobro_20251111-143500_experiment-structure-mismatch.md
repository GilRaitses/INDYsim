# Handoff: Experiment Structure Mismatch Issue

**From:** larry  
**To:** mechanobro  
**Date:** 2025-11-11 14:35:00  
**Priority:** High  
**Status:** ⚠️ New Issue Found

## Summary

The `openDataFile` error handling works correctly, but a new issue was discovered: the experiment structure loaded from the `.mat` file doesn't match what the MATLAB code expects. The experiment appears empty (0 tracks, 0 frames via `getInfo()`) even though 64 tracks were loaded successfully.

## Test Results

### ✅ What Works Now

1. **openDataFile Error Handling** ✅
   - Error is caught and handled gracefully
   - Script continues past the error
   - Warning messages displayed correctly

2. **Track Loading** ✅
   - Successfully loaded 64 tracks
   - Track interpolation completed
   - Tracks are in MATLAB workspace

### ❌ New Issue

**Error:** Experiment structure mismatch

```
warning: loaded 0 experiments from ...mat
Experiment: 0 tracks, 0 frames
Unrecognized field name "globalQuantity".
```

**Problem:** The `.mat` file structure doesn't match what the MATLAB code expects:
- Experiment structure appears empty (`loaded 0 experiments`)
- `app.getInfo()` returns 0 tracks/0 frames (even though 64 tracks were loaded)
- Cannot access `globalQuantity` field (needed for LED values)
- Experiment structure doesn't have expected fields

## Analysis

**What's Happening:**
1. `.mat` file loads but experiment structure is empty
2. Tracks load successfully from separate directory (64 tracks)
3. But tracks aren't accessible via `app.getInfo()`
4. Cannot access `globalQuantity` for LED values
5. Export fails because experiment structure is incomplete

**Possible Causes:**
1. `.mat` file format is different than expected
2. Experiment structure needs different initialization
3. Data is organized differently in these `.mat` files
4. MATLAB code expects a different experiment structure version

## Recommendations

### Option 1: Investigate .mat File Structure

**Action:** Inspect the `.mat` file structure directly to see what's actually in it:

```python
import scipy.io
mat_data = scipy.io.loadmat('path/to/file.mat', simplify_cells=True)
print(mat_data.keys())
# Inspect structure to see what fields are present
```

**Goal:** Understand what fields are actually in the `.mat` file vs. what the code expects.

### Option 2: Fix MATLAB Code

**Files to Check:**
- `D:\mechanosensation\scripts\2025-10-16\@DataManager\loadExperiment.m`
- How experiment structure is initialized
- How `globalQuantity` is accessed

**Possible Fixes:**
1. Handle missing `globalQuantity` field
2. Properly initialize experiment structure from `.mat` file
3. Ensure tracks are linked to experiment structure
4. Handle different experiment structure versions

### Option 3: Alternative Conversion Method

If MATLAB code can't be fixed, consider:
- Direct `.mat` file reading with `scipy.io.loadmat()`
- Direct `.bin` file reading for LED values
- Direct track data reading from `matfiles/` directories
- Manual H5 structure building

## Test Details

**Test Command:**
```bash
python src/@matlab_conversion/convert_matlab_to_h5.py \
    --eset-dir "data/matlab_data/GMR61@GMR61/T_Re_Sq_0to250PWM_30#T_Bl_Sq_5to15PWM_30" \
    --mat-file "btdfiles/btd_GMR61@GMR61_T_Re_Sq_0to250PWM_30#T_Bl_Sq_5to15PWM_30_202510301513.mat" \
    --output-dir "data/h5_files"
```

**Key Observations:**
- `warning: loaded 0 experiments` - Experiment structure is empty
- `⌂ Loaded 64 tracks` - Tracks loaded successfully
- `Experiment: 0 tracks, 0 frames` - But not accessible via API
- `Unrecognized field name "globalQuantity"` - Field doesn't exist

## Next Steps

1. **mechanobro:** Investigate `.mat` file structure to understand what's actually in it
2. **mechanobro:** Check if experiment structure needs different initialization
3. **mechanobro:** Consider if alternative conversion method is needed
4. **larry:** Awaiting investigation results

## Status

**openDataFile Error:** ✅ Fixed  
**Experiment Structure:** ❌ Mismatch - Needs Investigation  
**Overall:** ⚠️ Blocked by experiment structure incompatibility

---

**larry**  
**Date:** 2025-11-11 14:35:00  
**Status:** ⚠️ Experiment Structure Mismatch - Awaiting Investigation

