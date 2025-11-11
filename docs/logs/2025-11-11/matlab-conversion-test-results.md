# MATLAB to H5 Conversion - Test Results

**Date:** 2025-11-11  
**Test Status:** ⚠️ MATLAB Code Compatibility Issue  
**Prepared by:** larry

## Test Summary

**Test:** Single experiment conversion  
**ESET:** `T_Re_Sq_0to250PWM_30#T_Bl_Sq_5to15PWM_30`  
**MAT File:** `btd_GMR61@GMR61_T_Re_Sq_0to250PWM_30#T_Bl_Sq_5to15PWM_30_202510301513.mat`  
**Output Directory:** `data/h5_files`

## Test Results

### ✅ What Works

1. **File Discovery Logic** ✅
   - Correctly found `.mat` file in `btdfiles/` subdirectory
   - Correctly identified tracks directory: `GMR61@GMR61_202510301513 - tracks`
   - Correctly identified `.bin` file: `GMR61@GMR61_T_Re_Sq_0to250PWM_30#T_Bl_Sq_5to15PWM_30_202510301513.bin`
   - All file paths constructed correctly

2. **MATLAB Bridge Initialization** ✅
   - MATLAB Engine started successfully
   - MAGAT codebase added to path correctly
   - BehavioralVideoExplorer created successfully
   - All managers initialized

3. **Track Loading** ✅
   - Successfully loaded 64 tracks from tracks directory
   - Track interpolation completed successfully

### ❌ Issue Found

**Error:** MATLAB code compatibility issue

```
Unrecognized field name "openDataFile".

Error in DataManager/loadExperiment (line 52)
    obj.eset.expt(1).openDataFile;
```

**Location:** `D:\mechanosensation\scripts\2025-10-16\@DataManager\loadExperiment.m`, line 52

**Root Cause:** The MATLAB `DataManager` class is trying to access a field `openDataFile` that doesn't exist in the experiment structure loaded from the `.mat` file.

**Impact:** Conversion cannot proceed past track loading because the experiment structure doesn't match what the MATLAB code expects.

## Analysis

### File Structure Verification

**Verified Structure:**
```
T_Re_Sq_0to250PWM_30#T_Bl_Sq_5to15PWM_30/
├── btdfiles/
│   └── btd_GMR61@GMR61_T_Re_Sq_0to250PWM_30#T_Bl_Sq_5to15PWM_30_202510301513.mat ✅
├── matfiles/
│   └── GMR61@GMR61_202510301513 - tracks/ ✅ (64 tracks loaded)
├── GMR61@GMR61_T_Re_Sq_0to250PWM_30#T_Bl_Sq_5to15PWM_30_202510301513.bin ✅
└── GMR61@GMR61_T_Re_Sq_0to250PWM_30#T_Bl_Sq_5to15PWM_30_202510301513 sup data dir/ ✅
```

### Conversion Script Status

**Python Script:** ✅ Working correctly
- File discovery logic works
- Path construction correct
- MATLAB bridge initialization successful
- Track loading successful

**MATLAB Code:** ❌ Compatibility issue
- `DataManager.loadExperiment()` expects `openDataFile` field
- Experiment structure from `.mat` file doesn't have this field
- Needs fix in MATLAB classes

## Next Steps

### Option 1: Fix MATLAB Code (Recommended)

**File:** `D:\mechanosensation\scripts\2025-10-16\@DataManager\loadExperiment.m`

**Issue:** Line 52 tries to access `obj.eset.expt(1).openDataFile` which doesn't exist

**Fix Options:**
1. Make field access optional (check if field exists before accessing)
2. Add field to experiment structure if missing
3. Use alternative method to open data file

### Option 2: Alternative Conversion Method

If MATLAB code cannot be fixed, consider:
- Direct H5 file creation from `.mat` and `.bin` files without MATLAB bridge
- Use scipy.io to read `.mat` files directly
- Extract LED values from `.bin` files directly
- Build H5 structure manually

### Option 3: Update MATLAB Classes

Update `DataManager` class to handle experiment structures without `openDataFile` field:
- Add field existence check
- Provide default value if missing
- Handle different experiment structure versions

## Recommendations

1. **Immediate:** Fix MATLAB `DataManager.loadExperiment()` to handle missing `openDataFile` field
2. **Short-term:** Test with other `.mat` files to see if issue is consistent
3. **Long-term:** Consider alternative conversion method if MATLAB code continues to have compatibility issues

## Test Output

```
================================================================================
CONVERTING: GMR61@GMR61_T_Re_Sq_0to250PWM_30#T_Bl_Sq_5to15PWM_30_202510301513
================================================================================
MAT file: btd_GMR61@GMR61_T_Re_Sq_0to250PWM_30#T_Bl_Sq_5to15PWM_30_202510301513.mat
Tracks: GMR61@GMR61_202510301513 - tracks
BIN file: GMR61@GMR61_T_Re_Sq_0to250PWM_30#T_Bl_Sq_5to15PWM_30_202510301513.bin
Output: GMR61@GMR61_T_Re_Sq_0to250PWM_30#T_Bl_Sq_5to15PWM_30_202510301513.h5

Initializing MATLAB...
Starting MATLAB Engine...
Adding MAGAT codebase to path: d:\magniphyq\codebase\Matlab-Track-Analysis-SkanataLab
Creating BehavioralVideoExplorer...
=== Behavioral Video Explorer ===
Initializing managers...
⌂ All managers initialized
MAGAT Bridge ready!
=== DATA MANAGER: Loading Experiment ===
Loading experiment: D:\INDYsim\data\matlab_data\GMR61@GMR61\T_Re_Sq_0to250PWM_30#T_Bl_Sq_5to15PWM_30\btdfiles\btd_GMR61@GMR61_T_Re_Sq_0to250PWM_30#T_Bl_Sq_5to15PWM_30_202510301513.mat
Elapsed time is 1.751238 seconds.
warning: loaded 0 experiments from D:\INDYsim\data\matlab_data\GMR61@GMR61\T_Re_Sq_0to250PWM_30#T_Bl_Sq_5to15PWM_30\btdfiles\btd_GMR61@GMR61_T_Re_Sq_0to250PWM_30#T_Bl_Sq_5to15PWM_30_202510301513.mat
finished - 1.8961
Loading tracks from: D:\INDYsim\data\matlab_data\GMR61@GMR61\T_Re_Sq_0to250PWM_30#T_Bl_Sq_5to15PWM_30\matfiles\GMR61@GMR61_202510301513 - tracks
⌂ Loaded 64 tracks
Interpolating tracks...
⌂ Interpolation complete
Unrecognized field name "openDataFile".
```

## Status

**Conversion Script:** ✅ Working  
**MATLAB Code:** ❌ Needs fix  
**Overall:** ⚠️ Blocked by MATLAB code compatibility issue

---

**Last Updated:** 2025-11-11  
**Status:** ⚠️ MATLAB Code Compatibility Issue - Needs Fix

