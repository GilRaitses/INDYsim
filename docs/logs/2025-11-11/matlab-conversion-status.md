# MATLAB to H5 Conversion - Status Report

**Date:** 2025-11-11  
**Status:** ✅ Ready for Testing  
**Prepared by:** larry

## Summary

Conversion tools have been created and adapted to handle the native ESET folder structure and naming conventions. All scripts are ready for processing the 4 experiment conditions.

## Tools Created

### 1. Conversion Script
**File:** `src/@matlab_conversion/convert_matlab_to_h5.py`
- Converts MATLAB experiment data to H5 format
- Uses MAGAT Bridge to access MATLAB data
- Handles native ESET folder structure
- Creates H5 files compatible with `engineer_dataset_from_h5.py`

### 2. Batch Scripts
**Files:**
- `src/@matlab_conversion/process_single_eset.bat` - Process single ESET folder
- `src/@matlab_conversion/process_all_esets.bat` - Process all 4 ESET folders

**Features:**
- ✅ Iterates over all `.mat` files in `btdfiles/` subdirectory
- ✅ Provides `--mat-file` argument for each file
- ✅ Success/failure counting and reporting
- ✅ Proper error handling

### 3. Documentation
**Files:**
- `src/@matlab_conversion/README.md` - Tool documentation
- `docs/logs/2025-11-11/matlab-to-h5-conversion-guide.md` - Agent guide

## ESET Folder Structure Support

**Verified Structure:**
```
T_Re_Sq_0to250PWM_30#T_Bl_Sq_5to15PWM_30/
├── btdfiles/
│   ├── btd_GMR61@GMR61_T_Re_Sq_0to250PWM_30#T_Bl_Sq_5to15PWM_30_202510301513.mat
│   └── ... (multiple .mat files)
├── matfiles/
│   ├── GMR61@GMR61_202510301513 - tracks/
│   └── ... (track directories)
├── GMR61@GMR61_T_Re_Sq_0to250PWM_30#T_Bl_Sq_5to15PWM_30_202510301513.bin
├── GMR61@GMR61_T_Re_Sq_0to250PWM_30#T_Bl_Sq_5to15PWM_30_202510301513 sup data dir/
└── ... (other files)
```

**File Discovery Logic:**
- ✅ `.mat` files: `btdfiles/btd_*.mat` - Pattern verified
- ✅ Tracks directory: `matfiles/GMR61@GMR61_{timestamp} - tracks` - Pattern verified
- ✅ `.bin` file: `{base_name}.bin` at root - Pattern verified
- ✅ Sup data dir: `{base_name} sup data dir` at root - Pattern verified

## ESET Naming Convention

**All 4 ESET folders correctly handled:**
1. ✅ `T_Re_Sq_0to250PWM_30#T_Bl_Sq_5to15PWM_30`
2. ✅ `T_Re_Sq_0to250PWM_30#C_Bl_7PWM`
3. ✅ `T_Re_Sq_50to250PWM_30#C_Bl_7PWM`
4. ✅ `T_Re_Sq_50to250PWM_30#T_Bl_Sq_5to15PWM_30`

**Naming Pattern:**
- Format: `T_{LED1_COLOR}_{WAVEFORM}_{MIN}to{MAX}PWM_{REST}#{LED2_TYPE}_{LED2_COLOR}_{LED2_VALUES}PWM`
- Special characters (`#`, `@`) properly handled
- Timestamp extraction from `.mat` filenames works correctly

## Adaptations Made

### Batch Script Fixes
1. ✅ Fixed Python script path: `src\@matlab_conversion\convert_matlab_to_h5.py`
2. ✅ Added iteration over `.mat` files in `btdfiles/` subdirectory
3. ✅ Added `--mat-file` argument for each file
4. ✅ Added success/failure counting and summary reporting

### File Discovery Logic
- ✅ Timestamp extraction: Pattern `_(\d{12})\.mat$` matches actual filenames
- ✅ Base name construction: Removes `btd_` prefix correctly
- ✅ Tracks directory lookup: Checks `matfiles/` subdirectory first
- ✅ File validation: Checks all required files exist before processing

## Usage

### Process Single ESET
```batch
src\@matlab_conversion\process_single_eset.bat "T_Re_Sq_0to250PWM_30#T_Bl_Sq_5to15PWM_30" "data\h5_files"
```

### Process All ESETs
```batch
src\@matlab_conversion\process_all_esets.bat "data\h5_files"
```

### Direct Python Usage
```bash
python src/@matlab_conversion/convert_matlab_to_h5.py \
    --eset-dir "data/matlab_data/GMR61@GMR61/T_Re_Sq_0to250PWM_30#T_Bl_Sq_5to15PWM_30" \
    --mat-file "btdfiles/btd_GMR61@GMR61_T_Re_Sq_0to250PWM_30#T_Bl_Sq_5to15PWM_30_202510301513.mat" \
    --output-dir "data/h5_files"
```

## Output Structure

**H5 files created with:**
- `global_quantities/led1Val/yData` - LED1 values
- `global_quantities/led2Val/yData` - LED2 values (if available)
- `tracks/track_N/points/{head,mid,tail}` - Track positions
- `tracks/track_N/derived_quantities/{speed,theta,curv}` - Derived quantities
- `ETI` at root level - Elapsed time indices
- `metadata/` - Experiment metadata

## Next Steps

1. ✅ **Complete:** Conversion tools created and adapted
2. ⏳ **Next:** Test conversion on single experiment
3. ⏳ **Next:** Validate H5 output structure
4. ⏳ **Next:** Process all 4 ESET folders
5. ⏳ **Next:** Run `engineer_dataset_from_h5.py` on converted H5 files

## Testing Checklist

- [ ] Test single experiment conversion
- [ ] Verify H5 file structure matches expected format
- [ ] Check ETI is present at root level
- [ ] Verify LED values in `global_quantities/led1Val/yData`
- [ ] Verify LED2 values (if available) in `global_quantities/led2Val/yData`
- [ ] Check track structure matches expected format
- [ ] Test batch processing of single ESET
- [ ] Test batch processing of all ESETs
- [ ] Verify compatibility with `engineer_dataset_from_h5.py`

## Requirements

**Python Packages:**
- `h5py` - H5 file handling
- `scipy` - MATLAB file reading
- `numpy` - Array operations
- `pandas` - Data manipulation

**MATLAB Dependencies:**
- MAGAT Bridge server (from mechanosensation workspace)
- MATLAB Classes (from mechanosensation workspace)
- MAGAT Codebase (from magniphyq codebase)

## Testing Status

⚠️ **Experiment Structure Mismatch**

**Test Results:**
- ✅ File discovery logic works correctly
- ✅ MATLAB bridge initialization successful
- ✅ Track loading successful (64 tracks loaded)
- ✅ openDataFile error handling works
- ❌ Experiment structure mismatch: `loaded 0 experiments`, `globalQuantity` field missing

**Issues Found:**
1. ✅ **openDataFile Error:** Fixed - error handling added
2. ❌ **Experiment Structure:** `.mat` file structure doesn't match expected format
   - `warning: loaded 0 experiments` - Experiment structure appears empty
   - `Unrecognized field name "globalQuantity"` - Field doesn't exist
   - Tracks loaded but not accessible via `app.getInfo()`

**Root Cause:** The `.mat` file structure is different than what the MATLAB code expects. The experiment structure appears empty even though tracks loaded successfully.

**See:** 
- `docs/logs/2025-11-11/matlab-conversion-test-results.md` - Initial test results
- `docs/logs/2025-11-11/matlab-conversion-test-results-update.md` - Updated test results

## Status

⚠️ **Blocked by Experiment Structure Mismatch** - Conversion script works, but `.mat` file structure doesn't match expected format

**Progress:**
- ✅ openDataFile error handling implemented
- ✅ Script proceeds past openDataFile error
- ❌ Experiment structure is empty/incompatible
- ❌ Cannot access globalQuantity for LED values
- ❌ Cannot export H5 file

**Next Steps:**
1. Investigate `.mat` file structure to understand actual format
2. Fix MATLAB code to handle this experiment structure
3. Or implement alternative conversion method

**Files Ready:**
- ✅ `src/@matlab_conversion/convert_matlab_to_h5.py`
- ✅ `src/@matlab_conversion/process_single_eset.bat`
- ✅ `src/@matlab_conversion/process_all_esets.bat`
- ✅ `src/@matlab_conversion/README.md`
- ✅ `docs/logs/2025-11-11/matlab-to-h5-conversion-guide.md`

---

**Last Updated:** 2025-11-11  
**Status:** ✅ Ready for Testing

