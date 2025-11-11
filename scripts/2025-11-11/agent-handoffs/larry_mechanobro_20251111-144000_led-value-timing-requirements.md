# Handoff: LED Value Timing Requirements for H5 Export

**From:** larry  
**To:** mechanobro  
**Date:** 2025-11-11 14:40:00  
**Priority:** Critical  
**Status:** Requirements - LED Value Timing Integration Needed

## Summary

Based on today's findings and the critical P0 LED alignment work, the H5 export script needs to properly extract and align LED values with track timecode. This is essential for accurate stimulus-response analysis with proper temporal resolution.

## Critical Context: LED Value Timecode Alignment

**Background:** This is a **critical P0 task** that was identified because we have biologically implausible stimulus-response event dynamics due to misaligned LED value timecode.

**Reference Work:**
- Task 0.1 (osito-tender): Investigated Track Extraction Software LED value format
- Task 0.2 (conejo-code): Implemented MATLAB `GlobalLookupTable` alignment method in Python
- See: `docs/logs/2025-11-11/LED_TIMECODE_ALIGNMENT_ISSUE.md`

## What We Learned Today

### 1. Native Folder Structure and Naming Convention

**Critical:** The conversion script must handle the native lab folder structure and naming conventions.

**Directory Hierarchy:**
```
data/matlab_data/
└── {GENOTYPE}@{GENOTYPE}/          (e.g., GMR61@GMR61)
    └── {ESET_FOLDER}/               (e.g., T_Re_Sq_0to250PWM_30#C_Bl_7PWM)
        ├── btdfiles/
        │   └── btd_{GENOTYPE}@{GENOTYPE}_{ESET}_{TIMESTAMP}.mat
        ├── matfiles/
        │   └── {GENOTYPE}@{GENOTYPE}_{TIMESTAMP} - tracks/
        ├── {GENOTYPE}@{GENOTYPE}_{ESET}_{TIMESTAMP} sup data dir/
        │   ├── {GENOTYPE}@{GENOTYPE}_{ESET}_{TIMESTAMP} led1 values.bin
        │   └── {GENOTYPE}@{GENOTYPE}_{ESET}_{TIMESTAMP} led2 values.bin
        └── {GENOTYPE}@{GENOTYPE}_{ESET}_{TIMESTAMP}.bin
```

**Key Naming Components:**

1. **Genotype Pattern:** `{GENOTYPE}@{GENOTYPE}` (e.g., `GMR61@GMR61`)
   - Represents optogenetic variant of larva
   - Used in parent folder name and file names
   - Currently hardcoded as `GMR61@GMR61` in `find_experiment_files()` - **needs to be parsed dynamically**

2. **ESET Folder Naming:** `T_{LED1_COLOR}_{WAVEFORM}_{MIN}to{MAX}PWM_{REST}#{LED2_TYPE}_{LED2_COLOR}_{LED2_VALUES}PWM`
   - **Example:** `T_Re_Sq_0to250PWM_30#C_Bl_7PWM`
   - **LED1:** `T_Re_Sq_0to250PWM_30` = Time-based, Red, Square wave, 0-250 PWM, 30s rest
   - **LED2:** `C_Bl_7PWM` = Constant Blue at 7 PWM
   - **Separator:** `#` separates LED1 and LED2 specifications
   - **Reference:** See `docs/logs/2025-11-11/lab-eset-naming-convention.md` for full format details

3. **Base Name Construction:** `{GENOTYPE}@{GENOTYPE}_{ESET}_{TIMESTAMP}`
   - **Example:** `GMR61@GMR61_T_Re_Sq_0to250PWM_30#C_Bl_7PWM_202510291652`
   - Used to construct all file and directory names
   - Extracted from `.mat` filename (remove `btd_` prefix and `.mat` extension)

4. **Timestamp Format:** `YYYYMMDDHHMM` (12 digits, e.g., `202510291652`)

**File Discovery Patterns:**

**Current Implementation Issues:**
- `find_experiment_files()` hardcodes `GMR61@GMR61` for tracks directory
- Should parse genotype from mat filename or ESET folder path
- Needs to handle variations in naming (spaces, underscores)

**Required File Discovery Logic:**
1. **Parse genotype** from mat filename or ESET parent folder
   - Pattern: `{GENOTYPE}@{GENOTYPE}` (e.g., `GMR61@GMR61`)
   - Extract from mat filename: `btd_{GENOTYPE}@{GENOTYPE}_...`
   - Or from parent folder path: `.../{GENOTYPE}@{GENOTYPE}/{ESET}/`

2. **Extract timestamp** from mat filename
   - Pattern: `_(\d{12})\.mat$` (12 digits before `.mat`)

3. **Construct base_name** from mat filename
   - Remove `btd_` prefix
   - Remove `.mat` extension
   - Result: `{GENOTYPE}@{GENOTYPE}_{ESET}_{TIMESTAMP}`

4. **Find tracks directory** using pattern matching
   - Pattern: `{GENOTYPE}@{GENOTYPE}_{TIMESTAMP} - tracks`
   - Location: `matfiles/` subdirectory (or root if not found)

5. **Find sup data directory** using pattern matching
   - Pattern: `{base_name} sup data dir`
   - Location: Root level of ESET folder

6. **Find LED bin files** within sup data directory
   - Pattern: `{base_name} led1 values.bin`
   - Pattern: `{base_name} led2 values.bin` (may not exist)

**Example Full Path Structure:**
```
data/matlab_data/GMR61@GMR61/T_Re_Sq_0to250PWM_30#C_Bl_7PWM/
├── btdfiles/
│   └── btd_GMR61@GMR61_T_Re_Sq_0to250PWM_30#C_Bl_7PWM_202510291652.mat
├── matfiles/
│   └── GMR61@GMR61_202510291652 - tracks/
├── GMR61@GMR61_T_Re_Sq_0to250PWM_30#C_Bl_7PWM_202510291652 sup data dir/
│   ├── GMR61@GMR61_T_Re_Sq_0to250PWM_30#C_Bl_7PWM_202510291652 led1 values.bin ✅
│   └── GMR61@GMR61_T_Re_Sq_0to250PWM_30#C_Bl_7PWM_202510291652 led2 values.bin ✅
└── GMR61@GMR61_T_Re_Sq_0to250PWM_30#C_Bl_7PWM_202510291652.bin
```

**Naming Convention Reference:**
- See `docs/logs/2025-11-11/lab-eset-naming-convention.md` for complete ESET naming format breakdown
- Includes LED color codes (`Re`, `Bl`), waveform types (`Sq`), PWM ranges, rest intervals
- Handles constant LED2 (`C_Bl_7PWM`) vs time-varying LED2 (`T_Bl_Sq_5to15PWM_30`)

### 2. LED Value File Locations

**Found in ESET folder structure:**
- **LED1 values:** `* sup data dir/*led1 values.bin`
- **LED2 values:** `* sup data dir/*led2 values.bin`
- **Main .bin file:** Root level `{base_name}.bin` (format may be different - got buffer size error)

**Example:**
```
T_Re_Sq_0to250PWM_30#T_Bl_Sq_5to15PWM_30/
├── GMR61@GMR61_T_Re_Sq_0to250PWM_30#T_Bl_Sq_5to15PWM_30_202510301513 sup data dir/
│   ├── GMR61@GMR61_T_Re_Sq_0to250PWM_30#T_Bl_Sq_5to15PWM_30_202510301513 led1 values.bin ✅
│   └── GMR61@GMR61_T_Re_Sq_0to250PWM_30#T_Bl_Sq_5to15PWM_30_202510301513 led2 values.bin ✅
└── GMR61@GMR61_T_Re_Sq_0to250PWM_30#T_Bl_Sq_5to15PWM_30_202510301513.bin (main file - different format?)
```

### 3. LED Value File Format

**Format:** Binary uint16 values (PWM values 0-255)

**Reading:**
```python
with open(bin_path, 'rb') as f:
    data = np.frombuffer(f.read(), dtype=np.uint16)
```

**Test Results:**
- ✅ Successfully read LED1 values from `*led1 values.bin`
- ✅ Successfully read LED2 values from `*led2 values.bin`
- ❌ Main `.bin` file gave buffer size error (may be different format)

### 4. Critical Timing Alignment Issue

**Problem:** LED values must be aligned with track timecode (ETI - Elapsed Time Indices) for accurate temporal analysis.

**Why It Matters:**
- Stimulus-response analysis requires proper temporal alignment
- Misalignment causes biologically implausible dynamics
- Period-relative timing (`led12Val_ton/toff`) depends on accurate alignment

**Reference Implementation:**
- MATLAB `GlobalLookupTable` method (from `Matlab-Track-Analysis-SkanataLab` codebase)
- See `docs/logs/2025-11-11/LED_TIMECODE_ALIGNMENT_ISSUE.md` for MATLAB reference code
- Uses `averageInPrecedingBin` MATLAB function for binning/interpolation

### 5. Required H5 Structure

**LED values must be in:**
```
h5_file.h5
├── global_quantities/
│   ├── led1Val/
│   │   └── yData (aligned LED1 values array)
│   └── led2Val/
│       └── yData (aligned LED2 values array)
└── eti (at root level - Elapsed Time Indices for alignment)
```

**Critical:** LED values must be aligned with ETI for proper temporal analysis.

## Requirements for H5 Export Script

### 1. Handle Native Folder Structure and Naming

**Critical:** The script must robustly handle the native ESET folder structure and naming conventions.

**Required Functionality:**
- **Parse genotype** from mat filename or ESET parent folder (don't hardcode `GMR61@GMR61`)
- **Extract timestamp** from mat filename (12-digit pattern)
- **Construct base_name** dynamically from mat filename components
- **Discover files** using pattern matching (not hardcoded paths):
  - Find `.mat` files in `btdfiles/` subdirectory
  - Find tracks directories in `matfiles/` subdirectory (pattern: `{GENOTYPE}@{GENOTYPE}_{TIMESTAMP} - tracks`)
  - Find sup data directories (pattern: `{base_name} sup data dir`)
  - Locate LED bin files within sup data directories
- **Handle naming variations:**
  - Spaces vs underscores
  - Timestamp variations
  - Missing components (e.g., no LED2, constant values)

**Current Implementation:**
- `find_experiment_files()` function handles file discovery
- **Issue:** Hardcodes `GMR61@GMR61` for tracks directory (line 60)
- **Fix Needed:** Parse genotype dynamically from mat filename or parent folder

**Reference:** See `src/@matlab_conversion/convert_matlab_to_h5.py` `find_experiment_files()` function (lines 32-94)

### 2. Extract LED Values from .bin Files

**Current Status:** ✅ Partially working
- Can read `*led1 values.bin` and `*led2 values.bin` from sup data dir
- Main `.bin` file format needs investigation

**Required:**
- Read LED1 values from `*led1 values.bin`
- Read LED2 values from `*led2 values.bin` (if available)
- Handle main `.bin` file format (may need different parsing)

### 3. Align LED Values with Track Timecode

**Critical Requirement:** LED values must be aligned with track ETI (Elapsed Time Indices)

**Reference Methods:**
- MATLAB `GlobalLookupTable` alignment (see `Matlab-Track-Analysis-SkanataLab` codebase)
- Uses `ledFrame` and `bufnum` metadata for timecode mapping
- Uses `bytesperframe`/`bitsperframe` calculations for mapping LED values to time
- Uses `averageInPrecedingBin` MATLAB function for binning logic

**Alignment Process:**
1. Get track ETI (elapsed time indices) - typically from experiment structure or calculated from frame rate
2. Get global LED values (from .bin files)
3. Align LED values to track timecode using interpolation/binning
4. Handle length mismatches (e.g., 23991 global LED vs 24001 track frames)

**Key MATLAB Functions (reference):**
- `GlobalLookupTable` - Main alignment class
- `averageInPrecedingBin` - Binning logic for interpolation
- `createLedTableFromBitFile` - Creates lookup table from bit files

**Key Concepts:**
- `ledFrame` and `bufnum` - Metadata fields used for timecode mapping
- `bytesperframe`/`bitsperframe` - Calculations for mapping LED values to time
- Buffer axis to bit axis mapping - Maps buffer numbers to LED value positions

### 4. Export Aligned LED Values to H5

**Structure:**
```python
# In global_quantities group
led1_grp = gq_grp.create_group('led1Val')
led1_grp.create_dataset('yData', data=aligned_led1_values.astype(np.float32))

if led2_available:
    led2_grp = gq_grp.create_group('led2Val')
    led2_grp.create_dataset('yData', data=aligned_led2_values.astype(np.float32))
```

### 5. Export ETI for Alignment Reference

**Critical:** ETI must be at root level for timing calculations

```python
# At root level
f.create_dataset('eti', data=eti_array, **comp)
```

**Note:** If ETI is not available from experiment structure, it may need to be calculated from:
- Frame rate (typically 10 fps)
- Number of frames in tracks
- Or extracted from track data

## Implementation Guidance

### Reference Files

1. **LED Alignment Documentation:**
   - `docs/logs/2025-11-11/LED_TIMECODE_ALIGNMENT_ISSUE.md` - Problem statement and root cause analysis
   - `docs/work-trees/2025-11-11-work-tree.md` - Task breakdown (Task 0.1 and 0.2)

2. **MATLAB Reference:**
   - `D:\magniphyq\codebase\Matlab-Track-Analysis-SkanataLab` - Contains `GlobalLookupTable` method
   - Key files: `@GlobalLookupTable/createLedTableFromBitFile.m`, `@GlobalLookupTable/GlobalLookupTable.m`

3. **Current Python Implementation:**
   - `scripts/engineer_dataset_from_h5.py` - Processes H5 files (assumes LED values are pre-aligned)
   - Uses `extract_stimulus_timing()` to get LED values from H5
   - Period-relative timing calculations depend on proper alignment

### Key Considerations

1. **Length Mismatches:** Global LED values and track frames may have different lengths
   - Example: 23991 global LED vs 24001 track frames
   - Use interpolation/binning to align

2. **Timecode Source:** ETI may come from:
   - Experiment structure (`elapsedTime` field) - currently missing
   - Track data (calculate from frame rate)
   - Or needs to be calculated from track timestamps

3. **Binning Logic:** Use `averageInPrecedingBin` MATLAB method for MATLAB-compatible alignment (see MATLAB reference)

4. **Period-Relative Timing:** Once aligned, LED values can be used for:
   - Period detection
   - `led12Val_ton/toff` calculation
   - Stimulus-response analysis

## Current Test Results

**What Works:**
- ✅ Can read LED1/LED2 values from `*led1 values.bin` / `*led2 values.bin`
- ✅ Can export LED values to H5 structure
- ✅ Tracks load successfully (64 tracks)

**What Needs Work:**
- ❌ LED values not aligned with track timecode (just raw values)
- ❌ ETI not available from experiment structure
- ❌ Main `.bin` file format unclear

## Next Steps for mechanobro

1. **Fix Genotype Parsing:** Update `find_experiment_files()` to parse genotype dynamically from mat filename or parent folder (currently hardcoded as `GMR61@GMR61`)

2. **Verify Folder Structure Handling:** Ensure the script robustly handles all ESET folder naming variations and file discovery patterns

3. **Integrate LED Alignment:** Implement the MATLAB `GlobalLookupTable` alignment method to align LED values with track ETI (see MATLAB reference code in `docs/logs/2025-11-11/LED_TIMECODE_ALIGNMENT_ISSUE.md`)

4. **Handle ETI:** Either extract ETI from experiment structure or calculate from track data/frame rate

5. **Test Alignment:** Verify that aligned LED values match MATLAB reference output

6. **Export to H5:** Ensure aligned LED values are exported to `global_quantities/led1Val/yData` and `global_quantities/led2Val/yData`

7. **Test with All ESET Folders:** Verify the script works with all 4 ESET folder structures and naming conventions

8. **Provide Updated Script:** Once complete, provide the updated conversion script for testing

## Success Criteria

**H5 file must have:**
- ✅ `global_quantities/led1Val/yData` - Aligned LED1 values (length matches track frames)
- ✅ `global_quantities/led2Val/yData` - Aligned LED2 values (if available)
- ✅ `eti` at root level - Elapsed time indices for timing calculations
- ✅ Tracks exported with proper structure
- ✅ LED values temporally aligned with tracks (critical for analysis)

**Validation:**
- Compare aligned LED values with MATLAB `GlobalLookupTable` output
- Verify temporal alignment (stimulus onset should align with LED value changes)
- Check that period-relative timing calculations work correctly

## References

**Critical Documents:**
- `docs/logs/2025-11-11/LED_TIMECODE_ALIGNMENT_ISSUE.md` - Original problem statement and MATLAB reference code
- `docs/work-trees/2025-11-11-work-tree.md` - Task breakdown (Task 0.1 and 0.2 details)

**Code References:**
- MATLAB: `D:\magniphyq\codebase\Matlab-Track-Analysis-SkanataLab` - `GlobalLookupTable` method
  - `@GlobalLookupTable/GlobalLookupTable.m` - Main alignment class
  - `@GlobalLookupTable/createLedTableFromBitFile.m` - Creates lookup table from bit files
  - `@GlobalLookupTable/averageInPrecedingBin.m` - Binning logic
- Track Extraction Software: `D:\magniphyq\codebase\Track-Extraction-Software` - Generates LED value bin files
- Current Conversion Script: `src/@matlab_conversion/convert_matlab_to_h5.py` - File discovery logic (needs genotype parsing fix)

## Status

**Current:** LED values can be read from .bin files but are not aligned with track timecode  
**Required:** Integrate LED alignment logic into H5 export script  
**Priority:** Critical (P0) - Blocks accurate stimulus-response analysis

---

**larry**  
**Date:** 2025-11-11 14:40:00  
**Status:** Requirements - Awaiting LED Alignment Integration

**Note:** Please integrate the LED alignment logic and provide the updated conversion script. The alignment is critical for proper temporal analysis of stimulus-response dynamics.

