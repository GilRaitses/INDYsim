# Handoff: Request for MATLAB Conversion Tools Setup

**From:** larry  
**To:** mechanobro  
**Date:** 2025-11-11 14:23:48  
**Priority:** High  
**Status:** Request - Infrastructure Setup Required

## Summary

We need to process 4 new experiment conditions from MATLAB data in the `data/matlab_data/` directory. The conversion scripts need to be organized in a proper folder structure with `@` prefix for the script class.

## Request

**Please create the following folder structure:**

```
INDYsim/
└── src/
    └── @matlab_conversion/
        ├── convert_matlab_to_h5.py
        ├── process_all_esets.bat
        ├── process_single_eset.bat
        └── README.md
```

**Folder Purpose:**
- `src/` - Source code root directory
- `@matlab_conversion/` - Script class folder (using `@` prefix convention)
- Contains all MATLAB→H5 conversion tools

## Context

**Current Situation:**
- 4 ESET folders in `data/matlab_data/GMR61@GMR61/`
- Each ESET contains:
  - `.mat` files (track data) in `btdfiles/` subdirectory
  - `.bin` files (LED values) in root and `* sup data dir/` subdirectories
  - Native lab folder structure with specific naming conventions

**Required Output:**
- H5 files compatible with `scripts/engineer_dataset_from_h5.py`
- Must handle native ESET folder structure
- Must extract LED values from `.bin` files
- Must extract track data from `.mat` files

## Files to Create

**1. `src/@matlab_conversion/convert_matlab_to_h5.py`**
- Python script to convert MATLAB data to H5 format
- Handles native ESET folder structure
- Reads `.mat` files from `btdfiles/` subdirectory
- Reads LED `.bin` files from root and `* sup data dir/` subdirectories
- Creates H5 files with structure compatible with `engineer_dataset_from_h5.py`

**2. `src/@matlab_conversion/process_all_esets.bat`**
- Batch script to process all 4 ESET folders
- Usage: `process_all_esets.bat [output_dir]`
- Default output: `data/h5_files/`

**3. `src/@matlab_conversion/process_single_eset.bat`**
- Batch script to process single ESET folder
- Usage: `process_single_eset.bat [eset_name] [output_dir]`

**4. `src/@matlab_conversion/README.md`**
- Documentation for the conversion tools
- Usage instructions
- Troubleshooting guide

## ESET Folder Structure

**Example ESET:** `T_Re_Sq_0to250PWM_30#T_Bl_Sq_5to15PWM_30`

```
T_Re_Sq_0to250PWM_30#T_Bl_Sq_5to15PWM_30/
├── btdfiles/
│   ├── btd_*_202510301513.mat
│   ├── btd_*_202510311441.mat
│   └── ... (multiple .mat files)
├── * sup data dir/
│   ├── * led1 values.bin
│   ├── * led2 values.bin
│   └── ...
├── *.bin (main .bin file)
├── *.mdat
└── ... (other files)
```

## Required H5 Structure

**Output H5 files must have:**
```
h5_file.h5
├── global_quantities/
│   ├── led1Val/
│   │   └── yData (LED1 values array)
│   └── led2Val/
│       └── yData (LED2 values array)
├── tracks/
│   └── track_N/
│       ├── points/
│       │   ├── head (N_frames, 2)
│       │   ├── mid (N_frames, 2)
│       │   └── tail (N_frames, 2)
│       └── derived_quantities/
│           ├── speed
│           ├── theta
│           └── curv
└── metadata/
    └── attrs (frame_rate, etc.)
```

## Next Steps

1. **mechanobro:** Create `src/@matlab_conversion/` folder structure
2. **larry:** Will provide conversion script adapted from yesterday's work
3. **larry:** Will create batch scripts and agent guide
4. **Next Agent:** Use batch scripts to convert all 4 ESET folders

## Status

**larry:** Requesting infrastructure setup  
**mechanobro:** Please create folder structure  
**Next:** Conversion scripts will be added once folder structure exists

---

**larry**  
**Date:** 2025-11-11 14:23:48  
**Status:** Request - Awaiting folder structure creation

