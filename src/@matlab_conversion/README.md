# MATLAB to H5 Conversion Tools

This directory contains tools for converting MATLAB experiment data (ESET format) to H5 files compatible with `scripts/engineer_dataset_from_h5.py`.

## Folder Structure

```
src/@matlab_conversion/
├── convert_matlab_to_h5.py    # Main conversion script
├── process_all_esets.bat      # Batch process all ESET folders
├── process_single_eset.bat    # Process single ESET folder
└── README.md                   # This file
```

## ESET Folder Structure

ESET folders are located in `data/matlab_data/GMR61@GMR61/` and have the following structure:

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

## Required H5 Output Structure

The conversion script must create H5 files with the following structure (compatible with `engineer_dataset_from_h5.py`):

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

## Usage

### Process Single ESET

```batch
process_single_eset.bat T_Re_Sq_0to250PWM_30#T_Bl_Sq_5to15PWM_30 [output_dir]
```

Example:
```batch
process_single_eset.bat T_Re_Sq_0to250PWM_30#T_Bl_Sq_5to15PWM_30 data\h5_files
```

### Process All ESETs

```batch
process_all_esets.bat [output_dir]
```

Example:
```batch
process_all_esets.bat data\h5_files
```

Default output directory: `data/h5_files/`

### Direct Python Usage

```bash
python src/@matlab_conversion/convert_matlab_to_h5.py \
    --eset-dir "data/matlab_data/GMR61@GMR61/T_Re_Sq_0to250PWM_30#T_Bl_Sq_5to15PWM_30" \
    --mat-file "btdfiles/btd_GMR61@GMR61_T_Re_Sq_0to250PWM_30#T_Bl_Sq_5to15PWM_30_202510301513.mat" \
    --output-dir "data/h5_files"
```

## Available ESET Folders

Located in `data/matlab_data/GMR61@GMR61/`:

1. `T_Re_Sq_0to250PWM_30#C_Bl_7PWM`
2. `T_Re_Sq_0to250PWM_30#T_Bl_Sq_5to15PWM_30`
3. `T_Re_Sq_50to250PWM_30#C_Bl_7PWM`
4. `T_Re_Sq_50to250PWM_30#T_Bl_Sq_5to15PWM_30`

## Implementation Status

**Status:** ✅ Complete - Ready for Testing

**Completed:**
1. ✅ Folder structure created (`src/@matlab_conversion/`)
2. ✅ Conversion script implemented (`convert_matlab_to_h5.py`)
3. ✅ Batch scripts created and adapted for ESET folder structure
4. ✅ File discovery logic verified against actual folder structure
5. ✅ ESET naming convention support verified

**Next Steps:**
1. ⏳ Test conversion on single experiment
2. ⏳ Validate H5 output structure
3. ⏳ Process all 4 ESET folders
4. ⏳ Validate compatibility with `engineer_dataset_from_h5.py`

## Troubleshooting

### File Not Found Errors
- Ensure ESET folder exists in `data/matlab_data/GMR61@GMR61/`
- Check that `.mat` files are in `btdfiles/` subdirectory
- Verify LED `.bin` files exist in root or `* sup data dir/` subdirectories

### H5 Structure Errors
- Verify output H5 files match expected structure
- Check that `engineer_dataset_from_h5.py` can read the files
- Ensure LED values are in `global_quantities/led1Val/yData` and `global_quantities/led2Val/yData`

## References

- Target script: `scripts/engineer_dataset_from_h5.py`
- Reference H5 structure: See `engineer_dataset_from_h5.py` lines 125-150
- MATLAB data location: `data/matlab_data/GMR61@GMR61/`

## Adaptations Made

**Batch Script Updates (2025-11-11):**
- ✅ Fixed Python script path: `src\@matlab_conversion\convert_matlab_to_h5.py`
- ✅ Added iteration over all `.mat` files in `btdfiles/` subdirectory
- ✅ Added `--mat-file` argument for each file
- ✅ Added success/failure counting and summary reporting
- ✅ Proper error handling and validation

**File Discovery Logic:**
- ✅ Timestamp extraction: Pattern `_(\d{12})\.mat$` matches actual filenames
- ✅ Base name construction: Removes `btd_` prefix correctly
- ✅ Tracks directory lookup: Checks `matfiles/` subdirectory first
- ✅ File validation: Checks all required files exist before processing

---

**Created:** 2025-11-11  
**Authors:** mechanobro (infrastructure setup), larry (adaptations)  
**Status:** ✅ Complete - Ready for Testing










