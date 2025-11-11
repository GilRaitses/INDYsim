# MATLAB to H5 Conversion Guide for Agents

**Date:** 2025-11-11  
**Prepared by:** larry  
**Purpose:** Guide for agents on how to convert MATLAB ESET data to H5 format

## Overview

This guide explains how to use the conversion scripts to convert MATLAB experiment data (ESET folders) to H5 format compatible with `engineer_dataset_from_h5.py`.

## Prerequisites

**Required Python Packages:**
```bash
pip install h5py scipy numpy pandas
```

**Required Files:**
- `scripts/2025-11-11/convert_matlab_to_h5.py` - Python conversion script
- `scripts/2025-11-11/process_all_esets.bat` - Batch script for all ESETs
- `scripts/2025-11-11/process_single_eset.bat` - Batch script for single ESET

## Quick Start

### Option 1: Process All 4 ESET Folders (Recommended)

**Using Batch Script:**
```batch
REM Process all ESETs to default output directory (data/h5_files)
scripts\2025-11-11\process_all_esets.bat

REM Or specify custom output directory
scripts\2025-11-11\process_all_esets.bat data\my_h5_files
```

**What it does:**
- Processes all 4 ESET folders automatically
- Creates H5 files in `data/h5_files/` (or specified directory)
- Shows progress for each ESET

### Option 2: Process Single ESET Folder

**Using Batch Script:**
```batch
REM Process single ESET
scripts\2025-11-11\process_single_eset.bat T_Re_Sq_0to250PWM_30#T_Bl_Sq_5to15PWM_30

REM Or specify custom output directory
scripts\2025-11-11\process_single_eset.bat T_Re_Sq_0to250PWM_30#T_Bl_Sq_5to15PWM_30 data\my_h5_files
```

**Using Python Script Directly:**
```bash
python scripts/2025-11-11/convert_matlab_to_h5.py \
    --eset-dir "data/matlab_data/GMR61@GMR61/T_Re_Sq_0to250PWM_30#T_Bl_Sq_5to15PWM_30" \
    --output-dir "data/h5_files"
```

## ESET Folder Structure

**Location:** `data/matlab_data/GMR61@GMR61/`

**Available ESET Folders:**
1. `T_Re_Sq_0to250PWM_30#T_Bl_Sq_5to15PWM_30`
2. `T_Re_Sq_0to250PWM_30#C_Bl_7PWM`
3. `T_Re_Sq_50to250PWM_30#C_Bl_7PWM`
4. `T_Re_Sq_50to250PWM_30#T_Bl_Sq_5to15PWM_30`

**Each ESET folder contains:**
```
ESET_NAME/
├── btdfiles/
│   └── btd_*.mat (track data files)
├── * sup data dir/
│   ├── * led1 values.bin
│   ├── * led2 values.bin
│   └── ... (other files)
├── *.bin (main .bin file)
├── *.mdat
└── ... (other files)
```

## Conversion Process

**Step 1: Find LED Values**
- Script searches for `*led1 values.bin` and `*led2 values.bin` files
- Looks in root directory and `* sup data dir/` subdirectories
- Reads binary format (uint16 PWM values)

**Step 2: Find Track Data**
- Script searches for `.mat` files in `btdfiles/` subdirectory
- Reads MATLAB structure containing track positions and derived quantities

**Step 3: Create H5 File**
- Creates H5 file with structure compatible with `engineer_dataset_from_h5.py`:
  ```
  h5_file.h5
  ├── global_quantities/
  │   ├── led1Val/yData
  │   └── led2Val/yData
  ├── tracks/
  │   └── track_N/
  │       ├── points/{head,mid,tail}
  │       └── derived_quantities/{speed,theta,curv}
  └── metadata/
  ```

## Output

**H5 Files Created:**
- Location: `data/h5_files/` (or specified output directory)
- Naming: `{ESET_NAME}.h5`
- Example: `T_Re_Sq_0to250PWM_30#T_Bl_Sq_5to15PWM_30.h5`

**Next Steps:**
After conversion, use `engineer_dataset_from_h5.py` to process the H5 files:
```bash
python scripts/engineer_dataset_from_h5.py \
    --h5-dir data/h5_files \
    --output-dir data/engineered
```

## Troubleshooting

### Error: "No LED1 bin files found"
**Cause:** LED bin files not found in expected locations  
**Solution:**
- Check that ESET folder contains `*led1 values.bin` files
- Verify files are in root or `* sup data dir/` subdirectories
- Check file naming matches pattern

### Error: "No .mat files found in btdfiles/"
**Cause:** Track data files not found  
**Solution:**
- Verify `btdfiles/` subdirectory exists
- Check that `.mat` files are present
- Verify file naming matches pattern `btd_*.mat`

### Error: "Failed to read LED1 values"
**Cause:** Binary file format issue  
**Solution:**
- Verify bin files are readable
- Check file permissions
- Ensure files are not corrupted

### Error: "h5py not installed"
**Solution:**
```bash
pip install h5py scipy numpy pandas
```

### Error: "scipy not installed"
**Solution:**
```bash
pip install scipy
```

## Batch Script Options

### process_all_esets.bat

**Usage:**
```batch
process_all_esets.bat [output_dir]
```

**Arguments:**
- `output_dir` (optional): Output directory for H5 files
  - Default: `data/h5_files`

**Example:**
```batch
REM Use default output directory
process_all_esets.bat

REM Use custom output directory
process_all_esets.bat data\my_h5_files
```

### process_single_eset.bat

**Usage:**
```batch
process_single_eset.bat [eset_name] [output_dir]
```

**Arguments:**
- `eset_name` (required): Name of ESET folder to process
- `output_dir` (optional): Output directory for H5 files
  - Default: `data/h5_files`

**Example:**
```batch
REM Use default output directory
process_single_eset.bat T_Re_Sq_0to250PWM_30#T_Bl_Sq_5to15PWM_30

REM Use custom output directory
process_single_eset.bat T_Re_Sq_0to250PWM_30#T_Bl_Sq_5to15PWM_30 data\my_h5_files
```

## Python Script Options

### convert_matlab_to_h5.py

**Usage:**
```bash
python convert_matlab_to_h5.py --eset-dir <path> --output-dir <path>
```

**Arguments:**
- `--eset-dir` (required): Path to ESET folder
- `--output-dir` (optional): Output directory for H5 files
  - Default: `data/h5_files`

**Example:**
```bash
python scripts/2025-11-11/convert_matlab_to_h5.py \
    --eset-dir "data/matlab_data/GMR61@GMR61/T_Re_Sq_0to250PWM_30#T_Bl_Sq_5to15PWM_30" \
    --output-dir "data/h5_files"
```

## Verification

**After conversion, verify H5 files:**
```python
import h5py

h5_path = "data/h5_files/T_Re_Sq_0to250PWM_30#T_Bl_Sq_5to15PWM_30.h5"
with h5py.File(h5_path, 'r') as f:
    print("LED1 values:", f['global_quantities/led1Val/yData'].shape)
    if 'led2Val' in f['global_quantities']:
        print("LED2 values:", f['global_quantities/led2Val/yData'].shape)
    print("Tracks:", list(f['tracks'].keys()))
```

## Notes for Agents

1. **Always use batch scripts** - They handle paths and error checking automatically
2. **Check output directory** - Verify H5 files are created successfully
3. **Process all ESETs** - Use `process_all_esets.bat` to convert all 4 conditions
4. **Verify before proceeding** - Check H5 files before running `engineer_dataset_from_h5.py`
5. **Report issues** - If conversion fails, note the error and ESET name

## Related Files

- `scripts/engineer_dataset_from_h5.py` - Processes H5 files after conversion
- `docs/logs/2025-11-11/lab-eset-naming-convention.md` - ESET naming details
- `scripts/2025-11-11/agent-handoffs/conejo-code_larry_20251111-131500_new-experiments-processing-handoff.md` - Context

---

**Status:** Ready for use  
**Last Updated:** 2025-11-11

