# MATLAB to H5 Conversion Pipeline - Agent Guide

## Overview

This pipeline converts MATLAB ESET (Experiment Set) data to H5 format compatible with INDYsim's analysis scripts. The conversion exports complete MAGAT (MATLAB Track Analysis) structure including tracks, global quantities, stimulus data, and **critical ETI (Experiment Time Index) at root level**.

## Source Code Locations

### Adapted Scripts (INDYsim)
- **Main Export Script:** `D:\INDYsim\src\@matlab_conversion\convert_matlab_to_h5.py`
- **Batch Processor:** `D:\INDYsim\src\@matlab_conversion\batch_export_esets.py`

### Original Source (mechanosensation)
- **Original Export Script:** `D:\mechanosensation\scripts\2025-11-10\H5_clone.py`
- **Original Batch Script:** `D:\mechanosensation\scripts\2025-11-11\batch_export_indysim.py`

### MAGAT Bridge (mechanosensation)
- **MAGAT Bridge:** `D:\mechanosensation\mcp-servers\magat-bridge\server.py`
- **MATLAB Classes:** `D:\mechanosensation\scripts\2025-10-16`
- **MAGAT Codebase:** `d:\magniphyq\codebase\Matlab-Track-Analysis-SkanataLab`

### MATLAB Compatibility Fixes
- **loadExperiment.m:** `D:\mechanosensation\scripts\2025-10-16\@DataManager\loadExperiment.m`
  - Handles missing `openDataFile` method gracefully
  - Handles missing `globalQuantity` field gracefully

## Folder Structure

### ESET Folder Structure (Native MATLAB)
```
data/matlab_data/
└── {GENOTYPE}@{GENOTYPE}/          (e.g., GMR61@GMR61)
    └── {ESET_FOLDER}/               (e.g., T_Re_Sq_0to250PWM_30#C_Bl_7PWM)
        ├── matfiles/
        │   ├── {GENOTYPE}@{GENOTYPE}_{ESET}_{TIMESTAMP}.mat
        │   └── {GENOTYPE}@{GENOTYPE}_{TIMESTAMP} - tracks/
        ├── {base_name} sup data dir/
        │   ├── {base_name} led1 values.bin
        │   └── {base_name} led2 values.bin
        └── {base_name}.bin
```

### Key Points
- **MAT files:** Located in `matfiles/` directory (NOT `btdfiles/`)
- **Tracks:** Subdirectory in `matfiles/` named `{GENOTYPE}_{TIMESTAMP} - tracks`
- **FID .bin:** Root level of ESET folder
- **LED values:** In `{base_name} sup data dir/` directory

## Usage

### Batch Processing - Full Genotype Folder (RECOMMENDED)

**Windows Batch Script:**
```bash
cd D:\INDYsim\src\@matlab_conversion
process_genotype.bat GMR61@GMR61
```

**Python Script:**
```bash
cd D:\INDYsim\src\@matlab_conversion
python batch_export_esets.py --genotype "GMR61@GMR61"
```

### Batch Processing - All Genotypes

**Windows Batch Script:**
```bash
cd D:\INDYsim\src\@matlab_conversion
process_all_genotypes.bat
```

This will automatically find and process all genotype folders in `data/matlab_data/`.

### Batch Processing - Specific ESET

```bash
cd D:\INDYsim\src\@matlab_conversion
python batch_export_esets.py \
    --genotype "GMR61@GMR61" \
    --eset "T_Re_Sq_0to250PWM_30#C_Bl_7PWM"
```

### Single Experiment Export

```bash
python convert_matlab_to_h5.py \
    --mat "D:\INDYsim\data\matlab_data\GMR61@GMR61\T_Re_Sq_0to250PWM_30#C_Bl_7PWM\matfiles\GMR61@GMR61_T_Re_Sq_0to250PWM_30#C_Bl_7PWM_202510291652.mat" \
    --tracks "D:\INDYsim\data\matlab_data\GMR61@GMR61\T_Re_Sq_0to250PWM_30#C_Bl_7PWM\matfiles\GMR61@GMR61_202510291652 - tracks" \
    --bin "D:\INDYsim\data\matlab_data\GMR61@GMR61\T_Re_Sq_0to250PWM_30#C_Bl_7PWM\GMR61@GMR61_T_Re_Sq_0to250PWM_30#C_Bl_7PWM_202510291652.bin" \
    --output "D:\INDYsim\data\GMR61@GMR61_T_Re_Sq_0to250PWM_30#C_Bl_7PWM_202510291652.h5"
```

### Batch Export - All ESETs in Genotype

```bash
cd D:\INDYsim\src\@matlab_conversion
python batch_export_esets.py --genotype "GMR61@GMR61"
```

### Batch Export - Specific ESET

```bash
cd D:\INDYsim\src\@matlab_conversion
python batch_export_esets.py \
    --genotype "GMR61@GMR61" \
    --eset "T_Re_Sq_0to250PWM_30#C_Bl_7PWM"
```

### Custom Output Directory

```bash
python batch_export_esets.py \
    --genotype "GMR61@GMR61" \
    --output-dir "D:\output\h5_files"
```

## Output Structure

### H5 File Structure
```
{base_name}.h5
├── eti                          # CRITICAL: Experiment Time Index at root
├── experiment_info/             # Experiment metadata
├── global_quantities/            # LED values, derivatives, etc.
│   ├── led1Val/
│   ├── led1ValDeriv/
│   ├── led2Val/
│   └── ...
├── tracks/                      # All track data
│   ├── track_1/
│   │   ├── points/              # mid, head, tail, loc, area, contours, spine
│   │   ├── derived_quantities/  # All derived fields
│   │   ├── state/               # State arrays
│   │   └── metadata/            # Track metadata
│   └── track_N/
├── stimulus/                    # Stimulus onset frames
├── led_data/                    # Raw LED data array
└── metadata/                    # Export metadata (num_tracks, num_frames, has_eti, etc.)
```

### Critical: ETI at Root
- **Location:** Root level dataset `eti`
- **Source:** `expt.elapsedTime` from MATLAB experiment
- **Purpose:** Accurate time calculations for simulation scripts (replaces `frame / fps`)
- **Shape:** 1D array `(N,)` where N = number of frames

## Dependencies

### Required Python Packages
- `h5py` - HDF5 file I/O
- `numpy` - Array operations
- `matlab.engine` - MATLAB Engine API

### Required MATLAB Code
- **MAGAT Bridge:** Must be accessible from `D:\mechanosensation\mcp-servers\magat-bridge\server.py`
- **MATLAB Classes:** Must be accessible from `D:\mechanosensation\scripts\2025-10-16`
- **MAGAT Codebase:** Must be accessible from `d:\magniphyq\codebase\Matlab-Track-Analysis-SkanataLab`

### MATLAB Compatibility
- MATLAB R2024a or compatible
- MATLAB Engine API installed and configured
- `loadExperiment.m` must handle missing fields gracefully (already fixed)

## File Discovery Logic

### Genotype Parsing
1. Extract from `.mat` filename: `GMR61@GMR61_T_Re_Sq_*.mat` → `GMR61@GMR61`
2. Fallback to parent folder: `{GENOTYPE}@{GENOTYPE}/`

### Timestamp Extraction
- Pattern: `_(\d{12})\.mat$` (12-digit timestamp)
- Example: `GMR61@GMR61_T_Re_Sq_0to250PWM_30#C_Bl_7PWM_202510291652.mat` → `202510291652`

### Base Name Construction
- Remove `.mat` extension
- Use full filename (no `btd_` prefix in `matfiles/`)
- Example: `GMR61@GMR61_T_Re_Sq_0to250PWM_30#C_Bl_7PWM_202510291652`

### File Path Resolution
1. **MAT file:** `matfiles/{base_name}.mat`
2. **Tracks:** `matfiles/{GENOTYPE}_{TIMESTAMP} - tracks/`
3. **FID .bin:** `{base_name}.bin` (root level)
4. **Sup data dir:** `{base_name} sup data dir/`
5. **LED bins:** `{base_name} sup data dir/{base_name} led1 values.bin`

## Validation

### Structure Validation
Compare output H5 files to reference format:
- Root keys: `eti`, `experiment_info`, `global_quantities`, `led_data`, `metadata`, `stimulus`, `tracks`
- ETI at root: `(N,)` shape, 1D array
- Track structure: `points/`, `derived_quantities/`, `state/`, `metadata/`
- Global quantities: LED values and derivatives

### Example Validation Script
```python
import h5py

with h5py.File('output.h5', 'r') as f:
    print("Root keys:", list(f.keys()))
    print("ETI:", 'eti' in f, f['eti'].shape if 'eti' in f else 'N/A')
    print("Tracks:", len(list(f['tracks'].keys())))
    print("Global quantities:", list(f['global_quantities'].keys()))
    print("Metadata:", dict(f['metadata'].attrs))
```

## Troubleshooting

### File Locking (Windows Error 33)
- **Symptom:** `OSError: Unable to lock file, errno = 33`
- **Cause:** Another process has the file open (HDFView, MATLAB, Python)
- **Solution:** Close all programs accessing the file, or delete manually

### MATLAB Errors
- **`openDataFile` not found:** Already handled in `loadExperiment.m` - should not occur
- **`globalQuantity` not found:** Already handled in `loadExperiment.m` - should not occur
- **0 experiments loaded:** Check that `.mat` files are in `matfiles/` directory (not `btdfiles/`)

### Missing Files
- **No experiments found:** Verify `.mat` files exist in `matfiles/` directory
- **Tracks directory not found:** Check naming pattern: `{GENOTYPE}_{TIMESTAMP} - tracks`
- **LED bins not found:** Check `{base_name} sup data dir/` directory exists

### Path Issues
- **MAGAT Bridge not found:** Verify `D:\mechanosensation\mcp-servers\magat-bridge\server.py` exists
- **MATLAB classes not found:** Verify `D:\mechanosensation\scripts\2025-10-16` exists
- **MAGAT codebase not found:** Verify `d:\magniphyq\codebase\Matlab-Track-Analysis-SkanataLab` exists

## Integration with INDYsim

### Output Location
- Default: `D:\INDYsim\data\{base_name}.h5`
- Custom: Use `--output-dir` argument

### Downstream Usage
- H5 files are compatible with `engineer_dataset_from_h5.py`
- ETI at root enables accurate time calculations
- Track structure matches expected format for analysis scripts

## Notes

- **No FID Export:** FID pixel data is NOT exported (Tier 2 export)
- **ETI Critical:** ETI must be at root for simulation scripts to work correctly
- **File Naming:** Output filename matches experiment `.bin` filename (with `.h5` extension)
- **Strict Validation:** Scripts use NO FALLBACKS - missing files cause export to skip

## References

- **Original Development:** `D:\mechanosensation\scripts\2025-11-10\H5_clone.py`
- **Batch Processing:** `D:\mechanosensation\scripts\2025-11-11\batch_export_indysim.py`
- **MAGAT Bridge:** `D:\mechanosensation\mcp-servers\magat-bridge\server.py`
- **MATLAB Schema:** `D:\mechanosensation\docs\magat_schema.md` (if exists)

