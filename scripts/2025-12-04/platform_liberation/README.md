# Platform Liberation: MATLAB → Python Pipeline Transfer

**Date:** 2025-12-04  
**Status:** Validated and Production-Ready

## Overview

This directory contains the complete prototype for transferring MAGAT analysis pipelines from MATLAB to Python. The framework includes:

1. **H5 Export** - Convert MATLAB experiment files to HDF5 format
2. **Validation** - Prove numerical equivalence between MATLAB and Python
3. **Analysis** - Python implementation of reverse crawl detection

## Validation Results

| Metric | Status |
|--------|--------|
| Schema validation | 10/10 experiments pass |
| Camera calibration match | 7/7 fields exact match |
| SpeedRunVel computation | Identical values |
| Reversal detection | Identical results |

## Directory Structure

```
platform_liberation/
├── README.md                    # This file
├── engineer_data.py             # Main Python analysis script
├── copy_from_mechanosensation.py # Setup script
│
├── h5_export/                   # MATLAB → H5 conversion
│   ├── convert_matlab_to_h5.py  # Full experiment export
│   ├── append_camcal_to_h5.py   # Add camera calibration
│   ├── batch_export_esets.py    # Batch processing
│   └── README.md
│
└── validation/                  # Validation framework
    ├── matlab/                  # MATLAB reference scripts
    │   ├── load_experiment_and_compute.m
    │   ├── compute_heading_unit_vector.m
    │   ├── compute_velocity_and_speed.m
    │   ├── compute_speedrunvel.m
    │   ├── detect_reversals.m
    │   └── rate_from_time_corrected.m
    │
    ├── python/                  # Python validation scripts
    │   ├── load_experiment_and_compute.py
    │   ├── validate_h5_schema.py
    │   ├── validate_camcal.py
    │   ├── validate_data_integrity.py
    │   └── camera_calibration.py
    │
    ├── run_full_validation.py   # Batch schema validation
    ├── run_matlab_validation.py # Run MATLAB via engine
    ├── FIELD_MAPPING.md         # H5 field documentation
    └── DISCREPANCY_REPORT.md    # Issues found and resolved
```

## Quick Start

### 1. Setup (First Time)

```powershell
cd D:\INDYsim\scripts\2025-12-04\platform_liberation
python copy_from_mechanosensation.py
```

### 2. Validate All H5 Files

```powershell
cd validation
python run_full_validation.py --base-dir "D:\rawdata\GMR61@GMR61"
```

### 3. Validate Camera Calibration

```powershell
cd validation/python
python validate_camcal.py --batch --base-dir "D:\rawdata\GMR61@GMR61"
```

### 4. Run Analysis

```powershell
python engineer_data.py "D:\INDYsim\data\h5_validated\*.h5" --output results/
```

## Key Findings (Manuscript Ready)

### Critical Data Source Issue

**Problem:** Using raw position data (`points/loc`) instead of smoothed position (`derived_quantities/sloc`) produces SpeedRunVel values 5-7x larger and causes false negatives in reversal detection.

**Solution:** Always use `derived_quantities/sloc` to match MATLAB's `getDerivedQuantity('sloc')`.

### H5 Schema Requirements

| MATLAB Field | H5 Path | Required |
|--------------|---------|----------|
| `camcalinfo.lengthPerPixel` | `/lengthPerPixel` | Yes |
| `camcalinfo.realx/realy/camx/camy` | `/camcalinfo/*` | Yes |
| `track.dq.sloc` | `/tracks/{n}/derived_quantities/sloc` | Yes |
| `track.dq.shead` | `/tracks/{n}/derived_quantities/shead` | Yes |
| `track.dq.smid` | `/tracks/{n}/derived_quantities/smid` | Yes |
| `globalQuantity('led1Val')` | `/global_quantities/led1Val/yData` | Yes |

## Validated H5 Files

Location: `D:\INDYsim\data\h5_validated\`

- 10 experiment files with MD5 checksums
- `manifest.json` with provenance tracking

## Dependencies

- Python 3.11+
- numpy, h5py, scipy
- matlab.engine (for validation only)

