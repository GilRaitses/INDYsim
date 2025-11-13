# Integration Guide: Using Converted H5 Files with Analysis Pipeline

**Date:** 2025-11-13  
**Purpose:** Guide for integrating converted H5 files with the INDYsim analysis pipeline

## Overview

This guide explains how to use the converted H5 files with the analysis pipeline, including LED alignment, period-relative timing, and stimulus-locked turn rate analysis.

## Prerequisites

- H5 files converted and validated (see `docs/logs/2025-11-13/h5-file-validation-report.md`)
- Python environment with required packages:
  - `h5py`
  - `numpy`
  - `pandas`
  - Analysis pipeline scripts

## File Locations

**H5 Files:** `data/h5_files/`  
**Analysis Scripts:** `scripts/`  
**Processed Data:** `data/processed/` (created during processing)

## Step 1: Verify H5 File Structure

Before processing, verify the H5 file has the correct structure:

```python
import h5py

def verify_h5_file(h5_path):
    """Verify H5 file structure."""
    with h5py.File(h5_path, 'r') as f:
        # Check critical requirements
        assert 'eti' in f, "ETI missing at root level"
        assert 'global_quantities' in f, "Global quantities missing"
        assert 'tracks' in f, "Tracks missing"
        
        # Verify LED values
        assert 'led1Val' in f['global_quantities'], "LED1 missing"
        assert 'led2Val' in f['global_quantities'], "LED2 missing"
        
        # Verify lengths match
        eti_length = f['eti'].shape[0]
        led1_length = f['global_quantities']['led1Val']['yData'].shape[0]
        assert eti_length == led1_length, "ETI and LED1 length mismatch"
        
        print(f"✅ File verified: {eti_length} frames, {len(f['tracks'])} tracks")

# Example usage
verify_h5_file('data/h5_files/GMR61@GMR61_T_Re_Sq_0to250PWM_30#C_Bl_7PWM_202510291652.h5')
```

## Step 2: Load LED Values

LED values are stored in `global_quantities` and aligned with ETI:

```python
import h5py
import numpy as np

def load_led_values(h5_path):
    """Load LED values from H5 file."""
    with h5py.File(h5_path, 'r') as f:
        led1 = f['global_quantities']['led1Val']['yData'][:]
        led2 = f['global_quantities']['led2Val']['yData'][:]
        eti = f['eti'][:]
    return led1, led2, eti

# Example usage
led1, led2, eti = load_led_values('data/h5_files/...h5')
```

## Step 3: LED Alignment and Period Detection

The analysis pipeline uses period-relative timing variables. These are computed in `engineer_dataset_from_h5.py`:

### Period-Relative Timing Variables

- **`led12Val`:** Combined LED1 and LED2 values
- **`led12Val_ton`:** Time within LED ON period (0-15 seconds)
- **`led12Val_toff`:** Time within LED OFF period (0-15 seconds)

### Usage in Analysis Pipeline

```python
# Period detection and timing calculation
# This is handled automatically by engineer_dataset_from_h5.py

# Expected period: 10 seconds ON, 30 seconds OFF
# Period-relative timing maps each frame to its position within the period
```

## Step 4: Process H5 Files with Analysis Pipeline

Use `engineer_dataset_from_h5.py` to process H5 files:

```python
# Run analysis pipeline
python scripts/engineer_dataset_from_h5.py \
    --input data/h5_files/GMR61@GMR61_T_Re_Sq_0to250PWM_30#C_Bl_7PWM_202510291652.h5 \
    --output data/processed/processed_data.parquet
```

### Features Computed

The pipeline computes the following features:
- **Turn rate:** Rate of reorientations
- **Latency:** Time to first response
- **Stop fraction:** Fraction of time paused
- **Tortuosity:** Path complexity
- **Dispersal:** Movement spread
- **Spine curve energy:** Body curvature metrics

## Step 5: Stimulus-Locked Turn Rate Analysis

For stimulus-locked analysis, use `run_stimulus_locked_analysis_production.py`:

```python
# Run stimulus-locked analysis
python scripts/run_stimulus_locked_analysis_production.py \
    --input data/h5_files/GMR61@GMR61_T_Re_Sq_0to250PWM_30#C_Bl_7PWM_202510291652.h5 \
    --output data/processed/stimulus_locked_results.parquet
```

### Expected Output

- Turn rate aligned to stimulus onset (spike at t=0)
- Period-relative timing variables
- Behavioral metrics aligned with LED periods

## Step 6: Batch Processing

To process all 14 H5 files:

```python
from pathlib import Path
import subprocess

h5_dir = Path('data/h5_files')
output_dir = Path('data/processed')
output_dir.mkdir(exist_ok=True)

for h5_file in sorted(h5_dir.glob('*.h5')):
    print(f"Processing: {h5_file.name}")
    
    # Process with engineer_dataset_from_h5.py
    output_file = output_dir / f"{h5_file.stem}_processed.parquet"
    
    subprocess.run([
        'python', 'scripts/engineer_dataset_from_h5.py',
        '--input', str(h5_file),
        '--output', str(output_file)
    ])
    
    print(f"✅ Completed: {output_file}")
```

## Common Workflows

### Workflow 1: Single File Analysis

```python
# 1. Load and verify
h5_file = 'data/h5_files/GMR61@GMR61_T_Re_Sq_0to250PWM_30#C_Bl_7PWM_202510291652.h5'
verify_h5_file(h5_file)

# 2. Load LED values
led1, led2, eti = load_led_values(h5_file)

# 3. Process with pipeline
# (run engineer_dataset_from_h5.py)

# 4. Analyze results
# (use processed data for modeling)
```

### Workflow 2: Batch Processing

```python
# Process all files
for h5_file in Path('data/h5_files').glob('*.h5'):
    process_h5_file(h5_file)
```

### Workflow 3: Stimulus-Locked Analysis

```python
# Run stimulus-locked analysis on test files
test_files = [
    'data/h5_files/GMR61@GMR61_T_Re_Sq_0to250PWM_30#C_Bl_7PWM_202510291652.h5',
    'data/h5_files/GMR61@GMR61_T_Re_Sq_0to250PWM_30#T_Bl_Sq_5to15PWM_30_202510301513.h5',
]

for h5_file in test_files:
    run_stimulus_locked_analysis(h5_file)
```

## Troubleshooting

### Issue: LED values don't align with tracks

**Solution:** Verify ETI length matches LED value length. Use ETI as the authoritative source for alignment.

### Issue: Period detection fails

**Solution:** Check LED patterns. Non-zero baseline patterns may require special handling. See LED alignment test report.

### Issue: Turn rate spike not at t=0

**Solution:** Verify timing alignment. Check that period-relative timing variables are computed correctly.

### Issue: File access errors

**Solution:** Ensure file is not locked. Use `force_unlock_h5.py` if needed.

## Validation

After processing, validate results:

1. **Check output files exist**
2. **Verify feature ranges are reasonable**
3. **Check turn rate alignment (spike at t=0)**
4. **Compare with MATLAB reference (if available)**

## Next Steps

After successful integration:

1. **Model Fitting:** Use processed data to fit event-hazard models
2. **Simulation:** Run DOE simulations with fitted models
3. **Validation:** Compare simulation results with experimental data

## References

- **H5 File Structure:** `docs/logs/2025-11-13/h5-file-structure-guide.md`
- **Validation Report:** `docs/logs/2025-11-13/h5-file-validation-report.md`
- **LED Alignment Test:** `scripts/2025-11-12/led_alignment_test_report.md`
- **Experiment Manifest:** `docs/logs/2025-11-11/experiment-manifest.md`

---

**Last Updated:** 2025-11-13

