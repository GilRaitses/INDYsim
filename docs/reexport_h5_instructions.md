# Re-export H5 Files with Updated Fields

## Location
**Source Directory**: `D:\rawdata\GMR61@GMR61\T_Re_Sq_0to250PWM_30#C_Bl_7PWM\`

This directory contains:
- `.mat` files in `matfiles/` subdirectory
- `.bin` files at root level
- Track directories: `matfiles/GMR61@GMR61_{timestamp} - tracks/`

## Batch Export Command

Use the reusable batch export script:

```bash
cd D:\mechanosensation\scripts\2025-11-10
python batch_export_h5_reusable.py D:\rawdata\GMR61@GMR61\T_Re_Sq_0to250PWM_30#C_Bl_7PWM
```

This will:
1. Auto-detect all 3 experiments in the directory:
   - `202509051125`
   - `202509051201`
   - `202509051237`
2. Export each to H5 format with **derivation_rules** and **segment_options** included
3. Save outputs to: `D:\rawdata\GMR61@GMR61\T_Re_Sq_0to250PWM_30#C_Bl_7PWM\h5_exports\`

## What's Fixed

The updated `H5_clone.py` now exports:
- ✅ **derivation_rules** group with `interpTime`, `smoothTime`, `derivTime` (CRITICAL for segmentation)
- ✅ **segment_options** group with `curv_cut`, `theta_cut`, `minRunTime`, `minRunLength`

## Manual Export (Single Experiment)

If you need to export a single experiment manually:

```bash
python D:\mechanosensation\scripts\2025-11-10\H5_clone.py \
  --mat "D:\rawdata\GMR61@GMR61\T_Re_Sq_0to250PWM_30#C_Bl_7PWM\matfiles\GMR61@GMR61_T_Re_Sq_0to250PWM_30#C_Bl_7PWM_202509051201.mat" \
  --tracks "D:\rawdata\GMR61@GMR61\T_Re_Sq_0to250PWM_30#C_Bl_7PWM\matfiles\GMR61@GMR61_202509051201 - tracks" \
  --bin "D:\rawdata\GMR61@GMR61\T_Re_Sq_0to250PWM_30#C_Bl_7PWM\GMR61@GMR61_T_Re_Sq_0to250PWM_30#C_Bl_7PWM_202509051201.bin" \
  --output "D:\rawdata\GMR61@GMR61\T_Re_Sq_0to250PWM_30#C_Bl_7PWM\h5_exports\GMR61@GMR61_T_Re_Sq_0to250PWM_30#C_Bl_7PWM_202509051201.h5"
```

## Expected Output Structure

After export, each H5 file will have:
```
experiment_info/
global_quantities/
tracks/
  track_1/
    metadata/
    state/
    points/
    derived_quantities/  ← All dq fields (curv, vel_dp, speed, sspineTheta, etc.)
    derivation_rules/     ← NEW: interpTime, smoothTime, derivTime
    segment_options/      ← NEW: curv_cut, theta_cut, minRunTime, minRunLength
eti (at root)             ← Experiment Time Index
stimulus/
metadata/
```

## Verification

After export, verify derivation_rules are present:
```python
import h5py
with h5py.File("path/to/exported.h5", 'r') as f:
    track = f['tracks/track_1']
    if 'derivation_rules' in track:
        print("✅ derivation_rules found")
        print(f"   interpTime: {track['derivation_rules'].attrs.get('interpTime', 'MISSING')}")
        print(f"   smoothTime: {track['derivation_rules'].attrs.get('smoothTime', 'MISSING')}")
        print(f"   derivTime: {track['derivation_rules'].attrs.get('derivTime', 'MISSING')}")
    else:
        print("❌ derivation_rules MISSING")
```

