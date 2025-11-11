# MATLAB Conversion Pipeline - Deployment Complete

**From:** mechanobro  
**To:** larry  
**Date:** 2025-11-11  
**Status:** ✅ Ready for Validation

## Summary

MATLAB to H5 conversion pipeline has been deployed to `D:\INDYsim\src\@matlab_conversion\` with all dependencies and comprehensive documentation. The pipeline is adapted from yesterday's working scripts in the mechanosensation repo and is ready for validation.

## Deployment Location

**Scripts:**
- `D:\INDYsim\src\@matlab_conversion\convert_matlab_to_h5.py` - Main export script
- `D:\INDYsim\src\@matlab_conversion\batch_export_esets.py` - Batch processor
- `D:\INDYsim\src\@matlab_conversion\AGENT_GUIDE.md` - Comprehensive usage guide

**Dependencies (External):**
- MAGAT Bridge: `D:\mechanosensation\mcp-servers\magat-bridge\server.py`
- MATLAB Classes: `D:\mechanosensation\scripts\2025-10-16`
- MAGAT Codebase: `d:\magniphyq\codebase\Matlab-Track-Analysis-SkanataLab`

## Source Code References

**Original Scripts (mechanosensation):**
- `D:\mechanosensation\scripts\2025-11-10\H5_clone.py` - Core export logic
- `D:\mechanosensation\scripts\2025-11-11\batch_export_indysim.py` - Batch processing logic

**MATLAB Compatibility Fixes:**
- `D:\mechanosensation\scripts\2025-10-16\@DataManager\loadExperiment.m` - Handles missing fields gracefully

## Validation Request

Please validate the following H5 files that were exported during testing:

1. **`D:\INDYsim\data\GMR61@GMR61_T_Re_Sq_0to250PWM_30#C_Bl_7PWM_202510291713.h5`**
2. **`D:\INDYsim\data\GMR61@GMR61_T_Re_Sq_0to250PWM_30#C_Bl_7PWM_202510301228.h5`**

### Validation Checklist

- [ ] **ETI at root:** Check that `eti` dataset exists at root level with shape `(N,)`
- [ ] **Root structure:** Verify root keys: `eti`, `experiment_info`, `global_quantities`, `led_data`, `metadata`, `stimulus`, `tracks`
- [ ] **Track structure:** Verify tracks have `points/`, `derived_quantities/`, `state/`, `metadata/`
- [ ] **Global quantities:** Verify LED values exported (led1Val, led2Val, derivatives)
- [ ] **Metadata:** Check `has_eti=True`, `export_tier=2`, `num_tracks`, `num_frames`
- [ ] **File size:** Should be ~100-300 MB per file
- [ ] **Compatibility:** Test with `engineer_dataset_from_h5.py` if available

### Quick Validation Script

```python
import h5py

files = [
    r"D:\INDYsim\data\GMR61@GMR61_T_Re_Sq_0to250PWM_30#C_Bl_7PWM_202510291713.h5",
    r"D:\INDYsim\data\GMR61@GMR61_T_Re_Sq_0to250PWM_30#C_Bl_7PWM_202510301228.h5"
]

for fpath in files:
    print(f"\n{'='*80}")
    print(f"Validating: {Path(fpath).name}")
    print(f"{'='*80}")
    
    with h5py.File(fpath, 'r') as f:
        print(f"Root keys: {sorted(list(f.keys()))}")
        print(f"ETI: {'eti' in f}, shape={f['eti'].shape if 'eti' in f else 'N/A'}")
        print(f"Tracks: {len(list(f['tracks'].keys()))}")
        print(f"Global quantities: {sorted(list(f['global_quantities'].keys()))}")
        print(f"Metadata: {dict(f['metadata'].attrs)}")
```

## Usage Examples

### Process All ESETs in Genotype

```bash
cd D:\INDYsim\src\@matlab_conversion
python batch_export_esets.py --genotype "GMR61@GMR61"
```

### Process Specific ESET

```bash
python batch_export_esets.py \
    --genotype "GMR61@GMR61" \
    --eset "T_Re_Sq_0to250PWM_30#C_Bl_7PWM"
```

### Single Experiment Export

```bash
python convert_matlab_to_h5.py \
    --mat "path/to/experiment.mat" \
    --tracks "path/to/tracks" \
    --bin "path/to/experiment.bin" \
    --output "output.h5"
```

## Key Features

1. **ETI at Root:** Critical for simulation scripts - exports `expt.elapsedTime` to root level
2. **Native Structure:** Handles MATLAB ESET folder structure correctly (`matfiles/` not `btdfiles/`)
3. **Dynamic Parsing:** Extracts genotype and timestamp from filenames automatically
4. **Strict Validation:** NO FALLBACKS - missing files cause export to skip
5. **Error Handling:** File locking errors handled gracefully

## Known Issues Resolved

- ✅ **File path mismatch:** Fixed to use `matfiles/` directory (MATLAB expects these)
- ✅ **MATLAB compatibility:** `openDataFile` and `globalQuantity` errors handled in MATLAB code
- ✅ **File locking:** Windows Error 33 handled with clear error messages

## Documentation

See `D:\INDYsim\src\@matlab_conversion\AGENT_GUIDE.md` for:
- Complete usage instructions
- Folder structure details
- File discovery logic
- Troubleshooting guide
- Integration notes

## Next Steps

1. **Validate** the two H5 files listed above
2. **Test** batch export on remaining ESETs if validation passes
3. **Integrate** with downstream analysis scripts
4. **Report** any issues or validation results

---

**Status:** ✅ Deployment Complete - Awaiting Validation  
**Contact:** See AGENT_GUIDE.md for detailed usage instructions

