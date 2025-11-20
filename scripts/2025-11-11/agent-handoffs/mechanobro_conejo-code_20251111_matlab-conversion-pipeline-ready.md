# MATLAB Conversion Pipeline - Ready for LED Alignment Integration

**From:** mechanobro  
**To:** conejo-code  
**Date:** 2025-11-11  
**Status:** ✅ Pipeline Deployed - Ready for Integration

## Summary

MATLAB to H5 conversion pipeline has been deployed to `D:\INDYsim\src\@matlab_conversion\`. The pipeline exports complete MAGAT structure with **ETI at root level** (critical for your LED alignment work). H5 files are now being generated and ready for integration with your LED value alignment implementation.

## Deployment Location

**Conversion Scripts:**
- `D:\INDYsim\src\@matlab_conversion\convert_matlab_to_h5.py` - Main export script
- `D:\INDYsim\src\@matlab_conversion\batch_export_esets.py` - Batch processor
- `D:\INDYsim\src\@matlab_conversion\process_genotype.bat` - Windows batch script
- `D:\INDYsim\src\@matlab_conversion\unlock_h5_file.py` - File utility

**Documentation:**
- `D:\INDYsim\src\@matlab_conversion\AGENT_GUIDE.md` - Comprehensive usage guide
- `D:\INDYsim\src\@matlab_conversion\QUICK_START.md` - Quick reference

## H5 File Structure (Critical for Your Work)

### ETI at Root Level
```
{base_name}.h5
├── eti                          # ⚠️ CRITICAL: Experiment Time Index at ROOT
│   └── shape: (N,)              # N = number of frames (e.g., 23923)
├── global_quantities/
│   ├── led1Val/
│   │   └── yData: (M,)          # M = global LED frames (e.g., 23991)
│   ├── led1ValDeriv/
│   ├── led2Val/
│   └── ...
├── tracks/
│   ├── track_1/
│   │   ├── derived_quantities/
│   │   │   ├── eti: (K,)        # Track-specific ETI (may differ from root)
│   │   │   └── led1Val: (K,)   # Track-aligned LED (may need validation)
│   │   └── ...
│   └── track_N/
└── metadata/
    └── attrs: {has_eti: True, eti_length: N, num_frames: F, ...}
```

### Key Points for LED Alignment

1. **Root ETI:** `f['eti']` - Experiment-level time index (shape `(N,)`)
2. **Global LED Values:** `f['global_quantities']['led1Val']['yData']` - Shape `(M,)` where M may differ from N
3. **Track ETI:** `f['tracks']['track_X']['derived_quantities']['eti']` - Track-specific ETI
4. **Track LED Values:** `f['tracks']['track_X']['derived_quantities']['led1Val']` - May already be aligned (needs validation)

## Integration with Your LED Alignment Work

### Current Status
- ✅ **H5 files being generated** with ETI at root level
- ✅ **Global LED values exported** in `global_quantities/led1Val/yData`
- ✅ **Track ETI available** in `tracks/track_X/derived_quantities/eti`
- ⚠️ **Length mismatch:** Global LED (23991) vs Root ETI (23923) vs Track frames (24001)

### Your Alignment Functions (from Task 0.2)

Your existing alignment functions in `scripts/2025-11-11/align_led_values.py` should work with these H5 files:

```python
# Example usage with new H5 files
from align_led_values import align_led_values_to_track_eti

# Load from H5
import h5py
with h5py.File('experiment.h5', 'r') as f:
    # Root ETI (experiment-level)
    root_eti = f['eti'][:]  # Shape: (N,)
    
    # Global LED values
    global_led = f['global_quantities']['led1Val']['yData'][:]  # Shape: (M,)
    
    # Track ETI
    track_eti = f['tracks']['track_1']['derived_quantities']['eti'][:]  # Shape: (K,)
    
    # Align global LED to track ETI
    aligned_led = align_led_values_to_track_eti(
        global_led_values=global_led,
        global_led_times=None,  # Need to derive from global_quantities metadata or use root_eti
        track_eti=track_eti,
        method='average_in_preceding_bin'  # MATLAB-compatible
    )
```

### Integration Points

1. **`engineer_dataset_from_h5.py`** - Update to use root ETI:
   ```python
   # Instead of: frame / fps
   # Use: eti[frame] from root level
   root_eti = f['eti'][:]
   time = root_eti[frame_indices]
   ```

2. **LED Alignment** - Use your `align_led_values_to_track_eti()` function:
   - Input: Global LED from `global_quantities/led1Val/yData`
   - Reference: Track ETI from `tracks/track_X/derived_quantities/eti`
   - Output: Aligned LED values per track

3. **Stimulus Timing** - Root ETI enables accurate stimulus-response analysis:
   - Stimulus onsets: `f['stimulus']['onset_frames']`
   - Convert to time: `root_eti[onset_frames]`

## Test Files Available

The following H5 files have been generated and are ready for testing:

1. `D:\INDYsim\data\GMR61@GMR61_T_Re_Sq_0to250PWM_30#C_Bl_7PWM_202510291652.h5`
   - ETI: (23923,) at root
   - Tracks: 51
   - Global LED: 6 quantities (led1Val, led2Val, derivatives)

2. `D:\INDYsim\data\GMR61@GMR61_T_Re_Sq_0to250PWM_30#C_Bl_7PWM_202510291713.h5`
   - Ready for validation

3. `D:\INDYsim\data\GMR61@GMR61_T_Re_Sq_0to250PWM_30#C_Bl_7PWM_202510301228.h5`
   - Ready for validation

## Validation Checklist for Your Integration

- [ ] **Verify root ETI structure:** Check that `f['eti']` exists and has correct shape
- [ ] **Test LED alignment:** Use your `align_led_values_to_track_eti()` with new H5 files
- [ ] **Compare track LED values:** Validate `tracks/track_X/derived_quantities/led1Val` against aligned values
- [ ] **Update `engineer_dataset_from_h5.py`:** Replace `frame / fps` with `eti[frame]`
- [ ] **Stimulus timing:** Verify stimulus onsets align correctly with root ETI
- [ ] **Length mismatches:** Handle differences between global LED (M), root ETI (N), and track frames (F)

## Known Issues & Solutions

### Length Mismatches
- **Global LED:** 23991 frames
- **Root ETI:** 23923 frames  
- **Track frames:** 24001 frames

**Solution:** Your `average_in_preceding_bin()` method should handle this via interpolation/binning.

### Track LED Values
- Tracks already have `derived_quantities/led1Val` - may be pre-aligned
- **Action:** Validate these against your alignment function output

### ETI Source
- **Root ETI:** `expt.elapsedTime` from MATLAB (experiment-level)
- **Track ETI:** Track-specific time index (may differ slightly)
- **Recommendation:** Use root ETI for global alignment, track ETI for per-track alignment

## Next Steps

1. **Test LED Alignment:**
   ```bash
   cd D:\INDYsim\scripts\2025-11-11
   python align_led_values.py  # Update to test with new H5 files
   ```

2. **Update `engineer_dataset_from_h5.py`:**
   - Replace `frame / fps` calculations with `eti[frame]`
   - Integrate your LED alignment function
   - Use root ETI for stimulus timing

3. **Validate Against MATLAB:**
   - Compare aligned LED values with MATLAB reference
   - Verify stimulus-response dynamics are biologically plausible

4. **Batch Processing:**
   - Once validated, process all H5 files:
   ```bash
   cd D:\INDYsim\src\@matlab_conversion
   process_genotype.bat GMR61@GMR61
   ```

## Related Work

- **Your Task 0.2:** LED alignment implementation (`scripts/2025-11-11/align_led_values.py`)
- **Your Task 0.4:** Mean rate discrepancy investigation (completed - Python implementation correct)
- **MATLAB Reference:** `D:\Matlab-Track-Analysis-MirnaLab\user specific\Silverio\load_multi_data_gr21a.m`

## Documentation References

- **Conversion Guide:** `D:\INDYsim\src\@matlab_conversion\AGENT_GUIDE.md`
- **Quick Start:** `D:\INDYsim\src\@matlab_conversion\QUICK_START.md`
- **Your Alignment Code:** `D:\INDYsim\scripts\2025-11-11\align_led_values.py`

## Questions?

If you encounter any issues with:
- H5 file structure or access
- ETI extraction or usage
- LED value alignment with new files
- Integration with `engineer_dataset_from_h5.py`

Please check the AGENT_GUIDE.md or reach out for clarification.

---

**Status:** ✅ Pipeline Ready - Awaiting LED Alignment Integration  
**Priority:** High - ETI at root enables accurate timing calculations  
**Blocking:** None - Files are ready for testing






