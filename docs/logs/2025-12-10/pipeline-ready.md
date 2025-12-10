# Pipeline Ready for Re-exported H5 Files

**Date:** 2025-12-10  
**Status:** Ready for new H5 files from MagatFairy

---

## Summary

Prepared INDYsim pipeline for new H5 files that include `derivation_rules`. Also scaffolded the NB-GLM hazard model and event generator for simulation.

---

## Changes Made

### 1. engineer_dataset_from_h5.py

**Updated `load_h5_file()`** to read derivation_rules from H5 root:
```python
if 'derivation_rules' in f:
    dr_group = f['derivation_rules']
    data['derivation_rules'] = {
        'smoothTime': float(dr_group.attrs.get('smoothTime', 0.2)),
        'derivTime': float(dr_group.attrs.get('derivTime', 0.1)),
        'interpTime': float(dr_group.attrs.get('interpTime', 0.05))
    }
```

**Updated `extract_trajectory_features()`** to:
- Accept `derivation_rules` parameter
- Attach to DataFrame attrs for segmentation

**Updated call site** to pass derivation_rules through.

### 2. magat_segmentation.py

**Verified correct configuration:**
- Speed thresholds: 0.2/0.3 cm/s (converted from MATLAB's 2/3 mm/s)
- Reads `derivation_rules` from `trajectory_df.attrs`
- Has sensible defaults as fallback

### 3. hazard_model.py (NEW)

NB-GLM hazard model implementing MiroThinker specification:

| Component | Implementation |
|-----------|----------------|
| Family | Negative Binomial with log link |
| Temporal kernel | Raised-cosine basis (3-5 bases, -3s to 0s window) |
| Phase | sin/cos of LED1 60s cycle |
| LED covariates | Continuous intensity + interaction |
| Instantaneous | Speed (standardized), curvature (clipped) |
| CV | Leave-one-experiment-out |

### 4. event_generator.py (NEW)

Event generation for simulation:

| Method | Use Case |
|--------|----------|
| Inversion-based | Recommended for low-rate events (λ ≈ 0.01-0.1/s) |
| Thinning | Fallback for highly time-varying rates |

Includes:
- `InversionEventGenerator`: Inverts cumulative hazard
- `ThinningEventGenerator`: Lewis-Shedler algorithm
- Attribute samplers: heading change, head swings, speed
- Full trajectory simulation with boundary reflection

---

## MagatFairy Handoff

Created handoff document for fairybro agent:
- Branch: `handoff/indysim-h5-fixes-2025-12-10`
- File: `docs/handoffs/fairybro-h5-export-fixes.md`

Fixes implemented in MagatFairy:
1. `export_derivation_rules()` added to `convert_matlab_to_h5.py`
2. Position units documentation fixed (pixels → cm)
3. Curvature field warning added
4. Speed threshold documentation added
5. `validate_h5_for_analysis.py` validation script created

---

## Next Steps

1. **Wait for H5 re-export** to complete on lab PC
2. **Run pipeline** on new H5s:
   ```bash
   python scripts/engineer_dataset_from_h5.py \
       --h5-dir /path/to/new/h5s \
       --output-dir data/engineered_validated
   ```
3. **Fit hazard model**:
   ```bash
   python scripts/hazard_model.py \
       --data-dir data/engineered_validated \
       --cv
   ```
4. **Run simulation** with fitted model

---

## References

- MiroThinker research (2025-12-10): Model specification, units, event hierarchy
- Gepner et al. (2015) eLife: LNP model, raised-cosine kernels
- Klein et al. (2015) PNAS: Turn detection, sensory determinants
- Lewis & Shedler (1979): Thinning algorithm
