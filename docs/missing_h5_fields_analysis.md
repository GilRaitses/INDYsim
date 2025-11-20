# Missing H5 Fields Analysis
## Comparison: MATLAB segmentTrack.m Requirements vs H5 Clone Export

### MATLAB segmentTrack.m Requirements

From `@MaggotTrack/segmentTrack.m`, the following fields are accessed:

#### Derived Quantities (track.getDerivedQuantity()):
1. **`curv`** - Line 42: `cv = track.getDerivedQuantity('curv');`
2. **`spineLength`** - Line 38: `median(track.getDerivedQuantity('spineLength'))` (if autoset_curv_cut)
3. **`periFreq`** - Line 48: `median(track.getDerivedQuantity('periFreq'))` (if smoothBodyFromPeriFreq)
4. **`spineTheta`** - Line 50: `track.getDerivedQuantity('spineTheta')` (if smoothBodyFromPeriFreq)
5. **`sspineTheta`** - Line 52: `track.getDerivedQuantity('sspineTheta')` (default)
6. **`vel_dp`** - Line 54: `track.getDerivedQuantity('vel_dp');`
7. **`speed`** (or speed_field) - Line 55: `track.getDerivedQuantity(mso.speed_field);`
8. **`eti`** - Lines 58, 101, 111, 134, 171: `track.dq.eti`

#### Derivation Rules (track.dr):
9. **`interpTime`** - Line 50: `track.dr.interpTime` (for lowpass1D)
10. **`smoothTime`** - Line 140: `track.dr.smoothTime` (for buffer calculation)
11. **`derivTime`** - Line 140: `track.dr.derivTime` (for buffer calculation)

#### Segment Options (track.so / mso):
12. **`curv_cut`** - Used throughout
13. **`theta_cut`** - Used throughout
14. **`stop_speed_cut`** - Used throughout
15. **`start_speed_cut`** - Used throughout
16. **`aligned_dp`** - Used throughout
17. **`minRunTime`** - Line 111
18. **`headswing_start`** - Line 153
19. **`headswing_stop`** - Line 156

### H5 Clone Export (getCompleteTrackData.m)

#### ✅ EXPORTED:
- **Derived Quantities**: ALL fields from `track.dq` (lines 109-118)
  - Includes: `curv`, `vel_dp`, `speed`, `sspineTheta`, `spineTheta`, `spineLength`, `periFreq`, `eti`, etc.
- **Derivation Rules**: `interpTime`, `smoothTime`, `derivTime` (lines 128-135)
- **Segment Options**: `curv_cut`, `theta_cut`, `minRunTime`, `minRunLength` (lines 137-145)

#### ❌ NOT EXPORTED TO H5:
- **Derivation Rules**: `derivation_rules` group is NOT written to H5 file
- **Segment Options**: `segment_options` group is NOT written to H5 file

### H5_clone.py Export Status

From `H5_clone.py` lines 187-195:
- ✅ Exports `derived_quantities` (all fields from `track_data['derived']`)
- ❌ **MISSING**: Does NOT export `derivation_rules` from `track_data['derivation_rules']`
- ❌ **MISSING**: Does NOT export `segment_options` from `track_data['segment_options']`

### Critical Missing Fields

#### 1. **Derivation Rules (derivation_rules)** - CRITICAL FOR SEGMENTATION
**Status**: ❌ NOT EXPORTED TO H5

**Required for**:
- Buffer calculation: `buffer = ceil((smoothTime + derivTime) / interpTime)` (MATLAB line 140)
- Body theta smoothing: `lowpass1D(spineTheta, st/interpTime)` (MATLAB line 50)

**Fields needed**:
- `interpTime` - REQUIRED
- `smoothTime` - REQUIRED  
- `derivTime` - REQUIRED

**Impact**: Segmentation will FAIL without these fields. Python code currently raises ValueError if missing.

#### 2. **Segment Options (segment_options)** - USEFUL BUT NOT CRITICAL
**Status**: ❌ NOT EXPORTED TO H5

**Contains**:
- `curv_cut` - Can be set manually
- `theta_cut` - Can be set manually
- `minRunTime` - Can be set manually
- `minRunLength` - Can be set manually

**Impact**: Low - these are parameters that can be set in Python code, but having them exported would ensure consistency.

### Required Derived Quantities Status

All required derived quantities SHOULD be exported (since getCompleteTrackData exports ALL dq fields):

- ✅ `curv` - Should be in `derived_quantities/curv`
- ✅ `vel_dp` - Should be in `derived_quantities/vel_dp`
- ✅ `speed` - Should be in `derived_quantities/speed`
- ✅ `sspineTheta` - Should be in `derived_quantities/sspineTheta`
- ✅ `spineTheta` - Should be in `derived_quantities/spineTheta`
- ✅ `spineLength` - Should be in `derived_quantities/spineLength`
- ✅ `periFreq` - Should be in `derived_quantities/periFreq` (if calculated)
- ✅ `eti` - Should be in `derived_quantities/eti` (also exported to root)

**Note**: Need to verify these are actually present in H5 files. Some may not be calculated if segmentation hasn't been run.

### Recommendations

#### CRITICAL FIX REQUIRED:
1. **Export derivation_rules to H5**:
   ```python
   # In H5_clone.py, after line 195 (derived_quantities export):
   
   # Export derivation rules (CRITICAL for segmentation)
   if 'derivation_rules' in track_data and track_data['derivation_rules']:
       dr_grp = track_grp.create_group('derivation_rules')
       dr_dict = track_data['derivation_rules']
       for key in ['interpTime', 'smoothTime', 'derivTime']:
           if key in dr_dict:
               dr_grp.attrs[key] = float(dr_dict[key])
   ```

2. **Export segment_options to H5** (optional but recommended):
   ```python
   # Export segment options (for consistency)
   if 'segment_options' in track_data and track_data['segment_options']:
       so_grp = track_grp.create_group('segment_options')
       so_dict = track_data['segment_options']
       for key in ['curv_cut', 'theta_cut', 'minRunTime', 'minRunLength']:
           if key in so_dict:
               so_grp.attrs[key] = float(so_dict[key])
   ```

#### VERIFICATION NEEDED:
1. Verify all required derived quantities are actually present in H5 files
2. Check if `periFreq` is calculated (may not be if segmentation hasn't run)
3. Verify `eti` is in both root and `derived_quantities/eti`

### Summary Table

| Field | MATLAB Source | getCompleteTrackData | H5_clone.py Export | Status |
|-------|--------------|---------------------|-------------------|--------|
| `curv` | `track.dq.curv` | ✅ Exported | ✅ Exported | ✅ OK |
| `vel_dp` | `track.dq.vel_dp` | ✅ Exported | ✅ Exported | ✅ OK |
| `speed` | `track.dq.speed` | ✅ Exported | ✅ Exported | ✅ OK |
| `sspineTheta` | `track.dq.sspineTheta` | ✅ Exported | ✅ Exported | ✅ OK |
| `spineTheta` | `track.dq.spineTheta` | ✅ Exported | ✅ Exported | ✅ OK |
| `spineLength` | `track.dq.spineLength` | ✅ Exported | ✅ Exported | ✅ OK |
| `periFreq` | `track.dq.periFreq` | ✅ Exported | ✅ Exported | ⚠️ May be missing |
| `eti` | `track.dq.eti` | ✅ Exported | ✅ Exported (root) | ✅ OK |
| `interpTime` | `track.dr.interpTime` | ✅ Exported | ❌ **NOT EXPORTED** | ❌ **CRITICAL MISSING** |
| `smoothTime` | `track.dr.smoothTime` | ✅ Exported | ❌ **NOT EXPORTED** | ❌ **CRITICAL MISSING** |
| `derivTime` | `track.dr.derivTime` | ✅ Exported | ❌ **NOT EXPORTED** | ❌ **CRITICAL MISSING** |
| `curv_cut` | `track.so.curv_cut` | ✅ Exported | ❌ **NOT EXPORTED** | ⚠️ Can set manually |
| `theta_cut` | `track.so.theta_cut` | ✅ Exported | ❌ **NOT EXPORTED** | ⚠️ Can set manually |
| `minRunTime` | `track.so.minRunTime` | ✅ Exported | ❌ **NOT EXPORTED** | ⚠️ Can set manually |

### Action Items

1. **IMMEDIATE**: Add derivation_rules export to H5_clone.py
2. **RECOMMENDED**: Add segment_options export to H5_clone.py
3. **VERIFY**: Check existing H5 files for presence of all derived quantities
4. **TEST**: Verify segmentation works with exported derivation_rules

