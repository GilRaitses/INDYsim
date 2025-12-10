# Deep Research Prompt: ETI Overflow Warning Investigation

**Date:** 2025-12-10  
**Status:** Needs investigation and resolution decision  
**Priority:** Medium (handled gracefully, but needs root cause fix)

---

## The Problem

When processing H5 files exported by MagatFairy, tracks consistently have **more frames than the global ETI array**:

```
WARNING: Track has 24001 frames but ETI has 23997 elements.
Track frames exceed ETI by 4 frames.
Using ETI for 23997 frames, last ETI value for overflow frames.
```

This warning appears for **all tracks** in every processed H5 file, with overflow ranging from **4 to 39 frames** depending on the experiment.

---

## Essential Context Gathered

### 1. Two Different ETI Sources

| Field | Location | Length | Step | Uniformity |
|-------|----------|--------|------|------------|
| **Global ETI** | `/eti` (root) | 23997 | 0.050010s ± 0.001331s | Non-uniform (actual camera timestamps) |
| **Track ETI** | `/tracks/track_N/derived_quantities/eti` | 24001 | 0.050000s (exactly) | Perfectly uniform (interpolated) |

### 2. The Math

- **Global ETI**: Actual camera timestamps, ~20 fps with jitter, ends at 1200.0383s
- **Track ETI**: Uniform grid `np.arange(0, 1200.05, 0.05)` = 24001 points, ends at exactly 1200.0s
- **Difference**: 24001 - 23997 = **4 frames** (for this experiment)

### 3. The Source: `derivation_rules.interpTime`

From the H5 file:
```
derivation_rules attributes:
  interpTime: 0.05    ← This defines the uniform interpolation grid
  smoothTime: 0.457
  derivTime: 0.229
```

The MATLAB pipeline interpolates all track data to a **uniform time grid** at `interpTime = 0.05s` intervals. This creates exactly 24001 points for a 1200-second experiment (`1200/0.05 + 1 = 24001`).

### 4. Why Global ETI is Shorter

The camera didn't capture at exactly 20 fps:
- Mean step: 0.050010s (slightly > 0.05s)
- Had 1 large gap: 0.25s at frame 12285
- Total actual frames captured: 23997 (not 24001)

### 5. Current Handling (in `engineer_dataset_from_h5.py`)

```python
if max_eti_index >= eti_length:
    # Track has more frames than ETI - handle overflow gracefully
    n_overflow = max_eti_index - eti_length + 1
    time[:len(valid_indices)] = eti[valid_indices]
    time[len(valid_indices):] = eti[-1]  # Last ETI value for overflow
```

**Problem:** The last ~4 frames all get assigned `time = 1200.0383s` (duplicate timestamps), which is technically incorrect for time-series analysis.

---

## Experiment Data Comparison

| Experiment | Global ETI | Track Frames | Overflow |
|------------|-----------|--------------|----------|
| 202510301228 | 23997 | 24001 | 4 frames |
| 202510291713 | 23962 | 24001 | 39 frames |

---

## Key Questions for Research

### Q1: What is the CORRECT time source for analysis?

**Option A: Use Global ETI (actual camera timestamps)**
- Pros: Real acquisition timing, accounts for camera jitter and gaps
- Cons: Track data was interpolated to different grid, mismatch creates artifacts
- Current approach: ✓ (with overflow handling)

**Option B: Use Track-level ETI (uniform interpolated time)**
- Pros: Matches track data perfectly, no overflow issues
- Cons: Not actual acquisition time, may not align with LED timing

**Option C: Interpolate track data back to Global ETI grid**
- Pros: Both time and data align
- Cons: Throws away some interpolated data, adds processing step

### Q2: Which ETI should MagatFairy export as the "global" ETI?

Currently, MagatFairy exports:
- `/eti` = actual camera timestamps (non-uniform)
- `/tracks/track_N/derived_quantities/eti` = uniform interpolated time

Should the global `/eti` be changed to the uniform grid to match track data?

### Q3: Does the LED timing use Global ETI or Track ETI?

The `led1Val`, `led2Val` arrays are in `/global_quantities/` and have length 23997 (matching Global ETI).
But track LED data (`/tracks/track_N/derived_quantities/led1Val`) has length 24001 (matching Track ETI).

**Critical:** If we use track LED for analysis, it's already interpolated to the uniform grid.
If we use global LED, it uses actual camera timestamps.

### Q4: What did the original MATLAB analysis pipeline use?

- Klein's original code: What time array was used for the `run table` and behavioral analysis?
- Did Klein use interpolated uniform time or actual camera time?
- Is `interpTime` a pre-processing step that should make uniform time the "truth"?

### Q5: Should we just use Track-level ETI?

**The simplest fix:** Instead of mapping track frames to Global ETI (and getting overflow), use the track's own `/derived_quantities/eti` which is already uniform and matches track length.

```python
# Instead of:
time = global_eti[track_frame_indices]  # Causes overflow

# Use:
track_eti = track_data['derived_quantities']['eti']
time = track_eti  # Perfect match, no overflow
```

---

## Related Documentation

- `/docs/ETI_TIME_CALCULATION_POLICY.md` - Current policy requiring global ETI
- `/docs/ETI_TRACK_MISMATCH_HANDLING.md` - Current overflow handling approach

---

## Code Locations

**Where ETI is loaded:**
```python
# scripts/engineer_dataset_from_h5.py, line ~42-49
if 'eti' in f:
    data['eti'] = f['eti'][:]  # Loads GLOBAL eti
```

**Where overflow is handled:**
```python
# scripts/engineer_dataset_from_h5.py, line ~294-322
if max_eti_index >= eti_length:
    # Track has more frames than ETI - handle overflow gracefully
    ...
```

**Where MagatFairy creates the uniform grid:**
```matlab
% magatfairy/src/matlab/core/@DataManager/getCompleteTrackData.m
% Uses derivation_rules.interpTime to interpolate to uniform grid
```

---

## Proposed Solutions (Ranked by Preference)

### Solution 1: Use Track-Level ETI (Recommended)

**Change:** Load `/tracks/track_N/derived_quantities/eti` instead of global `/eti` for each track.

**Pros:**
- Perfect match (no overflow warnings)
- Track data was interpolated to this grid, so it's the "correct" time for that data
- Simple code change

**Cons:**
- Different from current policy (requires policy update)
- Need to verify LED timing alignment

### Solution 2: Update MagatFairy to Export Uniform Global ETI

**Change:** Have MagatFairy export the uniform grid as `/eti` instead of camera timestamps.

**Pros:**
- One global time source that matches all track data
- Cleaner design

**Cons:**
- Requires MagatFairy change and re-export
- Loses actual camera timestamp information

### Solution 3: Interpolate Global Quantities to Uniform Grid

**Change:** When exporting, interpolate `led1Val`, `led2Val`, etc. to the uniform 24001-point grid.

**Pros:**
- Everything uses uniform time
- No data loss

**Cons:**
- Requires MagatFairy change
- LED values may have slight interpolation artifacts

### Solution 4: Accept the Current Handling

**Change:** Nothing - keep the warning but document it's expected.

**Pros:**
- No code changes
- Works for current analysis

**Cons:**
- Duplicate timestamps for overflow frames (bad for time-series)
- Warnings clutter output

---

## Request for Deep Research

Please investigate:

1. **MATLAB original behavior:** What did Klein's original MATLAB code use for the "time" variable in run tables and behavioral analysis? Was it the interpolated uniform time or camera timestamps?

2. **InterpolateTrackTimes function:** Is there a MATLAB function that does the interpolation? What exactly does it do?

3. **LED timing source:** Are LED values in the MATLAB analysis based on camera frames or interpolated frames?

4. **Best practice for time-series:** When data has been interpolated to a uniform grid, should analysis use that uniform grid or try to map back to original timestamps?

5. **Recommendation:** Given the findings, what is the cleanest architectural fix for MagatFairy and INDYsim?

---

## Summary of Issue

| Aspect | Global ETI | Track ETI | Problem |
|--------|------------|-----------|---------|
| Source | Camera timestamps | Uniform interpolation | Different time grids |
| Length | 23997 (varies) | 24001 (always) | 4-39 frame mismatch |
| Step | ~0.05s ± jitter | Exactly 0.05s | Non-uniform vs uniform |
| Matches | `global_quantities/*` | `tracks/*/derived_quantities/*` | Data/time mismatch |

**Core issue:** MagatFairy interpolates track data to uniform time but exports camera timestamps as the "global" time reference. This creates a mismatch between the time array and the data arrays.

---

## Appendix: Raw Data Evidence

```
============================================================
ETI Uniformity Analysis
============================================================

Global ETI step statistics:
  Mean: 0.050010s
  Std:  0.001331s
  Min:  0.050000s
  Max:  0.249998s
  Is uniform? False

Track ETI step statistics:
  Mean: 0.050000s
  Std:  0.000000s
  Min:  0.050000s
  Max:  0.050000s
  Is uniform? True

============================================================
Global ETI (root level):
  Length: 23997
  Range: 0.0000s to 1200.0383s
  Last 5 values: [1199.838 1199.888 1199.938 1199.988 1200.038]

Track-level ETI (derived_quantities/eti):
  Length: 24001
  Range: 0.0000s to 1200.0000s
  Last 5 values: [1199.8 1199.85 1199.9 1199.95 1200.0]
============================================================
```
