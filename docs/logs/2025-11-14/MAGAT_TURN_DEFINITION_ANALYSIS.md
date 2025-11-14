# MAGAT Turn Definition Analysis

**Date:** 2025-11-14  
**Status:** CRITICAL - Turn counts too strict  
**Issue:** Current implementation doesn't match MAGAT's turn definition

## MAGAT Turn Definition (From Source Code)

### From `@MaggotReorientation/MaggotReorientation.m`:
```matlab
% periods of reorientation between runs; in our publications, a
% reorientation is called a "turn"
% reorientations have zero or more headSwings ("head sweeps" in
%   publications)
```

**Key Points:**
- Reorientations = gaps between runs
- Reorientations can have 0 or more head swings (`numHS`)
- In publications, reorientations are called "turns"

### From `load_multi_data_gr21a.m` (Line 169):
```matlab
turnStartTime = t(j).getSubFieldDQ('reorientation', 'eti', 
    'indsExpression', '[track.reorientation.numHS] >= 1', 
    'position', 'start');
```

**Key Points:**
- A "turn" is a reorientation with `numHS >= 1` (at least 1 head swing)
- NOT a reorientation with a pause!

### From `playMovie.m` (Lines 324-327):
```matlab
if (track.reorientation(I).numHS == 0)
    t = [t ['\color{white}pause, \Delta\theta = ...']];
else
    t = [t ['\color{white}turn, \Delta\theta = ...']];
end
```

**Key Points:**
- If `numHS == 0`: it's a **pause** (reorientation without head swings)
- If `numHS > 0`: it's a **turn** (reorientation with head swings)

### From `MAGAT_ANALYZER_DEMO.m` (Line 116):
```matlab
track.reorientation([track.reorientation.numHS] > 0 & 
    diff(unwrap([[track.reorientation.prevDir];[track.reorientation.nextDir]])) > 0)
    .playMovie('nopause', true);
```

**Key Points:**
- Filters reorientations by `numHS > 0` (at least 1 head swing)
- Also filters by direction change (turning left in this example)
- This confirms: **Turns = reorientations with head swings**

### From `spatialMaggotAnalysis.m` (Line 67):
```matlab
[num2str(sum(ad.eset_stats.numReosFromDirection)) ' reorientations (' 
 num2str(sum(ad.eset_stats.numReosWithHSFromDirection)) ' with at least 1 headsweep)']
```

**Key Points:**
- Distinguishes between all reorientations and reorientations with head swings
- Reorientations with head swings are a subset of all reorientations

## Current Implementation (WRONG)

### Current Definition:
```python
# Turns = reorientations that contain pauses
for reo_start, reo_end in reorientations:
    reo_pause_frames = is_pause[reo_start:reo_end+1]
    if np.any(reo_pause_frames):
        # This reorientation contains a pause → it's a turn
        turns.append((reo_start, reo_end))
```

**Problem:** This is backwards! We're defining turns as reorientations with pauses, but MAGAT defines turns as reorientations with head swings.

### Correct MAGAT Definition Should Be:
```python
# Turns = reorientations that contain head swings (numHS >= 1)
for reo_start, reo_end in reorientations:
    # Count head swings within this reorientation
    reo_head_swings = [hs for hs in head_swings 
                       if hs[0] >= reo_start and hs[1] <= reo_end]
    numHS = len(reo_head_swings)
    if numHS >= 1:  # At least 1 head swing
        # This reorientation has head swings → it's a turn
        turns.append((reo_start, reo_end))
```

## Comparison Table

| Aspect | MAGAT Definition | Current Implementation | Status |
|--------|-----------------|----------------------|--------|
| **Reorientations** | Gaps between runs | Gaps between runs | ✅ CORRECT |
| **Turns** | Reorientations with numHS >= 1 | Reorientations with pauses | ❌ WRONG |
| **Pauses** | Reorientations with numHS == 0 | Low speed periods | ❌ WRONG |
| **Head Swings** | Detected within reorientations | Detected within reorientations | ✅ CORRECT |
| **Turn Counting** | Count reorientations with head swings | Count reorientations with pauses | ❌ WRONG |

## Impact

**Current Issue:**
- Turn counts are too strict (only counting reorientations with pauses)
- Many valid turns (with head swings but no pauses) are missed
- This leads to underestimation of turn rates

**Expected Behavior:**
- Turns should be reorientations with at least 1 head swing
- This matches MAGAT's published methodology
- Turn rates should match MATLAB reference

## Fix Required

1. **Change turn definition:**
   - OLD: Turns = reorientations with pauses
   - NEW: Turns = reorientations with numHS >= 1

2. **Update turn extraction code:**
   - Count head swings within each reorientation
   - Mark as turn if numHS >= 1

3. **Update Klein run table:**
   - Column `reo#HS` already tracks number of head swings
   - Use this to determine if reorientation is a turn

4. **Update documentation:**
   - Pseudocode guide
   - Policy documents
   - Integration guide

## References

- `@MaggotReorientation/MaggotReorientation.m` - Reorientation class definition
- `@MaggotTrack/segmentTrack.m` - Segmentation algorithm
- `load_multi_data_gr21a.m` - Turn extraction example
- `MAGAT_ANALYZER_DEMO.m` - Demo script showing turn filtering
- `spatialMaggotAnalysis.m` - Analysis using reorientations with head swings

