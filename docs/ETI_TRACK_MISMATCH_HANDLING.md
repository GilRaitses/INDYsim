# ETI-Track Frame Mismatch Handling

**Date:** 2025-11-13  
**Status:** CRITICAL DATA INTEGRITY ISSUE  
**Policy:** DO NOT TRUNCATE DATA - PRESERVE ALL DATA POINTS

## Problem

Some tracks have more frames than the ETI array has elements. This creates a mismatch where:
- Track has `n_frames` data points
- ETI has `len(eti)` time values
- `n_frames > len(eti)` (track exceeds ETI length)

**Example:**
- Track 10: startFrame=0, n_frames=24001
- ETI length: 23923
- Track frames [0:24000] would map to ETI indices [0:24000], but ETI only has indices [0:23922]

## Root Cause

This mismatch can occur due to:
1. **Data export timing:** Track data exported before ETI was finalized
2. **Frame counting differences:** Track frame count includes interpolated/filled frames not in ETI
3. **H5 file structure:** ETI and track data exported from different sources with slight frame count differences

## Handling Strategy

**CRITICAL:** We MUST preserve all track data points. We cannot truncate track frames.

### Solution: Use Available ETI, Handle Overflow Gracefully

1. **For frames within ETI bounds:** Use ETI values directly
   - Track frames [0:len(eti)] → ETI indices [startFrame:startFrame+len(eti)]
   
2. **For frames beyond ETI length:** 
   - Use the last available ETI value
   - This preserves all track data points
   - Warning logged for data quality monitoring

### Implementation

```python
# Map track frames to ETI indices
start_frame = int(track_data['metadata_attrs']['startFrame'])
track_eti_indices = np.arange(start_frame, start_frame + n_frames, dtype=int)

# Handle overflow: clip indices to ETI bounds
valid_indices = track_eti_indices < len(eti)
n_valid = np.sum(valid_indices)

if n_valid < n_frames:
    # Some frames exceed ETI length
    print(f"WARNING: Track has {n_frames} frames but ETI has {len(eti)} elements. "
          f"Using ETI for {n_valid} frames, last ETI value for {n_frames - n_valid} frames.")
    
    # Use ETI for valid indices
    time = np.zeros(n_frames)
    time[:n_valid] = eti[track_eti_indices[:n_valid]]
    
    # Use last ETI value for overflow frames (preserves all data points)
    if n_valid > 0:
        time[n_valid:] = eti[-1]  # Last ETI value
    else:
        raise ValueError("No valid ETI indices for track")
else:
    # All frames within ETI bounds
    time = eti[track_eti_indices].copy()
```

## Data Integrity

- **All track data points preserved:** No truncation
- **ETI policy maintained:** All available ETI values used
- **Overflow handled:** Last ETI value used for frames beyond ETI length
- **Warning logged:** Data quality issue documented

## Validation

After time calculation:
- Check that `len(time) == n_frames` (all frames have time values)
- Check that time values are monotonically increasing
- Check that max(time) <= 1200 seconds (20 minute experiment limit)

## Future Improvements

1. **Investigate root cause:** Why do some tracks exceed ETI length?
2. **Fix export pipeline:** Ensure ETI and track data have matching frame counts
3. **Add validation:** Check ETI-track frame alignment during H5 export

