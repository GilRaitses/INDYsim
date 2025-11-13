# CRITICAL POLICY: ETI MUST ALWAYS BE USED FOR TIME CALCULATION

**Date:** 2025-11-13  
**Status:** MANDATORY - NO EXCEPTIONS  
**Applies To:** All agents working on INDYsim project

## Policy Statement

**ETI (Experiment Time Index) MUST ALWAYS be used for time calculation. Using frame-based time calculation (`np.arange(n_frames) / frame_rate`) is FORBIDDEN.**

## Why This Policy Exists

1. **Experiment Duration Constraint:** Experiments are only 20 minutes long (1200 seconds). Using frame indices assumes continuous frames and can produce durations exceeding 20 minutes, which is physically impossible.

2. **Track Gaps:** Tracks can have gaps (missing frames). Frame-based calculation (`frames / frame_rate`) assumes continuous frames from 0 to n_frames-1, which is incorrect for tracks with gaps.

3. **Data Integrity:** ETI represents actual experiment time and accounts for gaps, pauses, and other discontinuities in the data.

4. **frame_rate is NOT a time source:** The `frame_rate` parameter is ONLY used for pulse duration calculations and other frame-based operations. It is NEVER used to calculate time arrays.

## Error Example

**WRONG (FORBIDDEN):**
```python
frames = np.arange(n_frames)
time = frames / frame_rate  # FORBIDDEN - causes 37+ minute tracks!
```

**CORRECT (REQUIRED):**
```python
# ETI must be loaded from H5 root level
eti = h5_data['eti']  # From load_h5_file()
time = eti.copy()  # Use ETI directly
```

## Implementation Requirements

### 1. Loading ETI

ETI MUST be loaded from the H5 root level in `load_h5_file()`:

```python
def load_h5_file(h5_path: Path) -> Dict:
    data = {}
    with h5py.File(h5_path, 'r') as f:
        # CRITICAL: Load ETI from root level
        if 'eti' in f:
            data['eti'] = f['eti'][:]
        else:
            raise ValueError("ETI not found at root level - H5 file is invalid")
        # ... rest of loading ...
    return data
```

### 2. Using ETI in Functions

**Function signatures MUST accept ETI parameter:**

```python
def extract_trajectory_features(track_data: Dict, frame_rate: float = 10.0, 
                                eti: np.ndarray = None) -> pd.DataFrame:
    # CRITICAL: ETI is REQUIRED, not optional
    if eti is None:
        raise ValueError("CRITICAL ERROR: ETI is REQUIRED for time calculation")
    
    # Use ETI for time
    time = eti.copy()  # Or eti[track_frame_indices] if mapping needed
```

**Function calls MUST pass ETI:**

```python
# CORRECT
h5_data = load_h5_file(h5_path)
traj_df = extract_trajectory_features(track_data, frame_rate=fps, eti=h5_data['eti'])

# WRONG - will raise ValueError
traj_df = extract_trajectory_features(track_data, frame_rate=fps)  # Missing ETI!
```

### 3. Validation

All time calculations MUST validate that time does not exceed 20 minutes (1200 seconds):

```python
max_time = time.max() if len(time) > 0 else 0
if max_time > 1200:  # 20 minutes
    raise ValueError(f"CRITICAL ERROR: Time exceeds 20 minutes: {max_time:.1f}s ({max_time/60:.1f} min)")
```

## Functions That Must Use ETI

1. **`extract_trajectory_features()`** - Uses ETI for track time array
2. **`extract_stimulus_timing()`** - Uses ETI for LED/stimulus time array
3. **Any function that calculates time from frame indices** - MUST use ETI instead

## Functions That Are Safe (Use Time from DataFrames)

These functions use time from DataFrames that already have ETI-based time:
- `align_trajectory_with_stimulus()` - Uses `trajectory_df['time']` and `stimulus_df['time']`
- `create_event_records()` - Uses `trajectory_df['time']`
- Any function that operates on DataFrames with a `time` column

## frame_rate Parameter Usage

**frame_rate is ONLY used for:**
- Pulse duration calculations (e.g., `pulse_duration_frames = int(pulse_duration * frame_rate)`)
- Frame-based indexing operations
- NOT for calculating time arrays

**frame_rate is NEVER used for:**
- Creating time arrays (`np.arange(n_frames) / frame_rate` is FORBIDDEN)
- Time calculations
- Duration calculations (use ETI instead)

## Enforcement

- **Code Review:** All code changes must be reviewed to ensure ETI is used
- **Runtime Checks:** Functions raise `ValueError` if ETI is missing
- **Validation:** Time arrays are validated to not exceed 20 minutes
- **Documentation:** This policy must be referenced in all relevant code

## Common Mistakes to Avoid

1. **❌ Using `np.arange(n_frames) / frame_rate`** - FORBIDDEN
2. **❌ Not passing ETI to functions** - Will raise ValueError
3. **❌ Assuming ETI is optional** - It's REQUIRED
4. **❌ Using frame_rate to calculate time** - Use ETI instead
5. **❌ Ignoring validation errors** - If time > 1200s, investigate ETI data

## Related Files

- `scripts/engineer_dataset_from_h5.py` - Core implementation
- `docs/logs/2025-11-13/h5-file-structure-guide.md` - H5 structure documentation
- `scripts/2025-11-13/docs/pseudocode/analysis-pipeline-pseudocode.qmd` - Pseudocode guide (includes error documentation)

## Questions?

If you're unsure whether your code follows this policy:
1. Check if you're calculating time from frame indices
2. Verify ETI is loaded from H5 root level
3. Ensure ETI is passed to all functions that need time
4. Validate that calculated time does not exceed 1200 seconds

**Remember: When in doubt, use ETI. Never use frame-based time calculation.**
