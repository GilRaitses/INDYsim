# INDYsim Processed Data Access Guide

## Overview

This guide explains how to access processed experiment data from the INDYsim project for visualization and analysis in sandbox environments.

## Data Location

**Base Path:** `/Users/gilraitses/INDYsim/data/engineered/`

All processed experiments are stored in this directory as CSV files.

## File Naming Convention

Files follow this pattern:
```
{experiment_id}_{file_type}.csv
```

**Example:**
- `GMR61@GMR61_T_Re_Sq_0to250PWM_30#C_Bl_7PWM_202510291652_events.csv`
- `GMR61@GMR61_T_Re_Sq_0to250PWM_30#C_Bl_7PWM_202510291652_trajectories.csv`
- `GMR61@GMR61_T_Re_Sq_0to250PWM_30#C_Bl_7PWM_202510291652_klein_run_table.csv`

The `experiment_id` is derived from the original H5 filename and uniquely identifies each experiment.

## Available File Types

### 1. Events CSV (`{experiment_id}_events.csv`)

**Purpose:** Binned event records for discrete-time analysis (50ms bins)

**Key Columns:**
- `time_bin`: Bin index (integer)
- `time`: Time in seconds (float)
- `x`, `y`: Position coordinates (float)
- `speed`: Instantaneous speed (float)
- `heading`: Heading angle in radians (float)
- `track_id`: Track identifier (integer)
- `experiment_id`: Experiment identifier (string)

**Behavioral Events (Boolean):**
- `is_turn`: True during turn periods
- `is_turn_start`: True at turn START events (False→True transitions)
- `is_reorientation`: True during reorientation periods
- `is_reorientation_start`: True at reorientation START events
- `is_pause`: True during pause periods
- `is_pause_start`: True at pause START events
- `is_reversal`: True during reversal periods

**Stimulus Alignment:**
- `stimulus_on`: Boolean indicating if stimulus is active
- `time_since_stimulus`: Time since last stimulus onset (seconds)
- `led1Val`, `led2Val`: LED intensity values (float, 0-1)
- `led1Val_ton`, `led1Val_toff`: LED on/off transitions (boolean)

**Trajectory Features:**
- `curvature`: Trajectory curvature (rad/cm)
- `spine_curve_energy`: Spine curve energy metric
- `turn_duration`: Duration of current turn (seconds)
- `pause_duration`: Duration of current pause (seconds)

**Usage:** Best for event-based analysis, hazard modeling, and behavioral state transitions.

---

### 2. Trajectories CSV (`{experiment_id}_trajectories.csv`)

**Purpose:** Full-resolution trajectory data with frame-level detail

**Key Columns:**
- `frame_x`, `frame_y`: Frame indices
- `time`: Time in seconds (float)
- `x`, `y`: Position coordinates (float)
- `head_x`, `head_y`: Head position (float)
- `tail_x`, `tail_y`: Tail position (float)
- `speed`: Instantaneous speed (float)
- `heading`: Heading angle in radians (float)
- `track_id`: Track identifier (integer)

**Spine Points (10 points per frame):**
- `spine_x_0` through `spine_x_9`: X coordinates of spine points
- `spine_y_0` through `spine_y_9`: Y coordinates of spine points

**MAGAT Body Bend Angles:**
- `spineTheta_magat`: Body bend angle (radians)
- `sspineTheta_magat`: Smoothed body bend angle (radians)

**Trajectory Features:**
- `curvature`: Trajectory curvature (rad/cm)
- `curvature_abs`: Absolute curvature
- `spine_curve_energy`: Spine curve energy metric
- `acceleration`: Acceleration (mm/s²)
- `heading_change`: Change in heading angle (radians)

**Behavioral States (Boolean):**
- `is_turn`: True during turn periods (MAGAT definition: reorientations with head swings)
- `is_reorientation`: True during reorientation periods
- `is_run`: True during run periods
- `is_pause`: True during pause periods
- `is_reversal`: True during reversal periods
- `is_reverse_crawl`: True during reverse crawling

**Event Quantification:**
- `turn_duration`: Duration of current turn (seconds)
- `pause_duration`: Duration of current pause (seconds)
- `turn_event_id`: Unique identifier for each turn event

**Stimulus Alignment:**
- `led1Val`, `led2Val`: LED intensity values (float, 0-1)
- `led1Val_ton`, `led1Val_toff`: LED on/off transitions (boolean)
- `led2Val_ton`, `led2Val_toff`: LED on/off transitions (boolean)
- `stimulus_on`: Boolean indicating if stimulus is active
- `stimulus_onset`: True at stimulus onset events
- `time_since_stimulus`: Time since last stimulus onset (seconds)
- `time_bin`: Bin index for alignment with events CSV

**Usage:** Best for trajectory visualization, spine shape rendering, and frame-by-frame analysis.

---

### 3. Klein Run Table CSV (`{experiment_id}_klein_run_table.csv`)

**Purpose:** Run-level statistics following Klein et al. analysis framework

**Key Columns:**
- `reoYN`: Reorientation occurred (boolean)
- `reo#HS`: Number of head swings in reorientation
- `runDur`: Run duration (seconds)
- `runDist`: Run distance (mm or cm)
- `runSpeed`: Average run speed (mm/s or cm/s)
- `track_id`: Track identifier
- `experiment_id`: Experiment identifier

**Usage:** Best for run-level statistics and comparison with published analyses.

---

### 4. Progress JSON (`{experiment_id}_progress.json`)

**Purpose:** Processing status and metadata

**Contains:**
- Processing status
- Current track being processed
- Total tracks
- Elapsed time
- ETA
- Processing messages

**Usage:** Check processing status and track progress.

---

### 5. Console Log (`{experiment_id}_console.log`)

**Purpose:** Full console output from processing

**Contains:**
- Step-by-step processing messages
- Track-by-track progress
- Statistics for each track
- Error messages (if any)

**Usage:** Debugging and detailed progress tracking.

## Loading Data in Python

### Basic Loading

```python
import pandas as pd
from pathlib import Path

# Base directory
DATA_DIR = Path("/Users/gilraitses/INDYsim/data/engineered")

# Experiment ID
experiment_id = "GMR61@GMR61_T_Re_Sq_0to250PWM_30#C_Bl_7PWM_202510291652"

# Load events
events_df = pd.read_csv(DATA_DIR / f"{experiment_id}_events.csv")
print(f"Loaded {len(events_df):,} event records")
print(f"Tracks: {events_df['track_id'].nunique()}")
print(f"Time range: {events_df['time'].min():.1f}s to {events_df['time'].max():.1f}s")

# Load trajectories
trajectories_df = pd.read_csv(DATA_DIR / f"{experiment_id}_trajectories.csv")
print(f"Loaded {len(trajectories_df):,} trajectory points")
print(f"Tracks: {trajectories_df['track_id'].nunique()}")

# Load Klein run table (if available)
klein_file = DATA_DIR / f"{experiment_id}_klein_run_table.csv"
if klein_file.exists():
    klein_df = pd.read_csv(klein_file)
    print(f"Loaded {len(klein_df):,} run records")
```

### Finding Available Experiments

```python
from pathlib import Path
import re

DATA_DIR = Path("/Users/gilraitses/INDYsim/data/engineered")

# Find all processed experiments
event_files = list(DATA_DIR.glob("*_events.csv"))
experiment_ids = [f.stem.replace("_events", "") for f in event_files]

print(f"Found {len(experiment_ids)} processed experiments:")
for exp_id in sorted(experiment_ids):
    print(f"  - {exp_id}")
```

## Key Data Relationships

### Time Alignment

- **Events CSV**: Binned at 50ms intervals (`time_bin` increments by 1)
- **Trajectories CSV**: Frame-level data (typically 10 Hz = 100ms per frame)
- Both include `time` column for temporal alignment

### Track Identification

- Both files include `track_id` column
- Multiple tracks per experiment (typically 10-50 tracks)
- Each track represents one animal trajectory

### Event Detection

- **START events**: Use `*_start` columns (e.g., `is_turn_start`, `is_reorientation_start`)
- **Duration events**: Use duration columns (e.g., `turn_duration`, `pause_duration`)
- **State indicators**: Use boolean columns (e.g., `is_turn`, `is_reorientation`)

## Visualization Recommendations

### Trajectory Plotting

```python
import matplotlib.pyplot as plt

# Plot all tracks
for track_id in trajectories_df['track_id'].unique():
    track_data = trajectories_df[trajectories_df['track_id'] == track_id]
    plt.plot(track_data['x'], track_data['y'], alpha=0.5, label=f'Track {track_id}')

plt.xlabel('X position')
plt.ylabel('Y position')
plt.title('Animal Trajectories')
plt.legend()
plt.show()
```

### Spine Shape Rendering

```python
import numpy as np

# Extract spine points for a single frame
frame_idx = 100
spine_x_cols = [f'spine_x_{i}' for i in range(10)]
spine_y_cols = [f'spine_y_{i}' for i in range(10)]

spine_x = trajectories_df.loc[frame_idx, spine_x_cols].values
spine_y = trajectories_df.loc[frame_idx, spine_y_cols].values

plt.plot(spine_x, spine_y, 'o-', linewidth=2)
plt.axis('equal')
plt.title('Spine Shape')
plt.show()
```

### Event Timeline

```python
# Plot behavioral events over time
track_data = trajectories_df[trajectories_df['track_id'] == 1]

fig, axes = plt.subplots(3, 1, figsize=(12, 8))

axes[0].plot(track_data['time'], track_data['speed'])
axes[0].set_ylabel('Speed')
axes[0].set_title('Speed Over Time')

axes[1].fill_between(track_data['time'], 0, track_data['is_turn'], alpha=0.5, label='Turn')
axes[1].fill_between(track_data['time'], 0, track_data['is_reorientation'], alpha=0.5, label='Reorientation')
axes[1].set_ylabel('Behavioral State')
axes[1].legend()

axes[2].plot(track_data['time'], track_data['led1Val'])
axes[2].set_ylabel('LED Intensity')
axes[2].set_xlabel('Time (s)')

plt.tight_layout()
plt.show()
```

## Data Units

- **Time**: Seconds (float)
- **Position**: Millimeters or centimeters (check metadata)
- **Speed**: mm/s or cm/s
- **Angles**: Radians (heading, curvature)
- **LED Intensity**: Normalized 0-1 (0 = off, 1 = max)

## Notes for Sandbox Agents

1. **File Size**: Trajectories CSV files can be large (100-500 MB) due to full spine point data
2. **Memory**: Consider loading specific tracks or time ranges if memory is limited
3. **Time Binning**: Events CSV is pre-binned; Trajectories CSV is frame-level
4. **Track Filtering**: Use `track_id` to filter specific tracks
5. **Stimulus Alignment**: Both files include stimulus timing for stimulus-locked analysis
6. **Missing Data**: Some experiments may not have Klein run tables (check file existence)

## Example: Complete Loading Function

```python
def load_experiment(experiment_id: str, data_dir: Path = None):
    """
    Load all data files for an experiment.
    
    Parameters
    ----------
    experiment_id : str
        Experiment identifier
    data_dir : Path, optional
        Base data directory (defaults to INDYsim/data/engineered)
    
    Returns
    -------
    dict
        Dictionary containing:
        - 'events': Events DataFrame
        - 'trajectories': Trajectories DataFrame
        - 'klein_runs': Klein run table DataFrame (if available)
        - 'progress': Progress JSON (if available)
    """
    if data_dir is None:
        data_dir = Path("/Users/gilraitses/INDYsim/data/engineered")
    
    result = {}
    
    # Load events
    events_file = data_dir / f"{experiment_id}_events.csv"
    if events_file.exists():
        result['events'] = pd.read_csv(events_file)
    
    # Load trajectories
    trajectories_file = data_dir / f"{experiment_id}_trajectories.csv"
    if trajectories_file.exists():
        result['trajectories'] = pd.read_csv(trajectories_file)
    
    # Load Klein run table (optional)
    klein_file = data_dir / f"{experiment_id}_klein_run_table.csv"
    if klein_file.exists():
        result['klein_runs'] = pd.read_csv(klein_file)
    
    # Load progress (optional)
    progress_file = data_dir / f"{experiment_id}_progress.json"
    if progress_file.exists():
        import json
        with open(progress_file) as f:
            result['progress'] = json.load(f)
    
    return result

# Usage
data = load_experiment("GMR61@GMR61_T_Re_Sq_0to250PWM_30#C_Bl_7PWM_202510291652")
print(f"Events: {len(data['events']):,} records")
print(f"Trajectories: {len(data['trajectories']):,} points")
```

## Questions?

For issues or questions about the data format, refer to:
- `scripts/engineer_dataset_from_h5.py` - Data extraction code
- `scripts/queue/run_stimulus_locked_analysis_production.py` - Processing pipeline
- `scripts/config.py` - Path configuration


