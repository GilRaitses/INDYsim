# Validation Framework: MATLAB vs Python Pipeline

This framework validates that the Python `engineer_data.py` pipeline produces
**identical results** to the MATLAB reference implementation when processing
the same real experimental data.

## THREE-LAYER VALIDATION ARCHITECTURE

| Layer | Name | Purpose | Scripts |
|-------|------|---------|---------|
| 1a | H5 Schema | Verify H5 has all required fields | `validate_h5_schema.py` |
| 1b | Data Integrity | Verify H5 matches source .mat | `validate_data_integrity.py`, `validate_h5_against_mat.m` |
| 2 | Computation | Compare intermediate values | `load_experiment_and_compute.m`, `load_experiment_and_compute.py` |
| 3 | Results | Compare final analysis outputs | `run_full_validation.py` |

## Key Principle: NO SYNTHETIC DATA

All validation is performed on **real experiment files**, not generated test cases.

| Pipeline | Data Source |
|----------|-------------|
| MATLAB | `.mat` files from `matfiles/` |
| Python | `.h5` files from `h5_exports/` |

## Experiment Used for Validation

```
D:\rawdata\GMR61@GMR61\T_Re_Sq_50to250PWM_30#C_Bl_7PWM
└── 202506251614 (first experiment)
```

## Directory Structure

```
validation/
├── matlab/
│   ├── load_experiment_and_compute.m   # Load .mat, compute SpeedRunVel (Layer 2)
│   ├── validate_h5_against_mat.m       # Compare H5 to .mat (Layer 1b)
│   ├── compute_heading_unit_vector.m   # HeadVec normalization
│   ├── compute_velocity_and_speed.m    # Velocity and speed
│   ├── compute_speedrunvel.m           # SpeedRunVel (dot product)
│   ├── detect_reversals.m              # Reversal detection
│   ├── detect_turn_events.m            # Turn event detection
│   └── rate_from_time_corrected.m      # Corrected turn rate
├── python/
│   ├── load_experiment_and_compute.py  # Load .h5, compute SpeedRunVel (Layer 2)
│   ├── validate_h5_schema.py           # Verify H5 structure (Layer 1a)
│   ├── validate_data_integrity.py      # Compare raw data (Layer 1b)
│   └── [standalone function scripts]   # Matching MATLAB functions
├── test_data/
│   ├── matlab_validation_output.mat    # Full MATLAB data (generated)
│   ├── matlab_speedrunvel.csv          # MATLAB output (generated)
│   ├── python_validation_full.npz      # Full Python data (generated)
│   ├── python_speedrunvel.csv          # Python output (generated)
│   ├── python_validation_output.json   # Python summary (generated)
│   └── validation_report.json          # Machine-readable results (generated)
├── run_full_validation.py              # Master validation script (all layers)
├── FIELD_MAPPING.md                    # MATLAB field → H5 path mapping
├── VALIDATION_REPORT.md                # Human-readable report template
└── README.md                           # This file
```

## How to Run Validation

### Step 1: Run MATLAB Pipeline

```matlab
cd('D:\mechanosensation\scripts\2025-12-04\validation\matlab')
load_experiment_and_compute
```

This will:
- Load the real `.mat` experiment file
- Load tracks from the tracks directory
- Compute SpeedRunVel using the dot product method
- Detect reversals (SpeedRunVel < 0 for ≥ 3s)
- Save outputs to `test_data/matlab_speedrunvel.csv`

### Step 2: Run Python Pipeline

```bash
cd D:\mechanosensation\scripts\2025-12-04\validation\python
python load_experiment_and_compute.py
```

This will:
- Load the corresponding `.h5` file
- Extract track data from H5 structure
- Compute SpeedRunVel using identical method
- Detect reversals with same criteria
- Save outputs to `test_data/python_speedrunvel.csv`

### Step 3: Compare Results

```bash
cd D:\mechanosensation\scripts\2025-12-04\validation
python compare_real_data.py
```

This will:
- Load both output files
- Compare SpeedRunVel arrays element-by-element
- Compare reversal detection results
- Generate `VALIDATION_REPORT.md`

## What is Validated

### SpeedRunVel Computation

The signed velocity computed using Mason Klein's dot product method:

```
HeadVec = shead - smid
HeadUnitVec = HeadVec / ||HeadVec||
VelocityVec = [dx, dy] / ||[dx, dy]||
SpeedRun = ||[dx, dy]|| / dt
CosThetaFactor = VelocityVec · HeadUnitVec
SpeedRunVel = SpeedRun × CosThetaFactor
```

### Reversal Detection

Events where SpeedRunVel < 0 for duration ≥ 3 seconds.

### LED-Based Stimulus Timing

Both pipelines use the same LED values from the experiment to:
- Create ton/toff windows for stimulus periods
- Integrate behavior within stimulus windows

## Acceptance Criteria

| Metric | Threshold |
|--------|-----------|
| Max absolute difference (SpeedRunVel) | < 1e-6 |
| Time alignment | < 1e-6 seconds |
| Reversal count | Exact match |
| Reversal start times | < 0.1 seconds |

## Troubleshooting

### "H5 file not found"
Run the INDYsim H5 export pipeline first to generate `.h5` files.

### "Length mismatch"
The number of data points should match. Check that both pipelines
are loading the same track.

### "SpeedRunVel mismatch"
Check `lengthPerPixel` values - both pipelines must use identical
camera calibration.

