# Visualization Structure for INDYsim Experiments

**Date:** 2025-11-11  
**Prepared by:** larry  
**Purpose:** Define folder structure and visualization requirements for experiment data

## Directory Structure

```
data/
└── figures/
    └── {ESET_FOLDER}/
        ├── ledVals/                          # ESET-level LED value plots
        │   ├── {experiment}_ledVals.png     # One per experiment
        │   └── eset_summary_ledVals.png     # Aggregate LED value summary
        │
        ├── summary/                          # ESET-level summary statistics
        │   ├── eset_statistics.png          # Summary stats plot
        │   ├── track_count_distribution.png # Distribution of tracks per experiment
        │   ├── mean_turn_rate_by_experiment.png
        │   └── stimulus_parameters_summary.png
        │
        └── {experiment_timestamp}/           # Per-experiment visualizations
            ├── exact_reference_plots_matlab/ # MATLAB reference-style plots
            │   ├── experiment_composite.png  # Experiment-level composite
            │   ├── aggregate_composite.png   # Aggregate across all experiments
            │   └── track_{N}_composite.png   # Individual track composites
            │
            └── cycle_analysis/              # Cycle-based turn rate analysis
                └── turn_rate_bin_composite.png  # Turn rates per bin, all tracks
```

## Example Structure

```
data/figures/
└── T_Re_Sq_0to250PWM_30#C_Bl_7PWM/
    ├── ledVals/
    │   ├── GMR61@GMR61_T_Re_Sq_0to250PWM_30#C_Bl_7PWM_202510291652_ledVals.png
    │   ├── GMR61@GMR61_T_Re_Sq_0to250PWM_30#C_Bl_7PWM_202510291713_ledVals.png
    │   ├── GMR61@GMR61_T_Re_Sq_0to250PWM_30#C_Bl_7PWM_202510301228_ledVals.png
    │   ├── GMR61@GMR61_T_Re_Sq_0to250PWM_30#C_Bl_7PWM_202510301408_ledVals.png
    │   └── eset_summary_ledVals.png
    │
    ├── summary/
    │   ├── eset_statistics.png
    │   ├── track_count_distribution.png
    │   ├── mean_turn_rate_by_experiment.png
    │   └── stimulus_parameters_summary.png
    │
    ├── 202510291652/
    │   ├── exact_reference_plots_matlab/
    │   │   ├── experiment_composite.png
    │   │   ├── aggregate_composite.png
    │   │   ├── track_1_composite.png
    │   │   ├── track_2_composite.png
    │   │   └── ... (one per track)
    │   │
    │   └── cycle_analysis/
    │       └── turn_rate_bin_composite.png
    │
    ├── 202510291713/
    │   └── [same structure]
    │
    ├── 202510301228/
    │   └── [same structure]
    │
    └── 202510301408/
        └── [same structure]
```

## Visualization Requirements

### 1. ESET-Level LED Value Plots (`ledVals/`)

**Purpose:** Visualize stimulus pulse intensity parameters distribution

**Per-Experiment Plot:** `{experiment}_ledVals.png`
- X-axis: Time (seconds or frames)
- Y-axis: LED value (PWM 0-255)
- Plot LED1 and LED2 values over time
- Show stimulus ON/OFF periods
- Mark pulse boundaries
- Reference: `D:\mechanosensation\processed\ledVals\Re_Sq_0to50_20250303_1402_ledVals.png`

**ESET Summary Plot:** `eset_summary_ledVals.png`
- Overlay all experiments' LED values
- Show distribution of pulse intensities
- Highlight min/max/mean pulse intensities
- Show pulse duration distribution
- Show inter-pulse interval distribution

### 2. ESET-Level Summary Statistics (`summary/`)

**Plot 1: `eset_statistics.png`**
- Number of tracks per experiment
- Mean turn rate per experiment
- Mean latency per experiment
- Stop fraction per experiment
- Tortuosity per experiment
- Dispersal per experiment
- Spine curve energy per experiment
- Format: Multi-panel figure with statistics table

**Plot 2: `track_count_distribution.png`**
- Histogram of track counts across experiments
- Box plot showing distribution
- Mean and standard deviation

**Plot 3: `mean_turn_rate_by_experiment.png`**
- Bar plot or line plot showing mean turn rate for each experiment
- Error bars (standard deviation or standard error)
- X-axis: Experiment timestamp
- Y-axis: Mean turn rate (min⁻¹)

**Plot 4: `stimulus_parameters_summary.png`**
- Distribution of LED1 PWM values (min, max, mean)
- Distribution of LED2 PWM values (if applicable)
- Pulse duration distribution
- Inter-pulse interval distribution
- Duty cycle distribution

### 3. Experiment-Level Exact Reference Plots (`{experiment_timestamp}/exact_reference_plots_matlab/`)

**Plot 1: `experiment_composite.png`**
- Top panel: LED stimulus profile (Fictive CO2 equivalent)
- Bottom panel: Mean turn rate (min⁻¹) with variability shading
- X-axis: Time in cycle (s) 0-20
- Reference: `D:\mechanosensation\output\exact_reference_plots_matlab\experiment_1_composite.png`

**Plot 2: `aggregate_composite.png`**
- Aggregate across all experiments in ESET
- Same structure as experiment_composite.png
- Shows mean and variability across all experiments
- Reference: `D:\mechanosensation\output\exact_reference_plots_matlab\aggregate_composite.png`

**Plot 3: `track_{N}_composite.png` (one per track)**
- Individual track-level composite plots
- Top panel: LED stimulus profile
- Bottom panel: Turn rate for that specific track
- Shows individual track response variability
- Reference: `D:\mechanosensation\output\exact_reference_plots_matlab\track_1_composite.png`

### 4. Cycle Analysis (`{experiment_timestamp}/cycle_analysis/`)

**Plot: `turn_rate_bin_composite.png`**
- Composite plot showing turn rates in each time bin
- Counts all tracks present in each bin for a given cycle
- X-axis: Time in cycle (s) or bin number
- Y-axis: Turn rate (min⁻¹)
- Shows:
  - Mean turn rate per bin
  - Number of tracks contributing to each bin
  - Variability (shaded area or error bars)
- Useful for understanding cycle-averaged behavior

## Implementation Notes

### Data Sources

**For LED Value Plots:**
- Source: H5 files `global_quantities/led1Val/yData` and `global_quantities/led2Val/yData`
- After LED alignment is complete
- Need to extract pulse boundaries and parameters

**For Summary Statistics:**
- Source: Processed H5 files or `engineer_dataset_from_h5.py` output
- Track counts: From H5 file structure
- Turn rates: From processed trajectory data
- Other metrics: From feature extraction pipeline

**For Exact Reference Plots:**
- Source: Processed trajectory data with period-relative timing
- Need `led12Val_ton/toff` for cycle alignment
- Turn rate calculated per track, then aggregated

**For Cycle Analysis:**
- Source: Processed trajectory data
- Bin turn rates by cycle time
- Count tracks per bin
- Aggregate across cycles

### Plot Specifications

**Style:**
- Use Cinnamoroll color palette (`scripts/cinnamoroll_palette.py`)
- Avenir Ultralight font (if available)
- Follow style rules (`docs/style_rules.yaml`)

**Dimensions:**
- Standard: 1200x800 pixels (or equivalent DPI)
- Composite plots: May be larger (1600x1200)
- Individual track plots: Can be smaller (800x600)

**Format:**
- PNG format (high resolution for publications)
- Consider PDF for vector graphics

### Generation Script Structure

**Script:** `scripts/generate_experiment_visualizations.py`

**Functions:**
1. `create_figure_structure(eset_dir, experiment_timestamps)` - Create folder structure
2. `plot_led_values(h5_file, output_path)` - Generate LED value plots
3. `plot_eset_summary(eset_dir, output_dir)` - Generate ESET summary plots
4. `plot_experiment_composite(h5_file, output_path)` - Generate experiment composite
5. `plot_track_composites(h5_file, output_dir)` - Generate individual track composites
6. `plot_cycle_turn_rate_bins(h5_file, output_path)` - Generate cycle bin composite

**Dependencies:**
- `h5py` for reading H5 files
- `matplotlib` for plotting
- `numpy` for data processing
- `pandas` for data manipulation
- `cinnamoroll_palette.py` for colors

## Status

**Current:** Structure defined, implementation pending  
**Blocked by:** LED value alignment (P0)  
**Next Steps:**
1. Complete LED alignment integration
2. Convert all experiments to H5 format
3. Implement visualization generation script
4. Generate all plots for 14 experiments

## References

- **LED Value Plot Reference:** `D:\mechanosensation\processed\ledVals\`
- **Exact Reference Plot Reference:** `D:\mechanosensation\output\exact_reference_plots_matlab\`
- **Experiment Manifest:** `docs/logs/2025-11-11/experiment-manifest.md`
- **Color Palette:** `scripts/cinnamoroll_palette.py`
- **Style Rules:** `docs/style_rules.yaml`

---

**Status:** Structure defined, ready for implementation  
**Last Updated:** 2025-11-11

