# Comprehensive Research Prompt: Data Alignment and Final Validation

## Context

You are assisting with **INDYsim**, a hazard model for Drosophila larva reorientation behavior under optogenetic stimulation. The model is complete and produces realistic event rates, but **validation is failing due to a data alignment mismatch**.

---

## The Problem

### What We Have

1. **Fitted Hazard Model** (from `data/model/hybrid_model_results.json`):
   - 55 tracks
   - 1407 empirical events
   - Intercept: -6.76 (log events per frame at 20 Hz)
   - 12-basis raised-cosine kernel + rebound term
   - Track intercepts ranging from -7.65 to -6.23

2. **Analytic Kernel** (gamma-difference approximation):
   - R² = 0.968 fit to the learned kernel
   - 6 parameters with bootstrap CIs
   - Properly calibrated for frame-rate units
   - Produces ~1.5 events/min at baseline

3. **Empirical Data Files** (in `data/engineered/`):
   - 14 CSV files named `*_events.csv`
   - 31,119 total events when filtering `is_reorientation_start == True`
   - Mixed stimulus conditions (0-250 PWM, 50-250 PWM, temperature variants)

### The Mismatch

| Source | Tracks | Events | Events/Track |
|--------|--------|--------|--------------|
| Model fit | 55 | 1,407 | 25.6 |
| Engineered files | 14 files | 31,119 | ~2,223/file |

The model was fit on 55 tracks with 1,407 events, but the validation script loads 31,119 events from the engineered files. This causes the rate ratio to be 0.047 instead of ~1.0.

### Specific File Breakdown

```
data/engineered/*_events.csv:
  GMR61@GMR61_T_Re_Sq_0to250PWM_30#C_Bl_7PWM_202510291652: 14,617 events
  GMR61@GMR61_T_Re_Sq_0to250PWM_30#C_Bl_7PWM_202510291713:  9,214 events
  GMR61@GMR61_T_Re_Sq_0to250PWM_30#C_Bl_7PWM_202510301228:    737 events
  GMR61@GMR61_T_Re_Sq_0to250PWM_30#C_Bl_7PWM_202510301408:    670 events
  GMR61@GMR61_T_Re_Sq_0to250PWM_30#T_Bl_Sq_5to15PWM_30_*:  ~500 events each
  GMR61@GMR61_T_Re_Sq_50to250PWM_30#C_Bl_7PWM_*:           ~500-700 events each
  GMR61@GMR61_T_Re_Sq_50to250PWM_30#T_Bl_Sq_5to15PWM_30_*: ~500 events each
```

The first two files alone have 23,831 events (77% of total), suggesting either:
- Different experimental conditions
- Different event annotation criteria
- Multiple tracks per file with different durations

---

## What I Need to Understand

### TOPIC 1: Data Source for Model Fitting

**Question 1.1**: Where did the 55 tracks and 1,407 events used for model fitting come from?

Possibilities:
- A specific subset of the engineered files?
- A different data directory (e.g., `data/processed/`, `data/engineered_validated/`)?
- A different event definition (not `is_reorientation_start`)?
- Aggregated binned data rather than raw events?

**Question 1.2**: The `hybrid_model_results.json` contains 55 track intercepts and 55 track event rates. How were these tracks defined?
- Is each "track" a single larva trajectory?
- Or is it a track segment within a larger experiment?
- What is the typical duration of one track?

**Question 1.3**: The model validation section shows:
```json
"validation": {
  "emp_events": 1407,
  "sim_events": 840,
  "rate_ratio": 1.159
}
```

Where does this 1,407 event count come from? How was it computed during model fitting?

---

### TOPIC 2: Event Definition

**Question 2.1**: What exactly is an "is_reorientation_start" event?

The IEI analysis shows:
- Mean IEI: 0.53s
- Median IEI: 0.10s
- Min IEI: 0.00s

A 0.53s mean IEI implies ~113 events per minute per track, which is extremely high. This suggests `is_reorientation_start` may include:
- Head sweeps (frequent, small movements)
- Posture changes
- Frame-by-frame behavioral annotations

But the model expects reorientation EVENTS (discrete turns), not continuous annotations.

**Question 2.2**: Is there a different column or filter that gives the "true" reorientation events at ~1.2 events/min?

Possible columns to check:
- `is_turn_start`?
- `is_reversal`?
- A threshold on `turn_duration` or `curvature`?

**Question 2.3**: The first two event files have 14,617 and 9,214 "is_reorientation_start" events respectively. Why are these so much higher than the other files (which have 400-800)?

Possible explanations:
- Longer duration experiments?
- Different annotation algorithm?
- Different experimental conditions?

---

### TOPIC 3: Stimulus Conditions

**Question 3.1**: The file names contain different stimulus specifications:
- `0to250PWM_30#C_Bl_7PWM` - What does this mean exactly?
- `50to250PWM_30#C_Bl_7PWM` - Different intensity range?
- `#T_Bl_Sq_5to15PWM_30` - Temperature condition?

Which condition was the model fit on?

**Question 3.2**: The model assumes a 30s ON / 30s OFF square wave at 250 PWM. Which files match this exactly?

**Question 3.3**: Should validation be restricted to a single stimulus condition, or should the model generalize across conditions?

---

### TOPIC 4: Track Structure

**Question 4.1**: How are tracks defined within each experiment file?

The event files have columns:
- `track_id` (integer)
- `experiment_id` (string)

Are tracks:
- Individual larva trajectories within one video?
- Segments of a single larva's path?
- Multiple larvae tracked simultaneously?

**Question 4.2**: What is the typical duration of one track?

From the model's track event rates (1.0-2.3 events/min) and the total of 1,407 events across 55 tracks:
- Average: 25.6 events/track
- At 1.5 events/min: ~17 minutes per track
- At 1.0 events/min: ~26 minutes per track

Is this consistent with actual track durations?

**Question 4.3**: The model has 55 tracks, but the engineered directory has 14 files. How do these map?
- Are there multiple tracks per file?
- Which files were used for the 55 tracks?

---

### TOPIC 5: Binned Data vs Raw Events

**Question 5.1**: The NB-GLM was fit on binned/frame-wise data. Where is this binned data stored?

I found no files in `data/binned/`. Possible locations:
- `data/processed/*.parquet`?
- Computed on-the-fly during fitting?
- A different column in the event CSVs?

**Question 5.2**: The binned data should have:
- One row per frame (0.05s at 20 Hz)
- A binary response (0/1) for event occurrence
- Covariates: LED state, time since onset, kernel basis values

How was this binned data constructed from the raw events?

**Question 5.3**: For validation, should I:
- Use the binned data directly?
- Reconstruct binning from raw events?
- Compare only aggregate statistics?

---

### TOPIC 6: Correct Validation Procedure

**Question 6.1**: Given the data alignment issue, what is the correct procedure to validate the analytic hazard model?

Option A: Find and load the exact 55-track, 1,407-event dataset
Option B: Re-bin the raw events and compare frame-by-frame predictions
Option C: Use only aggregate metrics (total events, rate ratio) computed on matched subsets
Option D: Something else?

**Question 6.2**: The model produces ~1.5 events/min at baseline. The empirical data (from model results) shows track rates of 1.0-2.3 events/min. Is this close enough, or should the model be recalibrated?

**Question 6.3**: What validation metrics are most important for publication?
- Rate ratio (simulated/empirical)?
- PSTH correlation?
- IEI distribution match?
- Suppression timing (early vs late)?

---

### TOPIC 7: Integration with Simulation Pipeline

**Question 7.1**: How should the analytic hazard model be integrated into INDYsim's simulation pipeline?

Current structure:
- `scripts/analytic_hazard.py`: Standalone hazard model with `simulate_events_discrete()`
- `scripts/event_generator.py`: Existing event generation (thinning algorithm)
- `scripts/validate_simulation.py`: Validation against empirical data

Should I:
- Replace event_generator.py with analytic_hazard.py?
- Merge them?
- Keep both as options?

**Question 7.2**: For full trajectory simulation (not just event timing), what additional components are needed?
- Head-angle prediction?
- Run/turn state machine?
- Spatial movement model?

---

## Summary of Model Specification

For reference, here is the complete hazard model:

### Hazard Function
```
λ(t) = exp(β₀ + u_track + K_on(t_onset) + K_off(t_offset)) × frame_rate
```

Where:
- β₀ = -6.76 (log events per frame)
- u_track ~ N(0, 0.47²) (track random effect)
- frame_rate = 20 Hz

### LED-ON Kernel (Gamma-Difference)
```
K_on(t) = 0.456 × Γ(t; 2.22, 0.132) - 12.54 × Γ(t; 4.38, 0.869)
```

Timescales:
- Fast peak: 0.16s (sensory transduction)
- Slow peak: 2.94s (synaptic adaptation)
- Net effect: Strong suppression (~7× fewer events during LED-ON)

### LED-OFF Rebound
```
K_off(t) = -0.114 × exp(-t / 2.0)
```

### Simulation Output
- Baseline rate (LED-OFF): ~1.4 events/min
- Suppressed rate (LED-ON): ~0.2 events/min
- Suppression factor: ~7×

---

## What I Need From You

Please provide:

1. **Answers** to the questions above, with specific file paths, column names, or code snippets where applicable.

2. **Data Loading Code**: A Python function that loads the exact 55-track, 1,407-event dataset used for model fitting, so I can run matched validation.

3. **Event Definition Clarification**: The exact criteria for what constitutes a "reorientation event" in the model context (not just `is_reorientation_start`).

4. **Recommended Validation Procedure**: Step-by-step instructions for validating the analytic hazard model against the correct empirical data.

5. **Integration Guidance**: How to integrate the analytic hazard model into the existing INDYsim simulation pipeline.

---

## Available Data Directories

For reference, here are the data directories in the project:

```
data/
  engineered/           # 14 event CSV files, 31119 total events
  engineered_tier2/     # 2 CSV files
  engineered_validated/ # 42 CSV files
  processed/            # 44 parquet files
  model/                # Model results JSON files
  validation/           # Validation outputs
  simulation/           # Simulation outputs
  exports_pre-dr-fix/   # Earlier exports
```

The model was fit using data from one of these sources. Please identify which.

---

## Constraints

- The gamma-difference kernel parameters are **fixed** (already validated at R² = 0.968)
- The hazard model structure is **fixed** (NB-GLM with track random effects)
- The goal is to **validate** the existing model, not refit it
- Publication target: behavioral neuroscience methods paper

---

*Generated: 2025-12-11*
*Project: INDYsim*
*Issue: Data alignment for validation*

