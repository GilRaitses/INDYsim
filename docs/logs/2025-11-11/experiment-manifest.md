# Experiment Manifest: INDYsim Simulation Model Dataset

**Date:** 2025-11-11  
**Prepared by:** larry  
**Purpose:** Document all experiments and conditions used to build the simulation model

## Project Context

**Project:** INDYsim - Stimulus-Driven Behavioral Modeling of Drosophila Larvae  
**Objective:** Develop event-hazard models for larval behavioral responses to time-varying LED stimuli using generalized linear models with temporal kernels.

**Key Modeling Components:**
- Event-hazard modeling of reorientations, pauses, and reversals
- Raised-cosine temporal kernels capturing stimulus-response dynamics
- Key metrics: turn rate, latency, stop fraction, tortuosity, dispersal, spine curve energy

## Genotype

**Genotype:** `GMR61@GMR61`
- Optogenetic variant of larva
- Engineered sensory activation by red light
- Represents neural pathway variant

## Experiment Sets (ESETs)

**Total ESETs:** 4  
**Total Experiments:** 14  
**Location:** `data/matlab_data/GMR61@GMR61/`

### ESET 1: T_Re_Sq_0to250PWM_30#C_Bl_7PWM

**Condition Description:**
- **LED1:** Red, Square wave, 0-250 PWM range, 30s LED off (resting interval)
- **LED2:** Constant Blue at 7 PWM
- **Stimulus Period:** 10 seconds (to be validated)

**Experiments:** 4

| Experiment ID | Timestamp | MAT File | Status |
|--------------|-----------|----------|--------|
| EXP-001 | 202510291652 | `btd_GMR61@GMR61_T_Re_Sq_0to250PWM_30#C_Bl_7PWM_202510291652.mat` | Pending H5 conversion |
| EXP-002 | 202510291713 | `btd_GMR61@GMR61_T_Re_Sq_0to250PWM_30#C_Bl_7PWM_202510291713.mat` | Pending H5 conversion |
| EXP-003 | 202510301228 | `btd_GMR61@GMR61_T_Re_Sq_0to250PWM_30#C_Bl_7PWM_202510301228.mat` | Pending H5 conversion |
| EXP-004 | 202510301408 | `btd_GMR61@GMR61_T_Re_Sq_0to250PWM_30#C_Bl_7PWM_202510301408.mat` | Pending H5 conversion |

**Simplified Name:** `R_0_250_30_B_7`

**Modeling Purpose:**
- Baseline condition with full LED1 intensity range (0-250 PWM)
- Constant LED2 provides background illumination
- Tests response to varying red light intensity with constant blue background

---

### ESET 2: T_Re_Sq_0to250PWM_30#T_Bl_Sq_5to15PWM_30

**Condition Description:**
- **LED1:** Red, Square wave, 0-250 PWM range, 30s LED off (resting interval)
- **LED2:** Blue, Square wave, 5-15 PWM range, 30s LED off (resting interval)
- **Stimulus Period:** 10 seconds (to be validated)

**Experiments:** 4

| Experiment ID | Timestamp | MAT File | Status |
|--------------|-----------|----------|--------|
| EXP-005 | 202510301513 | `btd_GMR61@GMR61_T_Re_Sq_0to250PWM_30#T_Bl_Sq_5to15PWM_30_202510301513.mat` | Pending H5 conversion |
| EXP-006 | 202510311441 | `btd_GMR61@GMR61_T_Re_Sq_0to250PWM_30#T_Bl_Sq_5to15PWM_30_202510311441.mat` | Pending H5 conversion |
| EXP-007 | 202510311510 | `btd_GMR61@GMR61_T_Re_Sq_0to250PWM_30#T_Bl_Sq_5to15PWM_30_202510311510.mat` | Pending H5 conversion |
| EXP-008 | 202510311634 | `btd_GMR61@GMR61_T_Re_Sq_0to250PWM_30#T_Bl_Sq_5to15PWM_30_202510311634.mat` | Pending H5 conversion |

**Simplified Name:** `R_0_250_30_B_5_15_30`

**Modeling Purpose:**
- Full LED1 intensity range with time-varying LED2
- Tests interaction between red light intensity and blue light modulation
- Both LEDs have 10s stimulus periods with 30s rest intervals

---

### ESET 3: T_Re_Sq_50to250PWM_30#C_Bl_7PWM

**Condition Description:**
- **LED1:** Red, Square wave, 50-250 PWM range, 30s LED off (resting interval)
- **LED2:** Constant Blue at 7 PWM
- **Stimulus Period:** 10 seconds (to be validated)

**Experiments:** 4

| Experiment ID | Timestamp | MAT File | Status |
|--------------|-----------|----------|--------|
| EXP-009 | 202510291435 | `btd_GMR61@GMR61_T_Re_Sq_50to250PWM_30#C_Bl_7PWM_202510291435.mat` | Pending H5 conversion |
| EXP-010 | 202510291502 | `btd_GMR61@GMR61_T_Re_Sq_50to250PWM_30#C_Bl_7PWM_202510291502.mat` | Pending H5 conversion |
| EXP-011 | 202510291532 | `btd_GMR61@GMR61_T_Re_Sq_50to250PWM_30#C_Bl_7PWM_202510291532.mat` | Pending H5 conversion |
| EXP-012 | 202510291601 | `btd_GMR61@GMR61_T_Re_Sq_50to250PWM_30#C_Bl_7PWM_202510291601.mat` | Pending H5 conversion |

**Simplified Name:** `R_50_250_30_B_7`

**Modeling Purpose:**
- Reduced LED1 intensity range (50-250 PWM, no zero baseline)
- Constant LED2 provides background illumination
- Tests response to higher-intensity red light range without zero baseline

---

### ESET 4: T_Re_Sq_50to250PWM_30#T_Bl_Sq_5to15PWM_30

**Condition Description:**
- **LED1:** Red, Square wave, 50-250 PWM range, 30s LED off (resting interval)
- **LED2:** Blue, Square wave, 5-15 PWM range, 30s LED off (resting interval)
- **Stimulus Period:** 10 seconds (to be validated)

**Experiments:** 2

| Experiment ID | Timestamp | MAT File | Status |
|--------------|-----------|----------|--------|
| EXP-013 | 202511051636 | `btd_GMR61@GMR61_T_Re_Sq_50to250PWM_30#T_Bl_Sq_5to15PWM_30_202511051636.mat` | Pending H5 conversion |
| EXP-014 | 202511051713 | `btd_GMR61@GMR61_T_Re_Sq_50to250PWM_30#T_Bl_Sq_5to15PWM_30_202511051713.mat` | Pending H5 conversion |

**Simplified Name:** `R_50_250_30_B_5_15_30`

**Modeling Purpose:**
- Reduced LED1 intensity range with time-varying LED2
- Tests interaction between higher-intensity red light and blue light modulation
- Both LEDs have 10s stimulus periods with 30s rest intervals

---

## Experimental Design Summary

### Factorial Design Structure

**Factors:**
1. **LED1 Intensity Range:** 2 levels
   - Low range: 0-250 PWM (includes zero baseline)
   - High range: 50-250 PWM (no zero baseline)

2. **LED2 Type:** 2 levels
   - Constant: 7 PWM
   - Time-varying: 5-15 PWM square wave

**Total Conditions:** 2 × 2 = 4 ESETs

### Replication Structure

| ESET | Condition | Replications | Total Experiments |
|------|-----------|--------------|-------------------|
| ESET 1 | R_0_250_30_B_7 | 4 | 4 |
| ESET 2 | R_0_250_30_B_5_15_30 | 4 | 4 |
| ESET 3 | R_50_250_30_B_7 | 4 | 4 |
| ESET 4 | R_50_250_30_B_5_15_30 | 2 | 2 |
| **Total** | | **14** | **14** |

**Note:** ESET 4 has fewer replications (2 vs 4). This imbalance should be considered in statistical analysis.

## Stimulus Parameters

### Common Parameters (All ESETs)
- **Stimulus Period:** 10 seconds (to be validated from LED ON/OFF patterns)
- **Rest Interval:** 30 seconds (LED off duration)
- **Waveform:** Square wave
- **Frame Rate:** 10 fps (typical)

### LED1 Parameters

| ESET | Color | PWM Range | Min | Max | Rest Interval |
|------|-------|-----------|-----|-----|---------------|
| ESET 1, 2 | Red | 0-250 | 0 | 250 | 30s |
| ESET 3, 4 | Red | 50-250 | 50 | 250 | 30s |

### LED2 Parameters

| ESET | Type | Color | PWM Values | Rest Interval |
|------|------|-------|------------|---------------|
| ESET 1, 3 | Constant | Blue | 7 | N/A |
| ESET 2, 4 | Time-varying | Blue | 5-15 | 30s |

## Data Processing Pipeline

### Current Status

**Stage 1: MATLAB to H5 Conversion** ⏳ In Progress
- Conversion script: `src/@matlab_conversion/convert_matlab_to_h5.py`
- Status: LED value alignment integration pending
- Blocking issue: LED value timecode alignment (P0)

**Stage 2: H5 Processing** ⏳ Pending
- Processing script: `scripts/engineer_dataset_from_h5.py`
- Features: Turn rate, latency, stop fraction, tortuosity, dispersal, spine curve energy
- Status: Waiting for H5 files

**Stage 3: Model Fitting** ⏳ Pending
- Event-hazard models for reorientations, pauses, reversals
- Raised-cosine temporal kernels
- Status: Waiting for processed data

**Stage 4: Simulation** ⏳ Pending
- Full factorial DOE (45 conditions, 30 replications each)
- Arena-style summary statistics with confidence intervals
- Status: Waiting for fitted models

## Validation Requirements

### Stimulus Period Validation
- **Required:** Validate 10-second stimulus period using pattern recognition
- **Variables:** `led1Val_ton`, `led1Val_toff`, `led2Val_ton`, `led2Val_toff`
- **Process:**
  1. Detect LED ON/OFF transitions using threshold
  2. Calculate intervals between transitions
  3. Verify period is ~10 seconds
  4. Log start and end ETIs for each stimulus pulse
  5. Calculate duty cycle for each pulse

### Data Quality Checks
- Track count per experiment
- Frame count per experiment
- LED value alignment with track timecode
- Period detection consistency
- Missing data identification

## Model Building Strategy

### Training Data
- **Primary:** All 14 experiments from 4 ESETs
- **Purpose:** Fit event-hazard models and temporal kernels
- **Features:** Turn rate, latency, stop fraction, tortuosity, dispersal, spine curve energy

### Simulation Design
- **Target:** 45 conditions, 30 replications each (1,350 total simulations)
- **Factors:** Stimulus intensity × pulse duration × inter-pulse interval
- **Output:** Arena-style summary statistics with confidence intervals

### Validation Strategy
- Cross-validation within ESETs
- Comparison with experimental data
- Sensitivity analysis of model parameters

## File Naming Conventions

### H5 Output Files
**Pattern:** `{GENOTYPE}@{GENOTYPE}_{ESET}_{TIMESTAMP}.h5`

**Example:** `GMR61@GMR61_T_Re_Sq_0to250PWM_30#C_Bl_7PWM_202510291652.h5`

**Location:** `data/h5_files/`

### Processed Data Files
**Pattern:** `{ESET}_{TIMESTAMP}_processed.h5` or `.parquet`

**Location:** `data/processed/`

## Next Steps

1. **Complete H5 Conversion** (Blocked by LED alignment)
   - Integrate LED value timecode alignment
   - Convert all 14 experiments to H5 format
   - Validate H5 file structure and content

2. **Process H5 Files**
   - Extract features (turn rate, latency, etc.)
   - Validate stimulus period detection
   - Create processed dataset

3. **Fit Models**
   - Event-hazard models for behavioral events
   - Temporal kernel estimation
   - Model validation

4. **Run Simulations**
   - Generate 1,350 simulations (45 conditions × 30 replications)
   - Compute summary statistics
   - Generate Arena-style outputs

## References

- **Project Proposal:** [INDYsim Project](https://gilraitses.github.io/INDYsim/)
- **ESET Naming Convention:** `docs/logs/2025-11-11/lab-eset-naming-convention.md`
- **LED Alignment Issue:** `docs/logs/2025-11-11/LED_TIMECODE_ALIGNMENT_ISSUE.md`
- **Work Tree:** `docs/work-trees/2025-11-11-work-tree.md`

---

**Status:** Initial manifest created  
**Last Updated:** 2025-11-11  
**Next Update:** After H5 conversion completion

