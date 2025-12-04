# INDYsim: Stimulus-Driven Behavioral Simulation of Drosophila Larvae

**Event-Hazard Modeling and Design of Experiments for Predictive Simulation**

[![Project Status](https://img.shields.io/badge/status-active-success.svg)](https://github.com/GilRaitses/INDYsim)
[![Documentation](https://img.shields.io/badge/docs-available-blue.svg)](https://gilraitses.github.io/INDYsim/)

---

## 🎯 Project Overview

INDYsim (Interface Dynamics Simulation) is a **simulation modeling framework** that predicts how **Drosophila larvae** respond to **time-varying LED stimuli**. The project develops **event-hazard models** using **generalized linear models (GLMs)** with **temporal kernels** to simulate behavioral trajectories under controlled experimental conditions.

### Core Simulation Capabilities

- **Event-hazard modeling** of behavioral events (reorientations, pauses, reversals)
- **Stimulus-response dynamics** captured via raised-cosine temporal kernels
- **Full factorial design of experiments** (45 conditions, 30 replications each)
- **Simulated trajectory generation** with Arena-style summary statistics
- **Statistical validation** with confidence intervals for behavioral metrics

---

## 🔬 Simulation & Modeling Approach

### Event-Hazard GLM Framework

The core model predicts time-varying hazard rates for behavioral events:

```
λ_E(t) = exp{β₀,E + φ_E^T[s ⋆ κ](t) + x(t)^Tβ_E}
```

Where:
- **λ_E(t)**: Hazard rate for event type *E* ∈ {turn, stop, reverse} at time *t*
- **β₀,E**: Baseline log-hazard for event type *E*
- **s(t)**: Stimulus feature vector (intensity, on/off state, recent history)
- **κ**: Temporal kernel (raised-cosine basis expansion)
- **[s ⋆ κ](t)**: Convolution of stimulus with kernel (captures latency and adaptation)
- **x(t)**: Contextual features (speed, orientation, wall distance)
- **β_E**: Feature coefficients (estimated from data)

### Temporal Kernel Design

**Raised-cosine kernels** capture:
- **Latency effects**: Peak response at delay τ₀ ≈ 0.5-2 seconds
- **Adaptation**: Decay over longer delays (τ > 5 seconds)
- **Anticipation**: Pre-stimulus effects (if any)

### Design of Experiments (DOE)

**Factors:**
- **Stimulus Intensity**: 3 levels (PWM 250, 500, 1000)
- **Pulse Duration**: 5 levels (10s, 15s, 20s, 25s, 30s)
- **Inter-Pulse Interval**: 3 levels (5s, 10s, 20s)

**Design**: Full factorial (3 × 5 × 3 = **45 conditions**) with **30 replications** each

**Response Variables (KPIs):**
- Turn rate (reorientations per minute)
- Latency to first turn
- Stop fraction
- Pause rate
- Path tortuosity
- Spatial dispersal
- Mean spine curve energy
- Reversal rate and duration

---

## 📊 Recent Achievements (December 2025)

### ✅ Platform Liberation: MATLAB → Python Pipeline

**Status**: **Validated and Production-Ready**

Successfully transferred the entire analysis pipeline from MATLAB to Python with **numerical equivalence validation**:

| Validation Layer | Status | Details |
|------------------|--------|---------|
| **H5 Schema Validation** | ✅ PASS | 10/10 experiments validated |
| **Camera Calibration** | ✅ PASS | 7/7 fields exact match |
| **SpeedRunVel Computation** | ✅ PASS | Identical values (< 1e-10 tolerance) |
| **Reversal Detection** | ✅ PASS | Identical results (count, timing, duration) |
| **Turn Detection** | ✅ PASS | 45° threshold, identical event counts |

**Key Resolution**: Critical fix identified that using raw position data (`points/loc`) instead of smoothed data (`derived_quantities/sloc`) caused 5-7x errors in SpeedRunVel computation. Python pipeline now uses correct smoothed position data.

### ✅ Enhanced Analysis Pipeline

**New Capabilities**:
- **Stimulus-window analysis**: Per-track and population-level metrics within LED on/off windows
- **Reversal detection**: SpeedRunVel < 0 for > 3s duration
- **Turn detection**: 45° angle threshold with directional classification
- **Concurrency estimation**: Active tracks per time bin
- **Master H5 export**: Combined analysis ready for simulator input

**Output Structure**:
- Per-file JSON analysis (track-level, window-level, population aggregates)
- Combined analysis JSON (all experiments)
- Master H5 file for simulation intake

---

## 🚀 What Can Be Done With the Validated Data

### 1. **Fit Event-Hazard Models**

With 10 validated H5 files containing behavioral trajectories and stimulus timing:

- **Estimate temporal kernels** (φ_E) for each event type
- **Fit GLM coefficients** (β₀,E, β_E) using stimulus-locked analysis
- **Cross-validate** using leave-one-larva-out methodology
- **Characterize** latency, adaptation, and intensity-response relationships

### 2. **Run Simulation Experiments**

Using the fitted models:

- **Generate simulated trajectories** for all 45 DOE conditions
- **Produce 30 replications** per condition (1,350 total simulations)
- **Compute KPIs** with confidence intervals
- **Compare** simulated vs. empirical behavioral metrics

### 3. **Explore Stimulus Parameter Space**

The DOE framework enables:

- **Systematic exploration** of intensity × duration × interval effects
- **Response surface modeling** to predict behavior at untested conditions
- **Optimization** of stimulus protocols for desired behavioral outcomes
- **Sensitivity analysis** of model parameters

### 4. **Validate Model Predictions**

Compare simulation outputs to empirical data:

- **Turn rate predictions** vs. observed turn rates
- **Latency distributions** vs. empirical latencies
- **Reversal patterns** vs. observed reversal events
- **Spatial metrics** (tortuosity, dispersal) vs. empirical trajectories

### 5. **Extend to New Genotypes/Conditions**

The validated pipeline supports:

- **New genotype analysis** (beyond GMR61@GMR61)
- **Different stimulus protocols** (varying waveforms, frequencies)
- **Environmental conditions** (temperature, humidity effects)
- **Pharmacological interventions** (drug effects on behavior)

---

## 📁 Repository Structure

```
INDYsim/
├── data/
│   ├── h5_validated/              # ✅ 10 validated H5 files (ready for simulation)
│   │   ├── analysis/              # Enhanced analysis outputs
│   │   │   ├── *_analysis.json   # Per-file analysis
│   │   │   ├── combined_analysis.json
│   │   │   └── master_sim_input.h5
│   │   └── manifest.json          # Provenance tracking
│   ├── matlab_data/               # Source MATLAB files (gitignored)
│   └── engineered/                # Processed datasets
│
├── scripts/
│   ├── 2025-12-04/
│   │   └── platform_liberation/   # ✅ Validated Python pipeline
│   │       ├── engineer_dataset_from_h5.py  # Enhanced analysis
│   │       ├── validation/        # Validation framework
│   │       └── h5_export/         # MATLAB → H5 conversion
│   ├── engineer_dataset_from_h5.py # Original analysis script
│   └── queue/                     # Analysis scripts
│
├── config/
│   ├── doe_table.csv              # 45-condition DOE specification
│   └── model_config.json          # Model configuration (kernels, KPIs)
│
├── docs/
│   ├── project-proposal-indysim.qmd  # Full project proposal
│   ├── logs/                      # Daily progress logs
│   └── work-trees/                # Task planning documents
│
└── output/
    └── figures/                   # Generated visualizations
```

---

## 🔧 Technical Stack

- **Python 3.11+**: Data processing, analysis, and simulation
- **H5Py**: HDF5 file format for trajectory data
- **NumPy/Pandas**: Numerical computing and data manipulation
- **SciPy**: Statistical modeling and optimization
- **MAGAT Segmentation**: Larval track segmentation (runs, reorientations, head swings)
- **Quarto/LaTeX**: Report generation and documentation
- **MATLAB**: Reference implementation (validated against)

---

## 📚 Key Documentation

### Core Documentation
- **[Project Proposal](docs/project-proposal-indysim.qmd)**: Full methodology and theoretical framework
- **[Platform Liberation README](scripts/2025-12-04/platform_liberation/README.md)**: MATLAB → Python transfer details
- **[Validation Report](scripts/2025-12-04/platform_liberation/validation/VALIDATION_REPORT.md)**: Numerical equivalence validation
- **[Discrepancy Report](scripts/2025-12-04/platform_liberation/validation/DISCREPANCY_REPORT.md)**: Issues found and resolved
- **[Field Mapping](scripts/2025-12-04/platform_liberation/validation/FIELD_MAPPING.md)**: H5 schema documentation

### Recent Work Logs
- **[December 4, 2025](scripts/2025-12-04/platform_liberation/agent/handoff/README.md)**: Platform liberation handoff
- **[November 13, 2025](docs/logs/2025-11-13.md)**: Integration testing and validation
- **[November 12, 2025](docs/logs/2025-11-12.md)**: LED alignment and path cleanup
- **[November 11, 2025](docs/logs/2025-11-11.md)**: MATLAB to H5 conversion pipeline

---

## 🎓 Academic Context

**Course**: ECS630 - Simulation Modeling  
**Institution**: [Your Institution]  
**Term**: Fall 2025

This project applies **simulation modeling methods** from ECS630 to biological behavioral data, developing a **stimulus-response model** that simulates larval trajectories under different experimental conditions. The **event-hazard framework** models stochastic behavioral events, while the **DOE methodology** explores stimulus parameter space systematically.

### Simulation Modeling Concepts Applied

1. **Stochastic Process Modeling**: Event-hazard rates as time-varying stochastic processes
2. **Design of Experiments**: Systematic exploration of parameter space
3. **Monte Carlo Simulation**: Multiple replications for statistical inference
4. **Model Validation**: Comparison of simulated vs. empirical data
5. **Statistical Inference**: Confidence intervals and hypothesis testing

---

## 🧪 Dataset

**Genotype**: `GMR61@GMR61` (optogenetic variant)

**Validated Experiments**: **10 H5 files** ready for simulation
- **Location**: `data/h5_validated/`
- **Format**: HDF5 with validated schema
- **Provenance**: Tracked via `manifest.json`
- **Size**: ~3GB total

**Experimental Conditions**:
- **ESET 1**: T_Re_Sq_0to250PWM_30#C_Bl_7PWM (4 experiments)
- **ESET 2**: T_Re_Sq_0to250PWM_30#T_Bl_Sq_5to15PWM_30 (4 experiments)
- **ESET 3**: T_Re_Sq_50to250PWM_30#C_Bl_7PWM (4 experiments)
- **ESET 4**: T_Re_Sq_50to250PWM_30#T_Bl_Sq_5to15PWM_30 (2 experiments)

---

## 🚦 Current Status

### ✅ Completed

- **MATLAB to H5 Conversion**: All 14 MATLAB ESET files converted to H5 format
- **Platform Liberation**: Complete Python pipeline with numerical validation
- **Enhanced Analysis**: Stimulus-window analysis, reversal/turn detection
- **Data Validation**: 10 experiments validated and ready for simulation
- **Master H5 Export**: Combined analysis ready for simulator intake

### 🔄 Next Steps

1. **Fit Event-Hazard Models**: Estimate GLM coefficients from validated data
2. **Run Simulation Experiments**: Generate trajectories for 45 DOE conditions
3. **Validate Predictions**: Compare simulated vs. empirical behavioral metrics
4. **Generate Reports**: Statistical analysis and visualization of results

---

## 🔗 Quick Links

- **Live Documentation**: [https://gilraitses.github.io/INDYsim/](https://gilraitses.github.io/INDYsim/)
- **Repository**: [github.com/GilRaitses/INDYsim](https://github.com/GilRaitses/INDYsim)

---

## 📝 Usage Example

### Run Enhanced Analysis on Validated H5 Files

```bash
cd scripts/2025-12-04/platform_liberation
python engineer_dataset_from_h5.py data/h5_validated \
  -o data/h5_validated/analysis
```

### Export Master H5 for Simulator

```bash
cd scripts/2025-12-04/platform_liberation/agent/worktree
python export_master_h5.py \
  --combined data/h5_validated/analysis/combined_analysis.json \
  --output data/h5_validated/master_sim_input.h5
```

---

## 👥 Contributors

- **Gil Raitses** - Project Lead

---

**Last Updated**: December 4, 2025  
**Status**: ✅ **Ready for Simulation** - Validated data and pipeline complete

---

## 💡 Discussion Points for Professor Meeting

### What This Project Demonstrates

1. **Simulation Modeling**: Event-hazard framework for stochastic behavioral processes
2. **Design of Experiments**: Systematic parameter space exploration (45 conditions)
3. **Model Validation**: Numerical equivalence between MATLAB and Python implementations
4. **Data Engineering**: Robust pipeline from raw data to simulation-ready format
5. **Statistical Inference**: Confidence intervals and replication-based validation

### Potential Extensions

1. **Multi-genotype Comparison**: Extend to additional genotypes beyond GMR61@GMR61
2. **Temporal Dynamics**: Explore adaptation and habituation effects over longer timescales
3. **Spatial Modeling**: Incorporate arena geometry and wall interactions
4. **Machine Learning**: Deep learning approaches for event prediction
5. **Real-time Simulation**: Interactive simulation for experimental design

### Research Questions Addressable

- How do stimulus intensity, duration, and interval interact to affect behavioral responses?
- What are the optimal stimulus protocols for eliciting specific behaviors?
- How do individual differences affect population-level predictions?
- Can we predict behavior in novel experimental conditions?
