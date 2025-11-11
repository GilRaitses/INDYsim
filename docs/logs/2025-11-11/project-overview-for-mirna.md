# INDYsim Project Overview: Discussion Document for Mirna

**Date:** 2025-11-11  
**Prepared by:** larry  
**Purpose:** Overview of project objectives, DOE design, and research questions for discussion with Mirna

## Project Summary

**INDYsim: Stimulus-Driven Behavioral Modeling of Drosophila Larvae**

We are developing **event-hazard models** for larval behavioral responses to time-varying LED stimuli using generalized linear models with temporal kernels. The simulation framework will generate trajectories under factorial experimental designs and produce Arena-style summary statistics with confidence intervals.

## Current Status

### Data Available
- **4 ESETs** (experiment sets) with **14 total experiments**
- **Genotype:** GMR61@GMR61 (optogenetic variant)
- **Conditions:** 2×2 factorial design
  - LED1 Intensity Range: 2 levels (0-250 PWM vs 50-250 PWM)
  - LED2 Type: 2 levels (Constant 7 PWM vs Time-varying 5-15 PWM)
- **Replications:** 2-4 per condition (imbalanced)

### Current Pipeline Status
- ✅ MATLAB data available
- ⏳ H5 conversion (blocked by LED alignment - P0)
- ⏳ Data processing (pending H5 files)
- ⏳ Model fitting (pending processed data)
- ⏳ Simulation execution (pending fitted models)

## Key Research Questions

### 1. Stimulus-Response Dynamics

**What do we want to learn?**
- How do larvae respond to different LED intensity ranges?
- What is the temporal structure of behavioral responses (latency, duration, decay)?
- How do responses differ between full range (0-250 PWM) vs high range (50-250 PWM)?
- What is the role of LED2 (constant vs time-varying) in modulating responses?

**Hypotheses:**
- Turn rate increases with LED1 intensity
- Response latency is shorter for higher intensity ranges
- Time-varying LED2 modulates LED1 response dynamics
- Behavioral responses follow raised-cosine temporal kernels

### 2. Behavioral Event Modeling

**What do we want to learn?**
- Can we predict reorientations, pauses, and reversals from stimulus parameters?
- What are the hazard rates for different behavioral events?
- How do event rates vary across stimulus conditions?
- What is the relationship between stimulus intensity and event probability?

**Hypotheses:**
- Event hazard rates increase with stimulus intensity
- Different events (reorientations, pauses, reversals) have different temporal dynamics
- Event rates can be modeled with generalized linear models

### 3. Temporal Kernel Characterization

**What do we want to learn?**
- What is the shape of the temporal kernel for stimulus-response dynamics?
- How do raised-cosine kernels capture the response dynamics?
- What are the kernel parameters (width, phase, amplitude)?
- How do kernels differ across conditions?

**Hypotheses:**
- Raised-cosine kernels accurately capture response dynamics
- Kernel parameters vary systematically with stimulus conditions
- Kernel shape reflects underlying neural processing

### 4. Simulation-Based Predictions

**What do we want to learn?**
- Can we predict larval behavior under novel stimulus conditions?
- What is the uncertainty in our predictions (confidence intervals)?
- How do different stimulus parameters affect behavioral outcomes?
- What is the optimal experimental design for future experiments?

**Hypotheses:**
- Model can generalize to novel conditions
- Confidence intervals reflect model uncertainty appropriately
- Simulation can guide experimental design

## Design of Experiments (DOE) Considerations

### Current Experimental Design

**Factorial Structure:**
- **Factor 1:** LED1 Intensity Range
  - Level 1: 0-250 PWM (includes zero baseline)
  - Level 2: 50-250 PWM (no zero baseline)
- **Factor 2:** LED2 Type
  - Level 1: Constant (7 PWM)
  - Level 2: Time-varying (5-15 PWM square wave)

**Total Conditions:** 2 × 2 = 4 ESETs

**Current Replications:**
- ESET 1 (R_0_250_30_B_7): 4 experiments
- ESET 2 (R_0_250_30_B_5_15_30): 4 experiments
- ESET 3 (R_50_250_30_B_7): 4 experiments
- ESET 4 (R_50_250_30_B_5_15_30): 2 experiments ⚠️ **IMBALANCED**

### Proposed Simulation DOE

**Target:** 45 conditions, 30 replications each (1,350 total simulations)

**Factors to Consider:**
1. **Stimulus Intensity** (LED1 PWM)
   - Levels: 0, 50, 100, 150, 200, 250 PWM? (6 levels)
   - Or continuous range with specific test points?

2. **Pulse Duration**
   - Levels: 5s, 10s, 15s, 20s? (4 levels)
   - Current: 10s (to be validated)

3. **Inter-Pulse Interval** (Rest Duration)
   - Levels: 10s, 20s, 30s, 40s, 50s? (5 levels)
   - Current: 30s

**Potential DOE Structure:**
- **Full Factorial:** 6 × 4 × 5 = 120 conditions (too many)
- **Fractional Factorial:** Select subset of conditions
- **Response Surface:** Focus on regions of interest
- **Optimal Design:** D-optimal or I-optimal design

**Question for Mirna:** What stimulus parameter ranges are most interesting/important?

### Replication Requirements

**Current Data:** 2-4 replications per condition (insufficient for robust statistics)

**Simulation Target:** 30 replications per condition

**Rationale for 30 Replications:**
- **Central Limit Theorem:** ~30 samples for normal approximation
- **Confidence Intervals:** 30 provides reasonable CI width
- **Power Analysis:** Depends on effect size and desired power
- **Computational Cost:** Balance between precision and computation time

**CI Considerations:**
- **95% CI width** depends on:
  - Sample size (n)
  - Standard deviation (σ)
  - Formula: CI width ≈ 2 × 1.96 × σ/√n
- **For n=30:** CI width ≈ 0.72 × σ
- **For n=10:** CI width ≈ 1.24 × σ (wider, less precise)

**Question for Mirna:** What level of precision is needed? Is 30 replications sufficient, or do we need more?

## Key Metrics to Analyze

### Primary Metrics
1. **Turn Rate** (min⁻¹)
   - Mean turn rate per condition
   - Temporal dynamics (response over time)
   - Peak turn rate and latency

2. **Latency** (seconds)
   - Time to first behavioral response
   - Time to peak response
   - Response onset detection

3. **Stop Fraction**
   - Proportion of time spent paused
   - Pause frequency and duration
   - Relationship to stimulus parameters

### Secondary Metrics
4. **Tortuosity**
   - Path complexity measure
   - Relationship to stimulus conditions

5. **Dispersal**
   - Spatial spread of trajectories
   - Movement range

6. **Spine Curve Energy**
   - Body shape dynamics
   - Curvature measures

**Question for Mirna:** Which metrics are most important for your research questions?

## Model Outputs

### Arena-Style Summary Statistics

**Per Condition:**
- Mean ± CI for each metric
- Distribution plots
- Temporal dynamics plots
- Comparison across conditions

**Format:**
- Summary tables
- Multi-panel figures
- Statistical comparisons
- Effect size estimates

**Question for Mirna:** What format is most useful for your analysis/publication needs?

## Research Questions for Discussion

### Questions to Ask Mirna:

1. **Stimulus Parameters:**
   - What LED intensity ranges are most biologically relevant?
   - What pulse durations should we explore?
   - What inter-pulse intervals are interesting?
   - Are there specific parameter combinations you want to test?

2. **Behavioral Responses:**
   - What behavioral responses are most important to predict?
   - Are there specific response patterns you're looking for?
   - What time scales are most relevant (seconds, minutes)?

3. **Model Validation:**
   - How should we validate the model predictions?
   - What experimental data can we compare against?
   - What level of accuracy is acceptable?

4. **Experimental Design:**
   - How many conditions can we realistically simulate?
   - What replication level provides sufficient precision?
   - Should we use full factorial or fractional factorial design?

5. **Applications:**
   - What are the main applications of this model?
   - Will this guide future experimental design?
   - Are there specific hypotheses you want to test?

6. **Output Format:**
   - What visualization formats are most useful?
   - What statistical summaries are needed?
   - How should results be presented for publication?

## Next Steps

### Immediate (This Week)
1. ✅ Complete LED alignment (P0 blocking task)
2. ✅ Convert all 14 experiments to H5 format
3. ✅ Process H5 files to extract features
4. ✅ Fit event-hazard models
5. ✅ Validate models against experimental data

### Short-term (Next Week)
1. Design simulation DOE (45 conditions, 30 replications)
2. Run simulations
3. Generate Arena-style summary statistics
4. Create visualizations and reports

### Discussion Outcomes Needed
- **DOE Design:** Confirm 45 conditions or adjust
- **Replication Level:** Confirm 30 or adjust based on CI requirements
- **Research Questions:** Prioritize what to learn from the model
- **Metrics:** Confirm which metrics are most important
- **Output Format:** Confirm visualization and summary preferences

## References

- **Project Proposal:** [INDYsim Project](https://gilraitses.github.io/INDYsim/)
- **Experiment Manifest:** `docs/logs/2025-11-11/experiment-manifest.md`
- **Visualization Structure:** `docs/logs/2025-11-11/visualization-structure.md`

---

**Status:** Ready for discussion with Mirna  
**Last Updated:** 2025-11-11

