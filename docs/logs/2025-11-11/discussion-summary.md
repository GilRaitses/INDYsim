# Quick Discussion Summary: INDYsim Project

**Date:** 2025-11-11  
**For:** Mirna  
**Purpose:** Quick reference for project discussion

## Project Goal

Build **event-hazard models** to predict larval behavioral responses to LED stimuli, then use these models to simulate behavior under **45 different stimulus conditions** with **30 replications each** (1,350 total simulations).

## What We Have

- ✅ **14 experiments** across **4 conditions** (2×2 factorial design)
- ✅ **GMR61@GMR61** genotype (optogenetic variant)
- ⏳ **H5 conversion** in progress (blocked by LED alignment)

## What We Want to Learn

### Key Research Questions:

1. **How do larvae respond to different LED intensities?**
   - Turn rate, latency, stop fraction
   - Temporal dynamics (response over time)

2. **Can we predict behavioral events (reorientations, pauses, reversals)?**
   - Event hazard rates
   - Relationship to stimulus parameters

3. **What is the temporal structure of responses?**
   - Raised-cosine kernel parameters
   - Response shape and timing

4. **Can we predict behavior under novel conditions?**
   - Model generalization
   - Confidence intervals for predictions

## DOE Design Questions

### 1. How many conditions? (Target: 45)

**Factors to consider:**
- **Stimulus Intensity:** 6 levels? (0, 50, 100, 150, 200, 250 PWM)
- **Pulse Duration:** 3-4 levels? (5s, 10s, 15s, maybe 20s)
- **Inter-Pulse Interval:** 4-5 levels? (10s, 20s, 30s, 40s, maybe 50s)

**Options:**
- Full factorial: 6×4×5 = 120 (too many)
- Fractional factorial: ~45 conditions (recommended)
- Response surface: Focus on interesting regions

**Question:** What parameter ranges are most important?

### 2. How many replications? (Target: 30)

**Statistical considerations:**
- **n=30:** CI width ≈ 0.37 × SD, Power ≈ 0.8-0.9
- **n=20:** CI width ≈ 0.47 × SD, Power ≈ 0.7-0.8
- **n=50:** CI width ≈ 0.28 × SD, Power ≈ 0.9-0.95

**Computational cost:**
- 1,350 simulations (45×30): ~22-112 minutes
- Feasible with parallelization

**Question:** Is 30 replications sufficient, or do we need more precision?

### 3. Which conditions to simulate?

**Priority:**
1. ✅ Current experimental conditions (4) - for validation
2. ⭐ Extreme conditions - boundaries of parameter space
3. 📊 Intermediate conditions - fill gaps, explore response surface

**Question:** Are there specific conditions you want to test?

## Key Metrics

**Primary:**
- Turn rate (min⁻¹)
- Latency (seconds)
- Stop fraction

**Secondary:**
- Tortuosity
- Dispersal
- Spine curve energy

**Question:** Which metrics are most important for your research?

## Output Format

**Arena-style summary statistics:**
- Mean ± 95% CI for each metric
- Multi-panel figures
- Statistical comparisons
- Effect size estimates

**Question:** What format is most useful for your analysis/publication?

## Timeline

- **This Week:** Complete data processing, fit models
- **Next Week:** Run simulations, generate results

## Discussion Outcomes Needed

1. ✅ Confirm DOE design (45 conditions or adjust)
2. ✅ Confirm replication level (30 or adjust)
3. ✅ Prioritize research questions
4. ✅ Confirm important metrics
5. ✅ Confirm output format preferences

---

**Full Documents:**
- **Project Overview:** `docs/logs/2025-11-11/project-overview-for-mirna.md`
- **DOE Design:** `docs/logs/2025-11-11/doe-design-considerations.md`

