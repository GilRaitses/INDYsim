# Complete Review and Research Prompt

**Date**: 2025-12-11  
**Status**: Model validation passing, but absolute rate mismatch persists

---

## 1. Current State Summary

### What Works Well

| Metric | Value | Threshold | Status |
|--------|-------|-----------|--------|
| PSTH Correlation | **0.888** | ≥ 0.645 | PASS |
| W-ISE | **0.067** | ≤ 0.304 | PASS |
| Early Suppression (normalized) | 0.55 sim vs 0.55 emp | - | 1% diff |
| Late Suppression (normalized) | 0.36 sim vs 0.32 emp | - | 4% diff |
| Suppression Pattern | Building | Building | PASS |

### What Doesn't Work

| Issue | Observed | Expected | Gap |
|-------|----------|----------|-----|
| **Absolute Event Rate** | 1.26 events/min/track | 0.71 events/min/track | **1.77x too high** |
| **Intercept** | -6.66 | -7.44 (target) | 0.78 ln-units |

---

## 2. Detailed Problem Analysis

### 2.1 The Rate Mismatch Puzzle

**Observation**: The simulated rate is 1.77x higher than empirical, yet:
- The PSTH *shape* matches near-perfectly (correlation 0.888)
- The normalized suppression magnitudes match (1% and 4% diff)
- The temporal pattern is correct (building suppression)

**Attempted Fixes**:
1. Widened intercept bounds from `(-8, -6.5)` to `(-9, -5)` → No change
2. The optimizer genuinely prefers intercept = -6.66

**Implication**: The MLE intercept of -6.66 is what the data wants. This suggests either:
- The empirical rate calculation (0.71) is based on different data or method
- The fitting script's empirical rate (1.28) is the correct reference
- There's a systematic issue in how we count events or tracks

### 2.2 Empirical Rate Discrepancy - **BUG FOUND AND FIXED**

The fitting script was reporting:
```
Empirical rate: 1.28 events/min/track
```

But the correct rate is:
```
Empirical rate: 0.71 events/min/track
```

**Root Cause**: The fitting script used `data['track_id'].nunique()` which returned 55 because track IDs (0-43, 0-54) overlap across the two experiments. The correct calculation uses unique `(experiment_id, track_id)` pairs which gives 99 tracks.

**Fix Applied**: Updated `fit_extended_biphasic_model.py` to use `data.groupby(['experiment_id', 'track_id']).ngroups`.

**Implication**: The model was being compared against the wrong baseline rate. However, this doesn't change the fitted coefficients since the MLE doesn't depend on this calculation. The intercept of -6.66 is still the true MLE.

### 2.3 Intercept vs Rate Analysis

**Key Calculation**:
```
Current intercept: -6.66 → 1.54 events/min baseline
Target intercept:  -7.19 → 0.91 events/min baseline (accounting for suppression)
Difference: 0.53 ln-units → 1.7x rate difference
```

The MLE intercept produces a rate 1.7x higher than empirical. This is puzzling because the MLE should minimize deviance, which should naturally match the observed rate.

**Possible Explanations**:
1. The kernel is absorbing some of the baseline (e.g., negative kernel values during OFF phase)
2. The NB dispersion parameter is affecting the MLE
3. There's a systematic issue in the GLM formulation

### 2.4 Frame Rate and Hazard Units

**Current understanding**:
- GLM is fit on frame-wise data (20 Hz, dt = 0.05s)
- GLM output `exp(eta)` is interpreted as per-frame probability
- Simulation multiplies by `frame_rate=20` to convert to per-second hazard

**MiroThinker said** (in pseudocode):
```
# 2.1. Hazard Function (No LED Main Effect, No Frame‑Rate Scaling)
```

But also confirmed that:
- The GLM outputs per-frame probability
- Conversion to per-second hazard is needed

**Confusion**: Should we multiply by frame_rate or not? We kept it because removing it caused rates to drop to 0.10 events/min.

---

## 3. Specific Questions for Debugging

### Q1: Empirical Rate Calculation

The empirical event rate appears to be calculated two different ways:
- **0.71** events/min/track (1407 events / 99 tracks / 20 min)
- **1.28** events/min/track (from fitting script)

**Questions**:
1. Are all 99 tracks actually 20 minutes long, or are some shorter?
2. Is `track_id` unique across experiments or within experiments?
3. How many unique tracks are in the fitting data vs validation data?

### Q2: GLM Interpretation

The fitted intercept is -6.66, which implies:
- Per-frame hazard: exp(-6.66) = 0.00128
- Per-second rate: 0.00128 × 20 = 0.0257 events/s
- Per-minute rate: 0.0257 × 60 = 1.54 events/min

**Questions**:
1. Is the GLM fitting to per-frame binary events (0/1 per frame)?
2. Should the intercept be interpreted as log(events/frame) or log(events/second)?
3. Is there an offset term being used in the GLM that we're not accounting for in simulation?

### Q3: Simulation Hazard Integration

The inversion sampler does:
```python
cumulative += h * dt  # where h = exp(eta) * frame_rate
```

**Questions**:
1. Is this the correct interpretation of the fitted model?
2. Should we be using `h = exp(eta)` (per-frame) directly?
3. Or should the GLM be fit with an offset for exposure time?

### Q4: Biological Plausibility

The model shows:
- Strong early positive bump (+0.66 at 0.2s)
- Rapid suppression onset (-0.22 at 0.7s, -2.82 at 1.4s)
- Peak suppression (-2.85 at 3s)
- Recovery toward baseline (9s: -0.29)

**Questions**:
1. Is this temporal pattern consistent with known optogenetic responses in larvae?
2. Does the ~1.8x rate mismatch invalidate the model, or is shape more important?
3. Should we expect the absolute rate to match, or only the normalized pattern?

---

## 4. Potential Refinement Strategies

### Strategy A: Fix the Rate Calculation

Verify which empirical rate is correct:
1. Count unique tracks in the data files
2. Calculate total observation time per track
3. Recompute true empirical rate

### Strategy B: Add an Exposure Offset

If tracks have different durations, add an offset term to the GLM:
```python
# In fitting
offset = np.log(track_duration_frames)
model = GLM(y, X, offset=offset, family=NegativeBinomial())
```

### Strategy C: Reframe as Shape-Only Validation

Accept that absolute rate matching is not the goal:
- Normalize both empirical and simulated by their respective baselines
- Report only correlation, W-ISE, and pattern metrics
- Document rate mismatch as a known limitation

### Strategy D: Investigate Track-Level Variability

The 99 tracks may have different baseline rates:
1. Compute per-track event rates
2. Check for outlier tracks
3. Consider mixed-effects model (random intercept per track)

---

## 5. Research Agent Prompt

### Context

I am building a Linear-Nonlinear-Poisson (LNP) model for larval reorientation events during optogenetic stimulation in 2nd instar *Drosophila* larvae (GMR61 > CsChrimson). The model is implemented as a Negative-Binomial GLM with a triphasic raised-cosine temporal kernel.

### Current Status

The model passes all shape-based validation metrics:
- PSTH correlation: 0.888 (bootstrap threshold ≥ 0.645)
- W-ISE: 0.067 (bootstrap threshold ≤ 0.304)
- Normalized early suppression: 0.55 sim vs 0.55 emp (1% diff)
- Normalized late suppression: 0.36 sim vs 0.32 emp (4% diff)

However, the **absolute event rate** is mismatched:
- Simulated: 1.26 events/min/track
- Empirical: 0.71 events/min/track
- Ratio: 1.77x too high

The fitted GLM intercept is -6.66, but based on the empirical rate of 0.71 events/min/track, the expected intercept would be approximately -7.44.

### Model Details

**Kernel Structure** (triphasic):
- Early (0-1.5s): 3 bases at [0.2, 0.7, 1.4]s, width 0.4s. First constrained ≥0.
- Intermediate (1.5-3s): 2 bases at [2.0, 2.5]s, width 0.6s. Unconstrained.
- Late (3-10s): 4 bases at [3.0, 5.0, 7.0, 9.0]s, width 1.8s. Unconstrained.

**Fitted Coefficients**:
| Coefficient | Value | Interpretation |
|-------------|-------|----------------|
| intercept | -6.66 | Baseline log-rate |
| kernel_early_1 | +0.66 | Early positive bump |
| kernel_early_2 | -0.22 | Suppression onset |
| kernel_early_3 | -2.82 | Strong suppression |
| kernel_intm_1 | -1.38 | Suppression building |
| kernel_intm_2 | -0.61 | Suppression |
| kernel_late_1 | -2.85 | Peak suppression |
| kernel_late_2 | -1.97 | Sustained suppression |
| kernel_late_3 | -0.31 | Recovery begins |
| kernel_late_4 | -0.29 | Near baseline |

**Data**:
- 1,309,730 frames (at 20 Hz)
- 1,407 reorientation events
- ~99 tracks
- 20-minute experiments (10s ON / 20s OFF LED cycles)

**Simulation Method**:
- Inversion sampling with hazard = exp(intercept + kernel) × frame_rate × dt
- Generates events in continuous time

### Specific Questions

**Q1: Why Does MLE Intercept Not Match Empirical Rate?**

**UPDATE**: The 0.71 vs 1.28 discrepancy was a bug (track IDs not unique across experiments). The correct empirical rate is **0.71 events/min/track**.

However, the MLE intercept of -6.66 implies a baseline rate of ~1.54 events/min, which is 2.2x higher than empirical. After accounting for LED suppression (cycle average), the simulated rate is 1.26 events/min, which is 1.77x higher.

**Key Question**: Why doesn't the MLE produce an intercept that matches the empirical rate?

The MLE *should* minimize deviance, which generally means the predicted rate should match observed rate. Possible explanations:
1. The kernel is absorbing baseline (negative values during LED-OFF leak into intercept estimate)
2. The NB dispersion parameter is affecting the rate estimate
3. There's a systematic issue in the GLM formulation (exposure, offset, etc.)
4. The raised-cosine bases have non-zero values during LED-OFF periods

**Q2: GLM Intercept Interpretation**

The GLM is fit on frame-wise binary events (0/1 per 0.05s frame). The fitted intercept is -6.66.

- Should exp(-6.66) be interpreted as events/frame or events/second?
- When simulating, should the hazard be multiplied by frame_rate (20 Hz) to convert to events/second?
- Is there a standard convention in LNP literature for handling this?

**Q3: Is Rate Mismatch Acceptable?**

Given that:
- The PSTH shape matches nearly perfectly (0.888 correlation)
- The normalized suppression magnitudes match (1% and 4% diff)
- The temporal pattern is correct (building suppression)

Is a 1.77x rate mismatch acceptable for:
a) A course project / methods demonstration?
b) A scientific publication?
c) Use as a simulation tool for power analysis?

What are the standards in the larval LNP literature (e.g., Hernandez-Nunez 2015, Gepner 2015)?

**Q4: How to Fix the Rate**

If we want to fix the absolute rate, what are the recommended approaches?
1. Add an offset term to the GLM for track-level exposure?
2. Constrain the intercept to match empirical baseline?
3. Post-hoc scale the hazard (is this defensible)?
4. Accept the MLE and document the limitation?

**Q5: Mixed-Effects Alternative**

Should we consider a mixed-effects NB-GLMM with random intercepts per track? Would this:
1. Improve the rate match?
2. Account for individual larva variability?
3. Be necessary for a rigorous analysis?

**Q6: What Validation Metrics Are Standard?**

Are we missing any standard validation metrics for LNP/point-process models? Currently we compute:
- PSTH correlation (Pearson)
- W-ISE (Weighted Integrated Squared Error)
- Bootstrap threshold from empirical self-consistency
- IEI (Inter-Event Interval) K-S test

Should we also compute:
- Time-rescaling test (for point processes)?
- Deviance residuals?
- Quantile-quantile plots?

### Desired Output

1. **Root cause analysis** of the rate mismatch
2. **Clear recommendation** on whether to fix it or document it
3. **Specific implementation guidance** if a fix is recommended
4. **Literature-grounded assessment** of what is standard practice
5. **Any additional validation metrics** we should compute

---

## 6. Summary of Unknowns

| Unknown | Impact | Priority | Status |
|---------|--------|----------|--------|
| Why empirical rate was 0.71 vs 1.28 in fitting | High | 1 | **SOLVED** (track counting bug) |
| Why MLE intercept gives 1.77x rate | High | 2 | OPEN |
| Whether GLM intercept is per-frame or per-second | High | 3 | OPEN |
| Whether 1.77x rate mismatch invalidates the model | Medium | 4 | OPEN |
| Whether mixed-effects model is needed | Low | 5 | OPEN |

---

## 7. Files for Reference

- `scripts/fit_extended_biphasic_model.py` - Model fitting with triphasic kernel
- `scripts/simulate_extended_biphasic.py` - Simulation and validation
- `data/model/extended_biphasic_model_results.json` - Fitted coefficients
- `data/model/validation_results.json` - Validation metrics
- `docs/MODEL_SUMMARY.md` - Model documentation
- `docs/PROJECT_LIMITATIONS.md` - Known limitations



