# Cross-Condition Kernel Comparison

**Date:** 2025-12-11

**Purpose:** Assess kernel stability across the 2×2 factorial design to determine whether to proceed with pooled factorial analysis.

---

## Executive Summary

**Recommendation: PROCEED WITH FACTORIAL ANALYSIS**

The hazard kernel is stable across all 4 conditions in the 2×2 factorial design:
- Minimum correlation with reference: **0.943**
- Mean correlation: **0.971**
- All correlations exceed the 0.80 threshold for stability

The kernel shape (timescales, biphasic structure) is conserved; only the amplitude varies by condition. This supports a pooled factorial model with shared kernel shape and condition-specific amplitude parameters.

---

## Experimental Design

### 2×2 Factorial Structure

|  | **Control (7 PWM)** | **Temp (5-15 PWM)** |
|---|---|---|
| **0→250 PWM** | Reference condition | 1.74× stronger suppression |
| **50→250 PWM** | 0.73× weaker suppression | 1.15× suppression |

### Data Summary

| Condition | Experiments | Tracks | Events | Frames |
|-----------|-------------|--------|--------|--------|
| 0→250 \| Control | 2 (excl. anomalous) | 70 | 3,847 | 3.9M |
| 0→250 \| Temp | 4 | 65 | 3,441 | 3.9M |
| 50→250 \| Control | 4 | 70 | 2,440 | 2.6M |
| 50→250 \| Temp | 2 | 65 | 1,031 | 1.4M |
| **Total** | **12** | **270** | **10,759** | **11.7M** |

Note: Two experiments from 0→250 | Control were excluded due to anomalously high event counts (10-20× others).

---

## Kernel Comparison Results

### Correlation with Reference (0→250 | Control)

| Condition | Correlation | RMSE | Suppression Ratio | Peak Time |
|-----------|-------------|------|-------------------|-----------|
| 0→250 \| Control | 1.000 | 0.000 | 1.00× | 3.8s |
| 0→250 \| Temp | 0.975 | 0.598 | 1.74× | 3.3s |
| 50→250 \| Control | 0.966 | 0.258 | 0.73× | 4.0s |
| 50→250 \| Temp | 0.943 | 0.202 | 1.15× | 3.1s |

### Key Observations

1. **Kernel shape is conserved** (all correlations > 0.94)
   - The biphasic structure (fast rise, slow suppression) appears in all conditions
   - Timescales are similar (peak times range 3.1-4.0s, close to reference 3.8s)

2. **Amplitude varies by condition**
   - Temperature conditions show stronger suppression (1.74× and 1.15×)
   - 50→250 PWM (partial intensity) shows weaker suppression (0.73×)
   - This pattern is consistent with intensity-dependent modulation

3. **Baseline rates are similar**
   - Intercept means range from -6.93 to -7.25 (log-hazard per frame)
   - Track-level SD is consistent across conditions (~0.31-0.40)

---

## Intercept Comparison

| Condition | Global Intercept | Mean Track Intercept | Track SD |
|-----------|------------------|---------------------|----------|
| 0→250 \| Control | -6.77 | -6.94 | 0.37 |
| 0→250 \| Temp | -6.82 | -6.93 | 0.31 |
| 50→250 \| Control | -6.84 | -7.01 | 0.40 |
| 50→250 \| Temp | -7.06 | -7.25 | 0.40 |

**Interpretation:**
- 50→250 conditions have slightly lower baseline rates (more negative intercepts)
- This may reflect partial pre-stimulation at 50 PWM baseline

---

## Factorial Effects (Preliminary)

### Main Effect of LED Intensity (0→250 vs 50→250)

| Metric | 0→250 | 50→250 | Effect |
|--------|-------|--------|--------|
| Mean suppression | 1.37× | 0.94× | 50→250 has **31% weaker** suppression |
| Mean intercept | -6.93 | -7.13 | 50→250 has **lower baseline rate** |

**Interpretation:** Starting at 50 PWM reduces the suppression magnitude, possibly due to:
- Smaller intensity delta (200 vs 250 PWM)
- Partial adaptation to baseline stimulation

### Main Effect of Background (Control vs Temp)

| Metric | Control | Temp | Effect |
|--------|---------|------|--------|
| Mean suppression | 0.86× | 1.44× | Temp has **67% stronger** suppression |
| Mean intercept | -6.97 | -7.09 | Temp has **lower baseline rate** |

**Interpretation:** Temperature modulation (5-15 PWM cycling) enhances optogenetic suppression. This could indicate:
- Synergistic effect with thermal stimulation
- Altered arousal state affecting baseline behavior

### Interaction

The pattern suggests a possible interaction:
- At 0→250: Temp increases suppression by 74%
- At 50→250: Temp increases suppression by 58%

The interaction effect (16% difference) is modest but may be statistically significant in a pooled model.

---

## Stability Assessment

### Criteria for Proceeding with Factorial

| Criterion | Threshold | Observed | Status |
|-----------|-----------|----------|--------|
| Min correlation | > 0.80 | 0.943 | PASS |
| Mean correlation | > 0.90 | 0.971 | PASS |
| Peak time stability | within 1.0s | 0.9s range | PASS |
| Intercept consistency | SD < 0.5 | 0.13 | PASS |

**All criteria passed.**

---

## Recommended Next Steps

### Phase 2: Pooled Factorial NB-GLMM

Based on kernel stability, proceed with a pooled model:

```
log λ(t) = β₀ + β_I·I + β_T·T + β_{I×T}·(I·T) 
         + α_I·I·K_on(t) + α_T·T·K_on(t) 
         + γ·K_off(t) + u_track
```

Where:
- I = 0 for 0→250, 1 for 50→250
- T = 0 for Control, 1 for Temp
- K_on(t) is the shared gamma-difference kernel (fixed τ₁, τ₂)
- α_I, α_T allow condition-specific kernel amplitude

### Implementation Steps

1. **Pool all data** (12 experiments, 270 tracks, 10,759 events)
2. **Fix kernel shape** using reference parameters (τ₁=0.29s, τ₂=3.81s)
3. **Fit factorial effects** (β_I, β_T, β_{I×T}, α_I, α_T)
4. **Test main effects and interaction** via Wald tests
5. **Validate** by comparing predicted vs observed suppression per condition

### Paper Update

With factorial analysis:
- **Methods**: Add section on 2×2 design and pooled model
- **Results**: Add section on intensity and temperature effects
- **Discussion**: Interpret biological mechanisms

---

## Files Generated

| File | Description |
|------|-------------|
| `data/model/cross_condition_results.json` | Full fitting results for all conditions |
| `scripts/fit_cross_condition.py` | Cross-condition fitting script |
| `docs/logs/2025-12-11/CROSS-CONDITION-COMPARISON.md` | This report |

---

## Conclusion

The hazard kernel is **stable across the 2×2 factorial design** (min correlation 0.943). The kernel shape is conserved while amplitude varies by condition. This supports proceeding with a **pooled factorial analysis** that uses a shared kernel with condition-specific modulation.

The preliminary analysis suggests:
- **LED intensity effect**: 50→250 has ~31% weaker suppression than 0→250
- **Temperature effect**: Temp conditions have ~67% stronger suppression than Control
- **Interaction**: Modest (16% difference in temperature effect by intensity)

These effects are biologically plausible and enhance the scientific contribution of the paper beyond a single-condition methods demonstration.

