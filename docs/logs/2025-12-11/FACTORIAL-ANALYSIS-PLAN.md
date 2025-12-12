# Factorial Analysis Plan: 2×2 Design Extension

**Date:** 2025-12-11

**Status:** Planning phase, pending research guidance

---

## Executive Summary

The cross-condition analysis (Phase 1) confirms kernel stability across all 4 conditions (min correlation 0.943). This document outlines the plan for Phase 2: pooled factorial analysis to quantify main effects and interaction.

---

## Current State

### What We Have

| Asset | Status | Location |
|-------|--------|----------|
| Paper draft | Complete (single-condition) | `docs/paper/manuscript.qmd` |
| Cross-condition fits | Complete | `data/model/cross_condition_results.json` |
| Kernel stability | Confirmed (r > 0.94) | `docs/logs/2025-12-11/CROSS-CONDITION-COMPARISON.md` |
| Scripts | Ready | `scripts/fit_cross_condition.py` |

### Preliminary Effects Identified

| Effect | Direction | Magnitude |
|--------|-----------|-----------|
| LED intensity (50→250 vs 0→250) | Weaker suppression | -31% |
| Temperature (Temp vs Control) | Stronger suppression | +67% |
| Interaction | Modest | 16% |

---

## Proposed Factorial Model

### Model Structure

The pooled NB-GLMM with factorial effects:

```
log λ(t) = β₀ + β_I·I + β_T·T + β_{I×T}·(I·T) 
         + α·K_on(t) + α_I·I·K_on(t) + α_T·T·K_on(t)
         + γ·K_off(t) + u_track
```

Where:
- I = 0 for 0→250 PWM, 1 for 50→250 PWM (intensity factor)
- T = 0 for Control, 1 for Temp (temperature factor)
- K_on(t) = fixed gamma-difference kernel (shared τ₁, τ₂)
- α, α_I, α_T = kernel amplitude and modulation by condition
- u_track ~ N(0, σ²) = track random effect

### Parameters to Estimate

| Parameter | Interpretation |
|-----------|----------------|
| β₀ | Baseline log-hazard (reference condition) |
| β_I | Main effect of intensity on baseline |
| β_T | Main effect of temperature on baseline |
| β_{I×T} | Interaction on baseline |
| α | Reference kernel amplitude |
| α_I | Intensity modulation of kernel |
| α_T | Temperature modulation of kernel |
| γ | LED-OFF rebound amplitude |
| σ | Track random effect SD |

### Hypotheses to Test

1. **H1 (Intensity)**: β_I ≠ 0 or α_I ≠ 0
   - Does 50→250 PWM reduce suppression?
   
2. **H2 (Temperature)**: β_T ≠ 0 or α_T ≠ 0
   - Does temperature modulation enhance suppression?
   
3. **H3 (Interaction)**: β_{I×T} ≠ 0
   - Does intensity effect differ by temperature condition?

---

## Implementation Steps

### Phase 2A: Pooled Model Fitting (~1 day)

1. **Combine all data** (12 experiments, 270 tracks, 10,759 events)
2. **Create factorial design matrix** with I, T, I×T indicators
3. **Fix kernel shape** (τ₁ = 0.29s, τ₂ = 3.81s from reference)
4. **Fit NB-GLMM** with factorial fixed effects + track random effects
5. **Extract coefficients and CIs** for all parameters

### Phase 2B: Hypothesis Testing (~0.5 day)

1. **Wald tests** for each coefficient (β_I, β_T, β_{I×T}, α_I, α_T)
2. **Likelihood ratio tests** comparing nested models
3. **Effect size estimation** with 95% CIs
4. **Multiple comparison correction** (Bonferroni or FDR)

### Phase 2C: Validation (~0.5 day)

1. **Per-condition predictions** from pooled model
2. **Compare to separate fits** (are predictions improved?)
3. **Cross-validation** (leave-one-experiment-out)
4. **Residual diagnostics** (time-rescaling per condition)

### Phase 2D: Paper Update (~1 day)

1. **Add Results section** on factorial analysis
2. **Update Discussion** with biological interpretation
3. **Add figure** showing condition effects
4. **Update Abstract** to mention 2×2 design

---

## Gaps and Uncertainties

### Statistical Gaps

1. **How to model kernel amplitude modulation?**
   - Option A: Multiplicative (α_I multiplies entire kernel)
   - Option B: Additive (α_I added to kernel coefficients)
   - Option C: Per-component (separate α for fast and slow)
   - **Uncertainty:** Which is most interpretable and identifiable?

2. **Should kernel shape vary by condition?**
   - Current plan: fix τ₁, τ₂ globally
   - Alternative: allow condition-specific τ values
   - Cross-condition correlations (0.94-0.98) suggest shape is stable
   - **Uncertainty:** Is 6% shape variation meaningful?

3. **How to handle unbalanced design?**
   - 0→250|Control: 2 experiments (reference)
   - Other cells: 2-4 experiments each
   - Mixed-effects handles this, but power may differ
   - **Uncertainty:** Should we weight by N or use REML?

### Biological Gaps

4. **What is the temperature condition really doing?**
   - LED2 cycles 5-15 PWM (vs constant 7 PWM in Control)
   - "T_Bl_Sq" = Temperature Block Square wave?
   - Is this thermal stimulation or just light modulation?
   - **Uncertainty:** Biological mechanism unclear

5. **Why does 50→250 show weaker suppression?**
   - Smaller intensity delta (200 vs 250 PWM)?
   - Partial adaptation to 50 PWM baseline?
   - Different ChR2 activation dynamics?
   - **Uncertainty:** Cannot distinguish without more conditions

6. **Is the interaction biologically meaningful?**
   - 16% difference in temperature effect by intensity
   - Could be noise given small N
   - **Uncertainty:** Power to detect interaction?

### Implementation Gaps

7. **Track random effects in factorial model**
   - Current: random intercept per track
   - Should we allow random slopes?
   - Tracks are nested within experiments within conditions
   - **Uncertainty:** Correct random effects structure?

8. **Model selection criteria**
   - AIC/BIC for nested models?
   - Cross-validation?
   - Which metric is appropriate for hazard models?
   - **Uncertainty:** How to compare factorial vs separate fits?

---

## Questions for Research Agent

### Priority 1: Statistical Approach

1. What is the best way to model condition-dependent kernel amplitude in an NB-GLMM?

2. Should we use multiplicative or additive modulation of the kernel?

3. For a 2×2 factorial with unbalanced cells, what random effects structure is appropriate?

### Priority 2: Biological Interpretation

4. What does "temperature block" typically mean in Drosophila optogenetics experiments?

5. Is it common for LED intensity to have sub-linear effects on ChR2 activation?

6. Are there published examples of factorial designs in larval navigation studies?

### Priority 3: Literature References

7. What papers have used hazard models with factorial covariates for behavior?

8. Are there examples of gamma-difference kernels in sensory neuroscience?

9. What are standard effect sizes for optogenetic suppression in larvae?

---

## Risk Assessment

| Risk | Likelihood | Impact | Mitigation |
|------|------------|--------|------------|
| Interaction not significant | Medium | Low | Report as null result; focus on main effects |
| Model doesn't converge | Low | Medium | Simplify random effects; use separate fits |
| Kernel shape unstable | Low | High | Already checked: r > 0.94 across conditions |
| Results contradict single-condition | Low | Medium | Report discrepancy transparently |
| Biological mechanism unclear | High | Medium | Frame as empirical finding; suggest future work |

---

## Timeline

| Phase | Duration | Dependencies |
|-------|----------|--------------|
| 2A: Pooled fitting | 1 day | Research guidance on model structure |
| 2B: Hypothesis testing | 0.5 day | Phase 2A complete |
| 2C: Validation | 0.5 day | Phase 2A complete |
| 2D: Paper update | 1 day | Phases 2A-2C complete |
| **Total** | **3 days** | |

---

## Decision Points

1. **After Phase 2A:** If model doesn't converge or effects are very small, consider keeping single-condition paper

2. **After Phase 2B:** If interaction is not significant, simplify to main-effects-only model

3. **After Phase 2C:** If validation fails (rate ratio outside 0.8-1.25 for any condition), investigate per-condition calibration

---

## Next Steps

1. **Obtain research guidance** on statistical approach and literature
2. **Implement Phase 2A** (pooled factorial model)
3. **Update paper** based on results

