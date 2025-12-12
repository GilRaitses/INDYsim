# Research Prompt: Final Review of INDYsim Factorial Analysis

## Context

I am completing a computational neuroscience project modeling optogenetically-driven reorientation behavior in *Drosophila* larvae. I need guidance on whether the approach taken is optimal, what gaps remain, and how to interpret the results.

---

## Work Completed

### Phase 1: Single-Condition Hazard Model

**Data**: 2 experiments, 55 tracks, 1,407 events (0→250 PWM | Control condition)

**Model**: Negative-binomial GLM with gamma-difference kernel

```
log λ(t) = β₀ + K_on(t) + K_off(t)

K_on(t) = A·Γ(t; α₁, β₁) - B·Γ(t; α₂, β₂)  [gamma-difference]
K_off(t) = D·exp(-t/τ_off)                   [exponential rebound]
```

**Fitted parameters**:
- Fast timescale: τ₁ = 0.29 s (α₁ = 2.22)
- Slow timescale: τ₂ = 3.81 s (α₂ = 4.38)
- 6 total kernel parameters

**Validation**:
- Rate ratio: 0.97 (target: 0.8-1.25) ✓
- Kernel R²: 0.968 ✓
- PSTH correlation: 0.84 ✓

### Phase 2: Factorial Extension

**Data**: 12 experiments, 623 tracks, 7,288 events across 2×2 design

| Condition | LED Intensity | Background | Events |
|-----------|---------------|------------|--------|
| 0→250 \| Control | 0→250 PWM | Constant 7 PWM | 1,407 |
| 0→250 \| Temp | 0→250 PWM | Cycling 5-15 PWM | 2,410 |
| 50→250 \| Control | 50→250 PWM | Constant 7 PWM | 2,440 |
| 50→250 \| Temp | 50→250 PWM | Cycling 5-15 PWM | 1,031 |

**Model**: Extended hazard with factorial modulation

```
log λ(t) = β₀ + β_I·I + β_T·T + β_{IT}·(I×T) 
         + (α + α_I·I + α_T·T)·K_on(t) 
         + γ·K_off(t)
```

Where I = indicator for 50→250, T = indicator for Temp background.

**Key assumption**: Kernel SHAPE (timescales τ₁, τ₂) is fixed across conditions; only AMPLITUDE is modulated.

### Factorial Results (All Significant at p < 0.05)

| Effect | Estimate | 95% CI | Interpretation |
|--------|----------|--------|----------------|
| β_I (Intensity baseline) | -0.199 | [-0.266, -0.132] | 18% lower baseline hazard |
| β_T (Temp baseline) | -0.108 | [-0.174, -0.042] | 10% lower baseline hazard |
| β_{IT} (Interaction) | -0.119 | [-0.218, -0.019] | Synergistic reduction |
| α (Kernel amplitude) | 1.005 | [0.899, 1.110] | Reference suppression |
| α_I (Intensity mod.) | -0.665 | [-0.773, -0.557] | **66% weaker suppression** |
| α_T (Temp mod.) | +0.152 | [0.050, 0.254] | **15% stronger suppression** |
| γ (Rebound) | 1.669 | [0.470, 2.869] | Post-offset enhancement |

**Condition-specific amplitudes**:
- 0→250 | Control: 1.00 (reference)
- 0→250 | Temp: 1.16 (strongest)
- 50→250 | Control: 0.34 (weakest)
- 50→250 | Temp: 0.49
- Range: 3.4-fold

### Cross-Validation

- Mean rate ratio: 1.03 ± 0.31
- Pass rate: 7/12 (58%)
- High inter-experiment variance suggests session-level effects not captured by fixed-effects model

---

## Uncertainties and Gaps

### 1. Model Structure Decisions

**Fixed kernel shape assumption**: We assumed the gamma-difference timescales (τ₁, τ₂) are identical across all conditions, with only amplitude varying. This was based on preliminary cross-condition fits showing high kernel correlation (r > 0.94).

**Questions**:
- Is this assumption biologically justified? Could the 50→250 condition have genuinely different temporal dynamics (not just weaker amplitude)?
- Should we have fit separate kernels per condition and then tested for shape equivalence statistically?
- Is there a principled way to test "shape invariance" vs "amplitude modulation" hypotheses?

### 2. Factorial Design Interpretation

**Intensity effect (66% weaker for 50→250)**: We interpret this as "partial adaptation" - the sensory pathway has already adapted to 50 PWM baseline, so the step to 250 is less salient.

**Temperature effect (15% stronger for Temp)**: We interpret this as "cross-modal sensitization" - the cycling background somehow increases optogenetic gain.

**Questions**:
- Are these interpretations correct? The temperature manipulation cycles LED2 between 5-15 PWM. Is this truly a "temperature" effect or just a different light pattern?
- The interaction (β_{IT}) is significant but small. What does a synergistic baseline reduction mean biologically?
- Why do baseline effects (β_I, β_T) and amplitude effects (α_I, α_T) have opposite signs? Intensity reduces baseline but also reduces suppression; Temp reduces baseline but increases suppression.

### 3. Random Effects

**Current model**: Fixed-effects NB-GLM (statsmodels). No random effects for track or experiment.

**CV results**: 58% pass rate with σ = 0.31 suggests substantial unexplained between-experiment variance.

**Questions**:
- Should we have used a mixed-effects model (GLMM) with random intercepts for track/experiment?
- Bambi/PyMC failed to install due to Python 3.14 compatibility. Is statsmodels GLM sufficient, or are the results compromised without random effects?
- How should we report results from a fixed-effects model when we know there's hierarchical structure?

### 4. Event Definition

**Issue**: 77% of events have zero measured duration ("onset events"), while 23% have duration > 0.1s ("true turns").

**Current approach**: Fit hazard model on ALL 7,288 events; filter to "true turns" only for trajectory simulation.

**Questions**:
- Is this mixing of event types valid for the hazard model?
- Should we have fit separate models for "onset events" vs "true turns"?
- Could the zero-duration events be detection artifacts that bias the hazard estimates?

### 5. Poisson Assumption Violation

**Time-rescaling test**: Mean rescaled IEI = 0.87 (expected 1.0), indicating ~13% deviation from Poisson assumption.

**Interpretation**: Likely mild refractoriness or short-term dependencies not captured by the model.

**Questions**:
- Is 13% deviation acceptable for publication?
- Should we add a refractory kernel term to the model?
- Does this invalidate the negative-binomial assumption?

### 6. Rebound Kernel

**Current**: Single shared γ coefficient for K_off across all conditions.

**Questions**:
- Should we have allowed condition-specific rebound (γ_I, γ_T)?
- The rebound is significant (p = 0.006) but has wide CI [0.47, 2.87]. Is this estimate reliable?

---

## Alternative Approaches We Did NOT Try

### A. Separate Models Per Condition

Instead of a pooled factorial model with additive modulation, we could have:
1. Fit 4 completely separate hazard models (one per condition)
2. Compared parameters post-hoc

**Trade-off**: More flexibility but less statistical power; can't directly test interactions.

### B. Condition-Specific Kernel Shapes

Instead of fixing τ₁, τ₂ and modulating amplitude, we could have:
1. Allowed all 6 kernel parameters to vary by condition
2. Used hierarchical priors to share information

**Trade-off**: More parameters (24 vs 8), higher risk of overfitting.

### C. Different Kernel Functional Forms

Instead of gamma-difference, we could have tried:
1. Double-exponential
2. Alpha function
3. Non-parametric (raised-cosine basis)

**Trade-off**: Gamma-difference was chosen for interpretability, but may not be optimal for all conditions.

### D. Excluding Anomalous Experiments

Two experiments (202510291652, 202510291713) had 10-20× more events than others. We excluded them.

**Questions**:
- Was this the right decision?
- Could they have provided useful information about high-event-rate regimes?

---

## Questions About Project Direction

### 1. Is the Factorial Analysis the Right Focus?

The original goal was to create an "interpretable hazard kernel" for optogenetic simulation. We achieved this with the single-condition model (R² = 0.968).

**Question**: Does the factorial extension add scientific value, or is it scope creep? The 66% intensity effect is interesting, but the 15% temperature effect is modest.

### 2. Should We Have Used Different Conditions?

The 2×2 design (Intensity × Background) was determined by what experiments existed, not by experimental design.

**Questions**:
- Is this a meaningful factorial structure?
- The "Temp" condition cycles LED2 between 5-15 PWM. Is this a sensible manipulation to test, or a confound?
- Would it have been better to analyze intensity as a continuous variable rather than binary?

### 3. Publication Readiness

**Current state**: 
- Single-condition model: Strong (R² = 0.968, rate ratio = 0.97)
- Factorial extension: Moderate (all effects significant, but CV pass rate only 58%)

**Questions**:
- Is the factorial analysis ready for the main paper, or should it be supplementary?
- What additional validation would strengthen the factorial claims?
- Are there obvious controls or analyses we're missing?

### 4. Model Complexity

We went from 6 parameters (single-condition) to 8 parameters (factorial).

**Questions**:
- Is this the right level of complexity for 7,288 events?
- Should we have used model selection (AIC/BIC) to compare factorial vs simpler models?
- Is there value in even simpler models (e.g., just intensity effect, no temperature)?

---

## Requested Outputs

1. **Assessment**: Is the factorial approach sound, or should we have done something different?

2. **Interpretation guidance**: Are our biological interpretations of the effects reasonable?

3. **Gap prioritization**: Which gaps are critical to address before publication vs acceptable limitations?

4. **Statistical recommendations**: 
   - Is fixed-effects GLM acceptable given the hierarchical data structure?
   - How should we report the 58% CV pass rate?

5. **Literature pointers**: Are there similar factorial analyses of optogenetic behavioral data we should cite or compare to?

6. **Next steps**: Should we:
   - (A) Proceed with current results and write up
   - (B) Implement GLMM with random effects (requires different Python environment)
   - (C) Simplify to just intensity effect (drop temperature)
   - (D) Fit separate models per condition and compare

---

## Summary Statistics for Reference

| Metric | Value |
|--------|-------|
| Total frames | 7,783,717 |
| Total events | 7,288 |
| Experiments | 12 |
| Tracks | 623 |
| Conditions | 4 |
| Model parameters | 8 (fixed effects) |
| CV pass rate | 58% (7/12) |
| Mean rate ratio | 1.03 ± 0.31 |
| Strongest effect | α_I = -0.665 (66% reduction) |
| Weakest effect | α_T = +0.152 (15% increase) |

---

*This prompt requests a comprehensive review of the analysis approach, interpretation, and publication readiness.*

