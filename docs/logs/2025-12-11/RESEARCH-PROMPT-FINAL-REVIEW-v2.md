# Research Prompt: Final Review of INDYsim Factorial Analysis (Corrected)

## Context

I am completing a computational neuroscience project modeling optogenetically-driven reorientation behavior in *Drosophila* larvae. I need guidance on whether the approach taken is optimal, what gaps remain, and how to interpret the results.

**Important correction**: The experimental condition labeled "Temp" in our analysis refers to a **Timed/cycling background light pattern** (LED2 cycling 5-15 PWM), NOT a temperature manipulation. The "Control" condition has a **Constant background** (LED2 fixed at 7 PWM). This is a purely visual/optogenetic manipulation.

---

## Work Completed

### Phase 1: Single-Condition Hazard Model

**Data**: 2 experiments, 55 tracks, 1,407 events (0→250 PWM | Constant background)

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

| Condition | LED1 Intensity | LED2 Background | Events |
|-----------|----------------|-----------------|--------|
| 0→250 \| Constant | 0→250 PWM | Fixed 7 PWM | 1,407 |
| 0→250 \| Cycling | 0→250 PWM | Cycling 5-15 PWM | 2,410 |
| 50→250 \| Constant | 50→250 PWM | Fixed 7 PWM | 2,440 |
| 50→250 \| Cycling | 50→250 PWM | Cycling 5-15 PWM | 1,031 |

**Factorial Design**:
- **Factor I (Intensity)**: LED1 step size (0→250 vs 50→250 PWM)
- **Factor C (Cycling)**: LED2 background pattern (Constant 7 PWM vs Cycling 5-15 PWM)

**Model**: Extended hazard with factorial modulation

```
log λ(t) = β₀ + β_I·I + β_C·C + β_{IC}·(I×C) 
         + (α + α_I·I + α_C·C)·K_on(t) 
         + γ·K_off(t)
```

Where I = indicator for 50→250, C = indicator for Cycling background.

**Key assumption**: Kernel SHAPE (timescales τ₁, τ₂) is fixed across conditions; only AMPLITUDE is modulated.

### Factorial Results (All Significant at p < 0.05)

| Effect | Estimate | 95% CI | Interpretation |
|--------|----------|--------|----------------|
| β_I (Intensity baseline) | -0.199 | [-0.266, -0.132] | 18% lower baseline hazard |
| β_C (Cycling baseline) | -0.108 | [-0.174, -0.042] | 10% lower baseline hazard |
| β_{IC} (Interaction) | -0.119 | [-0.218, -0.019] | Synergistic reduction |
| α (Kernel amplitude) | 1.005 | [0.899, 1.110] | Reference suppression |
| α_I (Intensity mod.) | -0.665 | [-0.773, -0.557] | **66% weaker suppression** |
| α_C (Cycling mod.) | +0.152 | [0.050, 0.254] | **15% stronger suppression** |
| γ (Rebound) | 1.669 | [0.470, 2.869] | Post-offset enhancement |

**Condition-specific amplitudes**:
- 0→250 | Constant: 1.00 (reference)
- 0→250 | Cycling: 1.16 (strongest)
- 50→250 | Constant: 0.34 (weakest)
- 50→250 | Cycling: 0.49
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

**Cycling background effect (15% stronger for Cycling)**: The cycling background (LED2 oscillating 5-15 PWM) produces slightly stronger suppression than constant background (7 PWM).

**Questions**:
- Is the intensity interpretation (partial adaptation) correct?
- Why would a cycling background increase suppression amplitude? Possible explanations:
  - Prevents full adaptation to background (keeps pathway more responsive)?
  - Temporal contrast from cycling primes the system for detecting changes?
  - Mean intensity is similar (10 PWM cycling vs 7 PWM constant) but temporal structure differs?
- The interaction (β_{IC}) is significant but small. What does a synergistic baseline reduction mean biologically?
- Why do baseline effects (β_I, β_C) and amplitude effects (α_I, α_C) have opposite signs for the cycling condition? Cycling reduces baseline but increases suppression.

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
- Should we have allowed condition-specific rebound (γ_I, γ_C)?
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

**Question**: Does the factorial extension add scientific value, or is it scope creep? The 66% intensity effect is substantial, but the 15% cycling effect is modest.

### 2. Should We Have Used Different Conditions?

The 2×2 design (Intensity × Background Pattern) was determined by what experiments existed, not by experimental design.

**Questions**:
- Is this a meaningful factorial structure?
- The "Cycling" condition oscillates LED2 between 5-15 PWM. Is this a sensible manipulation to test?
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
- Is there value in even simpler models (e.g., just intensity effect, no cycling)?

---

## Requested Outputs

1. **Assessment**: Is the factorial approach sound, or should we have done something different?

2. **Interpretation guidance**: 
   - Is "partial adaptation" the right framing for the intensity effect?
   - What is the best interpretation for the cycling background effect (15% stronger suppression)?

3. **Gap prioritization**: Which gaps are critical to address before publication vs acceptable limitations?

4. **Statistical recommendations**: 
   - Is fixed-effects GLM acceptable given the hierarchical data structure?
   - How should we report the 58% CV pass rate?

5. **Literature pointers**: Are there similar factorial analyses of optogenetic behavioral data we should cite or compare to?

6. **Next steps**: Should we:
   - (A) Proceed with current results and write up
   - (B) Implement GLMM with random effects (requires different Python environment)
   - (C) Simplify to just intensity effect (drop cycling background)
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
| Modest effect | α_C = +0.152 (15% increase) |

---

## Correction Note

Previous versions of this analysis incorrectly referred to the "Cycling" condition as "Temperature" or "Temp". The experimental manipulation is purely optogenetic:
- **Constant (C)**: LED2 held at 7 PWM throughout
- **Cycling (T in filenames)**: LED2 oscillates between 5-15 PWM in a square wave pattern

There is no temperature manipulation in these experiments. The "T" in filenames stands for "Timed" (cycling pattern), not "Temperature".

---

*This prompt requests a comprehensive review of the analysis approach, interpretation, and publication readiness.*
