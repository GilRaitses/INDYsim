# Comprehensive Research Prompt: Fixing PSTH Correlation in LNP Model for Larval Reorientation

**Date**: 2025-12-11  
**Context**: LNP (Linear-Nonlinear-Poisson) model for Drosophila larval reorientation during optogenetic stimulation  
**Primary Issue**: PSTH correlation is 0.50 when target is 0.80  

---

## 1. Project Background

I am developing **INDYsim**, a simulation framework for predicting larval navigation behavior during optogenetic stimulation. The model is:

- **Architecture**: Linear-Nonlinear-Poisson (LNP) cascade
- **Linear filter**: Raised-cosine temporal basis functions (5 bases, 0-6s window)
- **Nonlinearity**: Exponential (log-link)
- **Point process**: Negative Binomial (handles overdispersion)
- **Driver line**: GMR61 > CsChrimson (novel, uncharacterized in larval literature)

The model was fit on ~2.3M observations with ~25k reorientation events from 4 experiments.

---

## 2. Current Model Specification

### Fitted Coefficients (Optimal Model)

| Coefficient | Estimate | SE | p-value |
|------------|----------|-----|---------|
| Intercept | -4.499 | 0.592 | <0.001 |
| LED1_scaled | +0.524 | 0.252 | 0.038 |
| kernel_1 (0-1.2s) | -6.466 | 1.127 | <0.001 |
| kernel_2 (1.2-2.4s) | -9.840 | 0.912 | <0.001 |
| kernel_3 (2.4-3.6s) | -11.029 | 3.210 | <0.001 |
| kernel_4 (3.6-4.8s) | -9.302 | 1.159 | <0.001 |
| kernel_5 (4.8-6.0s) | -2.578 | 0.535 | <0.001 |

### Key Observations

1. **LED main effect is positive** (+0.52): LED should increase turn rate
2. **All kernel coefficients are strongly negative** (-2.6 to -11.0): Immediate and sustained suppression
3. **Net effect during LED-ON**: Kernel suppression overwhelms the positive LED effect

---

## 3. The Problem: PSTH Mismatch

### Validation Results

| Metric | Value | Target | Status |
|--------|-------|--------|--------|
| Turn rate match | 0.54 vs 1.19 | Within CI | PASS |
| PSTH W-ISE | 7.90 | <2.8M | PASS |
| **PSTH correlation** | **0.50** | **≥0.80** | **FAIL** |

### Detailed PSTH Comparison

| Time Window | Empirical Rate | Simulated Rate | Correlation |
|-------------|----------------|----------------|-------------|
| Pre-onset (-5 to 0s) | 1.25 | 1.12 | 0.65 |
| **Early (0-3s)** | **1.08** | **0.40** | 0.97 |
| Late (3-10s) | 0.90 | 0.90 | 0.79 |
| Post-LED (10-15s) | 1.39 | 1.03 | 0.32 |

### Identified Mismatches

1. **Missing early peak**: 
   - Empirical peak at **t=0.2s** (immediate response)
   - Simulated peak at **t=8.2s** (delayed)
   
2. **Suppressed early response**:
   - 0-3s window: simulated is **63% lower** than empirical (0.40 vs 1.08)
   
3. **No post-LED recovery**:
   - After LED offset (t=10s), empirical shows rebound to 1.39
   - Simulated stays low at 1.03

---

## 4. Root Cause Analysis

### The Fundamental Issue

The fitted kernel is **monotonically suppressive** throughout the 0-6s window. But the empirical PSTH shows a **biphasic response**:

```
Empirical pattern:
t=0s: Brief INCREASE in turn rate (peak at 0.2s)
t=1-10s: SUPPRESSION below baseline
t=10s+: RECOVERY/rebound after LED off

Simulated pattern:
t=0s: Immediate SUPPRESSION (no early peak)
t=0-10s: Sustained suppression
t=10s+: No recovery
```

### Why the GLM Produces This

The GLM minimizes deviance globally, which means:
- The strong suppression during t=1-10s (majority of LED-ON period) dominates
- The brief early peak at t=0-0.5s gets averaged out
- The LED main effect (+0.52) tries to compensate but is insufficient

---

## 5. Proposed Fixes

### Fix 1: Reverse Correlation Kernel (Diagnostic)

**Rationale**: Before modifying the model, compute the empirical kernel via reverse correlation to see the "true" shape.

**Method** (per Hernandez-Nunez et al. 2015):
```
K(τ) = (1/N) Σᵢ LED(t_event_i - τ)
```
Where τ is the lag before each event.

**Expected outcome**: The reverse correlation kernel should show:
- Positive lobe near τ=0 (LED onset triggers events)
- Negative lobe at τ=1-4s (refractory/suppression)

**Uncertainty**: Reverse correlation assumes linear relationship. If the response is highly nonlinear (e.g., threshold-like), this may not capture the true dynamics.

### Fix 2: Biphasic Kernel Structure

**Rationale**: Split the temporal kernel into two components with different constraints.

**Implementation**:
```python
# Early kernel (0-2s): Constrained to allow positive values
kernel_early = create_kernel(n_bases=2, window=(0, 2), constraint='non_negative')

# Late kernel (2-6s): Unconstrained (can be negative)
kernel_late = create_kernel(n_bases=3, window=(2, 6), constraint=None)
```

**Expected outcome**: Early kernel can capture the positive peak; late kernel captures suppression.

**Uncertainty**: 
- Where to split? 2s is a guess based on typical larval response latencies
- May need cross-validation to find optimal split point

### Fix 3: LED-Off Rebound Term

**Rationale**: The post-LED recovery (t=10-15s) is not captured by the current model.

**Implementation**:
```python
# Add covariate for time since LED turned off
data['time_since_led_off'] = compute_time_since_offset(data)
data['led_off_recovery'] = np.exp(-data['time_since_led_off'] / tau) * (data['time_since_led_off'] > 0)
```

**Expected outcome**: Positive coefficient for `led_off_recovery` captures the rebound.

**Uncertainty**:
- What is the appropriate `tau` (recovery time constant)?
- Is the rebound exponential or something else (e.g., step function)?

### Fix 4: Constrained Kernel Optimization

**Rationale**: Force the first kernel basis to be non-negative.

**Implementation**:
```python
from scipy.optimize import minimize

# Bounds: kernel_1 >= 0, others unconstrained
bounds = [(-inf, inf)] * n_params
bounds[kernel_1_idx] = (0, inf)
```

**Expected outcome**: Prevents the model from using kernel_1 to suppress events at t=0.

**Uncertainty**:
- May hurt overall fit (deviance) if the constraint is too strong
- Identifiability: if kernel_1 is forced positive, other coefficients may compensate

---

## 6. Specific Questions for Research Agent

### Q1: Reverse Correlation vs GLM Kernel

In the LNP literature (Hernandez-Nunez 2015, Gepner 2015), kernels are often computed via **reverse correlation** rather than GLM fitting. 

- **Is GLM-based kernel estimation appropriate for our sparse event data (1.1% event rate)?**
- **Should we use reverse correlation as the primary method and GLM only for inference?**
- **What are the tradeoffs in terms of bias, variance, and interpretability?**

### Q2: Biphasic Response Structure

The empirical PSTH shows a clear biphasic pattern (increase → suppression → recovery). 

- **Is it standard to model this with separate kernel components for early vs late responses?**
- **Alternatively, should we use a parametric kernel form (e.g., difference of exponentials) that naturally captures biphasic responses?**
- **How do I determine the optimal split point (e.g., 2s) between phases?**

### Q3: Post-Stimulus Recovery

The post-LED recovery (t=10-15s after LED offset) is not captured.

- **Is this "rebound" a known phenomenon in optogenetic larval studies?**
- **Should I model it as:**
  - A separate LED-off kernel?
  - An asymmetric adaptation term?
  - A hidden state (e.g., responsive vs adapted)?
- **What is a reasonable functional form for recovery dynamics?**

### Q4: Kernel Constraints and Identifiability

If I constrain kernel_1 to be non-negative:

- **Does this introduce identifiability issues with the LED main effect?**
- **Should I remove the LED main effect and let the kernel fully capture the stimulus response?**
- **Are there regularization techniques (e.g., ridge, smoothness penalties) that would be more appropriate?**

### Q5: Model Selection

With multiple potential fixes, how should I select the best model?

- **Is AIC/BIC appropriate for comparing models with different kernel structures?**
- **Should I use cross-validation on held-out LED cycles (not just tracks)?**
- **What PSTH correlation threshold is realistic for behavioral LNP models?** (Is 0.80 achievable, or is 0.60-0.70 typical?)

---

## 7. Additional Context

### Experimental Design

- **Stimulus**: LED1 ramps from 0 to 250 PWM over ~5s, then plateau for 5s, then off for 20s
- **Cycle**: 10s ON / 20s OFF (30s period)
- **Duration**: 20 minutes per experiment
- **Detection**: Reorientation onsets detected from trajectory curvature

### Data Characteristics

| Metric | Value |
|--------|-------|
| Total observations | 2.27M |
| Total events | 25,238 |
| Event rate | 1.1% |
| Tracks | 177 |
| Experiments | 4 |

### Current Model Performance

| Metric | Value |
|--------|-------|
| Deviance | 216,667 |
| AIC | 269,600 |
| Pseudo-R² | 0.0285 |
| Dispersion ratio | 5.44 |

---

## 8. Suspected Gaps in My Understanding

1. **Kernel identifiability**: I'm not sure if the kernel and LED main effect are identifiable separately. The positive LED effect (+0.52) combined with negative kernel (-6 to -11) seems contradictory.

2. **Raised-cosine basis appropriateness**: The raised-cosine bases are smooth and overlapping. Maybe a more flexible basis (splines, discrete bins) would capture the sharp early peak better.

3. **Event detection artifacts**: The empirical "peak at 0.2s" could be a detection artifact (e.g., LED ramp causes apparent motion that triggers false positives). I have not verified this.

4. **Heterogeneity**: Some larvae may be "responders" (strong early peak) and others "non-responders" (suppression only). The pooled model averages these.

5. **Temporal resolution**: Events are detected at ~50ms resolution. The 0.2s peak is only 4 time bins wide, which may not be well-resolved.

---

## 9. What I Need From You

1. **Validate or refute my proposed fixes**: Are they grounded in the LNP literature?

2. **Recommend an implementation order**: Which fix should I try first?

3. **Provide specific guidance on**:
   - Reverse correlation implementation for sparse events
   - Biphasic kernel parameterization
   - Recovery term functional form
   - Regularization for constrained optimization

4. **Set realistic expectations**: What PSTH correlation is achievable for this type of data?

5. **Identify any gaps or errors in my reasoning** that I haven't noticed.

---

## 10. Summary Table of Issues and Proposed Fixes

| Issue | Evidence | Proposed Fix | Uncertainty |
|-------|----------|--------------|-------------|
| Missing early peak | Peak at 0.2s (emp) vs 8.2s (sim) | Reverse correlation; biphasic kernel | Is 0.2s peak real or artifact? |
| Suppressed 0-3s response | 0.40 (sim) vs 1.08 (emp) | Constrain kernel_1 ≥ 0 | Identifiability with LED effect |
| No post-LED recovery | 1.03 (sim) vs 1.39 (emp) | LED-off rebound term | Functional form unknown |
| All kernels negative | -6 to -11 | Biphasic structure | Optimal split point |
| Low overall correlation | 0.50 vs 0.80 target | All fixes combined | Is 0.80 realistic? |

---

## References

1. Hernandez-Nunez et al. 2015, Nat Commun - Reverse correlation for larval navigation
2. Gepner et al. 2015, eLife - LNP for larval phototaxis
3. Gepner et al. 2018, eLife - Variance adaptation in navigation
4. Pillow et al. 2008, Nature - Raised-cosine basis for neural encoding
5. Truccolo et al. 2005, J Neurophysiol - Point process GLM framework



