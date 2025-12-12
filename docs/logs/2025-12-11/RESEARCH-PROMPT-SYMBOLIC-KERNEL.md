# Research Prompt: Symbolic Regression for Interpretable Kernel Discovery

**Date**: 2025-12-11  
**Project**: INDYsim - LNP Model for Larval Optogenetic Response  
**Research Agent**: External (comprehensive analysis requested)

---

## Executive Summary

We have a validated LNP (Linear-Nonlinear-Poisson) model for predicting reorientation events in *Drosophila* larvae during optogenetic stimulation. The model achieves:

- **Rate ratio**: 1.16x (simulated/empirical)
- **Early suppression match**: 2% difference
- **Late suppression match**: 1% difference
- **PSTH correlation**: 0.810

The current kernel is represented as a weighted sum of 12 raised-cosine basis functions. We want to discover an **interpretable closed-form expression** using symbolic regression (PySR) or sparse identification (SINDy).

---

## Current Kernel Representation

### Learned Coefficients (Hybrid Model)

```
Kernel = Σ βⱼ · φⱼ(t)

where φⱼ(t) = 0.5 * (1 + cos(π(t - cⱼ)/wⱼ)) for |t - cⱼ| < wⱼ

Basis centers and coefficients:
  Early (width=0.30s):
    t=0.20s: β₁ = +1.228  (positive bump)
    t=0.63s: β₂ = +0.107  (small positive)
    t=1.07s: β₃ = -1.105  (suppression onset)
    t=1.50s: β₄ = -1.595  (suppression)
    
  Intermediate (width=0.60s):
    t=2.00s: β₅ = -1.137  (suppression)
    t=2.50s: β₆ = -1.184  (suppression)
    
  Late (width=2.49s):
    t=3.00s: β₇ = -1.626  (peak suppression)
    t=4.20s: β₈ = -1.694  (peak suppression)
    t=5.40s: β₉ = -0.086  (recovery)
    t=6.60s: β₁₀ = -0.823 (partial suppression)
    t=7.80s: β₁₁ = +0.596 (recovery)
    t=9.00s: β₁₂ = -0.578 (late fluctuation)
```

### Kernel Shape Description

When evaluated on a dense time grid, the kernel exhibits:

1. **Early positive bump** at t ≈ 0.2-0.5s (peak ~ +1.2)
2. **Rapid descent** crossing zero at t ≈ 0.8-1.0s
3. **Deep suppression** from t ≈ 1.5-5s (trough ~ -2.5)
4. **Gradual recovery** from t ≈ 5-9s (approaching 0)
5. **Oscillatory component** in late phase (possibly noise)

---

## Biological Context

### Experimental Setup
- **Genotype**: GMR61-Gal4 > UAS-CsChrimson
- **Stimulus**: 617nm LED, 0-250 PWM, 10s ON / 20s OFF cycles
- **Larval stage**: 2nd instar
- **Behavior measured**: Reorientation events (turn rate)
- **Duration**: 20 minutes per experiment

### Expected Physiology
- **GMR61**: Unknown driver (not characterized in literature)
- **CsChrimson**: Fast channelrhodopsin (τ_on ≈ 2ms, τ_off ≈ 16ms)
- **Larval circuits**: Multi-synaptic, typical delays 50-200ms per synapse

### Relevant Literature Timescales

| Source | System | τ_fast | τ_slow | Notes |
|--------|--------|--------|--------|-------|
| Gepner et al. 2015 | Phototaxis kernel | 0.3-0.5s | 2-4s | Bilobe filter |
| Klein et al. 2015 | Thermotaxis | 0.4s | 1-2s | Temperature gradient |
| Hernandez-Nunez 2015 | Chemotaxis | 0.2-0.4s | 1-3s | Odor response |
| Schulze et al. 2015 | Decision latency | 0.6-0.8s | - | Turn initiation |

---

## Candidate Functional Forms

### Form 1: Difference of Exponentials (DoE)

```
K(t) = A · exp(-t/τ_fast) - B · exp(-t/τ_slow)
```

**Parameters**: A, B, τ_fast, τ_slow (4 total)

**Properties**:
- Simple, interpretable
- Cannot capture recovery to baseline
- Common in receptor adaptation models

### Form 2: Triple Exponential

```
K(t) = A · exp(-t/τ₁) - B · exp(-t/τ₂) + C · exp(-t/τ₃)
```

**Parameters**: A, B, C, τ₁, τ₂, τ₃ (6 total)

**Properties**:
- Can capture recovery phase
- May overfit with 6 parameters
- Used in multi-timescale adaptation

### Form 3: Alpha Function + Exponential

```
K(t) = A · (t/τ₁) · exp(-t/τ₁) - B · exp(-t/τ₂)
```

**Parameters**: A, B, τ₁, τ₂ (4 total)

**Properties**:
- Smooth rise to peak (not instantaneous)
- Matches neural response shape
- Common in synaptic models

### Form 4: Gamma-Difference

```
K(t) = A · Γ(t; α₁, β₁) - B · Γ(t; α₂, β₂)

where Γ(t; α, β) = (β^α / Γ(α)) · t^(α-1) · exp(-βt)
```

**Parameters**: A, B, α₁, β₁, α₂, β₂ (6 total)

**Properties**:
- Flexible shape
- Mode at t = (α-1)/β
- Used in neural latency models

### Form 5: Piecewise Exponential

```
K(t) = {
  A · exp(-t/τ₁)           if t < t_switch
  -B · exp(-(t-t_switch)/τ₂)  if t ≥ t_switch
}
```

**Parameters**: A, B, τ₁, τ₂, t_switch (5 total)

**Properties**:
- Explicit phase transition
- May have discontinuity
- Matches observed early/late phases

---

## Questions for Research Agent

### Q1: Functional Form Selection

Given the kernel shape (positive bump → deep suppression → recovery), which functional form is most appropriate?

**Constraints**:
- Must be continuous
- Must have ≤6 parameters for interpretability
- Should have biological meaning

**Specific sub-questions**:
- Is difference-of-exponentials sufficient, or do we need a recovery term?
- Should the early bump have a smooth rise (alpha function) or instantaneous (exponential)?
- How do we handle the late-phase oscillations (noise or real)?

### Q2: Timescale Interpretation

The learned kernel has an empirical shape. How do we map discovered τ values to biology?

**What we know**:
- Early bump peaks at ~0.3s
- Suppression trough at ~3-4s
- Recovery begins ~5-6s

**Questions**:
- What is τ_fast likely representing? (sensory transduction? motor preparation?)
- What is τ_slow likely representing? (synaptic depression? network adaptation?)
- Is the recovery phase real or fitting noise?

### Q3: PySR Configuration

We plan to use PySR (Python Symbolic Regression). What configuration would you recommend?

**Our current plan**:
```python
PySRRegressor(
    niterations=100,
    binary_operators=["+", "-", "*", "/"],
    unary_operators=["exp", "neg"],
    maxsize=20,
    populations=20,
    timeout_in_seconds=3600
)
```

**Questions**:
- Should we include `sin`, `cos` for oscillatory components?
- Should we include `sqrt`, `log` for power-law decay?
- What complexity penalty balances accuracy vs interpretability?
- Should we use the derivative dK/dt as additional constraint?

### Q4: SINDy Library Design

For SINDy, we need to specify a library of candidate functions. What should be included?

**Our current plan**:
```python
library = [
    1,                    # constant
    exp(-t/0.3),          # fast decay
    exp(-t/0.8),          # matches IEI
    exp(-t/2.0),          # slow decay
    exp(-t/5.0),          # very slow decay
    t * exp(-t/0.5),      # alpha function
]
```

**Questions**:
- Are these τ values reasonable starting points?
- Should we include polynomial terms (t, t², t³)?
- Should we include interaction terms?
- How do we set the sparsity threshold (LASSO λ)?

### Q5: LED-Off Rebound

Our current model includes a separate LED-off rebound term:

```
rebound(t) = exp(-(t - t_off) / τ_rebound)  if t > t_off
```

With coefficient +1.12 (strong positive effect when LED turns off).

**Questions**:
- Should the rebound be part of the main kernel or separate?
- Is this a "release from inhibition" effect or a distinct response?
- How is this typically handled in sensorimotor literature?

### Q6: Validation Strategy

After finding an analytic kernel, how do we validate it's "good enough"?

**Our current plan**:
- R² ≥ 0.95 vs learned kernel
- Rate ratio within 1.25x
- PSTH correlation within 0.05 of hybrid model

**Questions**:
- Are these thresholds appropriate?
- Should we use cross-validation on held-out LED cycles?
- What statistical test compares two kernel representations?

### Q7: Known Gaps and Uncertainties

Please identify any gaps in our approach:

1. **Data limitations**: Only 1,407 events in dataset. Is this enough for symbolic regression?
2. **Identifiability**: Can we uniquely determine τ values from this data?
3. **Overfitting risk**: How do we prevent PySR from finding coincidental patterns?
4. **Generalization**: Will the analytic kernel work for other LED patterns (ramps, different duty cycles)?

---

## Data Available

| File | Description |
|------|-------------|
| `data/model/hybrid_model_results.json` | Learned coefficients |
| `data/model/bayesian_opt_results.json` | BO-optimal kernel config |
| `data/engineered/*_events.csv` | Raw event data (1.3M observations, 1,407 events) |
| `data/simulated/hybrid_model_events.csv` | Simulated events from hybrid model |

### Kernel Evaluation Code

```python
def evaluate_kernel(t, coefficients, config):
    """Evaluate learned kernel at time t."""
    K = 0
    for i, c in enumerate(config['early_centers']):
        if abs(t - c) < config['early_width']:
            phi = 0.5 * (1 + np.cos(np.pi * (t - c) / config['early_width']))
            K += coefficients[f'kernel_early_{i+1}'] * phi
    # ... similar for intm and late ...
    return K
```

---

## Expected Deliverables from Research Agent

1. **Recommended functional form** with justification
2. **Parameter bounds** for optimization
3. **PySR configuration** recommendations
4. **SINDy library** specification
5. **Interpretation template** for mapping τ → biology
6. **Validation protocol** for analytic kernel
7. **Literature references** for similar kernel discovery

---

## Context Files

For full context, the research agent should review:

- `docs/MODEL_SUMMARY.md` - Current model status
- `docs/logs/2025-12-11/PLAN-SYMBOLIC-REGRESSION-KERNEL.md` - Implementation plan
- `scripts/fit_analytic_kernel.py` - Existing double-exponential fitting code
- `scripts/fit_hybrid_model.py` - Hybrid model (current best)

---

## Constraints

- **Timeline**: 6.5 hours for implementation
- **Compute**: MacBook Pro M1, 16GB RAM
- **Packages available**: numpy, scipy, sklearn, pysr (can install), pysindy (can install)
- **Goal**: Publication-ready interpretable kernel

---

## Success Metrics

| Metric | Target |
|--------|--------|
| Expression complexity | ≤6 parameters |
| Fit quality (R²) | ≥0.95 |
| Simulation rate ratio | ≤1.25x |
| PSTH correlation | ≥0.75 |
| Biological interpretability | Parameters map to known timescales |

