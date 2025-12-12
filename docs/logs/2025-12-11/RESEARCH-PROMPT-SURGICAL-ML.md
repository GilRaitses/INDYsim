# Research Prompt: Surgical Deep Learning & Physics-Informed ML for LNP Optimization

**Date**: 2025-12-11  
**Context**: Follow-up after mixed-effects analysis and Bayesian kernel optimization

---

## New Findings Since Last Prompt

### 1. Mixed-Effects Analysis Results

We computed per-track baseline variability using a pure Python approximation:

| Metric | Value | Interpretation |
|--------|-------|----------------|
| Mean intercept | -6.95 | Close to MLE (-6.66) |
| Between-track SD | 0.48 | Substantial individual variability |
| Low-rate tracks | 0.38× mean | Some larvae rarely turn |
| High-rate tracks | 2.62× mean | Some larvae turn frequently |
| Implied baseline | 1.15 events/min | 1.6× empirical (0.71) |

**Key insight**: The 1.6× rate mismatch is **fundamental to the MLE trade-off**, not a bug. The model must choose between:
- Higher intercept: Better captures sparse event timing
- Lower intercept: Better matches overall rate

The between-track SD of 0.48 means individual larvae vary by ~2.6× in baseline rate. This is **real biological variability**.

### 2. Bayesian Optimization Progress (Ongoing)

Early iterations show promising improvements:

| Iteration | n_early | n_late | early_w | late_w | WISE | Rate Error | Corr |
|-----------|---------|--------|---------|--------|------|------------|------|
| 1 | 4 | 4 | 0.69 | 1.98 | 0.569 | 32% | 0.876 |
| 2 | 3 | 3 | 0.53 | 1.63 | 0.537 | 32% | 0.845 |
| 3 | 2 | 5 | 0.33 | 2.14 | 0.456 | 29% | 0.876 |

**Trend**: Fewer early bases + more late bases + narrower early width seems to help.

### 3. Current Model Performance

| Metric | Current Best | Target |
|--------|--------------|--------|
| PSTH Correlation | 0.88 | ≥ 0.85 ✓ |
| W-ISE | 0.094 | < 0.10 ✓ |
| Rate Ratio | 1.4-1.6× | ~1.0× (not achieved) |
| Early Suppression | 0.71 vs 0.55 | Match (16% off) |
| Late Suppression | 0.50 vs 0.32 | Match (18% off) |

---

## Surgical ML Approaches to Explore

Given the sparse data (1,407 events, 99 tracks), we need **targeted, low-parameter approaches** rather than heavy neural networks.

### Q1: Physics-Informed Refractory Penalty

The inter-event interval (IEI) distribution encodes a natural refractory period. Can we add a soft constraint?

**Empirical IEI statistics**:
- Mean: 0.84s
- CV (coefficient of variation): 1.35

**Proposed penalty**:
```python
def refractory_penalty(simulated_events, tau_refrac=0.5):
    """
    Penalize if post-event hazard doesn't decay exponentially.
    
    After an event, the hazard should be suppressed:
    λ(t | event at t₀) ≈ λ_baseline × (1 - exp(-(t-t₀)/τ))
    """
    ieis = np.diff(simulated_events)
    
    # Expected IEI if refractory: mean should be > tau_refrac
    # Penalize if too many short IEIs
    short_iei_fraction = np.mean(ieis < tau_refrac)
    
    # Also penalize if IEI distribution is too regular (CV should be ~1)
    cv = ieis.std() / ieis.mean()
    cv_penalty = (cv - 1.35)**2
    
    return short_iei_fraction + 0.1 * cv_penalty
```

**Questions**:
1. What is the biologically appropriate refractory timescale for larval reorientations?
2. Should this be a hard constraint (reject simulations) or soft penalty (add to loss)?
3. How do we integrate this with the NB-GLM fitting (not just simulation)?

### Q2: Neural Temporal Filter (Lightweight)

Instead of hand-crafted raised-cosine bases, can a small neural network learn the optimal filter shape?

**Proposed architecture** (minimal parameters):
```python
class LearnedTemporalFilter(nn.Module):
    def __init__(self, n_timepoints=200):  # 10s at 20Hz
        super().__init__()
        # Single-layer network: 200 -> 32 -> 1
        self.filter = nn.Sequential(
            nn.Linear(n_timepoints, 32),
            nn.Softplus(),  # Ensures smoothness
            nn.Linear(32, 1)
        )
        self.baseline = nn.Parameter(torch.tensor(-7.0))
    
    def forward(self, led_history):
        # led_history: (batch, 200) - past 10s of LED values
        kernel_response = self.filter(led_history)
        log_rate = self.baseline + kernel_response
        return log_rate
```

**Training objective**:
```python
# Point process negative log-likelihood
loss = -sum(log_rate[events]) + sum(exp(log_rate) * dt)
```

**Questions**:
1. Is 32 hidden units enough, or do we need more/fewer?
2. Should we regularize the learned filter to be smooth (add TV penalty)?
3. How do we extract an interpretable kernel from the trained network?

### Q3: Gaussian Process Prior on Kernel Shape

Instead of parametric bases, use a GP prior to learn K(t) nonparametrically:

```python
# GP prior on temporal kernel
K(t) ~ GP(0, k_SE(t, t'; lengthscale=1.0, variance=1.0))

# Log-rate model
log λ(t) = β₀ + ∫₀^{10} K(τ) × LED(t-τ) dτ
```

**Inference**: Use variational GP (sparse approximation with ~20 inducing points).

**Questions**:
1. What covariance function is most appropriate? (SE, Matern, periodic?)
2. How do we handle the point-process likelihood (binning? Poisson approximation?)
3. Can we jointly infer K(t) and the between-track variance σ²_track?

### Q4: Latent State Augmentation

The triphasic kernel (early bump → suppression → recovery) might reflect **latent state transitions**:

```
States: {ALERT, SUPPRESSED, REFRACTORY}

Transitions:
  ALERT → SUPPRESSED (on LED onset, fast)
  SUPPRESSED → REFRACTORY (during LED, slow)
  REFRACTORY → ALERT (after LED offset, exponential recovery)

Emissions:
  P(turn | ALERT) = high
  P(turn | SUPPRESSED) = low
  P(turn | REFRACTORY) = very low
```

**Questions**:
1. Can we fit a 3-state HMM to the event data?
2. With ~14 events per track on average, is there enough data?
3. How would latent states improve rate calibration?

### Q5: Symbolic Regression on Fitted Kernel

After Bayesian optimization finds the best raised-cosine configuration, can we fit a simpler analytic form?

**Current kernel shape** (qualitative):
- Peak at t ≈ 0.2s (early bump)
- Minimum at t ≈ 3s (maximum suppression)
- Recovery toward t ≈ 10s

**Candidate functional forms**:
```python
# Double exponential (difference of exponentials)
K(t) = A * exp(-t/τ_fast) - B * exp(-t/τ_slow)

# Damped oscillation
K(t) = A * exp(-t/τ) * cos(ω*t + φ)

# Alpha function with rebound
K(t) = A * (t/τ) * exp(-t/τ) - B * H(t-t_onset)
```

**Questions**:
1. Which functional form best approximates the learned kernel?
2. Can PySR discover this automatically?
3. What are the physiological interpretations of τ_fast, τ_slow?

---

## Specific Implementation Questions

### For Physics-Informed Regularization:

1. **Refractory constraint**: Should we:
   - Add penalty to NB-GLM loss during fitting?
   - Only apply during simulation (rejection sampling)?
   - Use as a post-hoc correction factor?

2. **Energy constraint**: The total event rate should be bounded. Should we:
   - Penalize if integrated hazard over 10s exceeds some threshold?
   - Normalize the kernel to have fixed integral?

### For Neural Approaches:

3. **Data augmentation**: With only 1,407 events, can we:
   - Bootstrap resample tracks?
   - Use time-shifted augmentation?
   - Synthesize pseudo-events from the current model?

4. **Interpretability**: After training a neural filter, how do we:
   - Extract the effective kernel (compute impulse response)?
   - Verify it matches the known triphasic shape?
   - Compare to the raised-cosine basis interpretation?

### For Hybrid Approaches:

5. **PIML + BO combination**: Can we add physics penalties to the BO objective?
   ```python
   score = WISE + 0.5*rate_error + 0.1*refractory_penalty + 0.1*smoothness_penalty
   ```

6. **GP + Mixed Effects**: Can we jointly estimate:
   - GP kernel K(t) (population level)
   - Random intercepts u_i ~ N(0, σ²) (track level)
   - In a single Bayesian model?

---

## Data Constraints

- **Events**: 1,407 reorientations
- **Tracks**: 55-99 (depending on counting method)
- **Observations**: 1.3M frames
- **Event rate**: 0.1% of frames have events
- **Stimulus**: 10s ON / 20s OFF, 0-250 PWM ramp

Given this sparsity, **simple models with good priors** will outperform complex models that overfit.

---

## Requested Outputs

1. **Feasibility ranking** of the 5 approaches (Q1-Q5) for our data size
2. **Recommended implementation order** (quick wins first)
3. **Specific hyperparameters** for:
   - Refractory timescale τ_refrac
   - Neural filter hidden size
   - GP lengthscale and variance
4. **Code snippets** for the most promising approach
5. **Expected improvement** in rate calibration from each method

---

## Success Criteria

We need to improve from:
- Rate ratio: 1.4-1.6× → **≤ 1.2×**
- Early suppression: 16% off → **≤ 10%**
- Late suppression: 18% off → **≤ 10%**

While maintaining:
- PSTH correlation ≥ 0.85
- W-ISE ≤ 0.10
- Interpretable kernel shape



