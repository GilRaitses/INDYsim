# Research Prompt: ML Methods for Optimizing LNP Behavioral Simulation

**Date**: 2025-12-11  
**Context**: Larval reorientation hazard model (NB-GLM with raised-cosine temporal kernel)

---

## Current State

We have a Linear-Nonlinear-Poisson (LNP) cascade model for predicting larval reorientation events during optogenetic stimulation:

```
log λ(t) = β₀ + Σⱼ βⱼ Bⱼ(time_since_LED_onset)
```

Where:
- `λ(t)` = hazard rate (events/s)
- `Bⱼ(t)` = raised-cosine basis functions (9 bases covering 0-10s)
- Fit via Negative Binomial GLM on 1.3M frame-level observations

### Current Validation Results

| Metric | Empirical | Simulated | Status |
|--------|-----------|-----------|--------|
| Event rate | 0.71 | 1.01 | 1.4x mismatch |
| PSTH correlation | - | 0.881 | PASS |
| W-ISE | - | 0.094 | PASS |
| Early suppression | 0.55 | 0.71 | 16% diff |
| Late suppression | 0.32 | 0.50 | 18% diff |

### Core Problem

There is a **rate-vs-shape trade-off**:
- MLE intercept (-6.66): Best PSTH shape but 1.8x rate
- Fixed intercept (-7.44): Correct rate but wrong suppression pattern
- Compromise (-7.0): 1.4x rate with reasonable shape

The sparse event data (1,407 events / 1.3M frames = 0.1%) limits the GLM's ability to simultaneously fit baseline rate AND temporal dynamics.

---

## Research Questions

### Q1: Physics-Informed Machine Learning (PIML)

Can we incorporate biological constraints as physics priors?

**Known constraints**:
1. Refractory period: P(event | recent_event) should decay exponentially
2. Adaptation: Response magnitude decreases with repeated stimulation
3. Energy conservation: Total neural activity should be bounded
4. Causal: Response cannot precede stimulus

**Specific questions**:
- How do we encode a refractory period as a differentiable constraint?
- Can we use a physics-informed loss that penalizes violations of these constraints?
- What PIML frameworks (DeepXDE, NeuralODE) are best suited for point-process data?

```python
# Hypothetical PIML loss
loss = NLL(observed, predicted) + λ_refrac * refractory_violation + λ_adapt * adaptation_violation
```

### Q2: SINDy (Sparse Identification of Nonlinear Dynamics)

Can SINDy discover the governing equations of the hazard dynamics?

**Setup**:
- State: `x(t) = [λ(t), LED(t), time_since_last_event, ...]`
- Dynamics: `dx/dt = f(x)` where f is sparse in a library of candidate functions

**Questions**:
- What candidate library should we use? (polynomials, trig, exponentials, thresholds)
- Can SINDy recover the raised-cosine kernel structure from data?
- How do we handle the discrete event observations (only observe events, not continuous λ)?

```python
# SINDy library for hazard dynamics
library = [1, LED, LED², sin(ωt), exp(-t/τ), H(t-t_event), ...]
```

### Q3: Symbolic Regression (PySR)

Can symbolic regression discover a closed-form expression for the optimal kernel?

**Target**: Find `K(t) = f(t)` such that:
```
log λ(t) = β₀ + ∫ K(τ) LED(t-τ) dτ
```

**Questions**:
- What's the best objective for PySR? (log-likelihood, PSTH correlation, W-ISE)
- How do we constrain the search to biologically plausible functions?
- Can PySR discover the triphasic structure (early bump → suppression → recovery)?

```python
# PySR search space
model = PySRRegressor(
    binary_operators=["+", "-", "*", "/"],
    unary_operators=["exp", "cos", "sin", "tanh"],
    constraints={"exp": (-10, 10)},  # Prevent overflow
    loss="negloglik"  # Custom loss for point process
)
```

### Q4: Neural Point Process with Fourier Neural Operator (FNO)

Can we replace the linear filter with a neural operator that learns the stimulus→response mapping?

**Architecture**:
```
LED(t) → FNO → K(t) → convolution → log λ(t) → NB likelihood
```

**Questions**:
- Is FNO appropriate for 1D temporal kernels, or is a simpler CNN sufficient?
- How do we ensure the learned kernel is interpretable?
- Can we use attention mechanisms to learn which time lags matter?

```python
# Conceptual architecture
class NeuralPointProcess(nn.Module):
    def __init__(self):
        self.kernel_net = FNO1d(modes=16, width=32)  # Learn kernel shape
        self.baseline = nn.Parameter(torch.tensor(-7.0))
    
    def forward(self, stimulus_history):
        kernel = self.kernel_net(stimulus_history)
        log_rate = self.baseline + (kernel * stimulus_history).sum(dim=-1)
        return log_rate
```

### Q5: Bayesian Optimization for Kernel Design

Can we automatically search the kernel basis configuration?

**Search space**:
- Number of bases: 3-15
- Centers: 0-10s
- Widths: 0.2-3.0s
- Basis type: raised-cosine, Gaussian, B-spline

**Objective**: Minimize W-ISE subject to rate constraint

```python
from bayes_opt import BayesianOptimization

def objective(n_early, n_late, early_width, late_width):
    # Fit model with this configuration
    # Return -WISE (we minimize)
    pass

optimizer = BayesianOptimization(
    f=objective,
    pbounds={
        'n_early': (2, 5),
        'n_late': (3, 8),
        'early_width': (0.2, 1.0),
        'late_width': (1.0, 3.0)
    }
)
```

### Q6: Stochastic Dynamic Programming (SDP)

Can we model the larva's behavioral state as an MDP/POMDP?

**State space**:
- Internal state: `s ∈ {alert, habituated, refractory}`
- Observable: LED intensity, time since last event

**Dynamics**:
```
P(s_{t+1} | s_t, LED_t) = transition_matrix[s_t, LED_t]
P(event | s_t) = emission_rate[s_t]
```

**Questions**:
- Can we infer the latent state sequence using HMMs or switching state-space models?
- Does adding latent states improve the rate-vs-shape trade-off?
- What's the value function interpretation for larval behavior?

```python
# Hidden Markov Model for latent behavioral states
class BehavioralHMM:
    n_states = 3  # alert, suppressed, refractory
    transition_matrix = ...  # P(s' | s, LED)
    emission_rates = ...  # λ(event | s)
```

### Q7: Gaussian Process Temporal Kernels

Can we use GP priors to learn the kernel shape nonparametrically?

```python
# GP prior on temporal kernel
K(t) ~ GP(0, k_SE(t, t'))  # Squared-exponential covariance

# Posterior after conditioning on event data
K(t) | events ~ GP(μ_post(t), k_post(t, t'))
```

**Questions**:
- What kernel function (SE, Matern, periodic) is most appropriate?
- How do we jointly infer the GP kernel and the baseline rate?
- Can we use sparse GP approximations for scalability?

### Q8: Mixed-Effects / Hierarchical Models

Can random effects explain the rate mismatch?

**Model**:
```
log λ_ij(t) = (β₀ + u_i) + Σ βⱼ Bⱼ(t)
u_i ~ N(0, σ²_track)  # Random intercept per track
```

**Questions**:
- What is the between-track variance in baseline rate?
- Does accounting for this variance improve the fixed-effect estimates?
- Can we use `glmmTMB` or `PyMC` for efficient inference?

---

## Requested Outputs

1. **Feasibility assessment** for each method given:
   - 1.3M observations, 1,407 events, 99 tracks
   - Python/R environment
   - Need interpretable results for paper

2. **Recommended workflow**:
   - Which methods to try first?
   - What's the expected improvement?
   - Computational requirements?

3. **Implementation priorities**:
   - Quick wins (1-2 hours)
   - Medium effort (1 day)
   - Research projects (1 week+)

4. **Hybrid approaches**:
   - Can we combine PIML + Bayesian optimization?
   - SINDy to discover structure, then neural net to refine?
   - GP kernel + HMM latent states?

---

## Data Available

- Frame-level observations: 1,309,730 rows × [time, track_id, LED, is_event]
- Event times: 1,407 reorientation onsets
- Stimulus protocol: 10s ON / 20s OFF, 0-250 PWM ramp
- 99 tracks across 2 experiments

---

## Success Criteria

1. **Rate match**: Simulated rate within 10% of empirical (0.71 events/min)
2. **Shape match**: PSTH correlation > 0.85, W-ISE < 0.10
3. **Interpretability**: Kernel shape has biological meaning
4. **Generalization**: Model predicts held-out tracks (CV correlation > 0.80)

---

## References to Explore

- **PIML**: Raissi et al. (2019) "Physics-informed neural networks"
- **SINDy**: Brunton et al. (2016) "Discovering governing equations from data"
- **PySR**: Cranmer et al. (2020) "Discovering symbolic models from deep learning"
- **Neural Point Process**: Shchur et al. (2021) "Neural temporal point processes"
- **FNO**: Li et al. (2021) "Fourier neural operator for parametric PDEs"
- **GP for point processes**: Adams et al. (2009) "Tractable nonparametric Bayesian inference"
- **HMM for behavior**: Wiltschko et al. (2015) "Mapping sub-second structure in mouse behavior"



