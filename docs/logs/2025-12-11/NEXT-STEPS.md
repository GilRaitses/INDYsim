# Next Steps for INDYsim

## Current Status

The analytic hazard model is validated and integrated:
- 6-parameter gamma-difference kernel for LED-ON response
- Exponential rebound for LED-OFF response
- Calibrated intercept (-6.23) matching empirical rates
- Rate ratio = 0.97, PSTH correlation = 0.84

## Immediate Priorities

### 1. Full Trajectory Simulation

The hazard model predicts **when** events occur. To simulate complete larval trajectories, add:

| Component | Description | Priority |
|-----------|-------------|----------|
| Run/Turn state machine | Binary state with hazard-driven transitions | HIGH |
| Turn angle distribution | Sample from empirical Δθ distribution | HIGH |
| Run kinematics | Forward motion + heading noise | MEDIUM |
| Turn duration | Empirical distribution (~0.5-2s) | MEDIUM |

**Suggested approach**:
```python
# Pseudocode for trajectory simulation
state = 'RUN'
for t in time_grid:
    if state == 'RUN':
        # Move forward with small heading noise
        x += v * cos(theta) * dt
        y += v * sin(theta) * dt
        theta += noise * dt
        
        # Check for turn event
        if hazard_event(t):
            state = 'TURN'
            turn_angle = sample_turn_distribution()
            turn_start = t
    
    elif state == 'TURN':
        # Execute turn
        theta += turn_angle * (t - turn_start) / turn_duration
        if t - turn_start > turn_duration:
            state = 'RUN'
```

### 2. Condition Generalization

Current model is fit to a single condition:
- 0→250 PWM square wave
- 10s ON / 20s OFF
- GMR61 optogenetic line

To generalize:

| Extension | Approach |
|-----------|----------|
| Different intensities | Scale kernel amplitude with intensity |
| Ramp stimuli | Convolve kernel with stimulus profile |
| Different genotypes | Refit model or adjust sensitivity parameter |

### 3. Model Validation Extensions

| Test | Purpose | Status |
|------|---------|--------|
| Time-rescaling | Verify Poisson assumption | Pending |
| Cross-validation | Out-of-sample performance | Done (CV R² = 0.96) |
| Fano factor | Event clustering | Pending |
| Residual diagnostics | Model adequacy | Pending |

### 4. Publication-Ready Outputs

| Deliverable | Description |
|-------------|-------------|
| Methods section | Model specification and fitting procedure |
| Figure 1 | Kernel shape with uncertainty bands |
| Figure 2 | PSTH comparison (empirical vs simulated) |
| Supplementary | Parameter table, validation metrics |

## Research Questions

### Biological Interpretation

1. **Fast excitatory component** (τ₁ ≈ 0.3s): What is the neural mechanism?
   - Sensory transduction time?
   - Motor preparation latency?

2. **Slow suppressive component** (τ₂ ≈ 3.8s): Why sustained suppression?
   - Adaptation in sensory neurons?
   - Neuromodulatory feedback?

3. **Rebound** (τ_off ≈ 2s): Post-inhibitory rebound?
   - Release from inhibition?
   - Return to baseline exploratory state?

### Model Extensions

1. **Refractory period**: IEI analysis shows no significant refractory period (exponential-like distribution). Current model is adequate.

2. **Track intercept variability** (σ = 0.47): What drives individual differences?
   - Genetic background?
   - Developmental state?
   - Experimental conditions?

3. **Head angle prediction**: Can the model predict turn direction?
   - Current model is for event timing only
   - Would need separate directional model

## Implementation Roadmap

### Phase 1: Trajectory Simulation (1-2 days)
- [ ] Define state machine (RUN/TURN)
- [ ] Sample turn angles from empirical distribution
- [ ] Implement run kinematics
- [ ] Generate synthetic trajectories
- [ ] Validate trajectory statistics

### Phase 2: Condition Generalization (2-3 days)
- [ ] Add intensity scaling parameter
- [ ] Implement stimulus convolution
- [ ] Test on 50→250 PWM condition
- [ ] Test on ramp stimuli

### Phase 3: Documentation (1 day)
- [ ] Write methods section
- [ ] Generate publication figures
- [ ] Prepare supplementary materials

## Technical Debt

| Item | Priority |
|------|----------|
| Clean up old validation scripts | LOW |
| Consolidate data loading functions | MEDIUM |
| Add unit tests for hazard model | MEDIUM |
| Profile simulation performance | LOW |

## Questions for Research Agent

If further research is needed:

1. **Literature**: What are typical turn angle distributions in Drosophila larvae under optogenetic stimulation?

2. **Methods**: What is the standard approach for simulating point process models with time-varying hazard?

3. **Validation**: What additional statistical tests are recommended for validating inhomogeneous Poisson process models?

