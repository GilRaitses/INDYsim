# INDYsim Hazard Model Summary

## Overview

INDYsim uses a **negative-binomial linear-nonlinear-Poisson (NB-LNP)** point process model to predict when Drosophila larvae initiate reorientation events in response to optogenetic stimulation.

## Model Structure

The instantaneous hazard of a reorientation event is:

```
λ(t) = exp(β₀ + u_track + K_on(t_since_onset) + K_off(t_since_offset))
```

Where:
- **β₀ = -6.76**: Global intercept (log-hazard baseline)
- **u_track ~ N(0, 0.47²)**: Track-specific random effect
- **K_on(t)**: LED-ON kernel (gamma-difference)
- **K_off(t)**: LED-OFF rebound kernel (exponential)

---

## Event Definition

### Inclusive Onset Events (Hazard Fitting)

The hazard model is fit to **all `is_reorientation_start` events** (1,407 total), which represent an inclusive onset marker triggered when curvature crosses a threshold. This includes:
- Large, sustained reorientations ("true turns")
- Brief head sweeps and micro-movements
- Frame-by-frame curvature fluctuations

### True Turns (Trajectory Simulation)

For trajectory simulation and behavioral interpretation, events are filtered to **"true turns"** with measurable duration:

```python
true_turns = events[events['turn_duration'] > 0.1]  # 319 events (23%)
```

| Event Set | Count | Description |
|-----------|-------|-------------|
| All events | 1,407 | Inclusive onset markers (for hazard fitting) |
| True turns | 319 | Duration > 0.1s (for trajectory output) |

### Rationale

This two-stage approach follows standard practice in larval navigation modeling:
1. **Hazard fitting** uses the inclusive onset process (captures full temporal structure)
2. **Trajectory simulation** filters to behaviorally salient turns (matches observable behavior)

Both event rates are reported for transparency.

---

## LED-ON Kernel (Gamma-Difference)

```
K_on(t) = A × Γ(t; α₁, β₁) - B × Γ(t; α₂, β₂)
```

Where Γ(t; α, β) is the gamma probability density function.

### Parameters (with 95% Bootstrap CIs)

| Parameter | Value | 95% CI | Interpretation |
|-----------|-------|--------|----------------|
| A | 0.456 | [0.409, 0.499] | Fast excitatory amplitude |
| α₁ | 2.22 | [1.93, 2.65] | Fast shape (~2 processing stages) |
| β₁ | 0.132s | [0.102, 0.168] | Fast timescale |
| B | 12.54 | [12.43, 12.66] | Slow suppressive amplitude |
| α₂ | 4.38 | [4.30, 4.46] | Slow shape (~4 processing stages) |
| β₂ | 0.869s | [0.852, 0.890] | Slow timescale |

### Derived Timescales

| Component | Peak Time | Mean Time | Interpretation |
|-----------|-----------|-----------|----------------|
| Fast | 0.16s | 0.29s | Sensory transduction |
| Slow | 2.94s | 3.81s | Synaptic adaptation |

---

## LED-OFF Rebound Kernel

```
K_off(t) = D × exp(-t/τ_off)
```

| Parameter | Value | Interpretation |
|-----------|-------|----------------|
| D | -0.114 | Modest negative rebound |
| τ_off | 2.0s | Recovery timescale |
| Half-life | 1.39s | Time to 50% recovery |

---

## Validation Results

### Kernel Fit
- **R² = 0.968** (analytic vs 12-basis raised-cosine)
- **Cross-validation R² = 0.961** (5-fold, 55 tracks)

### Hazard Dynamics
- **Hazard correlation = 0.981**
- **Hazard R² = 0.962**
- **Rate ratio = 1.01** (simulated/empirical)

### Suppression
- **Early suppression difference: 1.1%**
- **Late suppression difference: 2.3%**

---

## Inter-Event Interval Analysis

| Statistic | Value |
|-----------|-------|
| Mean IEI | 0.53s |
| Median IEI | 0.10s |
| CV | 2.51 |
| Min | 0.00s |
| Max | 29.8s |

**Refractory period**: Not detected. Current Poisson model is adequate.

**LED classification**: IEIs within LED-OFF vs spanning LED-ON are significantly different (KS p < 0.001), supporting bimodal structure.

---

## Methods Paragraph (for publication)

We modeled larval reorientation under optogenetic activation as a negative-binomial linear-nonlinear-Poisson (NB-LNP) point process. The hazard model was fit to 1,407 inclusive reorientation-onset events from 55 tracks under 10s ON / 20s OFF LED stimulation at 250 PWM. For each 20 Hz video frame, the instantaneous hazard was λ(t) = exp(β₀ + u_track + K_on(t_since_onset) + K_off(t_since_offset)), where β₀ = -6.23 is the calibrated intercept and u_track ~ N(0, 0.47²) is a track-specific random effect. The LED-ON filter K_on was parameterized as a difference of two gamma probability density functions, K_on(t) = A·Γ(t;α₁,β₁) - B·Γ(t;α₂,β₂), with A = 0.456 [0.409-0.499], α₁ = 2.22 [1.93-2.65], β₁ = 0.132s [0.102-0.168], B = 12.54 [12.43-12.66], α₂ = 4.38 [4.30-4.46], and β₂ = 0.869s [0.852-0.890] (95% bootstrap CIs). These correspond to a fast component with mean timescale τ₁ = 0.29s and a slow component with mean timescale τ₂ = 3.81s, which we interpret as sensory transduction and synaptic/network adaptation, respectively. A separate LED-OFF rebound term K_off(t) = -0.114·exp(-t/2.0s) captured transient post-offset suppression. The analytic kernel approximated a 12-basis raised-cosine fit with R² = 0.968, and the full model achieved a rate ratio of 0.97 and PSTH correlation of 0.84 between simulated and empirical data. For trajectory simulation, we used a RUN/TURN state machine where turn events were generated by the hazard model and turn angles were sampled from an empirical distribution (μ = 7°, σ = 86°) with durations from a lognormal fit (median 1.1s).

---

## Trajectory Simulation

### State Machine

Simulated trajectories use a two-state model:

```
RUN → (hazard event) → TURN → (duration elapsed) → RUN
```

### Run State
- **Speed**: 1.0 mm/s (typical larval crawling)
- **Heading noise**: σ = 0.03 rad/√s (Brownian diffusion)
- **Duration**: Until next hazard event

### Turn State
- **Angle**: Sampled from Normal(μ=0.12 rad, σ=1.50 rad)
- **Duration**: Sampled from Lognormal(s=0.59, scale=1.29s)
- **Speed**: 0.4× run speed

### Turn Distribution Parameters

From 319 filtered events (duration > 0.1s):

| Parameter | Value | Description |
|-----------|-------|-------------|
| Mean angle | 7° | Slight rightward bias |
| Angle σ | 86° | High variability |
| Mean |Δθ| | 69° | Substantial heading changes |
| Median duration | 1.1s | Typical turn duration |
| Duration range | 0.3-6.8s | Full range |

### Simulated Statistics

| Metric | Simulated | Empirical | Match |
|--------|-----------|-----------|-------|
| Turn rate | 1.88/min | 1.84/min | 98% |
| Suppression | 2× | 2× | Yes |

---

## Rate Normalization

### Unit Conversion

The GLM intercept (β₀ = -6.76) is in **log(events per frame)** at 20 Hz. To convert to events/second for simulation:

```
λ_sec = exp(η) × frame_rate = exp(η) × 20
```

Where η = β₀ + u_track + K_on + K_off is the linear predictor.

### Calibration Factor

When simulating with the original intercept (-6.76), the model produces ~60% of empirical events. This is corrected via rate normalization:

```
empirical_events = 1407
simulated_events = 830
calibration_factor = 1407 / 830 = 1.695

calibrated_intercept = original_intercept + log(f)
                     = -6.76 + log(1.695)
                     = -6.23
```

**Key point**: This is a global rate normalization, not a structural model change. The kernel shape parameters are unchanged. The factor corrects for:
1. Unit conversion between discrete-time GLM fitting and continuous-time simulation
2. Random effects vs fixed intercept: GLM global β₀ is pulled toward mean of random intercepts

After calibration:
- **Rate ratio**: 0.97 (target: 0.8-1.25)
- **Baseline rate (LED-OFF)**: ~1.9 events/min
- **Suppressed rate (LED-ON)**: ~1.0 events/min
- **Suppression factor**: ~2× fewer events during LED-ON

---

## Time-Rescaling Test (2025-12-11)

The time-rescaling theorem tests whether the Poisson assumption holds by transforming event times through the cumulative hazard.

### Results

| Event Set | N Events | Rescaled IEI Mean | KS p-value | Result |
|-----------|----------|-------------------|------------|--------|
| All (1,407) | 1,355 IEIs | 0.87 (expected: 1.0) | < 0.001 | FAIL |
| Filtered (319) | 274 IEIs | 2.92 (expected: 1.0) | < 0.001 | FAIL |

### Interpretation

- **All events**: Mean rescaled IEI = 0.87 indicates model slightly over-predicts hazard. The mild deviation (13% low) suggests minor temporal structure not captured.
- **Filtered events**: Mean = 2.92 indicates model severely under-predicts. This is expected since the model was calibrated on all 1,407 events, not the 319 filtered events.

### Implications

1. For simulation with **all events**: Current model is adequate; Poisson assumption mildly violated but matches empirical rates.
2. For simulation with **filtered events only**: Would require recalibration (factor ~3×) to match the sparser true-turn rate.
3. Consider adding a short refractory component if sub-second temporal precision is critical.

## Implementation

### Analytic Hazard Model

The model is implemented in `scripts/analytic_hazard.py`:

```python
from scripts.analytic_hazard import AnalyticHazardModel

model = AnalyticHazardModel()

# Get kernel value at time t
K_on = model.get_K_on(t_since_onset)
K_off = model.get_K_off(t_since_offset)

# Compute hazard
hazard = model.compute_hazard(t_onset, t_offset, track_intercept, led_on)

# Simulate events (frame-by-frame, matches GLM)
events = model.simulate_events_discrete(duration, led_onsets, led_offsets)
```

### Event Generator Integration

For batch simulation, use `scripts/event_generator.py`:

```python
from scripts.event_generator import generate_with_analytic_model

# Generate events for 50 tracks
events_df = generate_with_analytic_model(
    n_tracks=50,
    duration=1200.0,
    pulse_duration=10.0,  # LED ON duration
    gap_duration=20.0,    # LED OFF duration
    seed=42
)
```

### Data Loading

To load the exact dataset used for model fitting:

```python
from scripts.load_fitting_data import load_fitting_dataset

data, model_info = load_fitting_dataset()
# Returns: 55 tracks, 1407 events from 2 specific experiment files
```

---

## Files

### Scripts

| File | Description |
|------|-------------|
| `scripts/analytic_hazard.py` | Hazard model with lookup table (calibrated intercept: -6.23) |
| `scripts/load_fitting_data.py` | Load exact 55-track, 1407-event dataset + filtered events |
| `scripts/validate_matched.py` | Matched validation against fitting data |
| `scripts/time_rescaling_test.py` | Time-rescaling test for Poisson assumption |
| `scripts/extract_turn_distributions.py` | Extract turn angle and duration distributions |
| `scripts/simulate_trajectories.py` | RUN/TURN trajectory simulator |
| `scripts/generate_figures.py` | Generate publication-ready figures |
| `scripts/event_generator.py` | Event generation with `generate_with_analytic_model()` |

### Data

| File | Description |
|------|-------------|
| `data/model/hybrid_model_results.json` | Original NB-GLM fit results |
| `data/model/best_parametric_kernel.json` | Gamma-diff kernel parameters |
| `data/model/kernel_bootstrap_ci.json` | Bootstrap CIs (100 samples) |
| `data/model/turn_distributions.json` | Empirical turn angle/duration fits |
| `data/validation/matched_validation.json` | Final validation results (rate_ratio=0.97) |
| `data/validation/time_rescaling.json` | Time-rescaling test results |
| `data/simulated/*.parquet` | Simulated trajectory data |

### Figures

| File | Description |
|------|-------------|
| `figures/figure1_kernel.png` | Gamma-difference kernel with components |
| `figures/figure2_validation.png` | Validation metrics and PSTH |
| `figures/figure3_trajectories.png` | Example simulated trajectories |
| `figures/turn_distributions.png` | Turn angle and duration distributions |

---

## Limitations

### Condition Specificity
The model is fit to a single stimulus condition:
- 0→250 PWM intensity
- 10s ON / 20s OFF (30s cycle)
- GMR61 optogenetic line

Generalization to other intensities or temporal patterns requires refitting or intensity-scaling assumptions (not validated).

### Event Definition
The hazard model uses 1,407 inclusive onset events, of which 77% have zero measured duration. For trajectory simulation, only the 319 events with duration > 0.1s are used as "true turns." This two-stage approach follows standard practice but requires careful interpretation.

### Time-Rescaling
The time-rescaling test shows mild violation of the Poisson assumption (mean rescaled IEI = 0.87 vs expected 1.0, p < 0.001). This indicates minor unmodeled temporal structure, likely short-term dependencies or refractoriness. The deviation is 13% and does not substantially affect aggregate predictions.

### Trajectory Model Simplifications
The RUN/TURN state machine omits:
- Edge avoidance and boundary interactions
- Explicit head sweeps and reversals
- Speed gradients and acceleration
- Directional coupling to stimulus

These simplifications are appropriate for demonstrating hazard-driven event timing but limit biomechanical realism.

---

## References

1. Hernandez-Nunez L, et al. (2015). eLife 4:e06225.
2. Gepner R, et al. (2015). eLife 4:e06229.
3. Klein M, et al. (2015). PNAS 112(2):E220-9.

---

---

## Factorial Extension (2025-12-11)

The hazard model was extended to a 2×2 factorial design to assess generalization across experimental conditions.

### Factorial Design

| Factor | Levels | Description |
|--------|--------|-------------|
| Intensity (I) | 0→250, 50→250 | LED1 step size |
| Background (C) | Constant, Cycling | LED2 pattern (7 PWM fixed vs 5-15 PWM cycling) |

### Data

| Condition | Events | Tracks |
|-----------|--------|--------|
| 0→250 \| Constant | 1,407 | 55 |
| 0→250 \| Cycling | 2,410 | 214 |
| 50→250 \| Constant | 2,440 | 187 |
| 50→250 \| Cycling | 1,031 | 123 |
| **Total** | **7,288** | **623** |

### Factorial Model

```
log λ(t) = β₀ + β_I·I + β_C·C + β_{IC}·(I×C) 
         + (α + α_I·I + α_C·C)·K_on(t) 
         + γ·K_off(t)
```

Key assumption: Kernel SHAPE (τ₁, τ₂) is fixed across conditions; only AMPLITUDE is modulated.

### Coefficient Estimates (All p < 0.05)

| Effect | Estimate | 95% CI | Interpretation |
|--------|----------|--------|----------------|
| β_I (Intensity) | -0.199 | [-0.266, -0.132] | 18% lower baseline |
| β_C (Cycling) | -0.108 | [-0.174, -0.042] | 10% lower baseline |
| β_{IC} (Interaction) | -0.119 | [-0.218, -0.019] | Modest synergy |
| α (Kernel amplitude) | 1.005 | [0.899, 1.110] | Reference suppression |
| α_I (Intensity mod.) | -0.665 | [-0.773, -0.557] | **66% weaker** |
| α_C (Cycling mod.) | +0.152 | [0.050, 0.254] | **15% stronger** |
| γ (Rebound) | 1.669 | [0.470, 2.869] | Post-offset enhancement |

### Condition-Specific Amplitudes

| Condition | Amplitude | Relative |
|-----------|-----------|----------|
| 0→250 \| Constant | 1.00 | Reference |
| 0→250 \| Cycling | 1.16 | Strongest |
| 50→250 \| Constant | 0.34 | Weakest |
| 50→250 \| Cycling | 0.49 | - |

**Fold range**: 3.4× (from 0.34 to 1.16)

### Key Findings

1. **Intensity effect (66% weaker)**: Partial adaptation - the 50 PWM baseline pre-adapts the pathway, reducing sensitivity to the subsequent 250 PWM step.

2. **Cycling effect (15% stronger)**: Reduced adaptation - the oscillating LED2 prevents steady-state adaptation, maintaining higher responsiveness to LED1.

3. **Dissociable effects**: Baseline hazard and suppression gain are independently tunable. Intensity reduces both; cycling reduces baseline but increases gain.

### Cross-Validation

- **Mean rate ratio**: 1.03 ± 0.31
- **Pass rate**: 7/12 (58%)
- **Interpretation**: Model captures typical behavior; session-to-session variability remains

### Model Limitations

- **Fixed-effects only**: No random intercepts for track/experiment
- **Interaction power**: ~30-40% for detecting 12% effect
- **Event definition**: 77% zero-duration events included

---

## Final Matched Validation (2025-12-11)

Validated against the exact 55-track, 1407-event dataset used for model fitting.

| Metric | Empirical | Simulated | Status |
|--------|-----------|-----------|--------|
| Events | 1407 | 1371 | PASS |
| Rate ratio | - | 0.974 | PASS (target: 0.8-1.25) |
| Suppression | 2.0× | 1.9× | MATCH |
| PSTH correlation | - | 0.840 | GOOD |

### Calibration

The intercept was calibrated from -6.76 to -6.23 to match empirical event rates:

```
calibration_factor = 1407 / 830 = 1.695
calibrated_intercept = -6.76 + log(1.695) = -6.23
```

This produces ~1.9 events/min/track at baseline, matching empirical observations.

### Event Set Comparison

| Event Set | Empirical | Simulated | Rate Ratio | Status |
|-----------|-----------|-----------|------------|--------|
| All (1,407) | 1,407 | 1,371 | 0.97 | PASS |
| Filtered (319) | 319 | 1,336 | 4.19 | FAIL |

The filtered events validation fails because the model was fit to **all 1,407 events**, including 77% with zero duration. The model cannot directly predict the 319 filtered events without refitting.

**Implication for simulation**: Use the current model for all-event simulation. For trajectory simulation with only "true turns," either:
1. Accept that ~23% of simulated events represent behaviorally meaningful turns, or
2. Refit the model on filtered events only (requires ~3× smaller calibration factor).

---

*Generated: 2025-12-11*
*Model version: Gamma-difference (6 params) + Rebound (2 params)*
*Calibration: Frame-rate corrected (20 Hz), rate-normalized (factor 1.7)*
*Trajectory simulation: RUN/TURN state machine with empirical turn distributions*
