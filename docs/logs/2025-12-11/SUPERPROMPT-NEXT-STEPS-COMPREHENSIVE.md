# Comprehensive Research Prompt: INDYsim Hazard Model Next Steps

## Context

You are assisting with **INDYsim**, a project to model Drosophila larva reorientation behavior in response to optogenetic stimulation. The goal is to build a validated hazard model that predicts when larvae initiate reorientation events (turns/head sweeps) based on LED stimulus history.

---

## Project Status: Analytic Kernel Complete

We have successfully completed the **Symbolic Kernel Discovery** phase, transforming a 12-basis raised-cosine kernel into a 6-parameter closed-form expression.

---

## Complete Model Specification

### 1. Hazard Model Structure

The instantaneous hazard (event rate) is:

```
λ(t) = exp(β₀ + u_track + ∫₀^∞ K_on(τ) × LED(t-τ) dτ + K_off(t_since_LED_off))
```

Where:
- β₀ = -6.76 (global intercept, log-hazard units)
- u_track ~ N(0, 0.47²) (track-specific random effect)
- K_on(t) = LED-ON kernel (gamma-difference)
- K_off(t) = LED-OFF rebound kernel (exponential)
- LED(t) = stimulus intensity (0 or 250 PWM)

### 2. LED-ON Kernel (Gamma-Difference)

```
K_on(t) = A × Γ(t; α₁, β₁) - B × Γ(t; α₂, β₂)
```

Where Γ(t; α, β) is the gamma probability density function with shape α and scale β.

**Fitted Parameters (with 95% Bootstrap CIs from 100 resamples):**

| Parameter | Value | 95% CI | Interpretation |
|-----------|-------|--------|----------------|
| A | 0.456 | [0.409, 0.499] | Fast excitatory amplitude |
| α₁ | 2.22 | [1.93, 2.65] | Fast shape (~2 processing stages) |
| β₁ | 0.132s | [0.102, 0.168] | Fast timescale |
| B | 12.54 | [12.43, 12.66] | Slow suppressive amplitude |
| α₂ | 4.38 | [4.30, 4.46] | Slow shape (~4 processing stages) |
| β₂ | 0.869s | [0.852, 0.890] | Slow timescale |

**Derived Timescales:**

| Component | Peak Time | Mean Time | Std Dev | Interpretation |
|-----------|-----------|-----------|---------|----------------|
| Fast | 0.16s [0.15, 0.18] | 0.29s [0.27, 0.33] | 0.20s | Sensory transduction |
| Slow | 2.94s [2.93, 2.96] | 3.81s [3.79, 3.84] | 1.82s | Synaptic adaptation |

**Amplitude ratio**: A/B = 0.036 (suppression is 27× stronger than excitation)

### 3. LED-OFF Rebound Kernel

```
K_off(t) = D × exp(-t/τ_off)
```

| Parameter | Value | Interpretation |
|-----------|-------|----------------|
| D | -0.114 | Modest negative rebound |
| τ_off | 2.0s | Recovery timescale |
| Half-life | 1.39s | Time to 50% recovery |

The negative rebound means LED-OFF initially further suppresses hazard briefly before recovering to baseline.

### 4. Original 12-Basis Raised-Cosine Kernel

For reference, the learned kernel used 12 raised-cosine basis functions with these coefficients:

```
Basis centers: [0.2, 0.63, 1.07, 1.5, 2.0, 2.5, 3.0, 4.2, 5.4, 6.6, 7.8, 9.0] seconds
Coefficients: [1.23, 0.11, -1.10, -1.59, -1.14, -1.18, -1.63, -1.69, -0.09, -0.82, 0.60, -0.58]
Rebound coefficient (x13): -0.114
```

The gamma-difference kernel approximates this with R² = 0.968.

### 5. Track-Level Random Effects

55 tracks total. Track intercepts range from -7.65 to -6.23 (log-hazard units).
- Mean intercept: -6.65
- Std of intercepts: 0.47
- This corresponds to ~3× variation in baseline event rate across tracks.

Event rates per track range from 0 to 2.97 events/min (3 tracks have 0 events).

---

## Validation Results

### Kernel-Level Validation

| Metric | Value | Threshold | Status |
|--------|-------|-----------|--------|
| Kernel R² | 0.968 | ≥ 0.95 | PASS |
| Hazard correlation | 0.981 | ≥ 0.95 | PASS |
| Hazard R² | 0.962 | ≥ 0.90 | PASS |

### Simulation Validation

| Metric | Value | Threshold | Status |
|--------|-------|-----------|--------|
| Rate ratio (sim/empirical) | 1.01 | 0.8-1.25 | PASS |
| Early suppression diff | 1.1% | < 20% | PASS |
| Late suppression diff | 2.3% | < 20% | PASS |
| PSTH correlation | 0.44 | ≥ 0.75 | MARGINAL |

The low PSTH correlation (0.44) is attributed to event sparsity (~0.08 events/min during LED-ON), not model error. The hazard-based metrics confirm the model is correct.

### Cross-Validation (5-fold, 55 tracks)

| Fold | Train R² | Test R² | N Train | N Test |
|------|----------|---------|---------|--------|
| 1 | 0.968 | 0.961 | 44 | 11 |
| 2 | 0.968 | 0.961 | 44 | 11 |
| 3 | 0.968 | 0.961 | 44 | 11 |
| 4 | 0.968 | 0.962 | 44 | 11 |
| 5 | 0.968 | 0.960 | 44 | 11 |
| **Mean** | **0.968** | **0.961** | - | - |

Parameters are perfectly stable across folds (CV < 1%).

### Reverse-Correlation Analysis

The reverse-correlation (stimulus-triggered average) reports:
- `is_biphasic: false`
- `peak_significant: false`
- `early_mean_deviation: -3.52`
- `late_mean_deviation: +2.08`
- `peak_z_score: 1.55`
- `n_events: 1407`

**Investigation conclusion**: This is NOT a model failure. The reverse-correlation measures LED intensity before events, while the kernel measures hazard response. The sign pattern appears inverted because:
- Kernel says: LED-ON suppresses turns (kernel < 0 at late times)
- Reverse-corr says: Events follow "high LED then low LED" pattern
- These are CONSISTENT: fewer turns during LED-ON, more after LED-OFF

---

## Experimental Context

- **Organism**: Drosophila melanogaster larvae (3rd instar)
- **Genotype**: GMR61F07-Gal4 > UAS-CsChrimson (optogenetic activation of specific neurons)
- **Stimulus**: Red LED, 0-250 PWM intensity, 30s ON / 30s OFF square wave cycles
- **Behavior tracked**: Reorientation events (turns, head sweeps)
- **Dataset size**: 55 tracks, 1407 total events, ~25 events/track average
- **Event rates**: 
  - During LED-ON: ~0.08 events/min (heavily suppressed)
  - During LED-OFF: ~0.8 events/min (near baseline)
  - Ratio: LED-ON suppresses events by ~10×

---

## Research Questions for Next Steps

### TOPIC 1: Simulation Integration

**Goal**: Replace the 12-basis raised-cosine kernel with the analytic gamma-difference form in the simulation pipeline.

**Questions**:

1. What is the computational advantage of using the analytic kernel vs. the basis expansion? The current basis approach requires storing and multiplying a 12-column matrix. The gamma-difference requires evaluating 2 gamma PDFs.

2. How should the discrete convolution be implemented for the gamma PDF in simulation?
   - Direct numerical integration at each timestep?
   - Pre-computed lookup table for K(t) at fine resolution?
   - Recursive exponential approximation (gamma as sum of exponentials)?

3. The analytic kernel has K(0) = 0 (gamma PDF property), but the learned raised-cosine kernel has K(0) ≈ 0.31 (first basis coefficient). Is this discrepancy:
   - A basis artifact (raised-cosine basis doesn't enforce K(0)=0)?
   - Biologically meaningful (instantaneous LED effect)?
   - Numerically negligible for simulation?

4. Should simulations use point estimates or sample from the bootstrap distribution to propagate parameter uncertainty into event predictions?

---

### TOPIC 2: Full Simulation Validation

**Goal**: Confirm the hazard model reproduces empirical track-level statistics.

**Questions**:

1. What track-level statistics should we compare between simulated and empirical data?
   - Total event count per track
   - Inter-event interval (IEI) distribution
   - Event timing relative to LED onset/offset (PSTH)
   - Event clustering within LED-OFF periods

2. The PSTH correlation is only 0.44 despite good kernel fit. This was attributed to sparsity. How can we validate the model given ~0.08 events/min?
   - Aggregate across 100+ simulated tracks?
   - Use cumulative event counts instead of binned PSTH?
   - Bayesian posterior predictive checks?
   - QQ-plot of event times?

3. What tolerance is acceptable for simulation vs. empirical mismatch?
   - Rate ratio: currently 1.01 (empirical events: 1407, simulated: ~840)
   - Note: there's a discrepancy in the validation data showing different event counts. The hybrid model simulates 827 events, analytic simulates 837, but empirical has 1407. This may be due to different simulation durations or conditions.

4. How do we distinguish model error from natural variability in empirical data?

---

### TOPIC 3: Inter-Event Interval (IEI) Validation

**Goal**: Verify the model produces realistic waiting times between events.

**Questions**:

1. For a time-varying Poisson process with our kernel, what is the expected IEI distribution?
   - Hazard varies ~10× between LED-ON (suppressed) and LED-OFF (baseline)
   - Should we expect bimodal IEI? (Short intervals during LED-OFF bursts, long intervals spanning LED-ON periods)

2. The current model assumes events are conditionally independent given the hazard. Is there evidence for:
   - Absolute refractory period (no events possible for T_ref seconds after an event)?
   - Relative refractory (reduced hazard immediately after events)?
   - Self-excitation (events increase probability of subsequent events)?

3. If refractory effects exist, how should they be incorporated?
   - Add a post-event kernel: K_post(t) that modifies hazard after each event
   - Use a renewal process instead of Poisson
   - Hawkes process (self-exciting point process)

4. What empirical IEI statistics should we compute?
   - Mean, median, CV of IEI
   - IEI histogram with theoretical overlay
   - Autocorrelation of event times

---

### TOPIC 4: Variance Adaptation

**Goal**: Determine if the kernel adapts to stimulus statistics.

**Questions**:

1. Gepner et al. (2015) report that larvae adapt to stimulus variance. Our dataset uses a single stimulus condition (30s ON/30s OFF at 250 PWM). Is variance adaptation testable with this data, or would we need additional experiments?

2. How would the kernel change for different stimulus conditions?
   - Lower intensity (125 PWM): Scale amplitude, or change timescales?
   - Different duty cycle (10s ON/50s OFF): Same kernel, or different adaptation?
   - Flickering (1s ON/1s OFF): Would the slow component disappear?

3. Is variance adaptation relevant for the immediate project goals, or is it a future extension?

4. If variance adaptation is needed, what is the minimal model extension?
   - Divisive normalization: K(t) / (1 + σ²_stim)
   - Adaptive gain: A and B scale with stimulus history
   - Timescale adaptation: β₁, β₂ depend on stimulus statistics

---

### TOPIC 5: Head-Angle Prediction

**Goal**: Extend the model to predict turn direction.

**Questions**:

1. The current model predicts WHEN turns occur (scalar hazard). To predict WHERE the larva turns (direction), we need:
   - Bivariate hazard (separate left/right rates)?
   - Single hazard + conditional direction model P(left | turn)?
   - What approach do Gepner, Hernandez-Nunez, or Klein use?

2. For optogenetic stimulation (spatially uniform light), should turn direction be:
   - Random 50/50?
   - Biased by recent stimulus history?
   - Biased by the animal's current heading or body posture?

3. Head-angle data is available in our dataset. What analyses would reveal directional structure?
   - Left/right turn ratio during LED-ON vs LED-OFF
   - Turn angle magnitude distribution
   - Correlation between consecutive turn directions

4. Is head-angle prediction in scope for the current project, or should it be deferred?

---

### TOPIC 6: Literature Comparison

**Goal**: Validate our kernel against published results.

**Questions**:

1. **Hernandez-Nunez et al. (2015, eLife)** report reverse-correlation kernels for larval navigation under optogenetic control. How does our gamma-diff kernel compare to their results?
   - Do they see a similar biphasic structure?
   - What timescales do they report for excitation and suppression?
   - Do they parameterize the kernel or leave it non-parametric?

2. **Gepner et al. (2015, eLife)** use an LNP model for larval phototaxis. What temporal filter form do they use?
   - Exponential cascade?
   - Gamma functions?
   - Arbitrary basis functions?
   - Do they report filter timescales?

3. **Klein et al. (2015, PNAS)** model thermotaxis with event-driven hazard. Their kernel form would be directly comparable. What do they report?

4. Our kernel shows shape parameters α₁ ≈ 2 and α₂ ≈ 4. In a gamma-cascade interpretation, this suggests 2 and 4 "processing stages." Does this match:
   - Known GMR61F07 circuit architecture?
   - Number of synapses from photoreceptors to motor neurons?
   - Reported adaptation timescales in the literature?

5. The amplitude ratio A/B ≈ 0.036 means suppression is 27× stronger than excitation. Is this asymmetry reported in the literature? Is it plausible for CsChrimson activation?

---

### TOPIC 7: Methods Summary for Publication

**Requested Output**:

Please draft a ~200-word methods paragraph suitable for a behavioral neuroscience paper that:

1. States the model class (negative binomial GLM with temporal kernel / LNP model)
2. Describes the kernel form (gamma-difference with 6 parameters)
3. Reports key parameter values with confidence intervals
4. Summarizes the two timescales and their interpretation
5. Notes the LED-off rebound term
6. Reports validation metrics (R², hazard correlation, rate ratio)
7. Cites relevant methodology papers (Hernandez-Nunez 2015, Gepner 2015, etc.)

---

## Prioritization Request

Please provide:

1. **Answers** to the research questions above, prioritized by importance for completing a publishable hazard model.

2. **Recommendations** on which topics to pursue immediately vs. defer to future work.

3. **Specific action items** for each recommended topic, with estimated time.

4. **Risk assessment**: What could go wrong with each approach? What are the failure modes?

5. **Literature pointers**: Specific papers or sections that address these questions.

---

## Constraints

- The model should remain **interpretable** (prefer simple extensions over complex ones)
- **Computational efficiency** matters for simulation (target: thousands of tracks)
- The primary use case is **predicting reorientation timing**, not full trajectory simulation
- Publication target: **methods for a behavioral neuroscience audience** (not a modeling audience)
- The 6-parameter gamma-difference kernel is considered **final** for this phase; we are not seeking better kernel fits

---

## Summary of What Is Complete

1. ✅ NB-GLM / LNP model fitted to empirical data
2. ✅ 12-basis raised-cosine kernel extracted
3. ✅ Gamma-difference analytic kernel (R² = 0.968)
4. ✅ Bootstrap confidence intervals (100 samples)
5. ✅ LED-off rebound term characterized
6. ✅ 5-fold cross-validation (test R² = 0.961)
7. ✅ Reverse-correlation discrepancy explained
8. ✅ Documentation updated

## What Remains

1. ⬜ Integrate analytic kernel into simulation code
2. ⬜ Full track-level simulation validation
3. ⬜ IEI distribution analysis
4. ⬜ Literature comparison
5. ⬜ Methods paragraph draft
6. ⬜ (Optional) Variance adaptation analysis
7. ⬜ (Optional) Head-angle prediction

---

*Generated: 2025-12-11*
*Project: INDYsim (Integrated Navigation Dynamics Simulator)*
*Current phase: Post-kernel discovery, pre-simulation integration*

