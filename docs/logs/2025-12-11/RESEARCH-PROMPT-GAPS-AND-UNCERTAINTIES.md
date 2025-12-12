# Research Prompt: Gaps, Uncertainties, and Confounding Factors

## Context

We have developed and validated an analytic hazard model for Drosophila larval reorientation events under optogenetic stimulation. The model achieves good validation metrics (rate ratio = 0.97, PSTH correlation = 0.84) on the matched dataset. However, several uncertainties and potential confounding factors remain before proceeding with full trajectory simulation and publication.

---

## TOPIC 1: Calibration Discrepancy

### What We Observed

The fitted NB-GLM intercept (-6.76) produced only ~60% of empirical events when used in simulation. We applied a post-hoc calibration factor of 1.7× to achieve matched rates:

```
original_intercept = -6.76  →  ~830 events
calibrated_intercept = -6.23  →  ~1371 events
empirical = 1407 events
```

### Uncertainty

**Why was calibration necessary?**

Possible explanations:
1. **Unit mismatch**: The GLM fits log(events/bin) where bin = 0.5s, but simulation uses 0.05s steps. Did we correctly account for this?
2. **Offset term**: The GLM may have absorbed some variance into the intercept that doesn't transfer to continuous simulation.
3. **Numerical integration**: The discrete simulation may systematically underestimate the continuous hazard integral.
4. **Random effects**: The GLM used 55 track-specific intercepts. Are we correctly incorporating these in simulation?

### Questions for Research

1. When converting a discrete-time NB-GLM to continuous-time simulation, what is the correct procedure for scaling the intercept?
2. Is a 1.7× calibration factor within expected range, or does it indicate a fundamental model misspecification?
3. Should calibration be done on rate (events/time) or on the intercept directly?

---

## TOPIC 2: LED Timing (RESOLVED)

### Verified from Data

Checked `led1Val` column transitions in the raw data:

```
First 5 onsets:  [21.3, 51.3, 81.3, 111.3, 141.3]
First 5 offsets: [31.3, 61.3, 91.3, 121.3, 151.3]
ON duration: 10.0s
Cycle: 30.0s
```

**Confirmed: 10s ON / 20s OFF (30s cycle), first onset at ~21.3s**

Earlier documentation mentioning 30s ON / 30s OFF was incorrect or referred to a different experiment set.

### Remaining Question

The filename contains `0to250PWM_30` - does the `_30` refer to:
- 30s total cycle? (consistent with 10s ON + 20s OFF)
- 30s ramp duration? (inconsistent with observed step function)
- Something else?

This is low priority since we verified the actual timing from data.

---

## TOPIC 3: Event Definition (PARTIALLY RESOLVED)

### What We Found

Checked the two fitting files directly:

```
Total events (is_reorientation_start == True): 1407

Turn duration distribution:
  min: 0.000s, 25%: 0.000s, 50%: 0.000s, 75%: 0.000s, max: 6.85s
  
Events with turn_duration > 0: only 319 (23%)
```

**Key finding**: No duration filtering was applied. All 1407 `is_reorientation_start == True` events were used, regardless of `turn_duration`.

The 31,119 events across all files vs 1407 in the fitting set is explained by:
- **File selection**: Only 2 of 14 files were used (matching `*_0to250PWM_30#C_Bl_7PWM_2025103*`)
- **Condition filtering**: Other files have different LED protocols

### Remaining Questions

1. Why do 77% of events have `turn_duration = 0`? Is this a data issue or expected?
2. What triggers `is_reorientation_start` in the upstream pipeline?
3. Should we be filtering by `turn_duration > 0` for trajectory simulation?
4. What is the difference between `is_turn_start` and `is_reorientation_start`?

### Questions for Research

1. What is the precise definition of a reorientation event in the Drosophila larva literature?
2. How are "turns" vs "reorientations" distinguished in larval behavior?
3. Is it standard to include zero-duration events in hazard model fitting?

---

## TOPIC 4: Trajectory Simulation Approach

### What We Proposed

A simple run/turn state machine:
```
RUN → (hazard event) → TURN → (duration elapsed) → RUN
```

With:
- Run: Forward motion + small heading noise
- Turn: Instantaneous heading change sampled from distribution

### Uncertainty

**Is this the correct model for larval locomotion?**

Concerns:
1. **Continuous vs discrete turns**: Do larvae make discrete turns or continuous heading adjustments?
2. **Turn dynamics**: Is heading change instantaneous or gradual over the turn duration?
3. **Speed modulation**: Does speed change during turns? Before/after?
4. **Coupled behaviors**: Are there other behaviors (pauses, reversals, head sweeps) that should be modeled?

### Questions for Research

1. What is the standard kinematic model for Drosophila larval locomotion in the literature?
2. How are run and turn states typically defined and detected?
3. What is the empirical distribution of turn angles under optogenetic stimulation?
4. Are there published models we should follow or compare against (e.g., larvaworld, MaggotTracker)?

---

## TOPIC 5: Kernel Biological Interpretation

### What We Have

The gamma-difference kernel has two components:

| Component | Timescale | Interpretation |
|-----------|-----------|----------------|
| Fast (τ₁) | 0.29s | Initial response |
| Slow (τ₂) | 3.8s | Sustained suppression |

Peak suppression at t* = 1.7s, recovery by ~8s.

### Uncertainty

**What biological mechanisms produce these timescales?**

Speculative interpretations:
- τ₁ ≈ 0.3s: Sensory transduction + motor preparation?
- τ₂ ≈ 3.8s: Neuromodulatory feedback? Adaptation?

But these are guesses, not grounded in literature.

### Questions for Research

1. What are the known timescales of optogenetic activation in GMR61-expressing neurons?
2. What downstream circuits are involved in reorientation suppression?
3. Are there published models of larval sensorimotor integration with similar timescales?
4. How do our kernel parameters compare to other GLM fits in the larval behavior literature?

---

## TOPIC 6: Condition Generalization

### What We Proposed

To generalize beyond 0→250 PWM square wave:
- Scale kernel amplitude with intensity
- Convolve kernel with stimulus profile for ramps

### Uncertainty

**Is linear intensity scaling biologically justified?**

Concerns:
1. **Saturation**: Optogenetic response may saturate at high intensities
2. **Threshold effects**: May be nonlinear at low intensities
3. **Adaptation**: Prolonged stimulation may cause adaptation
4. **Different genotypes**: GMR61 may have different sensitivity than other lines

### Questions for Research

1. What is the dose-response curve for optogenetic activation in Drosophila larvae?
2. Is there evidence for saturation or threshold effects?
3. How should we handle ramp stimuli - convolution or instantaneous rate?
4. Are there published intensity-response curves for GMR61 or similar lines?

---

## TOPIC 7: Statistical Validation Gaps

### What We Have Done

| Test | Status | Result |
|------|--------|--------|
| Rate ratio | Done | 0.97 (PASS) |
| PSTH correlation | Done | 0.84 (GOOD) |
| Suppression ratio | Done | 1.9x vs 2.0x (MATCH) |
| IEI KS test | Done | stat=0.12, p<0.001 |
| Bootstrap CIs | Done | 100 samples |
| Track-wise CV | Done | R² = 0.96 |

### What We Have NOT Done

| Test | Purpose | Why It Matters |
|------|---------|----------------|
| Time-rescaling | Verify Poisson assumption | Core model assumption |
| Deviance residuals | Model adequacy | Detect systematic misfit |
| Fano factor | Event clustering | Overdispersion check |
| Q-Q plots | Distribution fit | Residual normality |
| Held-out validation | Generalization | Different experiments |

### Questions for Research

1. What is the standard validation suite for inhomogeneous Poisson process models?
2. How should we interpret the IEI KS test failure (p<0.001)?
3. What residual diagnostics are most informative for NB-GLM hazard models?
4. Should we validate on held-out experiments, or is within-experiment validation sufficient?

---

## TOPIC 8: Potential Confounding Factors

### Known Confounds

| Factor | Concern | Mitigation |
|--------|---------|------------|
| Track quality | Poor tracking → spurious events | Quality filtering? |
| Edge effects | Events near LED transitions | Exclude transition windows? |
| Habituation | Response may decrease over experiment | Time-varying baseline? |
| Individual variability | Track intercepts vary 3× | Random effects (done) |

### Unknown Confounds

| Factor | Concern |
|--------|---------|
| Temperature | May affect behavior, not recorded |
| Developmental stage | Age may affect sensitivity |
| Satiation state | Feeding history not controlled |
| Time of day | Circadian effects? |
| Arena position | Edge vs center behavior differences |

### Questions for Research

1. What experimental variables should be controlled or recorded for larval optogenetic experiments?
2. Are there known circadian or developmental effects on larval reorientation behavior?
3. How should we handle potential habituation over the course of a 20-min experiment?

---

## TOPIC 9: Comparison to Existing Models

### What We Should Compare Against

| Model | Reference | Key Features |
|-------|-----------|--------------|
| larvaworld | Klein et al. | Full agent-based simulation |
| Gepner et al. | 2015 | Optogenetic behavioral analysis |
| Ohyama et al. | 2015 | Neural circuit + behavior |

### Uncertainty

**How does our model compare to published work?**

We have not systematically compared:
- Our kernel shape to published kernels
- Our event rates to literature values
- Our suppression metrics to prior studies

### Questions for Research

1. What are the published hazard/kernel parameters for larval reorientation?
2. How do our suppression ratios compare to Gepner et al. and others?
3. Are there standard benchmarks for larval behavior models?

---

## Summary of Key Uncertainties

| Priority | Topic | Risk Level | Status |
|----------|-------|------------|--------|
| HIGH | Calibration discrepancy (1.7× factor) | May indicate model error | OPEN |
| ~~HIGH~~ | ~~LED timing confusion~~ | ~~Fundamental~~ | RESOLVED (10s ON / 20s OFF) |
| MEDIUM | Event definition (77% zero-duration) | Affects interpretation | PARTIALLY RESOLVED |
| MEDIUM | Trajectory simulation approach | Determines output validity | OPEN |
| MEDIUM | IEI KS test failure (p<0.001) | May indicate model misfit | OPEN |
| MEDIUM | Validation completeness | Missing time-rescaling etc. | OPEN |
| LOW | Biological interpretation | Affects narrative, not function | OPEN |
| LOW | Condition generalization | Future extension | OPEN |
| LOW | Literature comparison | Context for publication | OPEN |

---

## Requested Outputs

Please provide:

1. **Direct answers** to the questions above, with citations where available
2. **Risk assessment**: Which uncertainties are most likely to cause problems?
3. **Recommended actions**: What should we do before proceeding with trajectory simulation?
4. **Literature pointers**: Key papers for larval locomotion modeling and validation
5. **Red flags**: Any aspects of our approach that seem incorrect or non-standard

