# Deep Research Prompt: NB-GLM Hazard Model for Larval Reorientation

## Context

We have successfully processed 14 larval behavior experiments through the INDYsim pipeline:
- 8.7M trajectory frames (50ms resolution, 20min experiments)
- 8.5M event records with behavioral state labels
- 8,822 segmented runs with Klein-format metrics

The goal is to fit a Negative Binomial GLM hazard model to predict reorientation event rates as a function of optogenetic stimulus (LED) and behavioral covariates.

## Data Structure

### Consolidated Dataset (`consolidated_dataset.h5`)

```
/trajectories/
  - time: experiment time (s)
  - track_id: unique track identifier
  - experiment_id: source experiment
  - x, y: position (cm)
  - speed: instantaneous speed (cm/s)
  - curvature: path curvature (1/cm)
  - led1Val: Red LED intensity (0-255 PWM)
  - led2Val: Blue LED intensity (0-255 PWM)
  - stimulus_onset: boolean (LED1 turned ON this frame)
  - stimulus_offset: boolean (LED1 turned OFF this frame)
  - time_since_stimulus: seconds since last LED1 onset (0-30s range)
  - spineTheta: body bend angle (radians)
  - spineCurv: body curvature (1/cm)

/events/
  - Same columns as trajectories
  - is_run: boolean (in run state)
  - is_reorientation: boolean (in reorientation state)
  - is_head_swing: boolean (head swing detected)
  - reo_dtheta: heading change during reorientation (degrees)

/klein_run_tables/
  - run_id, track_id, experiment_id
  - run_start_time, run_end_time, run_duration
  - reo_start_time, reo_end_time, reo_duration
  - reoDTheta: net heading change (degrees)
  - reo#HS: number of head swings
  - reoYN: 1 if reorientation occurred, 0 otherwise
```

### Stimulus Protocol

- LED1 (Red, optogenetic): Pulsing with 30s ON / 30s OFF cycle (60s period)
- LED2 (Blue, tracking): Constant ON (100% duty cycle)
- PWM range: 0-250 for Red, 5-15 for Blue
- Total experiment duration: 1200s (20 minutes)
- Expected: ~40 stimulus pulses per experiment

## Research Questions

### 1. Model Specification

The current scaffold implements:
```
log(mu_i) = beta_0 
          + beta_1 * LED1_intensity 
          + beta_2 * LED2_intensity 
          + beta_3 * LED1 x LED2
          + beta_4 * sin(phase) + beta_5 * cos(phase)
          + sum_j(phi_j * B_j(time_since_stimulus))  # raised-cosine kernel
          + gamma_1 * speed 
          + gamma_2 * curvature
```

Questions:
- Is this specification complete for capturing stimulus-evoked reorientation modulation?
- Should we include track-level random effects (mixed model)?
- How should we handle the hierarchical structure (frames within tracks within experiments)?
- Should LED intensity be treated as continuous or binned?

### 2. Temporal Kernel Design

Current implementation uses raised-cosine basis functions:
```python
B_j(t) = 0.5 * (1 + cos(pi * (t - c_j) / width))  if |t - c_j| < width
       = 0                                          otherwise
```

Parameters to optimize:
- Number of bases (3, 4, or 5?)
- Window range (-2s to 0, -3s to 0, or -4s to 0?)
- Width parameter (0.6s default, controls overlap)

Questions:
- What is the expected latency from stimulus onset to behavioral response?
- Should we also model post-stimulus effects (stimulus offset response)?
- Is the raised-cosine parameterization optimal, or should we consider splines?

### 3. Event Definition

Current event types available:
- `is_reorientation`: Full reorientation episode (run-to-run transition)
- `is_head_swing`: Individual head sweep (can be multiple per reorientation)
- Klein `reoYN`: Binary reorientation with heading change > threshold

Questions:
- Which event type is most appropriate for hazard modeling?
- Should we model event rates at frame level (Bernoulli) or binned counts (Poisson/NB)?
- How to handle zero-inflation (many frames with no events)?

### 4. Cross-Validation Strategy

Current implementation supports:
- Leave-one-experiment-out CV
- K-fold CV with random splits

Questions:
- Is leave-one-experiment-out appropriate given experiment heterogeneity?
- Should we stratify folds by stimulus condition or genotype?
- What loss function to use for model comparison (deviance, AIC, pseudo-R2)?

### 5. Dispersion Estimation

NB dispersion parameter alpha controls overdispersion:
- Var(Y) = mu + alpha * mu^2
- alpha = 0 reduces to Poisson
- alpha > 0 allows for overdispersion

Current approach: Method of moments estimation
Questions:
- Should we use joint MLE for alpha?
- Is overdispersion expected given the temporal clustering of events?
- How sensitive are coefficient estimates to alpha misspecification?

## Existing Code

### hazard_model.py (650 lines)

Key functions:
- `raised_cosine_basis()`: Computes temporal kernel bases
- `build_design_matrix()`: Constructs full design matrix with all covariates
- `fit_nb_glm()`: Fits model using statsmodels
- `cross_validate_kernel_params()`: Leave-one-out CV for kernel hyperparameters

### event_generator.py (579 lines)

Key classes:
- `InversionEventGenerator`: Samples events by inverting cumulative hazard
- `ThinningEventGenerator`: Fallback using Lewis-Shedler thinning
- `HeadingChangeDistribution`: Samples reo_dtheta from empirical distribution

## Requested Analysis

1. **Literature Review**: What temporal kernel specifications have been used in similar larval behavior studies? (Gepner 2015, Klein 2015, Gershow 2012)

2. **Model Comparison**: Compare Poisson vs NB vs zero-inflated models for event count data

3. **Kernel Optimization**: Principled approach to selecting number of bases and window size

4. **Interpretation Guide**: How to interpret fitted coefficients in terms of behavioral modulation (e.g., "LED1 increases turn rate by X% relative to baseline")

5. **Simulation Validation**: How to validate that simulated events match empirical statistics (turn rate, heading change distribution, inter-event intervals)

## Expected Outputs

1. Fitted hazard model with coefficient table and confidence intervals
2. Temporal kernel shape showing stimulus response dynamics
3. Model diagnostics (residual plots, dispersion check, CV scores)
4. Simulation of synthetic larval tracks using fitted hazard
5. Comparison of simulated vs empirical turn statistics

## References

- Gepner et al. (2015) eLife - LNP model with raised-cosine kernels for C. elegans
- Klein et al. (2015) PNAS - Turn detection and stimulus-locked analysis
- Gershow et al. (2012) Science - Larval navigation decision-making
- Pillow et al. (2008) J Neurosci - Raised-cosine basis for spike train models

