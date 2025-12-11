# Comprehensive Research Prompt: NB-GLM Hazard Model for Drosophila Larval Reorientation

## Project Context

We are building **INDYsim**, a simulation framework for Drosophila larval navigation behavior. The goal is to fit a statistical hazard model to empirical reorientation events, then use this model to generate synthetic larval trajectories for experimental design optimization.

The larvae are tracked on an agar plate while optogenetic stimulation (red LED) is delivered in a pulsing pattern. We want to understand how stimulus timing affects the probability of reorientation (turning) events.

---

## Experimental Setup

### Recording Parameters
- Frame rate: 20 Hz (50 ms per frame)
- Experiment duration: 1200 seconds (20 minutes)
- Arena: Circular agar plate, larvae move freely
- Tracking: Centroid position + 11 spine points per larva

### Stimulus Protocol
- **LED1 (Red, 617nm)**: Optogenetic activation of mechanosensory neurons
  - Pulsing: 30s ON / 30s OFF (60s cycle period)
  - Intensity: 0-250 PWM (proportional to power)
  - Some experiments use ramp protocols (0-250 PWM over pulse)
  
- **LED2 (Blue, 470nm)**: Tracking illumination only
  - Constant ON at low intensity (5-15 PWM)
  - Some experiments have Blue also pulsing for control

### Genotype
- GMR61@GMR61: Basin neurons expressing CsChrimson (red-light activated)
- These neurons respond to gentle touch; optogenetic activation mimics touch

---

## Data Structure

### Processed Dataset Summary
- 14 experiments processed
- 8,741,825 trajectory frames (50ms resolution)
- 8,510,608 event records with behavioral state labels
- 8,822 segmented runs with Klein-format metrics
- Consolidated into single HDF5 file (1.92 GB)

### Trajectory Columns
| Column | Type | Units | Description |
|--------|------|-------|-------------|
| time | float | seconds | Experiment time (0-1200s) |
| track_id | int | - | Unique larva identifier within experiment |
| experiment_id | string | - | Source experiment identifier |
| x, y | float | cm | Centroid position |
| speed | float | cm/s | Instantaneous speed (typical: 0.01-0.15 cm/s) |
| curvature | float | 1/cm | Path curvature (signed) |
| led1Val | float | PWM | Red LED intensity (0-250) |
| led2Val | float | PWM | Blue LED intensity (0-15) |
| stimulus_onset | bool | - | True if LED1 turned ON this frame |
| stimulus_offset | bool | - | True if LED1 turned OFF this frame |
| time_since_stimulus | float | seconds | Time since last LED1 onset (0-30s range) |
| spineTheta | float | radians | Body bend angle (from spine points) |
| spineCurv | float | 1/cm | Body curvature (from spine points) |

### Event Columns (same as trajectory plus)
| Column | Type | Description |
|--------|------|-------------|
| is_run | bool | In run state (forward locomotion) |
| is_reorientation | bool | In reorientation state (turning) |
| is_head_swing | bool | Head swing detected this frame |
| reo_dtheta | float | Heading change during reorientation (degrees, -180 to +180) |

### Klein Run Table Columns
| Column | Type | Description |
|--------|------|-------------|
| run_id | int | Sequential run identifier |
| run_start_time | float | Run onset time (s) |
| run_end_time | float | Run offset time (s) |
| run_duration | float | Run duration (s) |
| reo_start_time | float | Reorientation onset time (s) |
| reo_end_time | float | Reorientation offset time (s) |
| reo_duration | float | Reorientation duration (s) |
| reoDTheta | float | Net heading change (degrees) |
| reo#HS | int | Number of head swings during reorientation |
| reoYN | int | 1 if reorientation occurred, 0 if run ended without turn |

---

## Behavioral Definitions

### Run
Forward locomotion characterized by:
- Speed > 0.2 cm/s (start threshold)
- Speed > 0.3 cm/s maintained (stop threshold with hysteresis)
- Low body curvature

### Reorientation
Turning behavior between runs:
- Speed drops below threshold
- Body bends (high spineTheta or spineCurv)
- One or more head swings may occur
- Ends when forward locomotion resumes

### Head Swing
Lateral head movement during reorientation:
- Rapid change in anterior spine angle
- Detected using derivative of head angle
- Larvae may make 0-10+ head swings per reorientation

### Typical Statistics
- Turn rate: ~1-3 reorientations per minute per larva
- Heading change: Roughly uniform distribution with slight bias toward larger turns
- Inter-event interval: Approximately exponential with mean ~20-40s

---

## Proposed Model Specification

### Outcome Variable
Binned reorientation counts with Negative Binomial distribution:

```
Y_it ~ NB(mu_it, alpha)
log(mu_it) = eta_it + log(delta_t)
```

Where:
- Y_it = number of reorientation events in bin i for track t
- mu_it = expected count
- alpha = dispersion parameter (overdispersion if alpha > 0)
- delta_t = bin width (exposure offset)

### Bin Size
0.5-1.0 second non-overlapping bins (configurable)

### Linear Predictor

```
eta_i = beta_0                                    # intercept
      + beta_1 * LED1_intensity_i                 # red LED main effect
      + beta_2 * LED2_intensity_i                 # blue LED main effect  
      + beta_3 * (LED1 x LED2)_i                  # interaction
      + beta_4 * sin(2*pi*phase_i)                # cycle phase (sin)
      + beta_5 * cos(2*pi*phase_i)                # cycle phase (cos)
      + sum_j(phi_j * B_j(time_since_stimulus_i)) # temporal kernel
      + gamma_1 * speed_i                         # speed effect
      + gamma_2 * curvature_i                     # curvature effect
```

Where:
- phase_i = (time mod 60) / 60 (position in 60s LED1 cycle)
- B_j() = raised-cosine basis functions for temporal kernel
- Covariates are bin-averaged values

### Temporal Kernel
Raised-cosine basis functions on time_since_stimulus:

```
B_j(t) = 0.5 * (1 + cos(pi * (t - c_j) / width))  if |t - c_j| < width
       = 0                                          otherwise
```

Parameters to optimize:
- Number of bases J: 3, 4, or 5
- Window: [0, 2s], [0, 3s], or [0, 4s]
- Width: ~0.6s (controls overlap)

### Hierarchical Structure
Data are nested: bins within tracks within experiments

Options:
1. Fixed-effects NB-GLM with cluster-robust SEs on experiment_id
2. Mixed-effects NB-GLMM with random intercepts:
   - b_track ~ N(0, sigma^2_track)
   - u_experiment ~ N(0, sigma^2_exp)

---

## Existing Code Scaffold

### Raised-Cosine Basis Function
```python
def raised_cosine_basis(t, centers, width):
    """
    t : ndarray - time points
    centers : ndarray - center positions for each basis
    width : float - controls overlap
    Returns: ndarray shape (len(t), len(centers))
    """
    n_times = len(t)
    n_bases = len(centers)
    basis = np.zeros((n_times, n_bases))
    
    for j, c in enumerate(centers):
        dist = np.abs(t - c)
        in_range = dist < width
        basis[in_range, j] = 0.5 * (1 + np.cos(np.pi * (t[in_range] - c) / width))
    
    return basis
```

### Phase Covariates
```python
def compute_phase_covariates(time, cycle_period=60.0):
    """Returns sin(2*pi*phase), cos(2*pi*phase)"""
    phase = (time % cycle_period) / cycle_period
    return np.sin(2 * np.pi * phase), np.cos(2 * np.pi * phase)
```

### Design Matrix Construction
```python
def build_design_matrix(data, n_temporal_bases=4, temporal_window=(-3.0, 0.0)):
    """
    Builds design matrix with:
    - Intercept
    - LED1, LED2 (normalized 0-1)
    - LED1 x LED2 interaction
    - Phase sin/cos
    - Temporal kernel bases
    - Speed (standardized)
    - Curvature (standardized, clipped)
    """
    # ... implementation exists
```

### Event Generator (Inversion Method)
```python
class InversionEventGenerator:
    """
    Generate events by inverting cumulative hazard.
    More efficient than thinning for low-rate events.
    
    Algorithm:
    1. Build dense vector of lambda(t) at fine time steps
    2. Compute cumulative hazard H(t) = integral_0^t lambda(s) ds
    3. Draw uniform U ~ [0,1], solve H(tau) = -log(1-U) for event time tau
    4. Use binary search to find tau
    """
    # ... implementation exists
```

---

## Research Questions

### 1. Model Validation
- How to assess goodness-of-fit for rare-event count data?
- What diagnostics indicate NB vs zero-inflated NB is needed?
- How to detect and handle serial correlation in residuals?

### 2. Kernel Interpretation
- Given fitted kernel weights phi_j, how to construct the stimulus response function?
- How to compute confidence bands on the kernel shape?
- How to identify peak response latency and duration?

### 3. Cross-Validation
- Is leave-one-experiment-out (LOEO) appropriate with 14 experiments?
- Should CV be stratified by LED condition (different intensity protocols)?
- What loss function for model selection: deviance, AIC, or pseudo-R2?

### 4. Mixed Effects
- When are random effects necessary vs cluster-robust SEs sufficient?
- How to fit NB-GLMM in Python (statsmodels, pymer4, or custom)?
- How to interpret variance components (track vs experiment)?

### 5. Simulation Validation
- What statistics to compare between simulated and empirical data?
- How to test if simulated turn rate matches empirical within confidence bounds?
- How to validate stimulus-locked dynamics (PSTH shape)?

### 6. Coefficient Interpretation
- How to convert log-link coefficients to percent change in rate?
- How to report LED intensity effect in biologically meaningful units?
- How to visualize interaction effects (LED1 x LED2)?

---

## Expected Outputs

### From Model Fitting
1. Coefficient table with estimates, SEs, p-values, rate ratios
2. Temporal kernel plot showing stimulus response dynamics
3. Model diagnostics (residual plots, dispersion ratio)
4. Cross-validation results for hyperparameter selection

### From Simulation
1. Synthetic larval trajectories with realistic turn statistics
2. Comparison plots: empirical vs simulated turn rates
3. PSTH comparison: stimulus-locked reorientation probability
4. Heading change distribution comparison

---

## References

### Larval Behavior
- Gepner et al. (2015) eLife: "Computations underlying Drosophila photo-taxis, odor-taxis, and multi-sensory integration" - LNP model with raised-cosine kernels
- Klein et al. (2015) PNAS: "Sensory determinants of behavioral dynamics in Drosophila thermotaxis" - Turn detection, MAGAT segmentation
- Gershow et al. (2012) Science: "Controlling airborne cues to study small animal navigation" - Larval navigation decision-making

### Statistical Methods
- Pillow et al. (2008) J Neurosci: "Spatio-temporal correlations and visual signalling in a complete neuronal population" - Raised-cosine basis for spike train models
- Lewis & Shedler (1979) Naval Research Logistics: "Simulation of nonhomogeneous Poisson processes" - Thinning algorithm

### Software
- statsmodels (Python): NegativeBinomial GLM
- pymer4 (Python): R lme4 wrapper for mixed models
- h5py (Python): HDF5 file access
- pyarrow (Python): Parquet file access

---

## Constraints and Preferences

1. **Python-first**: All implementation in Python 3.10+
2. **Reproducibility**: Random seeds set, results logged
3. **Efficiency**: Must handle 8M+ rows without memory issues
4. **Interpretability**: Coefficients must be explainable to biologists
5. **Validation**: Simulated data must match empirical statistics

---

## Specific Guidance Requested

Please provide:

1. **Complete model specification** with all terms explicitly defined
2. **Python code snippets** for any non-trivial calculations
3. **Diagnostic checklist** for model validation
4. **Interpretation template** for reporting results
5. **Simulation validation protocol** with specific statistical tests
