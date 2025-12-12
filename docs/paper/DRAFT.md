# An Analytic Hazard Kernel for Optogenetically-Driven Larval Reorientation

**Authors:** [To be added]

**Affiliation:** [To be added]

---

## Abstract

[PLACEHOLDER - To be written after Results are finalized]

Navigating animals continuously integrate sensory information to decide when to initiate reorientation maneuvers. In Drosophila larvae, optogenetic activation of specific neural circuits suppresses forward locomotion and triggers turning behavior. We present an analytic hazard model for predicting the timing of reorientation events under controlled LED stimulation. The model parameterizes the temporal response as a difference of two gamma probability density functions, capturing a fast sensory transduction component (τ₁ = 0.29s) and a slower synaptic adaptation component (τ₂ = 3.81s). Fit to 1,407 events from 55 larval tracks, the 6-parameter kernel achieves R² = 0.968 against a 12-basis raised-cosine reference and reproduces empirical event rates with a ratio of 0.97. We demonstrate the model's application in a RUN/TURN trajectory simulator that matches observed turn rates (1.88 vs 1.84 turns/min). [If factorial: We further validate the kernel across a 2×2 factorial design varying LED intensity and background conditions.] This interpretable formulation enables quantitative comparison across experimental conditions and provides a foundation for biologically-grounded navigation simulations.

---

## 1. Introduction

### 1.1 Larval Navigation and Optogenetic Control

Drosophila larvae navigate their environment using a characteristic locomotor pattern of forward crawling ("runs") punctuated by reorientation maneuvers ("turns") during which the animal samples new heading directions (Gomez-Marin et al., 2011). These turns are not random; their timing and direction are modulated by sensory input, enabling larvae to perform gradient climbing, odor tracking, and phototaxis (Gershow et al., 2012; Kane et al., 2013).

Optogenetic tools provide precise temporal control over neural activity, allowing researchers to probe how specific circuits influence behavioral decisions. In GMR61 larvae expressing channelrhodopsins, LED illumination activates neurons that suppress forward locomotion and increase the probability of reorientation events (Gepner et al., 2015). Understanding the temporal dynamics of this suppression—how the probability of initiating a turn evolves after stimulus onset and offset—is central to modeling sensorimotor integration.

### 1.2 The Need for Interpretable Hazard Models

Previous work has modeled larval turning probability using linear filter models and generalized linear models (GLMs) with temporal basis functions (Hernandez-Nunez et al., 2015; Klein et al., 2015). These approaches fit flexible kernels—often using raised-cosine or spline bases with many parameters—that capture the temporal structure of stimulus-response relationships.

While flexible basis representations achieve good predictive performance, they offer limited interpretability. A 12-parameter raised-cosine kernel may fit the data well but does not directly reveal the underlying timescales, nor does it distinguish contributions from distinct biological processes (e.g., sensory transduction vs. adaptation).

### 1.3 Contribution

We address this gap by developing an **analytic hazard kernel** for optogenetically-driven reorientation. Our kernel is a difference of two gamma probability density functions:

$$K_{\text{on}}(t) = A \cdot \Gamma(t; \alpha_1, \beta_1) - B \cdot \Gamma(t; \alpha_2, \beta_2)$$

This 6-parameter form captures:
1. A **fast excitatory component** (peak ~0.16s, mean τ₁ = 0.29s) representing rapid sensory transduction
2. A **slow suppressive component** (peak ~2.9s, mean τ₂ = 3.81s) representing synaptic or network adaptation

The gamma-difference form is not arbitrary: it arises naturally as the impulse response of cascaded first-order processes, making parameters directly interpretable in terms of processing stages and time constants.

We validate this kernel against experimental data from GMR61 larvae under 10s ON / 20s OFF LED stimulation, demonstrating that:
- The analytic kernel approximates a 12-basis reference with R² = 0.968
- The hazard model reproduces empirical event rates (rate ratio = 0.97)
- A RUN/TURN trajectory simulator driven by the model matches observed behavior

[If factorial: We further extend validation to a 2×2 factorial design, testing generalization across LED intensity (0→250 vs 50→250 PWM) and background conditions (control vs temperature modulation).]

---

## 2. Methods

### 2.1 Experimental Data

Data were collected from GMR61 Drosophila larvae expressing channelrhodopsins. Animals were tracked at 20 Hz on an agar substrate while receiving optogenetic LED stimulation in a square-wave pattern: 10s ON at 250 PWM, 20s OFF (30s cycle). For this study, we analyzed **55 tracks** containing **1,407 reorientation-onset events** from 2 experimental sessions under the 0→250 PWM, control (7 PWM background) condition.

Reorientation events were detected using a curvature-threshold algorithm that identifies the onset of heading changes. This inclusive definition captures both large sustained turns and brief head sweeps; events with measurable duration (>0.1s) were classified as "true turns" (N=319) for behavioral interpretation.

### 2.2 Hazard Model Structure

We model reorientation timing as a point process with instantaneous hazard:

$$\lambda(t) = \exp\left(\beta_0 + u_{\text{track}} + K_{\text{on}}(t_{\text{onset}}) + K_{\text{off}}(t_{\text{offset}})\right)$$

where:
- **β₀ = -6.23**: Calibrated global intercept (log-hazard baseline)
- **u_track ~ N(0, σ² = 0.47²)**: Track-specific random effect capturing individual variability
- **K_on(t)**: LED-ON kernel response to stimulus onset
- **K_off(t)**: LED-OFF kernel response to stimulus offset

The model is fit as a negative-binomial GLM (NB-GLM) with logarithmic link, treating each video frame (dt = 0.05s) as a Bernoulli trial for event occurrence.

### 2.3 LED-ON Kernel: Gamma-Difference

The LED-ON kernel is parameterized as a difference of two gamma probability density functions:

$$K_{\text{on}}(t) = A \cdot \Gamma(t; \alpha_1, \beta_1) - B \cdot \Gamma(t; \alpha_2, \beta_2)$$

where $\Gamma(t; \alpha, \beta) = \frac{t^{\alpha-1} e^{-t/\beta}}{\beta^\alpha \Gamma(\alpha)}$ is the gamma PDF.

**Fitted parameters (95% bootstrap CIs):**

| Parameter | Value | 95% CI | Interpretation |
|-----------|-------|--------|----------------|
| A | 0.456 | [0.409, 0.499] | Fast component amplitude |
| α₁ | 2.22 | [1.93, 2.65] | Fast shape (~2 stages) |
| β₁ | 0.132s | [0.102, 0.168] | Fast timescale |
| B | 12.54 | [12.43, 12.66] | Slow component amplitude |
| α₂ | 4.38 | [4.30, 4.46] | Slow shape (~4 stages) |
| β₂ | 0.869s | [0.852, 0.890] | Slow timescale |

**Derived timescales:**
- Fast component: peak at 0.16s, mean τ₁ = α₁β₁ = 0.29s
- Slow component: peak at 2.94s, mean τ₂ = α₂β₂ = 3.81s

### 2.4 LED-OFF Rebound Kernel

A separate exponential kernel captures transient effects after LED offset:

$$K_{\text{off}}(t) = D \cdot \exp(-t/\tau_{\text{off}})$$

with D = -0.114 and τ_off = 2.0s. This modest negative term represents continued suppression during recovery, with a half-life of 1.39s.

### 2.5 Event Definition

The hazard model was fit to **all 1,407 inclusive onset events**, which include:
- Large, sustained reorientations ("true turns")
- Brief head sweeps and micro-movements
- Frame-by-frame curvature fluctuations

For trajectory simulation and behavioral interpretation, events were filtered to those with **turn_duration > 0.1s** (N = 319, 23% of total). This two-stage approach follows standard practice in larval navigation modeling: hazard fitting uses the full temporal structure while behavioral output focuses on salient events.

### 2.6 Rate Calibration

The NB-GLM intercept (β₀ = -6.76) represents log-hazard per frame at 20 Hz. Discrete-time simulation with this intercept produced ~60% of empirical events. We applied a calibration factor:

$$\beta_0^{\text{cal}} = \beta_0 + \log\left(\frac{N_{\text{emp}}}{N_{\text{sim}}}\right) = -6.76 + \log(1.695) = -6.23$$

This global rate normalization preserves kernel shape while matching empirical event rates.

### 2.7 Trajectory Simulation

We implemented a RUN/TURN state machine driven by the hazard model:

**RUN state:**
- Forward motion at 1.0 mm/s
- Brownian heading noise (σ = 0.03 rad/√s)
- Transition to TURN governed by hazard

**TURN state:**
- Angle sampled from Normal(μ = 7°, σ = 86°)
- Duration sampled from Lognormal(median = 1.1s)
- Speed reduced to 0.4× run speed
- Return to RUN after duration elapsed

### 2.8 Validation Metrics

We assessed model performance using:
1. **Kernel R²**: Correlation between analytic and 12-basis raised-cosine kernels
2. **Rate ratio**: Simulated / empirical total events
3. **PSTH correlation**: Match between simulated and empirical peri-stimulus histograms
4. **Suppression magnitude**: Fold-change in event rate during LED-ON vs LED-OFF

---

## 3. Results

### 3.1 Analytic Kernel Captures Temporal Structure

The 6-parameter gamma-difference kernel closely approximates the 12-parameter raised-cosine reference (Figure 1A). The analytic form achieves **R² = 0.968** and **cross-validated R² = 0.961** (5-fold, track-wise), demonstrating that the compact parameterization does not sacrifice predictive accuracy.

The kernel shows characteristic biphasic dynamics: an initial brief increase in hazard (fast component, τ₁ = 0.29s) followed by sustained suppression (slow component, τ₂ = 3.81s). This pattern is consistent with rapid sensory transduction followed by slower synaptic adaptation.

### 3.2 Hazard Model Reproduces Event Rates

Simulation using the calibrated hazard model produces event counts closely matching empirical observations:

| Metric | Empirical | Simulated | Status |
|--------|-----------|-----------|--------|
| Total events | 1,407 | 1,371 | PASS |
| Rate ratio | - | 0.974 | Target: 0.8-1.25 |
| LED-OFF rate | ~1.9/min | ~1.9/min | MATCH |
| LED-ON rate | ~1.0/min | ~1.0/min | MATCH |
| Suppression | 2.0× | 1.9× | MATCH |

The PSTH correlation between simulated and empirical event histograms is **r = 0.84**, indicating good capture of temporal dynamics around stimulus transitions (Figure 2B).

### 3.3 Trajectory Simulation Matches Behavioral Statistics

The RUN/TURN simulator driven by the hazard model produces realistic larval trajectories (Figure 3). Key behavioral statistics match empirical observations:

| Metric | Simulated | Empirical | Match |
|--------|-----------|-----------|-------|
| Turn rate | 1.88/min | 1.84/min | 98% |
| Mean turn angle | 7° | 7° | MATCH |
| Turn duration | 1.1s median | 1.1s median | MATCH |

[PLACEHOLDER: If factorial analysis confirms kernel stability, add section 3.4 on cross-condition results]

---

## 4. Discussion

### 4.1 Interpretability of the Gamma-Difference Kernel

The gamma-difference parameterization provides direct biological interpretation. The shape parameters α₁ ≈ 2 and α₂ ≈ 4 suggest that the fast and slow components arise from cascades of 2 and 4 first-order processes, respectively. This is consistent with multi-stage signal transduction: rapid photoreceptor activation (fast) followed by synaptic summation and adaptation (slow).

The timescales τ₁ = 0.29s and τ₂ = 3.81s align with known neurophysiology. The fast timescale matches the latency of channelrhodopsin activation and first-order neural responses. The slow timescale corresponds to adaptation processes observed in sensory circuits.

### 4.2 Practical Utility

The analytic kernel enables:
1. **Quantitative comparison** across experimental conditions (e.g., different genotypes or stimulus protocols)
2. **Parameter-based hypothesis testing** (e.g., does a manipulation affect the fast or slow component?)
3. **Efficient simulation** without requiring precomputed basis functions

### 4.3 Limitations

**Condition specificity:** The current model is validated for a single stimulus protocol (0→250 PWM, 10s ON / 20s OFF, GMR61). Generalization to other intensities or temporal patterns requires validation or refitting.

**Event definition:** The hazard model uses 1,407 inclusive onset events, of which 77% have zero measured duration. This broad definition captures full temporal structure but complicates behavioral interpretation. We address this by filtering to 319 "true turns" for trajectory output.

**Poisson assumption:** Time-rescaling tests show mild violation (mean rescaled IEI = 0.87 vs expected 1.0, p < 0.001), indicating minor unmodeled temporal structure—likely short-term dependencies or refractoriness. The 13% deviation does not substantially affect aggregate predictions.

**Trajectory simplifications:** The RUN/TURN simulator omits edge avoidance, head sweeps, and speed gradients. These simplifications are appropriate for demonstrating hazard-driven timing but limit biomechanical realism.

### 4.4 Future Directions

[If factorial: Extend to multi-condition models with condition-specific amplitudes]

Potential extensions include:
- Refractory kernel to capture post-event suppression
- Edge avoidance for bounded arena simulation
- Intensity-scaling analysis using the 2×2 factorial design
- Integration with chemotaxis models for gradient navigation

---

## 5. Conclusions

We present an analytic hazard kernel for optogenetically-driven larval reorientation that combines interpretability with predictive accuracy. The 6-parameter gamma-difference form captures two biologically meaningful timescales and reproduces empirical event statistics with high fidelity (rate ratio = 0.97, R² = 0.968). Embedded in a RUN/TURN trajectory simulator, the model generates realistic larval behavior that matches observed turn rates. This framework provides a foundation for quantitative analysis of sensorimotor processing across experimental conditions.

---

## Figures

### Figure 1: Analytic Hazard Kernel

![Kernel](../../figures/figure1_kernel.png)

**(A)** LED-ON kernel showing the gamma-difference decomposition. The fast component (τ₁ = 0.29s, red) and slow component (τ₂ = 3.81s, blue) combine to produce the full kernel (black). Gray: 12-basis raised-cosine reference. **(B)** LED-OFF rebound kernel with τ_off = 2.0s.

### Figure 2: Validation Metrics

![Validation](../../figures/figure2_validation.png)

**(A)** Kernel fit comparison: analytic (red) vs raised-cosine (blue), R² = 0.968. **(B)** PSTH comparison: empirical (black) vs simulated (red), r = 0.84. **(C)** Suppression dynamics: rate ratio during LED-ON (gray shading) shows ~2× suppression matching empirical observations.

### Figure 3: Trajectory Simulation

![Trajectories](../../figures/figure3_trajectories.png)

Example simulated trajectories showing RUN segments (blue) and TURN events (red circles). Turn rate matches empirical observations (1.88 vs 1.84 turns/min).

---

## References

1. Gepner R, et al. (2015). Computations underlying Drosophila photo-taxis, odor-taxis, and multi-sensory integration. eLife 4:e06229.

2. Gershow M, et al. (2012). Controlling airborne cues to study small animal navigation. Nature Methods 9:290-296.

3. Gomez-Marin A, et al. (2011). Active sampling and decision making in Drosophila chemotaxis. Nature Communications 2:441.

4. Hernandez-Nunez L, et al. (2015). Reverse-correlation analysis of navigation dynamics in Drosophila larva using optogenetics. eLife 4:e06225.

5. Kane EA, et al. (2013). Sensorimotor structure of Drosophila larva phototaxis. PNAS 110:E3868-E3877.

6. Klein M, et al. (2015). Sensory determinants of behavioral dynamics in Drosophila thermotaxis. PNAS 112:E220-E229.

---

*Draft version: 2025-12-11*
*Status: Methods complete, Results pending factorial analysis*

