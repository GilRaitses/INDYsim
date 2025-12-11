# Comprehensive Research Prompt: Simulation Fidelity, Prediction, and Biological Interpretation

## Context

We have built a Negative Binomial GLM hazard model for Drosophila larval reorientation behavior during optogenetic stimulation. The model includes:
- Raised-cosine temporal kernel (3 bases, 0-4s window)
- AR(1) term for refractory effects
- Cluster-robust standard errors (14 experiments)
- LED1/LED2 covariates

### Current Model Performance

| Metric | Value |
|--------|-------|
| AIC | 88,735 |
| Dispersion ratio | 0.998 |
| Residual ACF lag-1 | -0.0002 |

### Kernel Coefficients

| Time Window | Rate Ratio | Interpretation |
|-------------|-----------|----------------|
| 0-1.33s | 7.31 (+631%) | Strong early spike |
| 1.33-2.67s | 0.25 (-75%) | Suppression |
| 2.67-4.0s | 0.06 (-94%) | Deep suppression |

### Identified Gaps

1. **Simulation rate mismatch:** Simulated 0.06 vs empirical 0.87 events/min/track
2. **Kernel-stimulus alignment:** Kernel computed from absolute time, not LED onset
3. **Limited DOE:** All experiments use LED1=250 PWM (high only)
4. **Delayed responders:** 5.6% of larvae show 480s latency
5. **No dose-response model:** Can't predict intermediate LED intensities
6. **PSTH validation missing:** Haven't compared stimulus-locked histograms

---

## Section 1: Simulation Fidelity

### Q1.1 Rate Mismatch Diagnosis

Our hazard-based simulation generates events at 0.06 events/min/track, but empirical data shows 0.87 events/min/track (14x difference).

**Questions:**
a) What are common causes of rate mismatch in point-process simulations?
b) Could the AR(1) term (coefficient -24.7, meaning RR=0 for bins following events) be suppressing too many events?
c) How should we handle the refractory period in continuous-time simulation vs. binned GLM fitting?

### Q1.2 Kernel-Stimulus Alignment

Currently, our kernel bases are centered at fixed times (0, 1.33, 2.67s) from simulation start, not from LED onset.

**Questions:**
a) How do standard implementations align temporal kernels to stimulus onsets in cyclic stimulus protocols?
b) For a 30s-on/30s-off LED cycle, should the kernel "reset" at each onset?
c) What is the standard approach: one kernel per cycle, or cumulative history?

### Q1.3 Validation Metrics Beyond ISI

We currently validate with:
- Turn rate comparison (mean ± CI)
- ISI/IEI distribution (KS test)

**Questions:**
a) What additional validation metrics are standard for behavioral point-process simulations?
b) How should we compute and compare peri-stimulus time histograms (PSTHs)?
c) What is an acceptable integrated squared error (ISE) for PSTH match?

---

## Section 2: Dose-Response Modeling

### Q2.1 Functional Forms

Our data includes only LED1=0 and LED1=250 PWM. We want to predict behavior at intermediate values (e.g., 50, 100 PWM).

**Questions:**
a) What functional forms are appropriate for optogenetic dose-response?
   - Linear: effect = β × LED1
   - Log-linear: effect = β × log(1 + LED1)
   - Sigmoid: effect = β_max / (1 + exp(-k(LED1 - LED50)))
b) Can we estimate dose-response parameters from binary (0/max) data?
c) What priors or constraints should we use?

### Q2.2 Saturation and Threshold Effects

**Questions:**
a) At what LED intensity do CsChrimson-expressing neurons typically saturate?
b) Is there a threshold below which no behavioral effect occurs?
c) How do we detect saturation from data with only one non-zero intensity?

### Q2.3 Extrapolation Bounds

**Questions:**
a) How far outside the training range can we safely extrapolate?
b) Should we report prediction intervals that widen with extrapolation distance?
c) What experimental validation would confirm extrapolation accuracy?

---

## Section 3: Delayed Responder Biology

### Q3.1 State-Dependent Modulation

Our cluster analysis found 5.6% of larvae with ~480s (8 min) latency to first stimulus response.

**Questions:**
a) What internal states could suppress stimulus responsiveness for minutes?
   - Satiety/hunger
   - Arousal/quiescence
   - Prior habituation
b) What neuromodulatory systems operate on minute timescales in larvae?
c) Are there behavioral signatures that distinguish "dormant" vs "suppressed" larvae?

### Q3.2 Modeling State Transitions

**Questions:**
a) How would we add a hidden "responsive state" to our hazard model?
b) What is the standard approach: HMM, switching GLM, or latent factor?
c) Can we estimate state transition rates from behavioral data alone?

### Q3.3 Biological Markers

**Questions:**
a) Are there genetic backgrounds associated with delayed responsiveness?
b) Could developmental stage (L2 vs L3) explain the 5.6% delayed cluster?
c) What experimental manipulations could test state-dependence hypotheses?

---

## Section 4: GMR61-Gal4 and Basin Neuron Biology

### Q4.1 Circuit Specifics

Our experiments use GMR61-Gal4 > CsChrimson.

**Questions:**
a) What neurons does GMR61-Gal4 label?
b) Are Basin neurons (mechanosensory interneurons) included in this expression pattern?
c) What is the expected behavioral output of Basin neuron activation?

### Q4.2 Published Characterization

**Questions:**
a) What behavioral phenotypes have been reported for GMR61-Gal4 activation?
b) Are there dose-response curves for this specific driver?
c) What is the typical response latency for GMR61-mediated behaviors?

### Q4.3 Circuit Interactions

**Questions:**
a) Do GMR61-labeled neurons interact with thermosensory or nociceptive pathways?
b) Could the strong suppression (RR 0.06 at 2.7-4s) reflect inhibitory feedback?
c) Are there known downstream targets that explain the behavioral refractory period?

---

## Section 5: Experimental Design Recommendations

### Q5.1 Factorial Design

To estimate LED1 main effects and LED1 x LED2 interactions, we need multiple LED1 levels.

**Questions:**
a) What is the minimum number of LED1 levels for dose-response estimation?
b) For a 2³ factorial (LED1 × LED2 × duration), how many larvae per condition?
c) What effect size (Cohen's d) should we power for?

### Q5.2 Power Analysis

**Questions:**
a) Given our observed between-track variance (SD = 0.97 events/min), what n is needed to detect a 0.5 events/min difference with 80% power?
b) How does clustering (larvae within experiments) affect required sample size?
c) Should we use a mixed-effects power calculation?

### Q5.3 Protocol Optimization

**Questions:**
a) What stimulus timing (on/off duration) maximizes effect detectability?
b) Should we use blocked or interleaved designs?
c) What is the recommended experiment duration to capture delayed responders?

---

## Section 6: Model Refinements

### Q6.1 Mixed-Effects Extension

We currently use cluster-robust SEs. A full mixed-effects model would add:
- Random intercepts by track
- Random intercepts by experiment
- Potentially random slopes for LED effects

**Questions:**
a) What R/Python packages support NB-GLMM with random effects?
b) How do we interpret random effect variances for simulation?
c) Is the complexity justified given our sample size (14 experiments, 701 tracks)?

### Q6.2 Alternative Kernel Parameterizations

We used raised-cosine bases. Alternatives include:
- B-splines
- Double exponential
- Log-Gaussian

**Questions:**
a) Which parameterization is most robust to bin width choice?
b) How do we test for biphasic vs. monophasic kernel shapes?
c) Should we allow asymmetric kernels (fast rise, slow decay)?

### Q6.3 Zero-Inflation Revisited

Our diagnostics showed no zero-inflation (observed zeros = expected zeros within 0.02%). However:

**Questions:**
a) Could the AR(1) term be absorbing what would otherwise appear as zero-inflation?
b) Should we test a ZINB model as a robustness check?
c) What would zero-inflation mean biologically (complete non-response vs. low rate)?

---

## Deliverables Requested

1. **Simulation Debugging Checklist:** Step-by-step diagnosis of rate mismatch
2. **Dose-Response Recommendations:** Functional form + estimation approach
3. **State-Switching Model Sketch:** How to add latent "responsive" state
4. **GMR61 Literature Summary:** 3-5 key papers on this driver
5. **Experimental Design Table:** LED levels, n per condition, expected power
6. **Model Comparison Guide:** NB-GLM vs NB-GLMM vs alternatives

---

## Formatting Requirements

- Number responses to match question numbers (Q1.1a, Q1.1b, etc.)
- Include citations where available (Author, Year, Journal)
- Flag speculative vs. literature-supported answers
- Provide concrete parameter values or ranges where possible

---

## Background Data Summary

For reference, here are key statistics from our analysis:

**Empirical Data:**
- 14 experiments, 701 tracks, 8.5M frames
- 7,867 reorientation onsets
- Mean turn rate: 0.87 ± 0.97 events/min/track
- LED1: 0 or 250 PWM (30s on/30s off cycles)
- LED2: 7 PWM constant

**Cluster Analysis:**
- k=4 optimal (silhouette 0.394)
- Non-responders: 25% (-69% response)
- Responders: 57% (+74% response)
- Fast responders: 12% (+65% response, 36s latency)
- Delayed responders: 6% (+68% response, 480s latency)

**Model:**
- NB-GLM with AR(1), cluster-robust SE
- 3 kernel bases, 0-4s window
- Y_lag1 coefficient: -24.7 (RR ≈ 0)
