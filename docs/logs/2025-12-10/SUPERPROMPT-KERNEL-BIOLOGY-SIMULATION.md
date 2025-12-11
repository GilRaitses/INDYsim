# Deep Research Superprompt: Kernel Interpretation, Biological Validation, and Simulation Fidelity

## Context

We have fit a Negative Binomial GLM hazard model to Drosophila larval reorientation behavior during optogenetic stimulation. The model uses raised-cosine temporal basis functions to capture stimulus-locked response dynamics.

### Current Findings

**Fitted Model (14 experiments, 874,711 bins, 7,867 events):**

| Feature | Rate Ratio | % Change | p-value |
|---------|-----------|----------|---------|
| LED1_scaled | 0.31 | -69% | 0.034 |
| kernel_1 (0-0.75s) | 7.31 | +631% | <0.001 |
| kernel_2 (0.75-1.5s) | 1.08 | +8% | 0.677 |
| kernel_3 (1.5-2.25s) | 0.29 | -71% | 0.008 |
| kernel_4 (2.25-3.0s) | 0.09 | -91% | <0.001 |

**Cluster Analysis (k=4):**
- Non-responders (25.3%): -69% response
- Responders (56.7%): +74% response, 86s latency
- Fast responders (12.4%): +65% response, 36s latency
- Delayed responders (5.6%): +68% response, 482s latency

---

## Research Questions

### Section 1: Sensory-Motor Latency Interpretation

**Q1.1:** What is the expected sensory-motor delay for Drosophila larvae responding to optogenetic stimulation?
- Include photoreceptor → CNS → motor output pathway timing
- Compare rhodopsin-based (ChR2) vs thermogenetic (dTRPA1) systems
- Report typical latencies in milliseconds from published studies

**Q1.2:** Our kernel shows +631% spike in the 0-0.75s window. Is this consistent with:
a) Direct sensory response?
b) Startle reflex?
c) Refractory period artifact (high prior probability of event)?
d) Something else?

**Q1.3:** The kernel shows strong suppression at 1.5-3.0s (-71% to -91%). What biological mechanisms could explain post-stimulus response suppression?
- Sensory adaptation
- Motor fatigue
- Inhibitory circuit engagement
- Habituation

---

### Section 2: Published Kernel Comparisons

**Q2.1:** Mason Klein's lab has published hazard/kernel models for Drosophila larval behavior. Please find and summarize:
- Peak latency values from their published kernels
- Kernel shape (monotonic decline vs. biphasic)
- Stimulus types used (thermal, optogenetic, chemosensory)

**Q2.2:** How does our fitted kernel (peak at 0-0.75s, suppression at 1.5-3.0s) compare to published results? Specifically:
- Are there discrepancies that suggest model misspecification?
- What would we expect for GMR61-Gal4 driver line (thermosensory)?

**Q2.3:** What alternative kernel parameterizations are used in the literature?
- Log-Gaussian kernels
- Alpha functions
- Double exponential
- Non-parametric (spline-based)

---

### Section 3: Responder/Non-Responder Phenotypes

**Q3.1:** Our cluster analysis identified 25% "non-responders" showing -69% turn rate during stimulation. Is this:
a) True non-response (genetic/circuit failure)?
b) Active suppression (freezing behavior)?
c) Ceiling effect (already at max turn rate)?

**Q3.2:** Are there known genetic variants or circuit manipulations that produce responder vs non-responder phenotypes in Drosophila larvae?
- Report specific Gal4 lines
- Relevant publications

**Q3.3:** The "delayed responder" cluster (5.6%, latency ~480s) shows very long response delays. What biological explanations exist for such long latencies?
- State-dependent modulation
- Circadian effects
- Developmental stage differences

---

### Section 4: Simulation Fidelity

**Q4.1:** For event-driven simulation using hazard functions, what validation metrics are standard in computational neuroscience?
- ISI distribution comparisons
- Auto-correlation structure
- Higher-order statistics (burst patterns)

**Q4.2:** Our model treats bins as independent given covariates, but ACF lag-1 = 0.999. What modifications enable modeling temporal autocorrelation?
- Autoregressive hazard models
- Hidden Markov models
- Gaussian Process modulated point processes

**Q4.3:** If we want to simulate untested conditions (e.g., LED1 at 50 PWM instead of 250 PWM), what extrapolation concerns apply?
- Linear vs. nonlinear dose-response
- Saturation effects
- Threshold effects

---

### Section 5: Mixed-Effects and Hierarchical Structure

**Q5.1:** Our data has hierarchical structure: frames within tracks within experiments. We used cluster-robust SEs, but didn't fit random effects. What are the trade-offs?
- Cluster-robust SE vs. mixed-effects models
- When does ignoring hierarchy lead to incorrect inference?

**Q5.2:** Should we model track-level random intercepts? What biological variation would this capture?

**Q5.3:** For simulation purposes, how would we generate synthetic "experiments" that maintain realistic between-experiment variance?

---

### Section 6: Alternative Statistical Approaches

**Q6.1:** The Negative Binomial assumes a specific mean-variance relationship. What alternatives exist for overdispersed count data with temporal structure?
- Poisson-Inverse Gaussian
- Conway-Maxwell-Poisson
- Zero-inflated models (we checked: no zero-inflation detected)

**Q6.2:** Are there concerns about using GLM for rare events (0.9% event rate per bin)? Should we consider:
- Exact logistic regression
- Firth's penalized likelihood
- Bayesian approaches

**Q6.3:** What are best practices for model selection when comparing kernel parameterizations?
- Cross-validation (we implemented LOEO)
- Information criteria (AIC/BIC)
- Posterior predictive checks

---

## Deliverables Requested

1. **Latency Reference Table:** Published sensory-motor latencies for Drosophila larvae across stimulus modalities (5-10 papers)

2. **Kernel Comparison:** Side-by-side comparison of our kernel to Mason Klein's published kernels (figure description or summary)

3. **Phenotype Genetics:** List of Gal4 lines associated with responder/non-responder phenotypes

4. **Simulation Recommendations:** Specific model modifications to improve temporal autocorrelation fidelity

5. **Extrapolation Guidance:** Practical rules for when hazard model extrapolation is/isn't appropriate

---

## Formatting Requirements

- Use numbered responses matching question numbers (Q1.1, Q1.2, etc.)
- Include specific citations where possible (Author, Year, Journal)
- For numerical values, include uncertainty ranges if available
- Flag speculative vs. literature-supported answers clearly

---

## Background References (for context, not exhaustive)

- Klein M, Afonso B, Vonner AJ, et al. (2015) "Sensory determinants of behavioral dynamics in Drosophila thermotaxis." PNAS
- Gershow M, Berck M, Mathew D, et al. (2012) "Controlling airborne cues to study small animal navigation." Nature Methods
- Gepner R, Skanata MM, Bernat NM, et al. (2015) "Computations underlying Drosophila photo-taxis, odor-taxis, and multi-sensory integration." eLife
- Louis M (2019) "Sensorimotor strategies for navigation in Drosophila larvae." J Comp Physiol A
