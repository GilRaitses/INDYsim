# Kernel Biology Research Findings

**Source:** MiroThinker external research agent  
**Date:** 2025-12-10  
**Grade:** B+/A-

---

## 1. Sensory-Motor Latency Reference Table

| Modality | Reported Latency | Notes | Reference |
|----------|------------------|-------|-----------|
| Optogenetic nociceptive (ChR2/CsChrimson) | 30-50 ms | Startle-like rolling escape | Honjo et al. 2012, PMC3751580 |
| Optogenetic ORN/photoreceptor | 0.6-1.0 s peak | Navigational decision latency | Gepner et al. 2015, eLife |
| Visual phototaxis (Bolwig's organ) | ~1 s delay | Return to baseline 2-4 s | Kane et al. 2013, PMC3791751 |
| Thermal nociception (probe) | 1-10 s | Distribution, not instantaneous | Bio-protocol 2737 |
| Thermogenetic (dTRPA1) | Seconds to minutes | Limited by heating ramp | Pulver et al. 2009 |

**Key finding:** For optogenetic stimulation in freely moving larvae, expected sensory-motor latency is **0.5-1.0 s**, even though synaptic delays are tens of ms.

---

## 2. Kernel Validation

### Our Fitted Kernel
| Window | Rate Ratio | Interpretation |
|--------|-----------|----------------|
| 0-0.75s | 7.31 | +631% spike |
| 0.75-1.5s | 1.08 | Baseline |
| 1.5-2.25s | 0.29 | -71% suppression |
| 2.25-3.0s | 0.09 | -91% suppression |

### Research Validation

**Peak timing (0-0.75s):** VALIDATED
- Matches literature peak at 0.6-0.8s before turn
- Consistent with direct sensory drive, not artifact
- Compatible with startle-like component (integrated at behavioral timescales)

**Suppression (1.5-3.0s):** BIOLOGICALLY PLAUSIBLE
Four mechanisms supported by literature:
1. **Sensory adaptation** - Visual/fictive-odor kernels decay to/below baseline within 2-3s
2. **Motor refractory** - Larvae run straight after turns (behavioral refractory period)
3. **Inhibitory circuits** - GABAergic interneurons shape navigational timing
4. **Short-term habituation** - Early phase, though more parsimoniously explained by refractory dynamics

### Comparison to Klein Lab Kernels

| Aspect | Our Kernel | Klein/Gepner | Assessment |
|--------|-----------|--------------|------------|
| Peak timing | 0-0.75s | 0.6-0.8s | Match |
| Peak magnitude | RR 7.31 | Monophasic lobe | Stronger in our data |
| Suppression depth | RR 0.09-0.29 | Modest undershoot | Stronger in our data |
| Shape | Biphasic | Monophasic + smooth decay | Different |

**Interpretation:** Stronger suppression may be real (high-intensity stimulus, GMR61 Basin neurons) OR artifact of coarse 4-basis parameterization. Consider testing with more basis functions or splines.

---

## 3. Responder/Non-Responder Phenotypes

### Our Cluster Analysis
| Cluster | Label | n (%) | Response |
|---------|-------|-------|----------|
| 0 | Non-Responder | 25.3% | -69% |
| 2 | Responder | 56.7% | +74% |
| 3 | Fast Responder | 12.4% | +65% |
| 1 | Delayed Responder | 5.6% | +68% |

### Biological Interpretation of Non-Responders (-69%)

Three explanations (all literature-supported):
1. **True non-response** - Low/absent CsChrimson expression (stochastic Gal4/UAS)
2. **Active suppression** - Overstimulated larvae freeze instead of turning
3. **Ceiling effect** - Less likely given modest baseline rates

**Likely:** Mixture of low expressers + freezing behavior

### Known Gal4 Lines with Phenotype Variation

| Gal4 Driver | Target | Phenotype | Reference |
|-------------|--------|-----------|-----------|
| Or42a-Gal4, Or42b-Gal4 | Olfactory neurons | Robust responders | Gepner et al. 2015 |
| Gr21a-Gal4 | CO2-sensitive | Strong aversive (responders) | Gepner et al. 2015 |
| Class IV md-Gal4 | Nociceptive | Near 100% responders | Honjo et al. 2012 |
| sr-Gal4 | Apodemes only | Non-responder | Epidermal studies |
| Weak/off-target Gal4s | Various | Non-responders | General surveys |

### Delayed Responders (480s latency)

Possible mechanisms:
- State-dependent modulation (low arousal at start)
- Slow neuromodulatory shifts (minutes timescale)
- Developmental stage variability (L2 vs L3)
- Habituation-sensitization dynamics

**Recommendation:** Model as agents with delayed onset of stimulus sensitivity (random delay ~400-500s).

---

## 4. Simulation Recommendations

### Validation Metrics (Standard in Point-Process Literature)

| Metric | Purpose | Target |
|--------|---------|--------|
| ISI distribution | Compare empirical vs simulated IEI | KS p > 0.05 |
| ACF of event counts | Match temporal structure | Lag-1 within 0.05 |
| Fano factor | Variance/mean across windows | Match empirical |
| Burst index | Clumping vs Poisson | Qualitative match |
| PSTH | Stimulus-locked turn rate | Within 95% CI |

### Handling ACF = 0.999 (High Temporal Autocorrelation)

**Problem:** Bin-to-bin counts extremely correlated after conditioning on covariates.

**Solutions (in order of complexity):**

1. **Autoregressive Hazard (AR)** - RECOMMENDED FIRST
   - Add Y_lag1 as predictor: `log(μ_t) = ... + ρ * log(1 + Y_{t-1})`
   - Captures short-range dependencies (refractoriness, burstiness)

2. **Hidden Markov Model (HMM)**
   - Latent states: RUN, TURN-PRONE, SUPPRESSED
   - Each state has own hazard; transitions generate clustering

3. **Gaussian Process modulated point process**
   - Log-hazard as continuous random function with GP prior
   - Captures long-range correlations without explicit states

**Practical progression:** Start with AR(1), check if residual ACF drops. If structure remains, try 2-3 state HMM.

### Extrapolation Guidance (Untested Conditions)

**Safe interpolation:**
- Interpolate between tested intensities (0 and 250 PWM)
- Use nonlinear terms: LED1 + LED1^2
- Validate shape with held-out data

**Unsafe extrapolation:**
- PWM above 250 (saturation unknown)
- Novel duty cycles not tested
- Treat as hypothesis-generating only

---

## 5. Mixed-Effects Recommendations

### Cluster-Robust SE vs Mixed-Effects

| Approach | Pros | Cons | Use When |
|----------|------|------|----------|
| Cluster-robust SE | Simple, corrects SEs | No variance modeling | Quick analysis |
| Random intercepts | Captures heterogeneity | More complex | Track-level variation matters |
| Full mixed-effects | Most flexible | Computational | Publication quality |

**Recommendation:** Given evident heterogeneity (responders, non-responders, delayed), track-level random intercepts are strongly justified.

### What Track-Level Random Intercepts Capture

- Individual differences in baseline activity
- Sensitivity to LED (expression levels)
- Internal state (metabolic, neuromodulatory)
- Reduces residual autocorrelation and overdispersion

### Generating Synthetic Experiments

1. Fit NB-GLMM with experiment and track random intercepts
2. Extract σ²_exp and σ²_track from fitted model
3. For each synthetic experiment:
   - Draw u_exp ~ N(0, σ²_exp)
   - For each track: draw b_track ~ N(0, σ²_track)
   - Use in linear predictor to generate events

---

## 6. Model Selection Best Practices

### Recommended Approach

1. Grid search (n_bases, window) with LOEO CV
2. Compute AIC/BIC for each configuration
3. For top 2-3 candidates:
   - Simulate event sequences
   - Compare PSTH, IEI, ACF, Fano factors
4. Choose model that is parsimonious AND reproduces key behaviors

### NB Appropriateness

- Verified no zero-inflation (NB appropriate)
- Event rate 0.9% per bin with 874K bins = 7,867 events (adequate for GLM asymptotics)
- Exact logistic/Firth penalization not needed at this sample size

---

## References

1. Honjo et al. 2012 - PMC3751580 (Optogenetic nociception)
2. Gepner et al. 2015 - eLife PMC4466338 (Photo/odor-taxis kernels)
3. Kane et al. 2013 - PMC3791751 (Phototaxis structure)
4. Bio-protocol 2737 (Thermo-nociceptive assay)
5. Pulver et al. 2009 - J Neurophysiol (dTRPA1 dynamics)
6. Kohsaka et al. 2018 - Neural Dev (Larval locomotion circuits)
7. Clark et al. 2018 - Front Behav Neurosci (Chemical/optogenetic responses)
8. Li et al. 2014 - Cell Rep (Gal4 driver resource)
9. Louis 2019 - J Comp Physiol A (Sensorimotor strategies review)
