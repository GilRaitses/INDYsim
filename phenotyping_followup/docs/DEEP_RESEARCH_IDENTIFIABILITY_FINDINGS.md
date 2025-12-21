# Deep Research Findings: Kernel Identifiability Analysis

**Source**: Deep Research Agent  
**Date**: 2025-12-18  
**Query**: Alternative approaches to kernel fitting given structural identifiability issues

---

## Executive Summary

The research agent **confirms our diagnostic conclusion**: the original 6-parameter gamma-difference kernel is **not suited for individual-level inference** under the current experimental design. This is a **structural identifiability problem**, not merely a sample size issue.

---

## Key Findings by Question

### 1. Alternative Kernel Parameterizations

**Recommended strategies (in order of practicality):**

| Strategy | Description | Implementation |
|----------|-------------|----------------|
| Fix τ₂, B at population | Treat slow inhibitory component as population-only | Hierarchical model with τ₁ random effects only |
| Constrain B/A ratio | Fix r = B/A ≈ 8 or use tight prior | Reduces effective parameters per individual |
| Single-gamma kernel | K(t) = A·Γ(t; α, τ₁/α) for individuals | Encode inhibition in baseline instead |

**Key insight**: With ~2 ON-events per track, even estimating τ₁ alone per individual is challenging; estimating both τ₁ and A/B is unrealistic.

### 2. Experimental Design Changes

**Current design problems:**
- 10s ON / 20s OFF → 33% duty cycle
- Strong inhibition (B >> A) → only ~20% of events in LED-ON
- ~2 informative events per larva

**Recommended changes:**

| Design Change | Example | Expected Improvement |
|---------------|---------|---------------------|
| Higher duty cycle | 5s ON / 5s OFF (50%) | Double ON-event fraction |
| Shorter cycles, more repeats | 3s ON / 3s OFF × 50 cycles | More independent "trials" |
| Pulse trains | Pairs with 0.5-1.0s spacing | Directly probes τ₁ recovery |

**Critical**: Longer tracks with same stimulus schedule will NOT help. The ratio of informative to uninformative events stays constant.

### 3. Alternative Statistical Approaches

**Phenotyping without individual kernels:**

1. **ON/OFF event rate ratio**: R = (events_ON/T_ON) / (events_OFF/T_OFF)
2. **First-event latency**: Median latency after LED onset per larva
3. **Non-parametric hazard**: Group-level hazard estimation, then per-larva deviation scores
4. **Feature-based classification**: Regularized logistic regression on summary features

**Most promising**: ON/OFF rate ratio is a 1-D statistic with reasonable power even with ~2 ON-events.

### 4. Hierarchical Models Interpretation

**Our interpretation is CORRECT**: 91% shrinkage to population mean is **strong evidence that data do not support meaningful individual τ₁ variation**.

- Shrinkage is diagnostic of low per-individual information
- Loosening priors would create prior-driven artifacts, not reveal real variation
- Fast-responder candidates (~8%) still have very low data support

### 5. Minimum Data Requirements

**For individual 6-parameter kernel fitting:**
- Need ~60 informative ON-events per larva
- With only 20% of events in ON window → need ~300 total events per larva
- Current data: ~18-25 total events (~4 ON-events)
- **Gap**: 12-15× more data needed

**Conclusion**: Individual-level inference is not realistically identifiable with current design.

---

## Manuscript-Ready Conclusion

> The original 6-parameter gamma-difference kernel is not suited for individual-level inference under the current experimental design. The combination of strong inhibition (B >> A), sparse ON-events (~2 per track), and a 6-parameter model yields a nearly flat likelihood and non-identifiable τ₁. Population-level estimation is robust and publishable. Individual phenotyping requires both a simpler model AND a redesigned experiment.

---

## References from Research Agent

| # | Citation | Topic |
|---|----------|-------|
| 1 | Daley & Vere-Jones (2003). An Introduction to the Theory of Point Processes | Point process identifiability theory |
| 2 | Marcon (2011). Gamma Kernel Intensity Estimation | Gamma kernel methods |
| 3 | Heckman (1984). The Identifiability of the Proportional Hazard Model | Identifiability conditions |
| 4 | Rebora et al. (2014). bshazard: Nonparametric Smoothing | Non-parametric hazard estimation |
| 5 | Bernabeu et al. (2025). Spatio-Temporal Hawkes Point Processes | Experimental design for point processes |
| 6 | Sandler et al. (2014). System Identification of Point-Process Neural Systems | Volterra kernel estimation |
| 7 | Wang & Eubank (1996). Hazard Rate Regression | Non-parametric methods |
| 8 | Wang (1999). Smoothing Hazard Rates | Hazard smoothing |
| 9 | Du et al. (2016). Recurrent Marked Temporal Point Processes | Learned embeddings |
| 10 | Gelman & Hill (2006). Data Analysis Using Hierarchical Models | Shrinkage interpretation |
| 11 | Tang et al. (2021). Multivariate Temporal Point Process Regression | Modern point process methods |
| 12 | Salehi et al. (2019). Learning Hawkes Processes from a Handful of Events | Sparse data methods |
| 13 | Li & Li (2018). Convergence Rates of Kernel Estimator | Theoretical foundations |

---

## Implications for Manuscript

### What to Keep
- Population-level kernel estimation (robust, publishable)
- Leave-one-experiment-out cross-validation results
- Hierarchical Bayesian analysis showing shrinkage

### What to Reframe
- Individual phenotyping as "exploratory hypothesis-generating"
- Fast-responder candidates as "pending confirmation with new data"
- The paper as a "methodological cautionary tale and strong null result"

### What to Add
- Fisher information analysis showing near-singularity
- Discussion of alternative parameterizations for future work
- Concrete experimental design recommendations
- Alternative phenotyping approaches (ON/OFF ratio, latency)

---

## Action Items

1. [ ] Update Discussion section with identifiability analysis
2. [ ] Add references to bibliography
3. [ ] Create Figure showing Fisher information / likelihood surface
4. [ ] Add experimental design recommendations for future work
5. [ ] Document ON/OFF ratio as alternative phenotype metric

