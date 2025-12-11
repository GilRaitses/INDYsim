# Gaps and Next Steps

## Identified Gaps

### 1. Distribution Fitting Quality
**Gap:** All K-S tests rejected (p=0.0) despite large sample sizes.
**Cause:** K-S is overpowered for N > 10,000; even tiny deviations are significant.
**Action:** Use Q-Q plots + relative fit metrics (AIC/BIC) instead of K-S p-values.

### 2. Limited DOE Structure
**Gap:** All 14 experiments use "high" LED1 (>200 PWM). Cannot estimate LED1 main effect.
**Cause:** Experimental design focused on one genotype/condition.
**Action:** Request additional experiments with low/medium LED1 intensities.

### 3. Cluster Interpretation
**Gap:** 4 clusters found but biological meaning unclear.
**Cause:** Purely statistical clustering without behavioral priors.
**Action:** Validate clusters against known behavioral phenotypes (e.g., thermotaxis, photophobia).

### 4. Kernel Peak at t=0
**Gap:** kernel_1 shows +631% spike immediately (0-0.75s window).
**Cause:** May be stimulus detection artifact vs. true behavioral response.
**Action:** Compare kernel shape to expected sensory-motor delay (~200ms in larvae).

### 5. Non-Responder Definition
**Gap:** "Non-responder" defined as <10% increase, but cluster 0 shows -69% (suppression).
**Cause:** Active suppression is different from non-response.
**Action:** Distinguish "suppressed" vs "non-responsive" phenotypes.

### 6. Simulation Validation Not Run
**Gap:** Created `validate_simulation.py` but didn't execute.
**Cause:** Need to run simulation first.
**Action:** Run event_generator → validate against empirical data.

### 7. Cross-Validation Not Run
**Gap:** `run_hazard_pipeline.py` supports `--run-cv` but wasn't used.
**Cause:** Time constraint.
**Action:** Run CV to optimize kernel parameters (n_bases, window).

---

## Next Steps (Priority Order)

### Phase 1: Model Refinement (2-3 hours)
1. **Run cross-validation** for kernel parameters
2. **Compare kernel to sensory-motor latency** literature (200-500ms expected)
3. **Refit with optimized parameters**

### Phase 2: Simulation (3-4 hours)
1. **Generate synthetic events** using fitted hazard model
2. **Run validation suite** (turn rate t-test, PSTH ISE, KS tests)
3. **Tune until validation passes**

### Phase 3: Biological Validation (research needed)
1. **Map clusters to known phenotypes** (literature search)
2. **Compare response profiles** to published data
3. **Identify genetic correlates** of responder/non-responder

### Phase 4: Experimental Design (external)
1. **Design factorial experiment** with LED1 low/medium/high
2. **Power analysis** for detecting interaction effects
3. **Protocol for new data collection**

### Phase 5: Inference Refinement (research needed)
1. **Alternative dispersion models** (Poisson-Inverse Gaussian)
2. **Mixed-effects structure** (random intercepts by experiment)
3. **Temporal autocorrelation models** (AR(1) errors)

---

## Research Questions for Deep Prompt

1. **How do Mason Klein's published kernel shapes compare to our fitted kernel?**
2. **What is the expected sensory-motor delay for Drosophila larvae?**
3. **Are there known genetic variants associated with responder/non-responder phenotypes?**
4. **What alternative GLM families handle count data with strong temporal structure?**
5. **How should we model the active suppression phenotype (cluster 0)?**

---

## Files Affected

| File | Status | Notes |
|------|--------|-------|
| `scripts/run_hazard_pipeline.py` | Ready | Add `--run-cv` flag |
| `scripts/event_generator.py` | Ready | Needs hazard function connection |
| `scripts/validate_simulation.py` | Ready | Run after event generation |
| `data/model/model_results.json` | Complete | May update after CV |
