# Research Prompt: Factorial Analysis Guidance for INDYsim

## Context

We are developing a publication-ready hazard model for larval reorientation under optogenetic stimulation. The core model is complete and validated for a single condition. We now seek guidance on extending to a 2×2 factorial design.

---

## PAPER STATUS

### Current State: Single-Condition Paper Draft Complete

| Component | Status | Quality |
|-----------|--------|---------|
| Abstract | Complete | Ready for review |
| Introduction | Complete | 3 subsections, clear gap statement |
| Methods | Complete | Full model specification with equations |
| Results | Complete | 3 subsections with validation metrics |
| Discussion | Complete | Limitations and future directions |
| Figures | Complete | 4 figures with comprehensive captions |
| References | Complete | 6 key citations |

**Paper file:** `docs/paper/manuscript.qmd` (renders to PDF via Quarto)

### Key Results Reported

- Gamma-difference kernel with R² = 0.968 vs 12-basis reference
- Rate ratio = 0.97 (simulated/empirical)
- PSTH correlation = 0.84
- Turn rate match: 1.88 vs 1.84 turns/min (98%)

### Questions About the Paper

1. **Is the single-condition paper publishable as-is?**
   - Target venue: methods-oriented behavioral neuroscience journal
   - Contribution: interpretable analytic kernel for point process model

2. **Does the factorial extension change the paper's identity?**
   - From: methods paper demonstrating analytic kernel
   - To: biology paper about intensity/temperature effects?

3. **Should factorial results go in main text or supplement?**
   - Main text: stronger paper but more complex story
   - Supplement: simpler core message, factorial as validation

---

## FACTORIAL DESIGN OVERVIEW

### Experimental Structure

| | Control (7 PWM) | Temp (5-15 PWM) |
|---|---|---|
| **0→250 PWM** | 2 expts, 70 tracks, 3847 events | 4 expts, 65 tracks, 3441 events |
| **50→250 PWM** | 4 expts, 70 tracks, 2440 events | 2 expts, 65 tracks, 1031 events |

**Total:** 12 experiments, 270 tracks, 10,759 events

### LED Protocol (Verified Identical Across Conditions)
- LED1 (optogenetic): 10s ON at 250 PWM / 20s OFF
- LED1 baseline: 0 PWM or 50 PWM depending on condition
- LED2 (background): 7 PWM constant (Control) or 5-15 PWM cycling (Temp)
- Frame rate: 20 Hz

### Preliminary Effects (from Phase 1)

| Condition | Correlation with Ref | Suppression Ratio | Intercept |
|-----------|---------------------|-------------------|-----------|
| 0→250 \| Control (ref) | 1.00 | 1.00× | -6.94 |
| 0→250 \| Temp | 0.98 | 1.74× | -6.93 |
| 50→250 \| Control | 0.97 | 0.73× | -7.01 |
| 50→250 \| Temp | 0.94 | 1.15× | -7.25 |

**Key observation:** Kernel SHAPE is stable (all r > 0.94); only AMPLITUDE varies.

---

## PROPOSED STATISTICAL MODEL

### Pooled NB-GLMM with Factorial Effects

```
log λ(t) = β₀ + β_I·I + β_T·T + β_{I×T}·(I·T) 
         + α·K_on(t) + α_I·I·K_on(t) + α_T·T·K_on(t)
         + γ·K_off(t) + u_track
```

Where:
- I ∈ {0, 1}: intensity indicator (0→250 vs 50→250)
- T ∈ {0, 1}: temperature indicator (Control vs Temp)
- K_on(t): fixed gamma-difference kernel (shared τ₁, τ₂)
- u_track ~ N(0, σ²): track random effect

### Interpretation of Parameters

| Parameter | Meaning |
|-----------|---------|
| β_I | Does 50→250 change baseline rate? |
| β_T | Does temperature change baseline rate? |
| β_{I×T} | Does intensity effect depend on temperature? |
| α_I | Does 50→250 weaken suppression amplitude? |
| α_T | Does temperature strengthen suppression amplitude? |

---

## QUESTIONS REQUIRING GUIDANCE

### SECTION 1: Statistical Modeling

**Q1.1** What is the best way to model condition-dependent kernel amplitude?

Options considered:
- A) Multiplicative: λ(t) = exp(β₀ + (α + α_I·I + α_T·T) · K_on(t) + ...)
- B) Additive: λ(t) = exp(β₀ + α·K_on(t) + α_I·I·K_on(t) + ...)
- C) Per-component: separate α for fast (τ₁) and slow (τ₂) components

We are leaning toward Option B (additive), but unsure if this is standard practice.

**Q1.2** Should kernel shape (τ₁, τ₂) be allowed to vary by condition?

Our Phase 1 analysis shows:
- Correlations > 0.94 across all conditions
- Peak times range 3.1-4.0s (reference 3.8s)
- Shape is qualitatively similar

We plan to fix τ₁ = 0.29s and τ₂ = 3.81s globally. Is this defensible?

**Q1.3** What random effects structure is appropriate for nested data?

Our data structure:
- Tracks nested within experiments
- Experiments nested within conditions
- Conditions crossed (2×2 factorial)

Current plan: random intercept per track only. Should we add:
- Random intercept per experiment?
- Random slope for kernel effect?

**Q1.4** How to test for interaction with adequate power?

We have:
- 2-4 experiments per cell
- Unbalanced design (1031 to 3847 events per cell)
- Moderate expected effect size (16% interaction from preliminary analysis)

Is there a power analysis framework for NB-GLMM with factorial covariates?

---

### SECTION 2: Biological Interpretation

**Q2.1** What does "temperature block" (T_Bl_Sq_5to15PWM) typically mean?

The filename pattern suggests:
- T_Bl = Temperature Block
- Sq_5to15PWM = Square wave 5-15 PWM

Possibilities:
- Thermal stimulation via IR LED
- Visual stimulation with different intensity range
- Synchronous/asynchronous with LED1

We do not have access to the original experimental protocol. Can you provide context on what this condition typically represents in Drosophila optogenetics?

**Q2.2** Why might 50→250 PWM show weaker suppression than 0→250?

Our preliminary finding: 50→250 produces 27% weaker suppression (ratio 0.73× vs 1.0×).

Possible explanations:
- A) Smaller intensity delta (200 vs 250 PWM)
- B) Partial adaptation to 50 PWM baseline
- C) Nonlinear ChR2 activation curve
- D) Different neural circuit engagement at baseline

Are there published studies on intensity-dependent optogenetic effects in larvae?

**Q2.3** Why might temperature condition enhance suppression?

Our preliminary finding: Temp conditions show 74% stronger suppression (0→250) or 58% (50→250).

Possible explanations:
- A) Temperature stress increases baseline arousal
- B) Synergistic effect with optogenetic activation
- C) LED2 visual input modulates circuit state
- D) Confound (different experimental days/batches)

Is there literature on thermal modulation of larval navigation?

---

### SECTION 3: Literature and Precedent

**Q3.1** Are there published examples of factorial designs in larval navigation studies?

We are looking for precedents that:
- Use 2×2 or higher factorial designs
- Model navigation behavior (turns, reorientations)
- Apply point process or hazard models

**Q3.2** Are there examples of gamma-difference kernels in sensory neuroscience?

Our kernel K_on(t) = A·Γ(t;α₁,β₁) - B·Γ(t;α₂,β₂) is interpretable but may be novel. Are there related formulations in:
- Visual neuroscience (temporal contrast sensitivity)
- Auditory processing (adaptation models)
- Motor control (response time kernels)

**Q3.3** What are standard effect sizes for optogenetic suppression in larvae?

We observe ~2× suppression during LED-ON. Is this:
- A) Typical for GMR61/ChR2?
- B) Strong or weak compared to other lines?
- C) Consistent with published studies?

---

### SECTION 4: Paper Strategy

**Q4.1** What is the strongest framing for this paper?

Options:
- A) Methods paper: "An analytic kernel for behavioral point processes"
- B) Systems paper: "Timescales of optogenetic suppression in larval navigation"
- C) Comparative paper: "Intensity and temperature modulation of reorientation"

We currently lean toward (A) with (C) as extension. Is this optimal?

**Q4.2** Should we include the factorial analysis in the first submission?

Arguments for:
- Stronger paper with more data
- Shows generalization across conditions
- Addresses potential reviewer criticism

Arguments against:
- Complicates the story
- May shift focus from methods to biology
- Delays submission

**Q4.3** What journal venues are appropriate?

Considering:
- eLife (methods section for behavior)
- PLOS Computational Biology
- Journal of Neuroscience Methods
- Frontiers in Behavioral Neuroscience

What is the typical scope for a methods-oriented paper on behavioral modeling?

---

## SPECIFIC UNCERTAINTIES

1. **I don't know** if the additive kernel modulation (Option B) is standard or if multiplicative is preferred

2. **I don't know** if fixing τ₁, τ₂ globally is defensible given 6% shape variation

3. **I don't know** what "temperature block" experimentally means in this context

4. **I don't know** if the interaction effect (16%) is statistically detectable with our N

5. **I don't know** if the paper is stronger with or without factorial extension

6. **I don't know** what related work exists on gamma-difference kernels

---

## DATA SUMMARY FOR REFERENCE

### Per-Condition Statistics

```
Condition               Tracks  Events  Intercept   AIC
0→250 | Control            70    3847    -6.938   60494
0→250 | Temp               65    3441    -6.928   54520
50→250 | Control           70    2440    -7.011   38689
50→250 | Temp              65    1031    -7.248   16837
```

### Kernel Parameters (Reference Condition)

```
Fast component:  A = 0.456, α₁ = 2.22, β₁ = 0.132s, τ₁ = 0.29s
Slow component:  B = 12.54, α₂ = 4.38, β₂ = 0.869s, τ₂ = 3.81s
Rebound:         D = -0.114, τ_off = 2.0s
```

### Validation Metrics (Reference Condition)

```
Kernel R² = 0.968
Rate ratio = 0.974
PSTH correlation = 0.84
Suppression = 2.0× (empirical) vs 1.9× (simulated)
```

---

## REQUESTED OUTPUTS

1. **Statistical guidance** on model specification (Q1.1-Q1.4)

2. **Biological context** for temperature and intensity effects (Q2.1-Q2.3)

3. **Literature references** for related methods and applications (Q3.1-Q3.3)

4. **Paper strategy advice** on framing and scope (Q4.1-Q4.3)

5. **Risk assessment** for proceeding with factorial extension

6. **Priority order** if we can only address some questions

