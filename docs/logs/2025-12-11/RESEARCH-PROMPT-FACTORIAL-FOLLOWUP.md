# Follow-Up Research Prompt: Factorial Implementation Details and Remaining Uncertainties

## Context

This is a follow-up to the initial factorial guidance. The prior response (graded A overall) provided clear recommendations on model structure, but several implementation details and literature gaps remain. This prompt seeks specific, actionable guidance to complete the factorial extension.

---

## PROJECT STATE SUMMARY

### What Is Complete

| Component | Status | Key Metrics |
|-----------|--------|-------------|
| Paper draft (single-condition) | Complete | `docs/paper/manuscript.qmd`, renders to PDF |
| Figures 1-4 | Complete | Kernel, validation, trajectories, distributions |
| Cross-condition fits (Phase 1) | Complete | All 4 conditions fit, r > 0.94 |
| Comparison report | Complete | `CROSS-CONDITION-COMPARISON.md` |
| Factorial plan | Complete | `FACTORIAL-ANALYSIS-PLAN.md` |

### What Is Pending

| Component | Status | Blocker |
|-----------|--------|---------|
| Pooled factorial model (Phase 2A) | Not started | Need implementation details |
| Per-condition validation (Phase 2B) | Not started | Depends on 2A |
| Paper update with factorial (Phase 2C) | Not started | Depends on 2A-2B |
| Figure 5 (factorial results) | Not started | Depends on 2A |

### Key Decisions Made (from prior guidance)

1. Use **additive kernel modulation** (Option B)
2. **Fix τ₁ = 0.29s, τ₂ = 3.81s** globally
3. **Track-level random intercepts only**
4. Treat **interaction as exploratory**
5. Include factorial in **main text** (one focused subsection)
6. Use **hybrid framing** (methods + biological application)

---

## GAPS IN PRIOR RESPONSE

### Gap 1: No Specific Literature Citations

The prior response mentioned precedents but provided no actual citations:

- "Bilobe gamma kernels are standard in temporal filters in vision"
- "Many lines show 1.5-3× changes in turning/locomotion rates"
- "Comparisons of light intensity × background conditions in LNP frameworks"

**Request:** Please provide 3-5 specific paper citations for:

1. **Gamma-difference or bilobe kernels** in sensory neuroscience (any modality)
2. **LNP/GLM hazard models** for larval navigation specifically
3. **Factorial designs** in Drosophila behavioral studies (any behavior)
4. **Optogenetic intensity-response curves** in larvae (if available)

Format: Author (Year). Title. Journal. DOI or key finding.

---

### Gap 2: No Power Calculation for Interaction

The prior response suggested simulation but did not provide:

- Expected power given our N (10,759 events, 270 tracks)
- Minimum detectable effect size for interaction
- Whether 16% interaction is likely detectable

**Request:** Please provide:

1. A rough power estimate for detecting a 16% interaction effect with our sample size
2. What interaction effect size would be detectable with 80% power?
3. Should we report a post-hoc power analysis in the paper?

---

### Gap 3: Temperature Mechanism Remains Ambiguous

The prior response correctly avoided overstatement but did not resolve:

- Is "T_Bl_Sq" a standard naming convention?
- Are there published protocols using similar LED2 modulation?
- What is the most likely biological interpretation?

**Request:** If you have access to Drosophila optogenetics literature, please clarify:

1. Is 5-15 PWM LED2 modulation typically thermal or visual?
2. Are there standard "temperature block" paradigms in larval studies?
3. What language should we use in the paper to describe this condition?

---

### Gap 4: No Specific Model Diagnostics Recommended

The prior response mentioned "residual diagnostics" but did not specify:

- Which residuals for NB-GLMM (Pearson, deviance, quantile)?
- How to assess random effects adequacy?
- What plots to include in supplement?

**Request:** Please specify:

1. Which residual type is most informative for point-process NB-GLMMs?
2. What diagnostic plots should we generate?
3. How to test for overdispersion in the factorial model?

---

## GAPS IN MY IMPLEMENTATION PLAN

### Gap 5: Exact Design Matrix Construction

I need to construct:

```
X = [1, I, T, I×T, K_on(t), I·K_on(t), T·K_on(t), K_off(t)]
```

**Uncertainties:**

1. Should K_on(t) be the full kernel value at each frame, or just the basis coefficients?
2. How to handle the interaction I·K_on(t) - is this element-wise multiplication?
3. Should I center I and T (0.5/-0.5 coding) or use 0/1 indicator coding?

**Request:** Please clarify the exact design matrix construction, including:

- Whether to use deviation coding vs indicator coding for I, T
- How to construct the kernel interaction terms
- Whether to include an I×T×K_on(t) three-way interaction (or is this overparameterized?)

---

### Gap 6: Fitting Procedure for NB-GLMM

My current infrastructure uses `statsmodels.GLM` with `NegativeBinomial` family. 

**Uncertainties:**

1. Does statsmodels support random effects, or do I need `statsmodels.MixedLM`?
2. Should I use `pymer4` or `bambi` for proper mixed-effects NB-GLMM?
3. What is the recommended Python package for NB-GLMM with random intercepts?

**Request:** Please recommend:

1. The best Python package for fitting NB-GLMM with random intercepts
2. Example code structure for the model specification
3. How to extract coefficient CIs and conduct LRT tests

---

### Gap 7: Handling the Rebound Kernel in Factorial Model

The prior response focused on K_on(t) modulation but did not address:

- Should K_off(t) also have condition-specific amplitudes (γ_I, γ_T)?
- Or should γ be shared across all conditions?

**Request:** Please clarify:

1. Should the LED-OFF rebound term be condition-modulated?
2. If yes, add γ_I and γ_T to the model?
3. Or is this overparameterization given limited OFF-period data?

---

### Gap 8: Cross-Validation Strategy

For per-condition validation (Phase 2B), I need to decide:

- Leave-one-experiment-out vs leave-one-track-out?
- Validate on held-out data or resubstitution?
- What metrics constitute "passing" validation?

**Request:** Please specify:

1. The recommended cross-validation strategy for this factorial model
2. Whether to use out-of-sample prediction or in-sample fit statistics
3. Pass/fail criteria for per-condition validation

---

## REMAINING UNCERTAINTIES

### Uncertainty 1: Baseline Rate vs Amplitude Effects

The model has separate terms for baseline effects (β_I, β_T) and amplitude effects (α_I, α_T).

**Question:** How do I interpret a scenario where:
- β_I is significant (baseline differs by intensity)
- α_I is NOT significant (amplitude does not differ)

Is this biologically meaningful, or does it suggest model misspecification?

---

### Uncertainty 2: Negative Binomial Dispersion

The NB-GLM has a dispersion parameter α (separate from kernel α).

**Question:**
- Should dispersion be fixed (as in current single-condition fits) or estimated?
- Should dispersion vary by condition?
- How does dispersion interact with the random effect variance σ²?

---

### Uncertainty 3: Event Definition in Pooled Model

In the single-condition model, we used 1,407 inclusive events.
In the pooled model, we have 10,759 events across conditions.

**Question:** Are these event definitions consistent across conditions?
- Same curvature threshold?
- Same annotation procedure?
- Could condition differences in event counts reflect annotation differences?

---

### Uncertainty 4: Reporting Non-Significant Interaction

If β_{I×T} is not significant (expected given power concerns):

**Question:**
- Do we report the full model with interaction, or drop it?
- If we drop it, do we need to justify model selection?
- Is AIC/BIC the right criterion, or should we use LRT?

---

### Uncertainty 5: Figure 5 Design

I need to create a figure summarizing factorial results.

**Question:** What is the most effective visualization?

Options:
- A) 2×2 heatmap of suppression amplitude
- B) Bar chart with error bars (4 conditions)
- C) Forest plot of coefficient estimates
- D) Overlay of 4 kernels (one per condition)

Which combination best tells the story?

---

## DATA CONTEXT (for reference)

### Per-Condition Summary

```
Condition               Expts  Tracks  Events  Intercept   Suppr Ratio
0→250 | Control            2      70    3847    -6.938      1.00×
0→250 | Temp               4      65    3441    -6.928      1.74×
50→250 | Control           4      70    2440    -7.011      0.73×
50→250 | Temp              2      65    1031    -7.248      1.15×
TOTAL                     12     270   10759
```

### Kernel Parameters (Reference)

```
Fast:    A = 0.456, α₁ = 2.22, β₁ = 0.132s → τ₁ = 0.29s
Slow:    B = 12.54, α₂ = 4.38, β₂ = 0.869s → τ₂ = 3.81s
Rebound: D = -0.114, τ_off = 2.0s
```

### Cross-Condition Kernel Correlations

```
0→250 | Control (ref)  →  r = 1.000
0→250 | Temp           →  r = 0.975
50→250 | Control       →  r = 0.966
50→250 | Temp          →  r = 0.943
```

---

## REQUESTED OUTPUTS

### Priority 1: Implementation Details

1. **Design matrix construction** (Gap 5): Exact specification with coding scheme
2. **Python package recommendation** (Gap 6): For NB-GLMM with random effects
3. **Rebound kernel treatment** (Gap 7): Shared or condition-specific γ?
4. **Cross-validation strategy** (Gap 8): Leave-one-out specification

### Priority 2: Statistical Clarifications

5. **Power estimate** (Gap 2): Rough power for 16% interaction
6. **Dispersion handling** (Uncertainty 2): Fixed or estimated?
7. **Model selection** (Uncertainty 4): Report full model or reduced?
8. **Diagnostics** (Gap 4): Which residuals and plots?

### Priority 3: Biological/Literature Context

9. **Specific citations** (Gap 1): 3-5 papers on gamma kernels, LNP models, factorial designs
10. **Temperature condition interpretation** (Gap 3): Standard language for paper

### Priority 4: Visualization

11. **Figure 5 design** (Uncertainty 5): Best combination of panels

---

## CONSTRAINTS

- Python implementation (statsmodels, pymer4, or bambi)
- Must integrate with existing codebase (`scripts/fit_cross_condition.py`)
- Paper must remain methods-focused (factorial as application, not main contribution)
- Timeline: 2.5 days for full implementation

---

## SPECIFIC QUESTIONS SUMMARY

| ID | Question | Priority |
|----|----------|----------|
| Q1 | Exact design matrix with coding scheme? | High |
| Q2 | Best Python package for NB-GLMM with random effects? | High |
| Q3 | Should γ (rebound) be condition-specific? | High |
| Q4 | Leave-one-experiment-out or leave-one-track-out CV? | High |
| Q5 | Power for detecting 16% interaction? | Medium |
| Q6 | Fixed or estimated NB dispersion? | Medium |
| Q7 | Report full model or drop non-significant interaction? | Medium |
| Q8 | Which residual diagnostics for NB-GLMM? | Medium |
| Q9 | 3-5 specific literature citations? | Medium |
| Q10 | Language for temperature condition in paper? | Medium |
| Q11 | Best figure design for factorial results? | Low |

Please address as many as possible, prioritizing High and Medium items.

