# Final Research Summary: ML Methods for Optimizing Your LNP Behavioral Simulation

**Source**: MiroThinker (MiroMind)  
**Date**: 2025-12-11

---

## Executive Summary

Your current NB‑GLM/LNP model already **meets the shape criteria** (PSTH correlation 0.881, W‑ISE 0.094) but falls short on **rate calibration** (1.4× too high) and slightly over/under‑estimates early/late suppression. Given only 1,407 events across 99 tracks, the **highest return** will come from:

1. Better **statistical structure around the GLM** (mixed effects, basis search, mild physics priors), not from very heavy neural methods.

2. Using complex methods (SINDy, PySR, FNO, GP, HMM) **surgically** where they add clear value, and not as primary engines.

---

## 1. Feasibility Assessment by Method

### Q8 First: Mixed‑Effects / Hierarchical Models (Most Impactful for You)

**Goal:** Explain rate mismatch by between‑track variability rather than forcing a single intercept to serve 99 larvae.

**Feasibility**

- Data size (99 tracks, 1,407 events) is **excellent** for a random‑intercept NB‑GLMM.
- Tools:
  - **R:** `glmmTMB`, `lme4` + `MASS`, easily handle NB with random intercepts.
  - **Python:** `PyMC`/`bambi`, `brms` (via R) if you want Bayesian inference.

- Model:
  ```
  log λ_ij(t) = (β₀ + u_i) + Σ β_k B_k(t)
  u_i ~ N(0, σ²_track)
  ```

- Expected effects:
  - Fixed‑effect kernel shape very similar to your current one.
  - Global rate brought much closer to empirical (because extreme tracks no longer drag the overall intercept up).
  - Better calibrated uncertainty on β₀ and the kernel.

**Interpretability:** Excellent. Same kernel, now with per‑track "baseline biases" factored out.

**Conclusion:** **High feasibility, high impact, low complexity. This is the first method you should apply.**

---

### Q5: Bayesian Optimization for Kernel Design

**Goal:** Let a Bayesian optimizer pick the **number/placement/width** of temporal bases to minimize W‑ISE while meeting a rate constraint.

**Feasibility**

- Search space is small:
  - n_early: 2–5
  - n_late: 3–8
  - early_width: 0.2–1.0 s
  - late_width: 1.0–3.0 s
  - (optional) basis_type ∈ {raised‑cosine, Gaussian, B‑spline} encoded as integers.

- One NB‑GLM fit on 1.3 M frames is seconds–minutes; BO needs maybe 20–40 evaluations → very manageable.

- You can hard‑enforce a **rate constraint** inside the objective:
  ```python
  def objective(params):
      # Fit model with basis params -> get rate_sim, WISE, corr
      if abs(rate_sim - rate_emp) > 0.10 * rate_emp:
          return -1e6    # heavy penalty
      return -WISE
  ```

**Interpretability:** Preserved – you still get a raised‑cosine or spline kernel, just tuned optimally.

**Conclusion:** **Very feasible and a true "quick optimization layer" on top of your current model.** Do it after adding random intercepts.

---

### Q1: Physics‑Informed Machine Learning (PIML)

**Goal:** Add **soft biological constraints** (refractory decay, adaptation, causality) into the loss.

Given your setting, the most useful constraints are:
1. **Refractory:** hazard after an event should start low and recover exponentially.
2. **Causality:** kernel support strictly for t ≥ 0 (you already enforce this).
3. **Adaptation:** hazard for long ON epochs should not grow unbounded (already discouraged by data).

**Feasibility**

- You do **not** need heavyweight PIML frameworks. You can implement physics‑informed terms directly around your GLM in PyTorch/JAX:
  ```python
  loss = NLL(observed, predicted) \
         + λ_refrac * refractory_penalty(beta, tau_refrac) \
         + λ_energy * energy_penalty(beta)
  ```

- Refractory penalty example:
  - Fit an exponential `r(Δt) = c0 * exp(-Δt / τ)` to your empirical IEI distribution.
  - Penalize deviations between simulated post‑event hazard and this target shape.

**Suitability of frameworks**
- **DeepXDE / PINN:** Overkill; they shine for PDEs on continuous fields, not 1‑D time kernels with 1 k events.
- **NeuralODE:** Only useful if you reparameterize the hazard dynamics as an ODE network, which is not necessary here.
- Your best PIML is a **custom regularized loss** on top of the GLM.

**Conclusion:** **Feasible, but incremental.** Add a **refractory‑shape penalty** only after mixed‑effects and BO.

---

### Q2: SINDy (Sparse Identification of Nonlinear Dynamics)

**Goal:** Discover a sparse ODE for the hazard/state dynamics (`dx/dt = f(x)`) from data.

**Feasibility**

- Challenge: SINDy expects **continuous trajectories** of `x(t)` and `dx/dt`. You only observe discrete events, not λ(t) directly.
- Workaround:
  - Use your fitted GLM to generate a **dense estimate of λ(t)**, then:
    - Smooth log λ(t) over time,
    - Use that as `x(t)` and a finite difference for `dx/dt`.
  - Use a library with: Polynomials, LED, Exponentials, Heaviside of time since last event

- Risk: with only 1.4 k events and a lot of zeros, SINDy may overfit noise.

**Conclusion:** **Methodologically interesting but not essential.** Good as a **research project**.

---

### Q3: Symbolic Regression (PySR)

**Goal:** Directly discover an analytic expression for `K(t)` that reproduces the observed PSTH/log‑likelihood.

**Feasibility**

- Data: still quite sparse for a flexible symbolic search.
- Reasonable objective: **W‑ISE + rate penalty** instead of full point‑process log‑likelihood.

**Main risk:** With only 9–15 effective DOF in the kernel and 1.4 k events, PySR may produce many equivalent forms.

**Conclusion:** **Interesting but heavy.** Keep it as a **longer‑term project**.

---

### Q4: Neural Point Process + FNO

**Goal:** Learn mapping from LED history to hazard via a neural operator instead of a linear filter.

**Feasibility**

- 1‑D temporal kernel over 0–10 s at 20 Hz → 200 timepoints. This is **too small to justify FNO**.
- A simple **1‑D CNN or small attention kernel** can easily learn the same mapping.
- Data limitation: 1,407 events is not much for training a flexible neural point process.

**Conclusion:** **A simple CNN‑based point process is feasible but will not solve your core rate‑vs‑shape trade‑off**, and will be less interpretable.

---

### Q6: SDP / HMM / Latent State Models

**Goal:** Model different internal states (alert/habituated/refractory) via MDP/HMM.

**Feasibility**

- You can fit a **3‑state HMM** on event times/LED state, but:
  - 1.4 k events across 99 tracks = ~14 events/track on average.
  - This is borderline for **robust** multi‑state parameter estimation.

**Conclusion:** **Nice for theory, not needed to reach your present success criteria.**

---

### Q7: Gaussian Process Temporal Kernels

**Goal:** Put a GP prior on K(t), learn its shape nonparametrically.

**Feasibility**

- Kernel domain is 0–10 s with maybe 50–100 support points → manageable with sparse GP.
- Considerably more implementation complexity than "just" a GLMM with bases.

**Conclusion:** **Research‑level method**, good for a methods‑heavy paper.

---

## 2. Recommended Workflow

### 2.1 Quick Wins (1–2 Hours)

1. **Fit an NB‑GLMM with random intercept per track**
   - Anchor the global intercept to the empirical baseline (≈ −7.44)
   - Re‑estimate the kernel βⱼ
   - Event rate → should come within 10–20 % of 0.71

2. **Use Bayesian Optimization to tune basis configuration**
   - Hyperparameters: n_early, n_late, early_width, late_width
   - Objective: minimise W‑ISE with hard penalty if |rate_sim–rate_emp| > 10 %

### 2.2 Medium Effort (≈ 1 Day)

3. **Add physics‑informed refractory/adaptation penalty**
4. **Add track‑wise cross‑validation** (CV correlation > 0.80)

### 2.3 Research Projects (≥ 1 Week)

5. SINDy or symbolic regression
6. GP temporal kernel
7. HMM/latent‑state model

---

## 3. Implementation Priorities

| Priority | Method | Time | Expected Improvement |
|----------|--------|------|---------------------|
| **1** | NB-GLMM random intercepts | 1-2 hrs | Rate within 10-20% |
| **2** | Bayesian Opt for basis | 1-2 hrs | W-ISE < 0.08 |
| **3** | PIML refractory penalty | 1 day | IEI match |
| **4** | Track-wise CV | 1 day | Generalization proof |
| **5** | SINDy / PySR | 1 week | Compact kernel formula |
| **6** | GP kernel | 1 week | Uncertainty quantification |
| **7** | HMM states | 1 week | Circuit interpretation |

---

## 4. Hybrid Approaches

- **PIML + Bayesian Optimisation:** Use PIML penalties as part of the objective in the BO loop

- **SINDy then GLM refinement:** Use SINDy to propose functional forms, then replace raised‑cosines with those as new basis functions

- **GP prior + Mixed‑Effects:** For a methods paper, place a GP prior over K(t) and random intercepts over tracks

---

## 5. Bottom‑Line Answers

1. **Is PIML useful?** Yes, as light‑weight regularization, not full PINN/NeuralODE
2. **Is SINDy feasible?** Yes, but only as secondary discovery tool
3. **Can PySR find closed‑form kernel?** Possibly, but data are sparse
4. **Is FNO appropriate?** No; use simple 1‑D CNN if needed
5. **Is Bayesian optimization worth it?** Yes – easiest way to improve W‑ISE
6. **Do you need HMM/MDP?** Not for current metrics
7. **Are GP kernels viable?** Yes, but overkill right now
8. **Will mixed‑effects solve rate‑vs‑shape?** They directly attack the main issue

---

## Most Effective Path

1. Upgrade to **mixed‑effects NB‑GLMM** with random intercepts
2. Use **Bayesian optimisation** to tune the raised‑cosine basis set
3. Optionally add **physics‑informed refractory penalty**
4. Reserve SINDy, PySR, GP kernels, HMM for **later research exploration**



