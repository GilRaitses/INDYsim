# MiroThinker Response: Final Uncertainties Review

*Received: 2025-12-12 00:30*
*Source: MiroThinker (MiroMind)*

---

## 1. Critical vs Acceptable Uncertainties

### 1.1 Fixed-effects NB-GLM (no random effects)

**Assessment**: For a methods-focused paper, this is **not a blocker**:
- The fixed-effects NB-GLM still gives you reasonable population-average effects.
- You already use cross-validation to show how well the model generalizes across experiments, which partially compensates for not modeling random effects explicitly.

**Status**: Acceptable limitation, not a critical blocker.

### 1.2 Event-definition mixing (onset vs "true turns")

**Assessment**: For hazard estimation:
- Including micro-events increases sample size but doesn't obviously distort the kernel; your validation metrics (R², rate ratio, PSTH correlation) remain strong.
- This is analogous to spike-train models where every threshold crossing is modeled, even if some are small.

**Status**: Acceptable with clear explanation. Not a blocker.

### 1.3 Interaction power (β_{IC})

**Assessment**: The main factorial story is driven by:
- A large intensity effect (~66% weaker suppression).
- A modest cycling effect (~15% stronger suppression).
- The interaction is small and naturally under-powered; it should not be a pillar of the narrative.

**Status**: Acceptable if framed as exploratory. Not a blocker.

### 1.4 Cycling background interpretation ("reduced adaptation")

**Assessment**: The safe claim is:
- Cycling background modestly increases suppression gain while lowering baseline.
- Mechanism could be reduced adaptation, temporal contrast effects, or general gain modulation.
- "Reduced adaptation" as a definitive explanation is too strong; as a hypothesis it is fine.

**Status**: Needs softening of language, but not a structural problem.

### 1.5 Time-rescaling violation (~13% deviation)

**Assessment**: This level of deviation is common in practical point-process models. Adding a post-event refractory kernel is a logical next step but:
- Adds complexity and parameters.
- Does not change your primary claims about LED-driven suppression timescales and amplitude.

**Status**: Acceptable as a clearly stated limitation. A refractory kernel is optional future work, not a prerequisite.

---

## 2. Doubts: Is the Factorial Extension Worth It?

### 2.1 Necessity of the factorial extension

**Assessment**: The factorial extension adds value if:
- It is one focused Results subsection.
- The single-condition kernel and validation remain the core of the paper.
- You treat the cycling and interaction effects as secondary and interpret them cautiously.

### 2.2 Overfitting and model-to-data ratio

**Assessment**: 8 fixed-effect parameters for 7,288 events is very conservative. Model complexity vs data volume is reasonable. Overfitting is not your main concern.

### 2.3 Kernel shape invariance assumption

**Assessment**: For this paper, it is enough to:
- Show cross-condition kernel overlays or correlations.
- State that full shape variation gave negligible gains, so you fixed τ₁, τ₂ to simplify interpretation.

A formal nested-model test is nice to have, not required.

---

## 3. Which "Secrets" Should Be Disclosed, and How?

### 3.1 Anomalous experiments excluded

**Recommendation**: Yes, disclose. In Methods:
- "Two experiments with anomalously high event counts (approximately 10–20× other sessions) were excluded from analysis because their annotation statistics were inconsistent with the remaining dataset."

### 3.2 Calibration factor origin

**Recommendation**: Frame it as rate-normalization:
- "We applied a constant multiplicative calibration to the intercept so that the model's baseline event rate matched the empirical rate. This scaling does not change kernel shape, suppression timing, or relative condition effects; it only aligns the overall rate scale."

### 3.3 LED timing assumption

**Recommendation**: State plainly:
- "We confirmed LED timing (10 s ON / 20 s OFF) directly from the recorded LED traces in all experiments."

### 3.4 "Cycling" vs "Temp" discovery

**Recommendation**: The analysis and manuscript now use the correct interpretation. This earlier misunderstanding does not need to be narrated in the paper; just ensure terminology is correct. No disclosure required.

### 3.5 Trajectory simulation not fully validated

**Recommendation**: Yes, disclose as scope:
- "Our trajectory simulator uses the hazard model to drive RUN/TURN transitions and reproduces event rates and timing; spatial statistics of trajectories (path shapes, occupancy) were not systematically validated and are presented as a demonstration of possible use rather than a fully calibrated locomotion model."

---

## 4. Reviewer Objections to Anticipate

| Likely Objection | How to Address |
|------------------|----------------|
| "Where are the random effects?" | Acknowledge hierarchy; present CV results (58% pass); state GLMM is future work |
| "Are micro-events biasing the hazard?" | Clarify event definition; emphasize strong fit metrics; explain two-tier usage |
| "Interaction seems under-powered." | Provide approximate power; present β_{IC} as exploratory |
| "Cycling effect is small and interpretation speculative." | Emphasize magnitude (~15%); frame as hypothesis |
| "Time-rescaling shows Poisson violation." | Show deviation is modest; note potential refractoriness |
| "Trajectory simulator looks unvalidated." | Label as demonstration; make clear hazard modeling is main contribution |

---

## 5. Direct Answers

1. **Critical blockers?** No. All identified uncertainties are manageable limitations.

2. **Mention excluded experiments in Methods?** Yes. Briefly state that two anomalous high-event sessions were excluded and why.

3. **Justify post-hoc intercept calibration?** Present it as rate normalization: matching global baseline rate while keeping kernel shape and relative condition effects unchanged.

4. **Factorial in main text or supplement?** Keep it in the main text as a concise subsection.

5. **Likely reviewer objections?** Expect questions about missing random effects, event definition, small interaction, speculative cycling interpretation, and time-rescaling.

6. **Citable precedent?** Yes: prior LNP/GLM work on larval navigation, factorial behavioral analyses in Drosophila, and bilobe/gamma temporal kernels in sensory systems.

---

## 6. Conclusion

INDYsim's single-condition hazard model is clearly publication-ready. The factorial extension, treated as a focused validation and application rather than the centerpiece, strengthens the paper by demonstrating shape invariance and interpretable intensity/background effects. With modest adjustments to framing and explicit acknowledgment of limitations, you are ready to proceed to submission.
