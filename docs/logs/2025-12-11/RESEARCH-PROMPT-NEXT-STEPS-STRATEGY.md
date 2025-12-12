# Research Prompt: Strategic Direction for INDYsim

## Context

We have developed a validated hazard model for Drosophila larval reorientation under optogenetic stimulation:

**Model**: Negative-binomial LNP with gamma-difference kernel (6 params) + exponential rebound (2 params)

**Validation Results**:
| Metric | Value | Status |
|--------|-------|--------|
| Rate ratio | 0.97 | PASS |
| PSTH correlation | 0.84 | GOOD |
| Suppression match | 2.0x vs 1.9x | GOOD |
| Time-rescaling | p < 0.001 | MILD FAIL |

**Key Parameters**:
- Fast component: τ₁ = 0.29s (sensory response)
- Slow component: τ₂ = 3.8s (adaptation)
- Peak suppression: 1.7s after LED onset
- Track variability: σ = 0.47 (3× spread in baseline rates)

**Critical Finding**: The model was fit to 1,407 events, but 77% have zero duration. Only 319 events (23%) represent behaviorally meaningful "true turns" with duration > 0.1s.

---

## DECISION POINT 1: Event Definition Strategy

### The Problem

We have two event populations:
1. **All events (1,407)**: Model validates well (rate ratio 0.97)
2. **Filtered events (319)**: Model fails validation (rate ratio 4.19)

The 77% zero-duration events are likely:
- Frame-by-frame curvature threshold crossings
- Head sweeps without sustained turns
- Detection noise

### Options

**Option A: Keep all-events model**
- Pros: Already validated, simpler
- Cons: Simulated "events" include many micro-movements, not just turns
- For trajectory simulation: ~23% of events are "real" turns

**Option B: Refit on filtered events**
- Pros: Events correspond to behaviorally meaningful turns
- Cons: Only 319 events (may be underpowered), requires refitting
- For trajectory simulation: All simulated events are "real" turns

**Option C: Post-hoc filtering**
- Simulate with all-events model, then subsample ~23% as "true turns"
- Pros: No refitting needed
- Cons: Arbitrary subsampling, temporal structure may be wrong

### Questions for Research

1. Which approach is standard in larval behavior modeling literature?
2. What is the minimum event count for reliable hazard model fitting?
3. If we refit on 319 events, what statistical power do we have?
4. Is there a principled way to distinguish "turn events" from "micro-movements" in simulation?

---

## DECISION POINT 2: Trajectory Simulation Architecture

### What We Need

To simulate full larval trajectories, we need:
1. **When**: Event timing (hazard model provides this)
2. **How much**: Turn angle distribution
3. **How long**: Turn duration
4. **Between turns**: Run kinematics

### Proposed Architecture

```
RUN state:
  - Forward motion: v ≈ 1 mm/s
  - Heading noise: small Brownian perturbations
  - Duration: until hazard event

TURN state:
  - Heading change: sampled from empirical Δθ distribution
  - Duration: sampled from empirical distribution (0.2-2s)
  - Speed: reduced (0.3-0.5× run speed)
```

### Uncertainties

1. **Turn angle distribution**: We have `turn_duration` but not `turn_angle` in our filtered events. Need to extract this.

2. **Run kinematics**: What is the empirical heading diffusion coefficient? Speed variability?

3. **State coupling**: Do turn angles depend on LED state? Prior heading? Time since last turn?

4. **Spatial effects**: Do larvae avoid edges? Respond to gradients?

### Questions for Research

1. What are the standard kinematic parameters for larval crawling in the literature?
2. Is there evidence for LED-dependent turn angle distributions (e.g., larger turns during suppression)?
3. What is the minimal viable trajectory model for behavioral neuroscience publication?
4. Should we compare against larvaworld or other established simulators?

---

## DECISION POINT 3: Scientific Focus

### Potential Directions

**Direction A: Methods Paper**
- Focus: Validate the gamma-difference kernel as a compact representation
- Novelty: 6-parameter analytic form vs 12-basis raised-cosine
- Audience: Computational neuroscience methods

**Direction B: Biological Interpretation**
- Focus: What do τ₁ = 0.29s and τ₂ = 3.8s tell us about circuit mechanisms?
- Novelty: Derive timescales from behavior, compare to neural recordings
- Audience: Systems neuroscience

**Direction C: Simulation Platform**
- Focus: Enable in silico experiments with virtual larvae
- Novelty: Predictive simulations for experimental design
- Audience: Behavioral neuroscience labs

**Direction D: Model Comparison**
- Focus: Compare hazard model to alternatives (diffusion, reinforcement learning)
- Novelty: Quantitative model selection
- Audience: Theoretical biology

### Current Assets

| Asset | Status | Publication-Ready |
|-------|--------|-------------------|
| Gamma-diff kernel (6 params) | Validated | Yes |
| Bootstrap CIs | Done | Yes |
| Rate calibration | Documented | Yes |
| Time-rescaling test | Done (mild fail) | Partially |
| Trajectory simulation | Not started | No |
| Condition generalization | Not started | No |

### Questions for Research

1. Which direction has the highest impact-to-effort ratio?
2. What are the key figures needed for each direction?
3. Are there recent publications we should cite or compare against?
4. What would reviewers expect to see for a behavioral neuroscience methods paper?

---

## DECISION POINT 4: Time-Rescaling Failure

### The Finding

Time-rescaling test failed (p < 0.001) for both event sets:
- All events: Mean rescaled IEI = 0.87 (model slightly over-predicts)
- Filtered events: Mean = 2.92 (model under-predicts due to wrong calibration)

### Interpretation

The mild failure (13% deviation from expected) suggests:
1. Minor unmodeled temporal structure (short-term dependencies)
2. Possible post-event refractory period
3. Or just sampling variability

### Options

**Option A: Accept and report**
- The deviation is mild; model captures main dynamics
- Transparent about limitations

**Option B: Add refractory component**
- Post-event suppression: λ(t) × f_ref(t - t_last_event)
- f_ref = exponential or step function
- Would improve IEI fit at cost of 1-2 more parameters

**Option C: Investigate further**
- Residual analysis by LED phase
- Track-specific deviations
- May reveal systematic patterns

### Questions for Research

1. What level of time-rescaling failure is acceptable for publication?
2. Is a refractory component standard in larval behavior models?
3. How much would adding refractoriness improve the fit?
4. Are there other model modifications that address the mild Poisson violation?

---

## DECISION POINT 5: Condition Generalization

### Current Scope

Model is fit to one condition:
- 0→250 PWM intensity
- 10s ON / 20s OFF
- GMR61 optogenetic line

### Available Data

We have 14 experiment files with different conditions:
- `0to250PWM_30`: Full intensity square wave (current model)
- `50to250PWM_30`: Partial intensity ramp
- `T_Bl_Sq_5to15PWM_30`: Temperature + low-intensity LED

### Generalization Approaches

**Approach A: Intensity scaling**
- K(t; intensity) = (intensity / 250) × K(t)
- Simple but may not capture saturation

**Approach B: Separate fits per condition**
- Fit kernel to each condition independently
- Compare parameters across conditions

**Approach C: Hierarchical model**
- Shared kernel shape, condition-specific amplitude/timing
- Most principled but complex

### Questions for Research

1. Is linear intensity scaling biologically plausible for optogenetics?
2. What is the expected dose-response curve for CsChrimson?
3. Would condition generalization significantly strengthen the paper?
4. How much additional data analysis would this require?

---

## SUMMARY: Key Strategic Questions

1. **Event definition**: Keep all 1,407 events or refit on 319 filtered events?

2. **Trajectory simulation**: What is the minimal viable model for publication?

3. **Scientific direction**: Methods paper, biological interpretation, or simulation platform?

4. **Time-rescaling**: Accept mild failure or add refractory component?

5. **Generalization**: Stay condition-specific or attempt generalization?

---

## Requested Outputs

Please provide:

1. **Recommended strategy**: Which decisions to prioritize and in what order

2. **Effort estimates**: Approximate time/complexity for each direction

3. **Impact assessment**: Which directions yield the most publishable results

4. **Literature context**: Key papers we should reference or compare against

5. **Red flags**: Anything about our current approach that seems problematic

6. **Minimal viable paper**: What is the smallest scope that would be publishable?

