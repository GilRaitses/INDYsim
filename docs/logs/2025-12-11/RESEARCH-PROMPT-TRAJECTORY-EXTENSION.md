# Research Prompt: Trajectory Model Extension and Prioritization

## Context

We have completed the minimal viable paper implementation for INDYsim, a hazard-based simulation of Drosophila larval reorientation under optogenetic stimulation. We now seek guidance on prioritization and trajectory model extensions.

---

## WHAT WE HAVE ACCOMPLISHED

### 1. Hazard Model (Complete)

**6-parameter gamma-difference kernel** for LED-ON response:
```
K_on(t) = A·Γ(t;α₁,β₁) - B·Γ(t;α₂,β₂)

Parameters:
  A = 0.456, α₁ = 2.22, β₁ = 0.132s  →  τ₁ = 0.29s (fast)
  B = 12.54, α₂ = 4.38, β₂ = 0.869s  →  τ₂ = 3.81s (slow)
```

**LED-OFF rebound**: K_off(t) = -0.114·exp(-t/2.0s)

**Calibrated intercept**: β₀ = -6.23 (rate-normalized)

### 2. Validation Results

| Metric | Value | Status |
|--------|-------|--------|
| Rate ratio | 0.97 | PASS |
| PSTH correlation | 0.84 | GOOD |
| Suppression | 2.0× emp / 1.9× sim | MATCH |
| Kernel R² | 0.968 | PASS |
| CV R² | 0.961 | PASS |
| Time-rescaling | p < 0.001 (13% deviation) | MILD FAIL |

### 3. Trajectory Simulation (Basic)

**RUN/TURN state machine**:
- Run speed: 1.0 mm/s
- Heading noise: 0.03 rad/√s
- Turn angles: Normal(μ=7°, σ=86°)
- Turn durations: Lognormal(median=1.1s)
- Simulated rate: 1.88 turns/min (matches empirical 1.84/min)

### 4. Publication Assets

- 3 publication-ready figures (kernel, validation, trajectories)
- Complete methods documentation
- Turn angle/duration distributions extracted
- 5 simulated trajectories saved

---

## WHAT GAPS REMAIN

### 1. Trajectory Model Simplifications

The current RUN/TURN model omits:

| Missing Component | Description | Biological Relevance |
|-------------------|-------------|---------------------|
| **Edge avoidance** | Larvae avoid arena boundaries | Required for realistic spatial patterns |
| **Head sweeps** | Sampling behavior before turning | May affect turn direction choice |
| **Reversals** | Backward crawling | Rare but distinct behavior |
| **Speed gradients** | Acceleration/deceleration | Smooth transitions |
| **Directional coupling** | Turn direction based on prior heading or stimulus | Chemotaxis-like behavior |

### 2. Event Definition Ambiguity

- 77% of events have `turn_duration = 0` (micro-movements)
- We use all 1,407 for hazard fitting but only 319 filtered for trajectory output
- Unclear if this two-stage approach is optimal

### 3. Condition Specificity

Model is fit to single condition only:
- 0→250 PWM, 10s ON / 20s OFF
- GMR61 line
- No intensity scaling validated

### 4. Time-Rescaling Violation

- Mean rescaled IEI = 0.87 (expected 1.0)
- 13% systematic deviation indicates unmodeled structure
- Possible short-term refractoriness not captured

### 5. Held-Out Validation

- Model validated within-experiment only
- No cross-experiment validation performed
- Other experimental files available but not used

---

## UNCERTAINTIES AND CONFUSIONS

### 1. Edge Avoidance Implementation

**Question**: How do larvae detect and respond to edges?

Options:
- **Hard boundary**: Reflect off walls
- **Soft repulsion**: Bias heading away from edges
- **Probabilistic turn**: Increase turn probability near edges
- **Vision-based**: Respond to edge contrast

**Uncertainty**: I don't know which mechanism is biologically accurate or commonly used in larval simulations.

### 2. Head Sweeps

**Question**: Are head sweeps a distinct state or part of turning?

From the data, I see `is_reorientation_start` events but unclear:
- Do head sweeps precede all turns?
- Are they separate from the turn itself?
- Should they affect turn direction choice?

**Uncertainty**: I don't have clear data on head sweep frequency or duration.

### 3. Turn Direction Bias

**Question**: Is turn direction purely random or influenced by:
- Prior heading?
- LED state?
- Position in arena?
- Time since last turn?

Current model: Turn angles are sampled independently from Normal(μ=7°, σ=86°).

The 7° rightward bias is small but consistent. I don't know if this is:
- Real behavioral asymmetry
- Artifact of data processing
- Should be modeled as 0°

### 4. Priority Trade-offs

**Question**: What provides more value for a behavioral neuroscience paper?

| Option | Adds | Costs |
|--------|------|-------|
| **A: Write paper** | Publication | Locks in current limitations |
| **B: More validation** | Confidence | Delays publication |
| **C: Extend model** | Realism | Complexity, potential bugs |

**Uncertainty**: I don't know what reviewers would expect or what would differentiate this work.

---

## THREE OPTIONS FOR NEXT STEPS

### Option A: Write Paper Draft

**What it involves**:
- Draft introduction, methods, results, discussion
- Use existing figures and MODEL_SUMMARY.md
- Submit as short methods paper

**Pros**:
- Fast (2-3 days)
- Uses current validated assets
- Publication is the goal

**Cons**:
- Trajectory model is basic (no edge avoidance)
- May lack novelty for high-impact venue
- Limitations become fixed

### Option B: Run Additional Validation

**What it involves**:
- Validate on held-out experiments (12 unused files)
- Cross-condition comparison (50→250 PWM)
- Add residual diagnostics

**Pros**:
- Strengthens claims
- May reveal model limitations
- More rigorous

**Cons**:
- Delays publication
- May uncover problems requiring fixes
- Diminishing returns if model already good

### Option C: Extend Trajectory Model

**What it involves**:
- Add edge avoidance (arena boundary)
- Add head sweeps (optional)
- Improve turn angle model (directional coupling?)

**Pros**:
- More realistic simulations
- Better visualization
- Platform for future work

**Cons**:
- 2-5 days additional work
- Adds complexity
- May require additional validation

---

## SPECIFIC QUESTIONS

### Prioritization

1. **Which option (A, B, or C) should we prioritize first?**
   - What is the highest impact-to-effort choice?
   - What would reviewers expect for a behavioral neuroscience methods paper?

2. **If Option C, what extensions are most important?**
   - Edge avoidance vs head sweeps vs directional coupling?
   - What is the minimal addition that significantly improves realism?

### Edge Avoidance

3. **How do larvae typically respond to arena edges?**
   - What is the biological mechanism?
   - What is the standard modeling approach in larvaworld or similar?

4. **What arena geometry should we assume?**
   - Circular vs rectangular?
   - Typical size (diameter/side length)?

### Head Sweeps

5. **Are head sweeps worth modeling explicitly?**
   - Do they affect turn direction choice?
   - What is the typical frequency and duration?

6. **How are head sweeps distinguished from turns in the data?**
   - Are they captured by `is_reorientation_start`?
   - Would modeling them change simulation output meaningfully?

### Turn Direction

7. **Is the 7° rightward bias real or artifact?**
   - Should we model it as 0° mean?
   - Are there published values for turn angle bias?

8. **Should turn direction depend on LED state?**
   - Do larvae show different turn patterns during suppression vs baseline?
   - Would this add meaningful biological insight?

---

## REQUESTED OUTPUTS

Please provide:

1. **Recommended priority order** for Options A, B, C with rationale

2. **If extending trajectory model (Option C)**:
   - Which components to add (ranked by importance)
   - Implementation guidance for edge avoidance
   - Whether head sweeps are worth the effort

3. **Edge avoidance specification**:
   - Recommended mechanism (hard/soft/probabilistic)
   - Typical arena size and geometry
   - Key parameters to tune

4. **Assessment of current assets**:
   - Are existing figures and validation sufficient for publication?
   - What would strengthen the paper most?

5. **Red flags**:
   - Anything about our approach that seems problematic?
   - Common mistakes in larval trajectory simulation?

