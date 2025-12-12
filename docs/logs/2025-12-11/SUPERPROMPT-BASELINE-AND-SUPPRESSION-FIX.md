# Comprehensive Research Prompt: Baseline Rate and Early Suppression Fixes

**Date**: 2025-12-11  
**Context**: Post-kernel-timing-fix analysis  
**Primary Issues**: Baseline rate mismatch (1.5x) and early suppression magnitude gap

---

## 1. Executive Summary

After fixing the kernel timing issue (PSTH correlation now 0.817), two residual problems remain:

1. **Baseline rate mismatch**: Simulated pre-onset rate (2.38 events/s) is 1.5x higher than empirical (1.58 events/s)
2. **Early suppression magnitude**: Simulated normalized rate at 0-3s (0.82) is weaker than empirical (0.55)

These issues affect the absolute accuracy of the model but not the relative temporal pattern.

---

## 2. Current Model State

### Extended Biphasic Kernel (Working)

| Component | Configuration |
|-----------|---------------|
| Early kernel | Centers [0.2, 0.7, 1.4]s, width 0.4s, **non-negative constraint** |
| Late kernel | Centers [3.0, 5.0, 7.0, 9.0]s, width 1.8s, unconstrained |
| LED main effect | **Removed** |
| Intercept | -6.689 |

### Fitted Coefficients

| Coefficient | Value | Constraint |
|-------------|-------|------------|
| intercept | -6.689 | None |
| kernel_early_1 | +0.698 | ≥ 0 |
| kernel_early_2 | **0.000** | ≥ 0 (binding) |
| kernel_early_3 | **0.000** | ≥ 0 (binding) |
| kernel_late_1 | -3.749 | None |
| kernel_late_2 | -1.876 | None |
| kernel_late_3 | -0.283 | None |
| kernel_late_4 | -0.253 | None |

### Validation Results

| Metric | Empirical | Simulated | Issue |
|--------|-----------|-----------|-------|
| Pre-onset rate | 1.58 | 2.38 | **+50% too high** |
| Early normalized (0-3s) | 0.55 | 0.82 | **+27% too weak suppression** |
| Late normalized (3-8s) | 0.32 | 0.40 | +8% (acceptable) |
| PSTH correlation | - | 0.817 | PASS |

---

## 3. Problem Analysis

### 3.1 Baseline Rate Mismatch

**Observation**: Simulated baseline (pre-onset) rate is 2.38 vs empirical 1.58.

**Potential Causes**:

1. **Intercept is too high**
   - Current intercept: -6.689
   - Expected rate: exp(-6.689) × 20 Hz = 0.00125 × 20 = 0.025 events/frame = 0.5 events/s
   - But simulated shows 2.38 events/s... where does the extra rate come from?

2. **Frame rate conversion in simulation**
   - Model was fit on frame-level data (20 Hz)
   - Simulation multiplies hazard by frame_rate (20) to convert to per-second
   - Is this double-counting?

3. **PSTH calculation method**
   - PSTH is computed as events per bin / (n_onsets × bin_size)
   - This gives events per second per LED cycle
   - But we have 99 tracks... should we divide by tracks too?

4. **Track normalization**
   - Empirical: 1,407 events across 99 tracks, 40 LED cycles
   - Expected per cycle per track: 1407 / 99 / 40 = 0.36 events/cycle
   - Over 30s cycle: 0.36 / 30 = 0.012 events/s per track
   - This doesn't match 1.58... so PSTH is summed across tracks

**Question**: How should the PSTH be normalized to compare empirical vs simulated fairly?

### 3.2 Early Suppression Magnitude Gap

**Observation**: Normalized rate at 0-3s is 0.82 (sim) vs 0.55 (emp).

**Potential Causes**:

1. **Non-negative constraint is too restrictive**
   - Early kernel bases 2 and 3 are constrained to ≥ 0
   - They are hitting the constraint (both = 0.000)
   - The model WANTS them to be negative but can't
   - This means we're forcing positive/neutral at 0.7-1.4s when empirical shows suppression

2. **Gap between early and late kernels**
   - Early kernel ends at 1.5s (split point)
   - Late kernel has first center at 3.0s
   - Gap from 1.5-3s has weak coverage
   - Suppression at 1.5-3s may be under-modeled

3. **Early positive response is real but masked**
   - The +0.698 at 0.2s is capturing a real early response
   - But the subsequent suppression (0.7-1.4s) is being blocked by constraint

**Question**: Should we relax the non-negative constraint, or add intermediate kernel bases?

---

## 4. Specific Questions for Research Agent

### Q1: Baseline Rate Normalization

I'm computing PSTH as:
```python
rates = counts / (n_onsets * bin_size)
```

Where `counts` is the sum across all 99 tracks. The simulated data also has 99 tracks.

- Is this the correct normalization for comparing empirical vs simulated PSTH?
- Should I divide by number of tracks to get per-track rate?
- Or is the mismatch coming from the hazard → event generation step?

### Q2: Intercept Adjustment Strategy

To reduce baseline rate from 2.38 to 1.58, I would need to shift intercept by:
```
Δ = ln(1.58 / 2.38) = -0.41
```

New intercept: -6.689 - 0.41 = -7.10

**But**: If I just shift the intercept, will the kernel coefficients remain valid? Or do I need to refit the entire model with a different intercept initialization?

### Q3: Non-Negative Constraint Relaxation

The early kernel constraint (≥ 0) was intended to enforce "early positive response." But:

- Early bases 2 and 3 are hitting the constraint at exactly 0
- This suggests the true early response is NOT uniformly positive
- Should I:
  - (A) Remove the constraint entirely and let early kernel be unconstrained?
  - (B) Only constrain the first early basis (0.2s) to be non-negative?
  - (C) Use a weaker constraint (e.g., kernel_early ≥ -1)?

### Q4: Intermediate Kernel Coverage

There's a gap between early kernel (ends at 1.5s) and late kernel (first center at 3.0s):

- Time 1.5-3s has weak basis function coverage
- Should I:
  - (A) Add an intermediate kernel with centers at [2.0, 2.5]s?
  - (B) Shift late kernel first center from 3.0s to 2.0s?
  - (C) Extend early kernel to cover [0, 3]s with more bases?

### Q5: Is 27% Early Suppression Gap Acceptable?

Given:
- Pseudo-R² is only 0.0104 (very low explained variance)
- This is typical for sparse event data (99.9% zeros)
- The correlation is 0.817 which exceeds the 0.80 target

Is a 27% difference in early phase magnitude (normalized) acceptable for this application? Or is this a fundamental model deficiency that must be fixed?

### Q6: What Is the Standard for "Good Enough"?

In the larval LNP literature (Hernandez-Nunez, Gepner, etc.):

- What PSTH correlation values are typically reported?
- What absolute rate match tolerances are considered acceptable?
- Do papers normalize PSTH before comparing, or compare absolute rates?

---

## 5. Candidate Solutions to Evaluate

### Solution A: Intercept Post-Hoc Adjustment

1. Compute scale factor: `scale = 1.58 / 2.38 = 0.664`
2. Multiply all simulated hazards by 0.664
3. Re-run simulation and validation

**Pro**: Simple, preserves kernel shape  
**Con**: Is this scientifically defensible?

### Solution B: Refit with Relaxed Early Constraint

1. Remove non-negative constraint on early kernel bases 2 and 3
2. Keep only kernel_early_1 ≥ 0 (the 0.2s basis)
3. Refit and validate

**Pro**: Allows model to capture early suppression  
**Con**: May cause identifiability issues with late kernel

### Solution C: Add Intermediate Kernel Bases

1. Add 2 bases at [2.0, 2.5]s with width 0.8s
2. Make them unconstrained
3. Refit and validate

**Pro**: Better coverage of 1.5-3s gap  
**Con**: More parameters, may not improve fit

### Solution D: Normalize Both PSTHs Before Comparison

1. Divide both empirical and simulated PSTH by their pre-onset baseline
2. Compare normalized shapes only
3. Accept baseline mismatch as a known limitation

**Pro**: Already done, and correlation is 0.817  
**Con**: Doesn't fix the underlying rate mismatch

### Solution E: Refit with Constrained Intercept

1. Add a constraint that intercept produces baseline rate matching empirical
2. Refit with this constraint
3. Allow kernel to adjust

**Pro**: Forces baseline match  
**Con**: May worsen kernel fit

---

## 6. Data for Debugging

### Hazard at Key Timepoints

| Time since onset | Current Hazard | Target (if baseline 1.5x lower) |
|------------------|----------------|--------------------------------|
| 0.0s | 0.00177 | 0.00118 |
| 0.5s | 0.00138 | 0.00092 |
| 1.0s | 0.00125 | 0.00083 |
| 2.0s | 0.00026 | 0.00018 |
| 3.0s | 0.00003 | 0.00002 |
| 5.0s | 0.00019 | 0.00013 |

### Empirical vs Simulated Event Counts

| Data | Total Events | Tracks | Duration | Rate/min/track |
|------|--------------|--------|----------|----------------|
| Empirical | 1,407 | 99 | 20 min | 0.71 |
| Simulated | 2,501 | 99 | 20 min | 1.26 |

**VERIFIED CALCULATION** (2025-12-11):

| Data | Events | Tracks | Duration | Rate/min/track |
|------|--------|--------|----------|----------------|
| Empirical | 1,407 | 99 | 20 min | **0.71** |
| Simulated | 2,501 | 99 | 20 min | **1.26** |
| **Ratio** | - | - | - | **1.78x** |

The earlier "1.19" figure was incorrect. The true empirical rate is 0.71 events/min/track.

**This means the simulated rate is 78% too high, not 50%.**

The baseline rate mismatch is more severe than initially thought.

---

## 7. Summary of Uncertainties

| Issue | Uncertainty Level | Impact |
|-------|-------------------|--------|
| Baseline rate mismatch cause | High | Affects absolute predictions |
| Non-negative constraint necessity | Medium | Affects early phase fit |
| PSTH normalization method | Medium | Affects validation interpretation |
| Acceptable tolerance for rate mismatch | Low | Reporting/framing only |
| Intermediate kernel necessity | Medium | May improve early phase |

---

## 8. What I Need from Research Agent

1. **Clarify PSTH normalization**: Should PSTH be per-track or summed across tracks?
2. **Recommend constraint strategy**: Keep, relax, or remove non-negative constraint?
3. **Validate intercept adjustment approach**: Is post-hoc scaling defensible?
4. **Set expectations**: What rate match tolerance is acceptable in published work?
5. **Identify calculation discrepancy**: Why is empirical rate 0.71 vs 1.19?

---

## 9. Files for Reference

| File | Description |
|------|-------------|
| `data/model/extended_biphasic_model_results.json` | Current model coefficients |
| `scripts/fit_extended_biphasic_model.py` | Fitting script |
| `scripts/simulate_extended_biphasic.py` | Simulation script |
| `docs/logs/2025-12-11/RESPONSE-KERNEL-TIMING-FIX.md` | Previous MiroThinker response |

---

## 10. Code Snippets for Context

### Current PSTH Calculation

```python
def compute_psth(events: np.ndarray, bin_size: float = 0.2, window: tuple = (-3.0, 8.0)) -> tuple:
    led_onsets = np.arange(FIRST_LED_ONSET, EXPERIMENT_DURATION, LED_CYCLE)
    bins = np.arange(window[0], window[1] + bin_size, bin_size)
    bin_centers = (bins[:-1] + bins[1:]) / 2
    
    counts = np.zeros(len(bin_centers))
    n_onsets = 0
    
    for onset in led_onsets:
        for i, bc in enumerate(bin_centers):
            t_start = onset + bc - bin_size / 2
            t_end = onset + bc + bin_size / 2
            counts[i] += np.sum((events >= t_start) & (events < t_end))
        n_onsets += 1
    
    # This sums across all tracks, divided by n_onsets and bin_size
    rates = counts / (n_onsets * bin_size)
    return bin_centers, rates
```

### Current Event Generation

```python
def generate_events_inversion(hazard_func, t_start, t_end, dt=0.05, frame_rate=20.0, rng=None):
    events = []
    t = t_start
    
    while t < t_end:
        u = rng.random()
        target_integral = -np.log(u)
        
        cumulative = 0.0
        while cumulative < target_integral and t < t_end:
            h = hazard_func(t)  # Per-frame hazard
            h_per_sec = h * frame_rate  # Convert to per-second
            cumulative += h_per_sec * dt
            t += dt
        
        if t < t_end:
            events.append(t)
    
    return np.array(events)
```

---

## 11. Gaps in Current Plan

### Gap 1: Intercept Adjustment May Break Kernel Timing

The plan proposes adjusting intercept by -0.41 to fix baseline. But:
- The kernel was fitted jointly with the intercept
- Changing intercept alone may shift the effective suppression magnitude
- May need full refit with constrained baseline

**Uncertainty**: Will post-hoc intercept adjustment preserve the kernel timing we just fixed?

### Gap 2: Constraint Relaxation Strategy Not Specified

The plan mentions relaxing the non-negative constraint, but:
- No specific strategy for which bases to unconstrain
- No analysis of identifiability risks
- No fallback if early kernel becomes negative and competes with late kernel

**Uncertainty**: How to relax constraints without causing identifiability issues?

### Gap 3: Gap Between Early and Late Kernels

There's weak coverage from 1.5-3.0s:
- Early kernel ends at 1.5s
- Late kernel first center is at 3.0s
- This 1.5s gap may cause the early suppression magnitude issue

**Uncertainty**: Should we add intermediate bases or shift existing bases?

### Gap 4: Rate Calculation Discrepancy Now Confirmed

The empirical rate was incorrectly reported as 1.19, but is actually **0.71** events/min/track.
- This means simulated rate (1.26) is **1.78x** too high, not 1.5x
- The mismatch is more severe than previously thought

**Uncertainty**: Why was the rate initially calculated incorrectly? Are there other calculation errors?

### Gap 5: Frame Rate Conversion Uncertainty

The simulation multiplies hazard by frame_rate (20) to convert per-frame to per-second:
```python
h_per_sec = h * frame_rate  # Convert to per-second
```

But the model was fitted at 20 Hz, so:
- Intercept exp(-6.689) = 0.00124 is already per-frame probability
- Multiplying by 20 gives 0.0248 events/second/track
- Over 1200s: 0.0248 × 1200 = 29.8 events/track
- For 99 tracks: 29.8 × 99 = 2,950 events (close to simulated 2,501)

But empirical has 1,407 events, suggesting the baseline probability should be:
- Target: 1407 / 99 / 1200 = 0.0118 events/second/track
- Per-frame at 20 Hz: 0.0118 / 20 = 0.00059
- Log: ln(0.00059) = -7.44

So intercept should be approximately **-7.44**, not -6.689.

**This explains the 1.78x mismatch!** Intercept is ~0.75 too high.

### Gap 6: Mixed-Effects Model May Not Converge

The plan suggests fitting NB-GLMM with extended kernel, but:
- More parameters than before
- Random effects add complexity
- May fail to converge

**Uncertainty**: Is mixed-effects model feasible with extended kernel?

---

## 12. Corrected Analysis

### Expected Intercept Calculation

Given:
- Empirical events: 1,407
- Tracks: 99
- Duration: 1,200s (20 min)
- Frame rate: 20 Hz

Empirical event rate:
- Per second per track: 1407 / 99 / 1200 = 0.0118 events/s/track
- Per frame per track: 0.0118 / 20 = 0.00059

Required intercept (without kernel):
- ln(0.00059) = **-7.44**

Current intercept: -6.689
Difference: -7.44 - (-6.689) = **-0.75**

To fix baseline, intercept should be reduced by approximately **0.75**, not 0.41.

### Revised Fix Strategy

1. **Refit model** with intercept initialized at -7.5
2. Allow kernel to adjust to new baseline
3. Validate PSTH correlation is maintained
4. If early suppression still weak, consider relaxing non-negative constraint

---

## 13. Bottom Line Questions

1. **Why is simulated rate 1.78x too high?** Is it purely an intercept issue, or is the frame rate conversion wrong?
2. **Should I refit with lower intercept, or adjust post-hoc?**
3. **Should I relax the non-negative constraint** to improve early suppression?
4. **Is the current 0.817 correlation "good enough"** even with 1.78x rate mismatch?
5. **What is the standard practice** for handling baseline rate mismatch in LNP validation?
6. **Are there other calculation errors** I should check for?



