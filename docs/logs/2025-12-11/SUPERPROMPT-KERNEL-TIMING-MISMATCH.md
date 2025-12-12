# Comprehensive Research Prompt: Kernel Timing Mismatch in LNP Model

**Date**: 2025-12-11  
**Context**: Debugging PSTH correlation failure in biphasic LNP model for larval reorientation  
**Primary Issue**: Kernel suppression timing is inverted relative to empirical data

---

## 1. Executive Summary

I have built a biphasic LNP (Linear-Nonlinear-Poisson) model for Drosophila larval reorientation during optogenetic stimulation. The model achieves:

- **Turn rate match**: 1.28 sim vs 1.19 emp (PASS)
- **PSTH correlation**: 0.72 (improved from 0.50, but still below 0.80 target)
- **Absolute rate mismatch**: Simulated baseline is ~2x empirical

**Critical finding**: The normalized PSTH reveals an **inverted suppression pattern**:
- Empirical: suppression builds over time (0.79 → 0.56)
- Simulated: suppression fades over time (0.56 → 0.68)

This is a fundamental kernel timing mismatch.

---

## 2. Experimental Context

### Stimulus Protocol
- **LED pattern**: 10s ON / 20s OFF (30s cycle)
- **Intensity**: 0-250 PWM ramp over ~5s, then plateau
- **First onset**: ~21s
- **Duration**: 20 minutes (40 cycles)

### Mirna's Analysis Window
- **Integration window**: -3s before to +8s after LED onset
- This captures pre-onset baseline, early response, and peak suppression phase

### Data Characteristics
- 1.3M observations, 1,407 events
- 99 tracks across 2 experiments
- Event rate: 1.19 events/min/track (0.11% of frames)
- Frame rate: 20 Hz (50ms per frame)

---

## 3. Current Model Specification

### Biphasic Kernel Structure
- **Early kernel**: 0-1.5s, 3 raised-cosine bases, constrained non-negative
- **Late kernel**: 1.5-6.0s, 4 raised-cosine bases, unconstrained

### Fitted Coefficients
```
intercept:       -6.668 (baseline log-hazard)
LED1_scaled:     -0.022 (near zero, not significant)
kernel_early_1:  +0.231 (early positive response)
kernel_early_2:  +0.886 (early positive response)
kernel_early_3:  +1.065 (early positive response)
kernel_late_1:   -3.224 (strong suppression at 1.5s)
kernel_late_2:   -3.489 (strong suppression at 3.0s)
kernel_late_3:   -2.822 (suppression at 4.5s)
kernel_late_4:   -0.943 (weak suppression at 6.0s)
led_off_rebound: -0.111 (negative, not significant)
```

### Kernel Center Locations
- Early centers: [0.0, 0.75, 1.5] seconds
- Late centers: [1.5, 3.0, 4.5, 6.0] seconds
- Early width: 0.6s
- Late width: 1.2s

---

## 4. The Problem: Detailed Diagnosis

### PSTH Comparison (using -3s to +8s window)

| Phase | Empirical | Simulated | Ratio | Issue |
|-------|-----------|-----------|-------|-------|
| Pre-onset (-3 to 0s) | 1.40 | 2.43 | 1.74x | Baseline too high |
| Early (0-3s) | 1.10 | 1.35 | 1.23x | Slightly high |
| Late (3-8s) | 0.78 | 1.66 | 2.13x | Way too high! |

### Normalized Pattern (relative to baseline)

| Phase | Empirical | Simulated | Expected |
|-------|-----------|-----------|----------|
| Pre-onset | 1.00 | 1.00 | Match |
| Early (0-3s) | 0.79 | 0.56 | Sim suppresses MORE than emp |
| Late (3-8s) | 0.56 | 0.68 | Sim suppresses LESS than emp |

### The Inversion Problem

The empirical data shows:
```
Time 0s → 3s: Mild suppression (21% below baseline)
Time 3s → 8s: Strong suppression (44% below baseline)
```

The simulated data shows:
```
Time 0s → 3s: Strong suppression (44% below baseline)
Time 3s → 8s: Recovery (only 32% below baseline)
```

**The kernel suppression peaks at the WRONG TIME.**

---

## 5. Hazard Function Analysis

I computed the hazard rate at different times since LED onset:

| Time since onset | Hazard (events/s) | Kernel contribution |
|------------------|-------------------|---------------------|
| 0.0s | 0.031 | +0.23 (early) |
| 0.5s | 0.036 | +0.57 (early peak) |
| 1.0s | 0.006 | -2.03 (late starts) |
| 1.5s | 0.003 | -3.22 (late max) |
| 3.0s | 0.001 | -3.49 (strongest suppression) |
| 4.0s | 0.003 | -2.01 |
| 5.0s | 0.004 | -1.84 |
| 6.0s | 0.010 | -0.94 (recovering) |
| 10.0s | 0.025 | 0.00 (back to baseline) |

**Observation**: Maximum suppression occurs at 1.5-3.0s, but empirical data shows maximum suppression at 3-8s.

---

## 6. Suspected Root Causes

### Cause 1: Kernel Window Is Too Short
The late kernel extends only to 6.0s, but empirical suppression continues through 8s.

**Fix**: Extend late kernel window to [1.5, 10.0] seconds.

### Cause 2: Basis Centers Are Misaligned
Late kernel centers at [1.5, 3.0, 4.5, 6.0] place most suppressive power in the 1.5-4.5s range.

**Fix**: Shift centers to [3.0, 5.0, 7.0, 9.0] to match empirical suppression timing.

### Cause 3: Split Point Is Too Early
The 1.5s split means early kernel covers 0-1.5s and late kernel covers 1.5-6s. But the empirical "early response" extends to 3s.

**Fix**: Use 3.0s split point (matching Mirna's window).

### Cause 4: Model Doesn't Capture Delayed Suppression
The suppression builds gradually in empirical data (adaptation-like), but the kernel imposes suppression immediately at the wrong phase.

**Fix**: Consider a parametric kernel with explicit latency parameter, or use difference-of-exponentials form.

### Cause 5: Baseline Rate Scaling
The simulated baseline is 2.4 vs empirical 1.4. Even with correct kernel shape, this offset causes all rates to be too high.

**Fix**: Adjust intercept or use rate normalization in validation.

---

## 7. Specific Questions

### Q1: Kernel Window Extension

The empirical suppression continues through 8s. Should I:
- Extend the late kernel to [1.5, 10.0]s?
- Or add a third "sustained suppression" kernel covering [6.0, 10.0]s?
- What is the biological rationale for suppression lasting this long?

### Q2: Basis Center Placement

My late kernel centers are at [1.5, 3.0, 4.5, 6.0]s, but suppression is strongest at 3-8s. Should I:
- Shift centers to [3.0, 5.0, 7.0, 9.0]s?
- Use non-uniform spacing (denser at 4-6s)?
- Let the data determine centers via cross-validation?

### Q3: Split Point Selection

I used 1.5s split based on AIC comparison of [1.0, 1.5, 2.0]s. But Mirna's window suggests the "early" phase extends to 3s. Should I:
- Use 3.0s split point?
- Define split point from reverse-correlation zero-crossing?
- Use three phases: early (0-1s), transition (1-3s), late (3-10s)?

### Q4: Parametric vs Basis Kernel

The raised-cosine basis may be too smooth to capture the sharp suppression onset. Should I consider:
- Difference-of-exponentials: K(t) = A₁exp(-t/τ₁) - A₂exp(-t/τ₂)
- Alpha function: K(t) = (t/τ)exp(-t/τ)
- Latency parameter: K(t-δ) where δ is fitted delay

### Q5: Baseline Rate Normalization

The simulated baseline is 74% higher than empirical. For validation purposes, should I:
- Normalize PSTH before comparing (divide by baseline)?
- Adjust the intercept post-hoc to match empirical rate?
- Consider this an acceptable model limitation?

### Q6: Integration Window

Mirna mentioned -3s to +8s as the integration window. Is this:
- The window for computing the PSTH?
- The window for fitting the kernel (stimulus-triggered average)?
- A standard in the field for larval optogenetics?

---

## 8. What I've Already Tried

1. **Single kernel (0-6s, 5 bases)**: PSTH correlation 0.50, wrong peak timing
2. **Biphasic kernel (0-1.5s + 1.5-6s)**: PSTH correlation 0.72, inverted suppression pattern
3. **Added LED-off rebound term**: Coefficient was negative and not significant
4. **Ridge + smoothness regularization**: Helped stability but didn't fix timing
5. **Non-negative constraint on early kernel**: Early coefficients are now positive

---

## 9. Additional Context

### Event Detection
Reorientation events are detected from trajectory curvature at 20 Hz. The 0.2s peak in earlier analysis may be a detection artifact rather than true biology.

### Reverse Correlation Results
Reverse correlation kernel was **not clearly biphasic** (z-score < 2 at peak), suggesting either:
- The signal is weak in this genotype
- The LED signal doesn't have strong temporal structure before events
- Event detection noise obscures the true kernel

### GMR61 Phenotype
GMR61 is an uncharacterized driver line in larvae. The overall LED effect appears to be **suppressive** (79% of events during LED OFF), which is novel and not predicted by prior literature.

---

## 10. Summary Table

| Issue | Evidence | Suspected Fix | Uncertainty |
|-------|----------|---------------|-------------|
| Suppression timing inverted | Sim peaks at 1.5s, emp at 5s | Shift kernel centers to 3-9s | Will this match biology? |
| Kernel window too short | Late kernel ends at 6s, emp suppression continues to 8s | Extend to 10s | How many bases needed? |
| Split point too early | 1.5s split, but early phase is 0-3s | Use 3.0s split | Is there a principled way to choose? |
| Baseline 74% too high | Intercept gives higher rate | Normalize or adjust intercept | Is this acceptable? |
| Rebound not captured | LED-off coefficient negative | May need offset-locked kernel | What's the recovery timescale? |

---

## 11. What I Need

1. **Validate my diagnosis**: Is the kernel timing mismatch the core issue?
2. **Recommend kernel redesign**: What window, centers, and basis functions to use?
3. **Clarify Mirna's window**: Is -3s to +8s standard? Should my kernel match this?
4. **Set realistic expectations**: Given this data, what PSTH correlation is achievable?
5. **Identify any conceptual errors**: Am I thinking about this correctly?

---

## References

1. Hernandez-Nunez et al. 2015, Nat Commun - Larval navigation LNP
2. Gepner et al. 2015, eLife - Larval phototaxis LNP  
3. Gepner et al. 2018, eLife - Variance adaptation
4. Pillow et al. 2008, Nature - Raised-cosine basis functions
5. Truccolo et al. 2005, J Neurophysiol - Point process GLM



