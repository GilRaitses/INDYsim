# MiroThinker Response: Baseline Rate and Early Suppression Fixes

**Date**: 2025-12-11  
**Source**: MiroThinker (MiroMind)  
**Topic**: Root cause of baseline rate mismatch and recommended fixes

---

## Executive Summary

The baseline rate mismatch is a **code bug**, not a model flaw:

- **Root Cause**: The event generator multiplies hazard by `frame_rate` when it shouldn't
- **Effect**: Inflates simulated event rate by ~20x (partially masked by other factors)
- **Fix**: Remove `* frame_rate` from hazard integration

---

## 1. Root Cause: Hazard Integration Bug

### Current Code (WRONG)

```python
h = hazard_func(t)          # per-frame hazard
h_per_sec = h * frame_rate  # "convert" to per-second
cumulative += h_per_sec * dt
```

### Problem

If `hazard_func` returns hazard per second (as GLM does), then:
- Correct integral increment: `h * dt`
- Current integral increment: `h * frame_rate * dt = h * 20 * 0.05 = h * 1.0`

This inflates the event rate by a factor of ~20!

### Correct Code

```python
h = hazard_func(t)       # hazard in events/s
cumulative += h * dt     # dimensionless integral
```

Remove the `* frame_rate` line entirely.

---

## 2. PSTH Normalization Fix

### Current Code (WRONG)

```python
rates = counts / (n_onsets * bin_size)
```

This gives population-level events/s (summed across 99 tracks).

### Correct Code

```python
rates_per_track = counts / (n_onsets * bin_size * n_tracks)
```

This gives per-track events/s, which is what the model predicts.

---

## 3. Early Kernel Constraint Strategy

### Recommended Approach

Only constrain the first early basis (0.2s) to be non-negative:

| Basis | Center | Current Constraint | Recommended |
|-------|--------|-------------------|-------------|
| early_1 | 0.2s | ≥ 0 | **Keep ≥ 0** |
| early_2 | 0.7s | ≥ 0 (binding at 0) | **Unconstrained** |
| early_3 | 1.4s | ≥ 0 (binding at 0) | **Unconstrained** |

This allows the model to capture early suppression at 0.7-1.4s while preserving the known early positive bump.

---

## 4. Intermediate Kernel Coverage

### Gap Identified

- Early kernel ends at ~1.5s
- Late kernel starts at 3.0s (first center)
- **1.5-3.0s is under-represented**

### Solution

Add intermediate bases at [2.0, 2.5]s with width ~0.6s, unconstrained.

Or extend early kernel to cover [0, 3]s with more bases.

---

## 5. Rate Discrepancy Explained

| Calculation | Result | Source |
|-------------|--------|--------|
| 1407 / 99 / 20 min | 0.71 events/min/track | Direct count (CORRECT) |
| ~1.19 events/min/track | INCORRECT | Likely from population PSTH without dividing by tracks |

Use 0.71 events/min/track as the ground truth.

---

## 6. Implementation Order

1. **Fix hazard integration**: Remove `* frame_rate` from `generate_events_inversion()`
2. **Fix PSTH normalization**: Divide by `n_tracks`
3. **Relax early kernel constraints**: Only keep first basis non-negative
4. **Add intermediate bases**: [2.0, 2.5]s with width 0.6s
5. **Refit model** with corrected baseline
6. **Validate**: Check PSTH correlation remains >0.8

---

## 7. Realistic Expectations

| Metric | Typical Value | Current | Target |
|--------|---------------|---------|--------|
| PSTH correlation | 0.7-0.85 | 0.817 | GOOD |
| Pseudo-R² | 0.02-0.05 | 0.0104 | Acceptable |
| Baseline rate match | Within 2x | 1.78x (to be fixed) | Within 20% |
| Early suppression error | 10-30% | 27% | Acceptable |

---

## 8. Key Quotes

> "The baseline over‑estimation is a bug, not a model flaw; fix it and then relax the early‑kernel constraints (or add intermediate bases)."

> "The correlation of 0.817 is good enough for publication – the early suppression magnitude is a known limitation of sparse larval event data."

> "Remove the * frame_rate line, or set frame_rate=1.0 and treat dt as the only time step."

---

## References

- Hernández‑Núñez et al. 2015, Nat Commun
- Gepner et al. 2015, eLife
- Gepner et al. 2018, eLife
- Truccolo et al. 2005, J Neurophysiol



