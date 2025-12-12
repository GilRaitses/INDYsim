# MiroThinker Response: Kernel Timing Fix

**Date**: 2025-12-11  
**Source**: MiroThinker (MiroMind)  
**Topic**: Kernel timing mismatch diagnosis and fix recommendations

---

## Key Findings

### 1. Diagnosis Confirmed
The kernel timing mismatch is the **core issue**:
- Current kernel: strongest suppression at 1.5-3s
- Empirical data: strongest suppression at 3-8s
- This phase inversion explains the poor PSTH correlation

### 2. Recommended Kernel Redesign

#### (A) Extend and Shift Late Kernel
- **Window**: Extend from 1.5-6s to **2-10s**
- **Centers**: Shift from [1.5, 3, 4.5, 6] to **[3, 5, 7, 9]s**
- **Widths**: 1.0-1.5s for early-late (2-4s), 2.0s for late-sustained (6-10s)

#### (B) Adjust Split Point
- Current: 1.5s (too early)
- Recommended: **2-3s** (matching Mirna's window)
- Test candidates: [1.5, 2.0, 2.5, 3.0]s via AIC and PSTH correlation

#### (C) Narrower Early Bases
- Current: 3 bases over 0-1.5s, width 0.6s
- Recommended: 2-3 bases over 0-2s, centers [0.2, 0.7, 1.4]s, width 0.3-0.5s

### 3. LED Main Effect
- Current coefficient: -0.022 (near zero, not significant)
- Recommendation: **Remove or de-emphasize** - let kernel carry all LED dynamics

### 4. Baseline Rate Fix
- Current: simulated baseline is 2x empirical
- Fix: Reduce intercept by ln(1.4/2.4) ≈ -0.54 in full refit
- For validation: use baseline-normalized PSTH as primary timing metric

### 5. Realistic Targets
- **r = 0.7-0.75** is realistic for this data
- Use bootstrap-derived W-ISE threshold rather than hard-coding r ≥ 0.8
- Pseudo-R² ≈ 0.03 indicates modest stimulus-explained variance

### 6. Action Order
1. Confirm early peak is real (manual check)
2. Redesign kernel with extended/shifted late window
3. Remove LED main effect
4. Refit NB-GLM with ridge + smoothness regularization
5. Evaluate on baseline-normalized PSTH correlation and W-ISE in -3 to +8s
6. Add offset-locked recovery term if needed
7. Recalibrate targets to r ≥ 0.7

---

## Implementation Notes

### Kernel Structure (New)
```
Early kernel:
  - Window: [0, 2] or [0, 3] seconds
  - Bases: 2-3 narrow raised-cosines
  - Centers: [0.2, 0.7, 1.4] seconds
  - Widths: 0.3-0.5 seconds
  - Constraint: non-negative

Late kernel:
  - Window: [2, 10] or [3, 10] seconds
  - Bases: 4 raised-cosines
  - Centers: [3, 5, 7, 9] seconds
  - Widths: 1.5-2.0 seconds
  - Constraint: unconstrained
```

### Model Simplification
- Remove `LED1_scaled` main effect
- Let temporal kernel encode all LED dynamics
- This improves identifiability

### Validation Metrics
- Primary: baseline-normalized PSTH correlation (r ≥ 0.7)
- Secondary: W-ISE within bootstrap CI
- Tertiary: pre-onset baseline match, IEI K-S test

---

## References

[1] Hernandez-Nuñez et al. - Larval navigation LNP
[2] Gepner et al. 2015 - Larval phototaxis
[3] Gepner et al. 2018 - Variance adaptation
[4] Pillow et al. 2008 - Raised-cosine basis
[5] Truccolo et al. 2005 - Point process GLM



