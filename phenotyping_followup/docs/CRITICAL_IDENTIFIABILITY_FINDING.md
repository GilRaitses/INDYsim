# Critical Finding: Kernel Parameter Identifiability Failure

**Date:** 2025-12-18
**Status:** Confirmed via simulation diagnostics

---

## Executive Summary

The power analysis revealed a **fundamental identifiability problem** with individual-level kernel fitting. The issue is not sample size or track duration - it is the **kernel structure itself**.

---

## Technical Details

### The Problem

With the gamma-difference kernel used in the original study:
- K(t) = A * Gamma(t; alpha, tau1/alpha) - B * Gamma(t; alpha, tau2/alpha)
- Parameters: A=1.5, B=12.0, alpha=2.0

The kernel is **mostly negative** for t > 0.2s:

| Time (s) | K(t) at tau1=0.63 |
|----------|-------------------|
| 0.1 | +0.38 (positive) |
| 0.3 | -0.09 |
| 0.5 | -1.06 |
| 1.0 | -2.85 |
| 1.5 | -3.30 |
| 2.0 | -3.06 |

### Consequence

Since B >> A (12 >> 1.5), the **slow inhibitory component dominates**, meaning:

1. **Events are SUPPRESSED during LED-ON** relative to LED-OFF
2. With baseline hazard exp(-3.5) = 0.03/s, only ~20% of events occur in LED window
3. Diagnostic run: 11 events total, only **2 in LED window**
4. These 2 events carry almost no information about tau1

### Likelihood Surface

The log-likelihood at different tau1 values for a typical track:

| tau1 | Log-likelihood |
|------|----------------|
| 0.3 | -55.34 |
| 0.5 | -54.14 |
| **0.63 (true)** | **-53.98** |
| 1.0 | -53.85 |
| **1.5** | **-53.83 (maximum!)** |
| 2.0 | -53.85 |

**The true tau1=0.63 is NOT the likelihood maximum!** The optimizer correctly finds the maximum at tau1~1.5, but this is wrong because the likelihood surface is flat.

### Why Longer Tracks Won't Help

With this kernel structure:
- Longer tracks = more events, but same proportion in LED window (~20%)
- More events outside LED window = more data with **no tau1 information**
- The fundamental ratio of informative to uninformative events remains constant

---

## Implications for the Manuscript

### What This Means

1. **Individual tau1 estimation is NOT feasible** with this kernel parameterization
2. The 8% "fast responders" identified via outlier detection are likely **fitting artifacts**, not real phenotypes
3. Population-level estimation works because it pools information across many individuals
4. The data:parameter ratio of 3:1 is not the main issue - the issue is **data structure**

### Key Quote for Discussion

> The gamma-difference kernel with B >> A (12 >> 1.5) creates a predominantly inhibitory response, suppressing reorientation events during LED-ON periods. Consequently, only ~20% of reorientation events occur during the LED-ON window where the kernel modulates behavior. These few events are insufficient to constrain individual tau1 estimates, resulting in a flat likelihood surface where MLE converges to incorrect values. This structural limitation cannot be overcome by collecting longer tracks - the problem is not event count but the proportion of events that carry tau1 information.

---

## Questions for Further Research

1. Could a **simplified kernel** (fewer parameters) be identifiable?
2. Would a **different experimental design** (higher-frequency LED pulses) help?
3. Is there a **hierarchical approach** that can borrow strength across individuals while still detecting outliers?
4. Should the analysis focus on a **different observable** that is more sensitive to tau1?

---

## Files

- Diagnostic script: `scripts/2025-12-18/aws_setup/power_analysis.py`
- Debug log: `/home/ubuntu/indysim/debug.log` (on AWS)
- Results: `/home/ubuntu/indysim/results/power_analysis/power_analysis_results.json`

