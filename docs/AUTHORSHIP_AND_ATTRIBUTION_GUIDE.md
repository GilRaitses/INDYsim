# Authorship, Attribution, and Data Permission Guide

## Executive Summary

Before submitting this manuscript to bioRxiv, you **must obtain explicit written permission** from Mirna Mihovilovic-Skanata (Lab PI) to publish the experimental data. This is a non-negotiable ethical requirement.

---

## Gemini Research Agent Summary

The INDYsim framework represents a significant advance in computational ethology. The core intellectual contribution is the resolution of the **"interpretability-plasticity dilemma"** in behavioral modeling:

| Prior Approaches | INDYsim Contribution |
|------------------|---------------------|
| 12-parameter raised-cosine GLMs | 6-parameter Gamma-difference kernel |
| Phenomenological (black box) | Mechanistic (cascaded first-order processes) |
| Coefficients lack biological meaning | Parameters map to timescales (τ₁=0.29s, τ₂=3.81s) |

**Key scientific claims**:
1. The hazard kernel decomposes into fast transduction (~2 stages) and slow adaptation (~4 stages)
2. R² = 0.968 against 12-basis reference validates the analytic approach
3. Factorial analysis reveals **dissociation**: kernel SHAPE is intrinsic, AMPLITUDE is context-modulated
4. Tonic excitability and phasic responsivity are orthogonal control axes

---

## 1. What You Did (Plain Language for Mirna)

### The Problem with Previous Approaches

In your 2015 eLife paper (Gepner et al.) and related work (Klein et al., Hernandez-Nunez et al.), the team modeled how larvae respond to optogenetic stimulation using **flexible filter models**. These work like this:

```
Response = w₁×(bump₁) + w₂×(bump₂) + ... + w₁₂×(bump₁₂)
```

You fit 12 weights to 12 "bump" functions (raised-cosine bases). This gives a flexible curve that matches the data well, but:
- The 12 weights don't mean anything biologically
- You can't say "the fast timescale is 0.3 seconds" directly
- Comparing across conditions means comparing 12 numbers

### What This Analysis Contributes

Instead of 12 flexible bumps, I used a **closed-form equation** that has biological meaning:

```
Response = A×Gamma(fast) − B×Gamma(slow)
```

This is the mathematical form you'd expect from **cascaded first-order neural processes** (like photoreceptor → interneuron → motor output). The 6 parameters directly tell you:

| Parameter | Value | Biological Meaning |
|-----------|-------|-------------------|
| τ₁ | 0.29 s | Fast sensory transduction timescale |
| τ₂ | 3.81 s | Slow adaptation timescale |
| α₁ ≈ 2 | 2.22 | ~2 stages in fast cascade |
| α₂ ≈ 4 | 4.38 | ~4 stages in slow cascade |
| A, B | fitted | Amplitudes |

**Key result**: This 6-parameter form achieves R² = 0.968 against the 12-parameter raised-cosine reference—same predictive accuracy, but now interpretable.

### Novel Finding from Factorial Analysis

Extending to a 2×2 factorial design (your 12 experiments with varying intensity and background), I found that:
- **Kernel SHAPE is conserved** across all conditions (same τ₁, τ₂)
- **Kernel AMPLITUDE is modulated** by condition (gain control)

This suggests the GMR61 circuit has **fixed temporal dynamics** but **adjustable gain**—a dissociation between tonic excitability and stimulus-locked modulation that wouldn't be visible from fitting independent 12-parameter kernels per condition.

---

## 2. CRediT Contribution Mapping

| Contributor | CRediT Roles |
|-------------|--------------|
| **Gil Raitses** | Conceptualization, Methodology, Software, Formal Analysis, Validation, Visualization, Writing – Original Draft, Writing – Review & Editing |
| **Devindi Goonawardhana** | Investigation (data collection), Data Curation |
| **Mirna Mihovilovic-Skanata** | Resources, Supervision, Funding Acquisition |
| **Ki Young Jeong** | (Course context only—see Acknowledgments) |

---

## 3. ICMJE Authorship Criteria

To be listed as an **author**, a person must meet ALL FOUR criteria:

| Criterion | Description |
|-----------|-------------|
| **1. Substantial contribution** | Conception, design, data acquisition, OR analysis |
| **2. Drafting/revising** | Participated in writing or critically revising the manuscript |
| **3. Final approval** | Approved the version to be published |
| **4. Accountability** | Agrees to be accountable for all aspects of the work |

### Current Status

| Person | Criterion 1 | Criterion 2 | Criterion 3 | Criterion 4 |
|--------|-------------|-------------|-------------|-------------|
| Gil | ✓ Analysis, methodology | ✓ Wrote draft | ✓ | ✓ |
| Devindi | ✓ Data acquisition | ❓ Pending | ❓ Pending | ❓ Pending |
| Mirna | ✓ Resources, supervision | ❓ Pending | ❓ Pending | ❓ Pending |
| Ki Young | ❌ (course instruction only) | ❌ | ❌ | ❌ |

**Key point**: Devindi and Mirna have made substantial contributions, but to be co-authors they must also review the manuscript, approve the final version, and agree to accountability.

---

## 4. Authorship Scenarios

### Scenario A: Full Collaboration (Recommended)
**Authors**: Gil Raitses, Devindi Goonawardhana, Mirna Mihovilovic-Skanata

- Mirna and Devindi review and approve manuscript
- All three take accountability
- Most ethically clear; recognizes everyone's contribution

### Scenario B: Mirna as Senior Author Only
**Authors**: Gil Raitses, Mirna Mihovilovic-Skanata
**Acknowledgments**: Devindi Goonawardhana (data collection)

- If Devindi declines to participate in manuscript review
- Must be Devindi's choice, not imposed

### Scenario C: Gil as Sole Author
**Authors**: Gil Raitses
**Acknowledgments**: Devindi Goonawardhana (data collection), Mirna Mihovilovic-Skanata (lab resources)

- Only if both decline co-authorship responsibilities
- Still requires explicit data permission from Mirna

### Scenario D: Permission Denied (Contingency)
If Mirna does not grant permission to publish the data:
- Reframe as **methods paper** using simulated data only
- Publish the gamma-difference kernel methodology
- Reference that it was validated on experimental data without publishing raw data

---

## 5. Acknowledgment Language

### For Data Collection (if not author)
> We thank Devindi Goonawardhana for collecting the experimental data used in this analysis.

### For Lab Resources (if not author)
> We thank Mirna Mihovilovic-Skanata for providing access to the GMR61 larval tracking data and laboratory resources.

### For Course Context
> This work was initiated as a project for [Course Name] taught by Ki Young Jeong at Syracuse University.

### For Funding (if applicable)
> This work was supported by [grant information from Mirna's lab].

---

## 6. Order of Operations

```
1. FIRST: Email Mirna requesting data permission + opening authorship discussion
   ↓
2. SECOND: Wait for Mirna's response
   ↓
3. THIRD: Based on response, email Devindi about co-authorship opportunity
   ↓
4. FOURTH: Finalize author list based on commitments
   ↓
5. FIFTH: All authors review and approve final manuscript
   ↓
6. SIXTH: Submit to bioRxiv
```

**Do NOT submit to bioRxiv before completing steps 1-5.**

---

## 7. Risk Assessment

| Risk | Consequence | Mitigation |
|------|-------------|------------|
| Publishing without permission | Retraction, misconduct finding | Get explicit written permission first |
| Authorship dispute | Journal investigation, damaged relationships | Follow ICMJE criteria transparently |
| Co-author doesn't respond | Indefinite delay | Set reasonable timeline in email |
| Permission denied | Cannot publish with this data | Pivot to methods paper with simulated data |

---

## 8. Key Ethical Points

1. **Data ownership**: The experimental data belongs to Mirna's lab/institution, not to you
2. **Good faith**: You did substantial novel work, but the data is the foundation
3. **Transparency**: Be upfront about what you've done and what you're asking
4. **Flexibility**: Be prepared for negotiation on authorship order, acknowledgment wording, etc.
5. **Documentation**: Keep records of all communications about permission and authorship

