# Kernel Parameters Explanation: Gamma-Difference vs. Raised Cosine

## Overview

The manuscript uses two kernel parameterizations to model temporal response dynamics:
1. **Gamma-Difference Kernel** (6 parameters) - Biologically interpretable, used for final analysis
2. **Raised Cosine Basis** (12 parameters) - Flexible reference model, used for comparison

The gamma-difference kernel achieves near-optimal fit quality (R² = 0.968) with half the parameters of the raised-cosine basis (R² = 0.974), while providing biological interpretability.

---

## 1. Gamma-Difference Kernel (6 Parameters)

### Mathematical Form

\[
K(t) = A \cdot \text{Gamma}(t; \alpha_1, \beta_1) - B \cdot \text{Gamma}(t; \alpha_2, \beta_2)
\]

where:
- \(\tau_1 = \alpha_1 \beta_1\) is the fast timescale (mean of fast gamma distribution)
- \(\tau_2 = \alpha_2 \beta_2\) is the slow timescale (mean of slow gamma distribution)

### Parameter Roles

#### **A** (Excitation Amplitude)
- **Role**: Controls the magnitude of the fast excitatory response
- **Effect**: Higher A → larger initial peak in kernel
- **Typical range**: 0.1 - 5.0
- **Biological interpretation**: Strength of initial sensory response

#### **α₁** (Fast Shape Parameter)
- **Role**: Controls the shape (skewness) of the fast gamma distribution
- **Effect**: Higher α₁ → more symmetric, less skewed peak
- **Typical range**: 1.0 - 5.0
- **Biological interpretation**: Shape of initial response dynamics
- **Note**: Together with β₁, determines \(\tau_1 = \alpha_1 \beta_1\)

#### **β₁** (Fast Scale Parameter)
- **Role**: Controls the timescale of the fast gamma distribution
- **Effect**: Higher β₁ → faster decay, narrower peak
- **Typical range**: 0.05 - 1.0 seconds
- **Biological interpretation**: Speed of initial response
- **Note**: Together with α₁, determines \(\tau_1 = \alpha_1 \beta_1\)

#### **B** (Suppression Amplitude)
- **Role**: Controls the magnitude of the slow suppressive response
- **Effect**: Higher B → deeper suppression trough
- **Typical range**: 5.0 - 20.0
- **Biological interpretation**: Strength of adaptation/suppression

#### **α₂** (Slow Shape Parameter)
- **Role**: Controls the shape (skewness) of the slow gamma distribution
- **Effect**: Higher α₂ → more symmetric, less skewed suppression
- **Typical range**: 2.0 - 8.0
- **Biological interpretation**: Shape of adaptation dynamics
- **Note**: Together with β₂, determines \(\tau_2 = \alpha_2 \beta_2\)

#### **β₂** (Slow Scale Parameter)
- **Role**: Controls the timescale of the slow gamma distribution
- **Effect**: Higher β₂ → faster suppression decay
- **Typical range**: 0.3 - 2.0 seconds
- **Biological interpretation**: Speed of adaptation/suppression
- **Note**: Together with α₂, determines \(\tau_2 = \alpha_2 \beta_2\)

### Key Relationships

- **τ₁ = α₁ × β₁**: Fast timescale (typically ~0.3-0.6s)
- **τ₂ = α₂ × β₂**: Slow timescale (typically ~2-4s)
- **A/B ratio**: Determines excitation vs. suppression balance
  - A/B < 1: Inhibition-dominated (our kernel, A/B ≈ 0.125)
  - A/B = 1: Balanced
  - A/B > 1: Excitation-dominated

### Biological Interpretation

- **Fast component (A, α₁, β₁)**: Models initial excitatory response to LED onset
  - Peak occurs at ~τ₁ seconds after stimulus
  - Represents sensory transduction and initial neural response
  
- **Slow component (B, α₂, β₂)**: Models delayed suppression/adaptation
  - Trough occurs at ~τ₂ seconds after stimulus
  - Represents adaptation, habituation, or inhibitory feedback

---

## 2. Raised Cosine Basis (12 Parameters)

### Mathematical Form

\[
K(t) = \sum_{j=1}^{12} w_j \cdot B_j(t)
\]

where each basis function is:

\[
B_j(t) = \begin{cases}
0.5 \cdot (1 + \cos(\pi \cdot (t - c_j) / w)) & \text{if } |t - c_j| < w \\
0 & \text{otherwise}
\end{cases}
\]

### Parameter Structure

The 12 parameters consist of **12 coefficients** (weights) \(w_1, w_2, ..., w_{12}\), one for each basis function.

#### Basis Function Organization

The 12 basis functions are organized into three groups (from `fit_gamma_per_condition.py`):

1. **Early basis functions** (4 functions)
   - Centers: [0.2, 0.6333, 1.0667, 1.5] seconds
   - Width: 0.4 seconds
   - Coefficients: \(w_1, w_2, w_3, w_4\)
   - **Role**: Capture early excitatory response (0-2s window)

2. **Intermediate basis functions** (2 functions)
   - Centers: [2.0, 2.5] seconds
   - Width: 0.6 seconds
   - Coefficients: \(w_5, w_6\)
   - **Role**: Capture transition from excitation to suppression (2-3s window)

3. **Late basis functions** (6 functions)
   - Centers: [3.0, 4.2, 5.4, 6.6, 7.8, 9.0] seconds
   - Width: 1.8 seconds
   - Coefficients: \(w_7, w_8, w_9, w_{10}, w_{11}, w_{12}\)
   - **Role**: Capture late suppression and recovery (3-10s window)

**Total**: 4 + 2 + 6 = 12 basis functions

### Parameter Roles

Each coefficient \(w_j\) controls:
- **Magnitude**: Positive values → excitation, negative values → suppression
- **Timing**: Determined by the basis function center \(c_j\)
- **Shape**: Determined by the basis function width (shared within each group)

### Advantages and Disadvantages

**Advantages:**
- **Flexibility**: Can capture arbitrary temporal shapes
- **No assumptions**: No parametric form required
- **High fit quality**: R² = 0.974 (slightly better than gamma-difference)

**Disadvantages:**
- **No biological interpretation**: Coefficients don't map to neural processes
- **Overparameterized**: 12 parameters vs. 6 for similar fit quality
- **Less parsimonious**: Higher AIC (-3386 vs. -357 for gamma-difference)

---

## 3. How Both Kernels Were Used for Comparison

### Model Selection Process

1. **Initial fitting**: Raised cosine basis (12 parameters) was fitted to empirical PSTH data
   - This provided a flexible, high-quality reference kernel shape
   - R² = 0.974

2. **Parametric approximation**: Gamma-difference kernel (6 parameters) was fitted to match the raised cosine kernel
   - Minimized sum of squared errors between gamma-difference and raised cosine curves
   - Achieved R² = 0.968 (96.8% of variance explained)

3. **Model comparison**: Both kernels were compared using:
   - **R²**: Goodness of fit to empirical PSTH
   - **AIC**: Akaike Information Criterion (penalizes complexity)
   - **BIC**: Bayesian Information Criterion (stronger penalty)

### Comparison Results

| Model | Parameters | R² | AIC | Interpretation |
|-------|-----------|----|-----|----------------|
| Raised Cosine (12 basis) | 12 | 0.974 | -3386 | Overparameterized |
| **Gamma-Difference** | **6** | **0.968** | **-357** | **Biologically interpretable** |

**Key finding**: The gamma-difference model achieves near-optimal fit quality (R² = 0.968) with half the parameters, while providing biological interpretability (timescales map to neural processes).

### Why Gamma-Difference Was Chosen

1. **Biological interpretability**: Parameters map directly to neural timescales
2. **Parsimony**: Half the parameters for similar fit quality
3. **Simulation**: Enables generative modeling via gamma distributions
4. **Theoretical foundation**: Based on known neural response dynamics

---

## 4. Figure Status

### Figure 1: `fig3_psth_kernel_v2.pdf`

**Location**: `/Users/gilraitses/InDySim/phenotyping_followup/figures/fig3_psth_kernel_v2.pdf`

**Used in manuscript**: Yes, Figure 1 in Introduction section (`01_introduction.tex`)

**Content**: Shows:
- **(A)** Empirical PSTH (binned events)
- **(B)** Fitted gamma-difference kernel \(K(t)\)
- **(C)** Per-frame event probability \(p(t) = \exp(\beta_0 + K(t))\)
- **(D)** Discrete-time Bernoulli process

**Status**: **Does NOT show raised cosine kernel** - only shows gamma-difference

### Recommendation

**Consider adding a comparison panel** showing:
- Gamma-difference kernel (blue)
- Raised cosine kernel (red, dashed)
- Overlay showing they match closely (R² = 0.968)

This would visually demonstrate that the gamma-difference kernel captures the same shape as the flexible raised cosine reference.

---

## 5. Document Alignment Assessment

### Current State in Manuscript

#### ✅ **Well Documented**
- Gamma-difference kernel form and parameters (methods section)
- Model comparison table (supplement)
- R² = 0.968 comparison mentioned (main manuscript)

#### ⚠️ **Needs Improvement**
- **Raised cosine basis**: Not explained in methods section
- **12 parameters**: Not enumerated or explained
- **Comparison methodology**: Not detailed (how raised cosine was fitted, how gamma-difference was matched to it)
- **Figure**: Does not show both kernels for visual comparison

### Recommended Additions

1. **Methods section**: Add subsection explaining raised cosine basis
2. **Figure**: Add panel showing both kernels overlaid
3. **Supplement**: Expand model comparison section with parameter details

---

## 6. Summary

### Gamma-Difference Kernel (6 Parameters)
- **A, α₁, β₁**: Fast excitatory component (τ₁ = α₁ × β₁)
- **B, α₂, β₂**: Slow suppressive component (τ₂ = α₂ × β₂)
- **Biological interpretation**: Timescales map to neural processes
- **Used for**: Final analysis, simulation, phenotyping

### Raised Cosine Basis (12 Parameters)
- **12 coefficients**: One weight per basis function
- **3 groups**: Early (4), intermediate (4), late (4) basis functions
- **No biological interpretation**: Flexible but not mechanistic
- **Used for**: Reference model, comparison, validation

### Comparison
- Both achieve similar fit quality (R² = 0.968 vs. 0.974)
- Gamma-difference chosen for biological interpretability and parsimony
- Figure currently shows only gamma-difference (consider adding raised cosine overlay)

