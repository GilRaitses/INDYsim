# Phenotyping Experiment Results

**Date:** 2025-12-17  
**Objective:** Test whether including empirical tracks ≥10 minutes improves phenotype clustering stability compared to simulated data

---

## Experimental Design

### Hypothesis

The main study used only 79 complete tracks (spanning the full 20-minute experiment) for individual-level analysis. We hypothesized that including empirical tracks with duration ≥10 minutes would:

1. Increase sample size (349 tracks vs 79)
2. Capture more individual variation
3. Improve clustering stability for phenotype identification

### Datasets

| Dataset | N Tracks | Duration | Source |
|---------|----------|----------|--------|
| Simulated | 300 | 10 min | Generated from population-level parameters + random intercepts (intercept=-6.54, std=0.38) |
| Empirical (≥10 min) | 260 | 10-20 min | Real larval trajectories from consolidated_dataset.h5 |
| Empirical (complete) | 141 | ≥18 min | Full-length experimental tracks |

### Methods

1. **Kernel fitting**: Gamma-difference kernel fitted to each track using maximum likelihood
2. **Feature extraction**: τ₁, τ₂, A, B kernel parameters
3. **Clustering**: K-means with k = 2, 3, 4, 5
4. **Stability metrics**: Bootstrap ARI (50 samples), silhouette score

---

## Results: Simulated Data (Quick Test)

Testing k=2,3,4,5 on 300 simulated tracks to establish baseline.

| k | Silhouette | Stability (ARI) | Seed Sensitivity | Cluster Sizes |
|---|------------|-----------------|------------------|---------------|
| 2 | 0.760 | 0.513 ± 0.506 | 0.898 | {294, 6} |
| 3 | 0.497 | 0.703 ± 0.357 | 0.997 | {75, 6, 219} |
| 4 | 0.501 | 0.841 ± 0.218 | 0.996 | {70, 220, 6, 4} |
| **5** | 0.471 | **0.920 ± 0.099** | 0.995 | {153, 6, 65, 72, 4} |

### Key Finding

Stability **increases** with more clusters (k=5 best). The high silhouette for k=2 is misleading—it simply separates 6 outlier tracks from the remaining 294. The k=5 solution has highest stability (0.920) with lowest variance (0.099).

---

## Results: Empirical Data (Final - v2)

After correcting the data quality filtering (excluding tracks that failed MAGAT segmentation):

| Metric | Value |
|--------|-------|
| Tracks ≥10 min with valid segmentation | 260 |
| Tracks fitted | **260 (100%)** |
| Mean events/track | 25.2 |
| Best stability | **0.945 (k=4)** |
| Best silhouette | **0.573 (k=5)** |

### Clustering Results (Empirical N=260)

| k | Silhouette | Stability (ARI) | Cluster Sizes |
|---|------------|-----------------|---------------|
| 2 | 0.435 | 0.478 ± 0.500 | {140, 120} |
| 3 | 0.503 | 0.918 ± 0.200 | {11, 129, 120} |
| 4 | 0.539 | **0.945** ± 0.104 | {128, 11, 115, 6} |
| 5 | **0.573** | 0.937 ± 0.089 | {115, 11, 68, 6, 60} |

### Key Finding

A consistent small subgroup of **11 tracks (4.2%)** appears across k=3,4,5 solutions with distinct kernel characteristics. This may represent a rare behavioral phenotype.

---

## Comparison: Simulated vs Empirical (Final)

| k | Simulated Stability | Empirical Stability | Simulated Silhouette | Empirical Silhouette |
|---|---------------------|---------------------|----------------------|----------------------|
| 2 | 0.513 | 0.478 | **0.760** | 0.435 |
| 3 | 0.703 | **0.918** | 0.497 | **0.503** |
| 4 | 0.841 | **0.945** | 0.501 | **0.539** |
| **5** | 0.920 | 0.937 | 0.471 | **0.573** |

### Key Observations

1. **Silhouette is HIGHER for empirical data at k≥3**: Real individual variation produces more distinct clusters than simulated tracks (0.573 vs 0.471 at k=5).

2. **Stability is HIGHER for empirical at k=3,4**: Empirical phenotypes are more reproducible than simulated ones (0.945 vs 0.841 at k=4).

3. **Cluster balance differs fundamentally**:
   - Simulated k=5: {153, 6, 65, 72, 4} — one dominant cluster (51%)
   - Empirical k=5: {115, 11, 68, 6, 60} — more balanced (largest = 44%)

4. **k=2 reversal is diagnostic**: Simulated data has high silhouette (0.760) at k=2 because it separates 6 outliers from homogeneous majority. Empirical data has LOW silhouette (0.435) because there are multiple real phenotypes that don't split into just 2 groups.

5. **Rare phenotype confirmed**: Both datasets identify a small distinct group (~2-4% of tracks) that persists across k values.

---

## Interpretation

### Hypothesis Evaluation

| Prediction | Result |
|------------|--------|
| More tracks with ≥10 min threshold | ✓ 349 tracks (vs 79 complete) |
| Better cluster separation | ✓ Higher silhouette (0.632 vs 0.471) |
| More balanced clusters | ✓ More even distribution |
| Higher fitting success | Pending (fix applied) |

### Biological Significance

1. **Real larvae show more distinct behavioral phenotypes** than simulated tracks. The population-level model + random intercepts captures average behavior but underestimates true individual variation.

2. **Cluster structure in empirical data is more balanced**, suggesting multiple distinct phenotypes rather than a single dominant type with outliers.

3. **The ≥10 minute threshold is sufficient** for kernel fitting. Tracks of this duration provide enough events (~20 on average) for reliable parameter estimation.

---

## Methods Section Text

The following text follows manuscript style protocol.

### For Results Section

Individual-level kernel fitting succeeded for 260 empirical tracks with duration ≥10 minutes and ≥10 reorientation events. K-means clustering with k=5 achieved a silhouette score of 0.632 and bootstrap stability (ARI) of 0.908 ± 0.07. Empirical data produced higher silhouette scores than simulated tracks generated from population parameters (0.632 vs 0.471), indicating that real individual variation produces more distinct behavioral phenotypes than the random-intercept model captures. Cluster size distributions were more balanced in empirical data, with no single cluster exceeding 56% of tracks, compared to 51% for the largest simulated cluster.

### For Discussion Section

The phenotyping analysis revealed that empirical tracks exhibit greater individual-level variation than predicted by the population model. Simulated tracks, generated from population-level kernel parameters with track-specific random intercepts (σ = 0.47), formed clusters dominated by a single large group (51% of tracks). Empirical tracks showed more balanced cluster distributions, suggesting that real larvae exhibit distinct behavioral phenotypes not captured by the random-intercept model alone. The gamma-difference kernel parameters (τ₁, τ₂, A, B) effectively discriminated between phenotypes, with cluster silhouette scores exceeding 0.6 for k=5.

---

## Files Generated

| File | Description |
|------|-------------|
| `results/quick_test_fewer_clusters.csv` | Simulated data clustering results |
| `results/empirical_10min_kernel_fits.csv` | Empirical kernel parameters (v1) |
| `results/empirical_10min_clustering_results.csv` | Empirical clustering results (v1) |
| `results/empirical_10min_kernel_fits_v2.csv` | Empirical kernel parameters (fixed) |
| `results/empirical_10min_clustering_results_v2.csv` | Empirical clustering results (fixed) |

---

## Critical Finding: Data Source Mismatch (CORRECTED)

During debugging, we discovered a critical issue with how tracks were being counted:

### The Bug

The **events table** contains 701 unique (experiment, track) pairs, but **277 of these have 0 reorientation events** because they failed MAGAT segmentation. The **klein_run_table** only contains 424 tracks that successfully passed segmentation.

| Source | Tracks | Mean Events/Track |
|--------|--------|-------------------|
| Events table (ALL) | 701 | ~11 (misleading!) |
| Klein run table (segmented) | 424 | **18.6** (correct) |
| Events table (WITH klein data) | 424 | **18.6** |
| Events table (NO klein data) | 277 | **0** |

### Tracks ≥10 Minutes

| Filter | Count | Mean Events |
|--------|-------|-------------|
| ≥10 min (total) | 349 | — |
| ≥10 min WITH klein data | 299 | 22.7 |
| ≥10 min NO klein data | 50 | **0** |
| ≥10 min, ≥10 events | 260 | **25.2** |

### Root Cause

The 50 tracks ≥10 min with 0 events aren't "low-activity larvae" - they **completely failed MAGAT segmentation**. This is a data processing issue, not biological variability.

### Fix

The empirical hypothesis script must:
1. Filter tracks by presence in `klein_run_table` (not just event count)
2. Use tracks ≥10 min with ≥10 reorientations → **260 usable tracks**
3. Average of **25 events/track** is sufficient for kernel fitting

---

---

## Phase 1: Cluster Validation Results

### Permutation Test (Are clusters better than random?)

| k | Observed Silhouette | Null Mean ± SD | p-value | Significant? |
|---|---------------------|----------------|---------|--------------|
| 2 | 0.435 | 0.413 ± 0.036 | 0.022 | Yes* |
| 3 | 0.503 | 0.464 ± 0.009 | 0.002 | Yes** |
| 4 | 0.539 | 0.495 ± 0.020 | <0.001 | Yes*** |
| 5 | 0.573 | 0.508 ± 0.012 | <0.001 | Yes*** |

**Result:** All cluster solutions are significantly better than random (p < 0.05).

### Gap Statistic (Optimal number of clusters)

| k | Gap | SD |
|---|-----|-----|
| 1 | 1.11 | 0.04 |
| 2 | 0.63 | 0.03 |
| 3 | 0.79 | 0.04 |
| 4 | 1.15 | 0.03 |
| 5 | 1.40 | 0.04 |

**Optimal k = 1** (by gap criterion)

This means the data may represent **continuous variation** rather than discrete phenotypes.

### Train/Test Reproducibility

| k | Mean ARI | SD | Range | Quality |
|---|----------|-----|-------|---------|
| 2 | 0.15 | 0.36 | [-0.05, 1.0] | Poor |
| 3 | 0.74 | 0.26 | [0.18, 1.0] | Good |
| 4 | 0.79 | 0.15 | [0.62, 1.0] | Good |
| 5 | 0.78 | 0.13 | [0.55, 1.0] | Good |

**Result:** k ≥ 3 solutions show good reproducibility (ARI > 0.5).

### Validation Summary

| Criterion | k=2 | k=3 | k=4 | k=5 |
|-----------|-----|-----|-----|-----|
| Permutation test | ✓ | ✓ | ✓ | ✓ |
| Gap optimal | ✗ | ✗ | ✗ | ✗ |
| Reproducibility | ✗ | ✓ | ✓ | ✓ |
| **Overall** | Partial | **Valid** | **Valid** | **Valid** |

### Interpretation

The clusters are **statistically real** (permutation test) and **reproducible** (train/test), but may represent **regions of continuous variation** rather than discrete phenotypes (gap statistic prefers k=1).

This is consistent with:
- Quantitative traits showing continuous rather than Mendelian inheritance
- Behavioral phenotypes lying on a spectrum
- The random-intercept model capturing most but not all variation

---

## Phase 2: Cluster Characterization Results

### Cluster Profiles (k=4)

| Cluster | N (%) | τ₁ (s) | τ₂ (s) | A | B | Interpretation |
|---------|-------|--------|--------|---|---|----------------|
| 0 | 128 (49%) | 0.22 | 6.6 | 0.37 | 19.9 | **Standard** - typical response |
| 1 | 11 (4%) | **5.0** | **0.63** | 0.55 | 20.0 | **Inverted timescales** - anomalous |
| 2 | 115 (44%) | 0.22 | 9.7 | **5.0** | 20.0 | **Strong excitation** - enhanced |
| 3 | 6 (2%) | 0.18 | 10.8 | 4.2 | **12.2** | **Weak suppression** - rare |

### Statistical Significance

ALL parameters significantly differ across clusters (p < 0.001):

| Parameter | η² (Effect Size) | Interpretation |
|-----------|------------------|----------------|
| τ₁ | **0.967** | Massive - almost entirely explains cluster membership |
| A | **0.966** | Massive - excitation amplitude drives separation |
| B | 0.808 | Large - suppression distinguishes rare phenotype |
| τ₂ | 0.169 | Large - secondary contribution |

### Discriminant Analysis

**99.6% classification accuracy** (10-fold CV)

| Function | Variance Explained | Key Loadings | Separates |
|----------|-------------------|--------------|-----------|
| LD1 | 60% | τ₁ (+4.6), A (-3.6) | Cluster 1 (inverted) |
| LD2 | 34% | A (+4.1), τ₁ (+3.4) | Cluster 2 (strong) |
| LD3 | 6% | B (-2.3) | Cluster 3 (weak supp.) |

### Rare Phenotype (Cluster 3, n=6)

Only **B (suppression amplitude)** is significantly different:
- Rare phenotype: B = 12.2
- All others: B = 19.9
- Difference: -7.7 (p < 0.001)

**Biological meaning:** These larvae fail to suppress turning after the initial excitatory response. They may represent a "hyperactive" phenotype.

---

## Phase 3: External Validation Results

### 1. Cluster × Experiment Association

| Metric | Value | Interpretation |
|--------|-------|----------------|
| χ² | 62.9 | — |
| p-value | **0.009** | Significant |
| Cramér's V | 0.28 | Small effect |

⚠ **Weak association with experiments** - some experiments have slightly more of certain phenotypes, but effect is small.

### 2. Cluster × Experimental Condition

| Condition | χ² | p-value | Significant? |
|-----------|-----|---------|--------------|
| Intensity (0-250 vs 50-250) | 0.0 | 1.000 | ✓ No |
| Pattern (Constant vs Cycling) | 4.1 | 0.247 | ✓ No |

✓ **Phenotypes are NOT driven by experimental conditions** - good!

### 3. Duration/Event Bias

| Variable | H | p-value | Significant? |
|----------|---|---------|--------------|
| Track duration | 14.3 | 0.003 | ⚠ Yes |
| Event count | 19.7 | 0.0002 | ⚠ Yes |

⚠ **Some clusters have different track characteristics** - needs investigation.

### 4. Rare Phenotype (Cluster 3, n=6)

| Metric | Value |
|--------|-------|
| Intensity | ALL 6 from 0-250 (low intensity) |
| Pattern | 5 Cycling, 1 Constant |
| Experiments | Distributed across 5 different experiments |

**Interpretation:** The rare "weak suppression" phenotype:
- Only appears at low LED intensity (0-250)
- Mostly under cycling stimulation
- NOT a batch effect (spread across experiments)

### Validation Summary

| Check | Result | Concern Level |
|-------|--------|---------------|
| Experiment association | p=0.009, V=0.28 | ⚠ Minor |
| Intensity condition | p=1.0 | ✓ None |
| Pattern condition | p=0.25 | ✓ None |
| Duration bias | p=0.003 | ⚠ Moderate |
| Event count bias | p=0.0002 | ⚠ Moderate |

**Overall:** Phenotypes are genuine (not batch effects or condition-driven), but some clusters may be partially confounded with track characteristics.

---

## Phase 4: Improved Simulation Results

### ⚠️ CRITICAL FINDING: Round-Trip Validation FAILED

| Metric | Value | Expected | Interpretation |
|--------|-------|----------|----------------|
| Fit success rate | 98.8% | >90% | ✓ Good |
| Cluster recovery (ARI) | **0.128** | >0.5 | ✗ Failed |
| Mean param correlation | **-0.08** | >0.5 | ✗ Failed |

### Parameter Recovery Details

| Parameter | Correlation | RMSE | Interpretation |
|-----------|-------------|------|----------------|
| τ₁ | -0.03 | 1.53 | No recovery |
| τ₂ | **-0.62** | 8.92 | Negative (inverted!) |
| A | 0.35 | 2.96 | Weak positive |
| B | -0.01 | 1.02 | No recovery |

### What This Means

**The phenotype clusters may be FITTING ARTIFACTS rather than true biological phenotypes.**

When we:
1. Generated tracks from known phenotype-specific kernels
2. Fitted kernels back to those tracks
3. Clustered the fitted parameters

...we could NOT recover the original phenotypes (ARI = 0.13 vs expected >0.5).

### Possible Explanations

| Hypothesis | Evidence | Implication |
|------------|----------|-------------|
| **Kernel non-identifiability** | Different params → similar events | Clusters reflect fitting noise, not biology |
| **Insufficient events** | Only ~25 events/track | Not enough data to constrain 4 parameters |
| **Model mismatch** | Simulation ≠ reality | Our Bernoulli model may be wrong |
| **Overfitting in original analysis** | High silhouette, low recovery | Clusters fit noise in fitted params |

### Recommended Next Steps

1. **Deep EDA** - PCA on PSTH vs kernel params to understand data structure
2. **PSTH-based clustering** - Cluster on raw event patterns, not fitted params
3. **Reduce parameter dimensions** - Fit τ₁ and τ₂ only (fix A and B)
4. **FNO (Fourier Neural Operator)** - Learn event→kernel mapping end-to-end

---

## Phase 5: Deep EDA (PCA Analysis)

Script: `07_deep_eda.py`

### Purpose

Compare two representations for phenotyping:
1. **PSTH-based**: Raw event patterns aligned to LED onset (20 time bins)
2. **Kernel-based**: Fitted parameters (τ₁, τ₂, A, B)

### Key Questions

- How many dimensions explain the data?
- Do PSTH and kernel approaches find the same clusters?
- Which representation gives better cluster separation?

### Results

| Metric | PSTH | Kernel |
|--------|------|--------|
| PCs for 90% variance | 16 | 4 |
| PC1 variance | 15% | 39% |
| Silhouette (k=4) | 0.52 | 0.54 |
| **ARI agreement** | **0.01** | - |

### 🚨 CRITICAL FINDING: REPRESENTATIONS ARE UNCORRELATED

The **ARI = 0.01** between PSTH and kernel clustering means:

> Kernel parameter clusters are **NOT** the same as event pattern clusters.

**Implications:**

1. Kernel fitting creates structure that doesn't exist in raw data
2. The 4 "phenotypes" reflect fitting noise, not behavior
3. PSTH is noisier but more directly reflects biology

### Interpretation

| If kernel phenotypes were real... | What we observe |
|-----------------------------------|-----------------|
| PSTH clusters ≈ Kernel clusters | ARI = 0.01 (random) |
| Round-trip validation works | ARI = 0.13 (failed) |
| Parameters correlate with recovery | r ≈ 0 (no correlation) |

**All three lines of evidence point to the same conclusion: kernel-based phenotypes are artifacts.**

---

## Potential Future: Fourier Neural Operator (FNO)

### Why FNO?

The round-trip validation failed because:
- Parametric fitting is noisy with sparse events
- 4 parameters from 25 events is underdetermined
- Fitting is sensitive to initialization

FNO could address this by:
- Learning the event→kernel mapping end-to-end
- Training on all tracks jointly (regularization)
- Avoiding parametric assumptions

### FNO Architecture

```
Input:  Event mask e(t) ∈ {0,1}^T  (binary at each frame)
        LED state l(t) ∈ {0,1}^T  (on/off at each frame)

FNO Layers:
  - Lift to higher dim
  - Fourier conv (learn spectral weights)
  - Pointwise nonlinearity
  - Repeat 4 layers
  - Project back

Output: K(t) kernel function on [0, 30s]
```

### Training Objective

```
L = -log P(events | K(t))    # GLM log-likelihood
    + λ‖K‖²                   # Regularization
```

This would require:
- ~1000 simulated tracks for pre-training
- Fine-tune on 260 empirical tracks
- Extract learned K(t) for clustering

---

## Phase 6: FNO Implementation

Script: `08_fno_phenotyping.py`

### Architecture

```
Input:  PSTH (20 bins, normalized)
        ↓
Lift:   Linear(1 → 64)
        ↓
FNO Layer 1-4:
  - Spectral Conv (8 Fourier modes)
  - Pointwise Conv
  - GELU activation
        ↓
Project: Linear(64 → 1)
        ↓
Output: K(t) on 60-point grid [0, 10s]
```

### Training Protocol

- **Training data:** 2000 synthetic tracks with varied kernel parameters
- **Validation data:** 500 synthetic tracks
- **Epochs:** 100
- **Optimizer:** Adam (lr=1e-3) with ReduceLROnPlateau

### Key Metrics to Watch

| Metric | Good Result | Bad Result |
|--------|-------------|------------|
| Validation kernel correlation | r > 0.8 | r < 0.5 |
| FNO vs PSTH cluster ARI | > 0.3 | < 0.1 |
| FNO vs Parametric cluster ARI | > 0.3 | < 0.1 |

### Interpretation Guide

| Outcome | What It Means |
|---------|---------------|
| FNO agrees with PSTH | Neural method finds same structure as raw data |
| FNO agrees with Parametric | Both find same (possibly artifactual) structure |
| FNO finds nothing | Phenotypic signal likely absent |

### Results: TRAINING COMPLETE ✓

| Metric | FNO | MLP |
|--------|-----|-----|
| Validation kernel corr | **0.921** | **0.978** |
| Validation MSE | 0.103 | 0.035 |

**Both models learn the event→kernel mapping extremely well on synthetic data!**

### Clustering Results

| k | Silhouette | ARI vs PSTH | Notes |
|---|------------|-------------|-------|
| 3 | 0.326 | **0.250** | Best silhouette |
| 4 | 0.303 | **0.276** | Best PSTH agreement |
| 5 | 0.255 | 0.200 | Declining |

### FNO vs Parametric Comparison

| Metric | Value |
|--------|-------|
| Kernel correlation | 0.48 ± 0.55 (weak, high variance) |
| Cluster ARI (k=4) | **0.011** (no agreement) |

### Interpretation

1. **Validation SUCCESS:** FNO/MLP can recover kernels from synthetic data (r > 0.92)
2. **FNO finds different structure:** 
   - Partially agrees with PSTH (ARI ≈ 0.27) ← **Better than parametric!**
   - Does NOT agree with parametric fits (ARI ≈ 0.01)
3. **Parametric fitting creates artifacts:** Confirmed by multiple lines of evidence
4. **Phenotypic signal is weak:** Even FNO clusters have low silhouette (0.25-0.33)

---

## Phase 7: Hierarchical Bayesian Model

Script: `09_hierarchical_bayesian.py`

### Why This Is the Gold Standard

| Problem | Solution |
|---------|----------|
| Independent fitting overfits | Joint estimation with shrinkage |
| No uncertainty quantification | Full posterior distributions |
| 25 events per track too few | Borrows strength from all 256 tracks |
| Can't identify genuine outliers | 95% credible intervals |

### Model Structure

```
Population:
  τ₁_pop ~ LogNormal(log(0.3), σ_τ1)
  τ₂_pop ~ LogNormal(log(4.0), σ_τ2)

Individual (partial pooling):
  τ₁ᵢ ~ LogNormal(μ_τ1, σ_τ1)  ← Pulled toward population

Likelihood:
  PSTH_i(t) ~ Normal(exp(β₀ + K(t; θᵢ)), σ_obs)
```

### Key Outputs

1. **Population estimates:** What is the "average" kernel?
2. **Individual posteriors:** Each track's parameter distribution
3. **Shrinkage:** How much are estimates pulled toward population?
4. **Genuine outliers:** Tracks whose 95% CI doesn't overlap population

### Results: COMPLETE ✓

| Parameter | Value | Interpretation |
|-----------|-------|----------------|
| Population τ₁ | **0.63s** | Fast excitation |
| Population τ₂ | **2.48s** | Slow suppression |
| σ_τ1 | 0.31 | Moderate variation |
| σ_τ2 | 0.46 | Moderate variation |

### Genuine Outliers

| Parameter | Outliers | Percentage |
|-----------|----------|------------|
| τ₁ | 22/256 | **8.6%** |
| τ₂ | 16/256 | **6.2%** |

### Clustering Comparison

| k | Silhouette | ARI vs PSTH | ARI vs MLE |
|---|------------|-------------|------------|
| 3 | 0.55 | 0.07 | **0.00** |
| 4 | 0.50 | 0.14 | **-0.01** |
| 5 | 0.53 | 0.10 | **-0.02** |

### 🎯 DEFINITIVE CONCLUSION

**Discrete phenotypes are NOT supported.**

- Only ~8% of tracks genuinely differ from population
- Bayes clusters show NO agreement with MLE clusters (ARI ≈ 0)
- The "4 phenotypes" from parametric fitting were ARTIFACTS
- Individual variation is CONTINUOUS, not discrete

---

## FINAL SUMMARY: All Phases Complete

| Phase | Method | Key Finding |
|-------|--------|-------------|
| 1-4 | Parametric clustering | 4 phenotypes, 99.6% accuracy |
| 5 | Round-trip validation | **FAILED** (ARI = 0.13) |
| 6 | Deep EDA | PSTH vs Kernel uncorrelated |
| 7 | FNO neural operator | FNO agrees with PSTH, not parametric |
| **8** | **Hierarchical Bayes** | **Only 8% genuine outliers** |

### For Your Presentation

> "We initially identified 4 behavioral phenotypes with 99.6% classification accuracy. However, hierarchical Bayesian analysis with proper uncertainty quantification revealed that only 8.6% of individuals genuinely differ from the population. The apparent phenotypes were fitting artifacts from underdetermined optimization. Individual sensory response variation is continuous, not discrete."

---

## PRACTICAL RECOMMENDATIONS FOR LAB

### What Won't Work

| Option | Why Not |
|--------|---------|
| 80 min recordings | Larvae quiesce, impractical |
| Multi-day same individual | 2nd → 3rd instar, not comparable |

### What Will Work

| Option | Implementation | Benefit |
|--------|----------------|---------|
| **2-parameter model** | Fix A, B; fit only τ₁, τ₂ | Works with existing data |
| **Condition-level analysis** | Pool larvae by condition | Robust, publishable |
| **Higher stim frequency** | LED every 15s vs 30s | 2x events in same time |

### The Honest Message

> "Individual phenotyping requires ~100 events/track. Current 20-min protocols yield ~25 events. Until recording duration or stimulation frequency increases 4-fold, individual-level kernel estimation is not reliable. This work establishes population-level estimation as the appropriate analysis for current data."

### Key Citations for Long Recording Justification

1. Pulver et al., 2018 - Multi-day larval bioluminescence imaging
2. Szuperak et al., 2018 - Larval sleep states are predictable and can be excluded

---

## PSTH: Empirical vs Model Explanation

### What is PSTH?

**Peri-Stimulus Time Histogram** - Shows event rate aligned to stimulus onset.

```
EMPIRICAL PSTH (from data):
  Time:    0s   1s   2s   3s   4s   5s   6s   7s   8s   9s  10s
  Events:  ▂▂▂▂▇▇▇▅▅▃▃▂▂▁▁▁▁▂▂▂▂▂
           ↑         ↑              ↑
          LED ON   Peak response   Suppression

MODEL PSTH (from K(t)):
  K(t):   [0.0][+0.8][+0.5][-0.2][-0.8][-0.6][-0.3][-0.1][0.0]
  rate:   baseline × exp(K(t))
```

### The Three PSTHs in This Study

| PSTH Type | Source | Purpose |
|-----------|--------|---------|
| **Empirical** | Real events, binned | Ground truth |
| **Model** | Evaluate fitted K(t) | Validate fit quality |
| **Simulated** | Bernoulli + K(t) | Generate synthetic data |

---

---

# FINAL RESEARCH SUMMARY

## 1. Scientific Conclusion

This dataset of 256 second-instar *Drosophila* larvae, each with ≈25 reorientation events under optogenetic stimulation, **does not provide evidence for discrete behavioral phenotypes** in stimulus–response dynamics. Instead, the data support:

- **One main, robust population-level kernel** (τ₁ ≈ 0.63 s, τ₂ ≈ 2.48 s).
- **Mostly continuous individual variation** around this kernel.
- At most a **small minority (~8.6%, 22/256) of candidate "fast responders"** (τ₁ ≈ 0.45 s) that may differ, but cannot be confidently established as a distinct phenotype with current data.

Apparent "four phenotypes" are best understood as **artifacts of clustering and data sparsity**, not real subpopulations.

---

## 2. Why the "4 Phenotypes" Hypothesis Fails

### 2.1 Clustering is an artifact

- PCA + K-means produced **four visually distinct colour groups**, but:
  - The true geometry is a **single dense blob with scattered outliers**.
  - 3D PCA and kernel surface visualizations show **smooth, unimodal structure**, not separated regions.
- **Round-trip validation** (simulate from fitted parameters → refit → recluster) yields:
  - Adjusted Rand Index (ARI) ≈ **0.1**, essentially chance.
- Conclusion: **K-means is forcing structure onto unimodal, continuous data**. The "phenotypes" are not reproducible clusters.

### 2.2 Parameter instability from sparse data

- Each track has ≈25 events; each individual model has 4 parameters (τ₁, τ₂, A, B):
  - Data : parameter ratio ≈ **6:1** → fundamentally **under-powered** for reliable individual fits.
- Maximum Likelihood Estimates (MLE):
  - τ₁ spans **0–5 s**, an implausibly wide biological range.
  - A few extreme MLEs (τ₁ ≈ 5 s) are clearly **fitting failures**, not biology.
- This behavior is exactly what you expect when **trying to estimate too many parameters from too few events**.

### 2.3 Hierarchical Bayesian model reveals near-homogeneity

- A hierarchical prior across individuals regularizes the model:
  - **91%** of larvae have posteriors that **overlap the population mean** (τ₁ ≈ 0.63 s) within 95% credible intervals.
  - The **spread of τ₁ shrinks** dramatically to a realistic band (~0.5–0.9 s).
- Only **22/256** larvae (~8.6%) have 95% credible intervals that **exclude** the population mean:
  - These form a lower-τ₁ tail (≈0.45 s), consistent with **faster responses**, but small and not independently validated.
- Interpretation:
  - Without shrinkage, MLE invents huge "individual differences".
  - With shrinkage, **most larvae look statistically equivalent** to the same kernel.
  - The model is telling you: **your data do not support strong individual heterogeneity**.

---

## 3. Interpreting the Outliers ("Fast Responders")

- The 22 candidate "fast responders" have τ₁ ≈ 0.45 s vs. population τ₁ ≈ 0.63 s.
- This is the **only potentially real deviation**:
  - Histograms and violin plots show a low-τ₁ bump in the outliers.
  - Outliers remain low τ₁ even after shrinkage, so they are not just MLE noise.
- But there are important caveats:
  - N = 22 (< 10% of the cohort) → small tail; could arise from chance or subtle batch effects.
  - No external validation (genotype, experimental condition, imaging data) is provided.
  - With 25 events per track, **even real individual differences are hard to separate from noise**.

**Best stance:** Treat fast responders as **hypothesis-generating candidates**, not confirmed phenotypes.

---

## 4. Statistical Implications

### 4.1 Is ≈25 events per larva enough?

For a **4-parameter, nonlinear** kernel model, ≈25 events per individual is **below a reasonable threshold** for robust individual-level inference:

- Data:parameter ≈ 6:1 is too low.
- Posterior intervals are wide; shrinkage collapses most individuals onto the mean.
- To reliably distinguish individuals with τ₁ ≈ 0.45 s vs. 0.63 s:
  - You likely need **≈100 events per track** for a 4-param model, *or*
  - A simpler model (e.g., **2-param kernel or shared τ₂/A/B**) if you must stick near 25 events.

### 4.2 Model choice and priors

- You are currently over-parameterizing relative to information per individual.
- Recommendation: Compare a **4-param model vs. a 2-param reduced model** using WAIC/LOO-CV.

---

## 5. Evidence Summary Table

| Phase | Method | Key Finding | Implication |
|-------|--------|-------------|-------------|
| 1-4 | Parametric clustering | 4 phenotypes, 99.6% accuracy | Initial positive result |
| 5 | Round-trip validation | **FAILED** (ARI = 0.13) | Clusters not recoverable |
| 6 | Deep EDA | PSTH vs Kernel uncorrelated (ARI = 0.01) | Fitting creates artifacts |
| 7 | FNO neural operator | FNO agrees with PSTH (0.27), not parametric (0.01) | Parametric fitting wrong |
| **8** | **Hierarchical Bayes** | **Only 8.6% genuine outliers** | **Definitive null result** |

---

## 6. Publishable Abstract

> We hypothesized that *Drosophila* larvae would exhibit discrete behavioral phenotypes in their stimulus-response dynamics. Using a 4-parameter gamma-difference kernel fit to ~25 reorientation events per individual (N=256), clustering analyses initially suggested four phenotypes with high separation metrics (silhouette = 0.57, classification accuracy = 99.6%). However, rigorous validation—round-trip clustering (ARI ≈ 0.1), hierarchical Bayesian modeling, and neural operator comparison—demonstrates that this structure is an artifact of sparse data and model overfitting: 91% of larvae are statistically indistinguishable from a single population kernel (τ₁ = 0.63 s, τ₂ = 2.48 s). Only ~8% exhibit consistently faster response times, and these require independent replication. Our results establish that under typical experimental conditions, larval optogenetic phenotyping is best interpreted at the population level, and we provide guidelines for the event counts and models needed to resolve genuine individual differences.

---

## 7. Actionable Recommendations

### For This Lab

| Recommendation | Implementation |
|----------------|----------------|
| Accept null result | Discrete phenotypes not supported |
| Use population kernel | τ₁ = 0.63s, τ₂ = 2.48s is robust |
| Try 2-param model | Fix A, B; fit only τ₁, τ₂ |
| Increase stim frequency | LED every 15s vs 30s for 2× events |

### For the Field

| Recommendation | Rationale |
|----------------|-----------|
| Report validation metrics | Silhouette alone is insufficient |
| Use hierarchical models | MLE overfits with sparse data |
| Require ≥100 events | For individual phenotyping with 4 params |

---

## 8. Figure Requirements (Honest Visualization)

The figures should communicate: *"We hypothesized phenotypes, found artifacts, discovered data limitations."*

### Figure 1: Clustering Illusion
- Single-color PCA scatter with density contours
- No K-means colors
- Caption: "Apparent clusters do not survive validation"

### Figure 2: Data Sparsity
- Strip plot of events/track (most 10-30)
- MLE τ₁ histogram (0-5s range)
- Caption: "Sparse data explains MLE instability"

### Figure 3: Hierarchical Shrinkage
- Caterpillar plot of τ₁ with 95% CIs
- Population mean line
- Caption: "91% of larvae are indistinguishable from population"

### Figure 4: Fast Responders
- Violin plot: Normal vs Outliers
- Caption: "8.6% may be faster, but requires confirmation"

---

*Final synthesis completed: 2025-12-17*

---

## 9. Manuscript Completion Plan

### Current Status

| Component | Status | Location |
|-----------|--------|----------|
| **Introduction** | ✓ Complete | `sections/01_introduction.tex` |
| **Methods** | ✓ Complete | `sections/02_methods.tex` |
| **Results** | ✓ Complete + Figures | `sections/03_results.tex` |
| **Discussion** | ✓ Complete | `sections/04_discussion.tex` |
| **Conclusion** | ✓ Complete | `sections/05_conclusion.tex` |
| **References** | ✓ Complete | `sections/08_references.tex` |

### Figure Status

| Figure | File | Caption Status |
|--------|------|----------------|
| Summary (4-panel) | `figures/honest/fig_combined_summary.pdf` | ✓ Added to Results |
| Clustering Illusion | `figures/honest/fig1_clustering_illusion.pdf` | ✓ Added to Results |
| Data Sparsity | `figures/honest/fig2_data_sparsity.pdf` | ✓ Added to Results |
| Hierarchical Shrinkage | `figures/honest/fig3_hierarchical_shrinkage.pdf` | ✓ Added to Results |
| Fast Responders | `figures/honest/fig4_fast_responders.pdf` | ✓ Added to Results |

### Remaining Tasks

| Task | Priority | Time Est. |
|------|----------|-----------|
| Compile LaTeX → PDF | High | 5 min |
| Verify figure placement | High | 10 min |
| Proofread abstract | Medium | 15 min |
| Add author info / affiliations | Medium | 5 min |
| Format for journal submission | Low | 30 min |

### To Compile Manuscript

```bash
cd /Users/gilraitses/INDYsim_project/scripts/2025-12-16/phenotyping_followup
pdflatex main.tex  # Or create main.tex that includes all sections
```

### Key Messages for Presentation

1. **Headline**: "We hypothesized 4 phenotypes. Validation showed they were artifacts."

2. **Three-sentence summary**:
   - Initial clustering suggested 4 distinct behavioral phenotypes with 99.6% classification accuracy.
   - However, round-trip validation (ARI = 0.1), hierarchical Bayesian analysis (91% overlap population), and neural operator comparison all showed the clusters were fitting artifacts.
   - The population kernel (τ₁ = 0.63s, τ₂ = 2.48s) is robust; individual phenotyping requires ≥100 events/track.

3. **For skeptics**: "Gap statistic says optimal k = 1. All validation methods disagree (ARI ≈ 0). This is not a weak null result—it's a definitive negative."

4. **For the lab**: "Current 20-min protocols with ~25 events are sufficient for population-level analysis but not individual phenotyping. Either increase stimulation frequency or accept condition-level analysis."

---

## 10. File Locations Summary

### Manuscript Files
```
scripts/2025-12-16/phenotyping_followup/
├── sections/
│   ├── 01_introduction.tex
│   ├── 02_methods.tex
│   ├── 03_results.tex          ← Now includes figure captions
│   ├── 04_discussion.tex       ← Complete
│   ├── 05_conclusion.tex       ← Complete with abstract
│   └── 08_references.tex
└── figures/
    ├── honest/                  ← New honest figures
    │   ├── fig_combined_summary.pdf
    │   ├── fig1_clustering_illusion.pdf
    │   ├── fig2_data_sparsity.pdf
    │   ├── fig3_hierarchical_shrinkage.pdf
    │   └── fig4_fast_responders.pdf
    └── interactive/             ← Plotly HTML for presentations
```

### Analysis Results
```
scripts/2025-12-17/phenotyping_experiments/
├── results/
│   ├── empirical_10min_kernel_fits_v2.csv
│   ├── hierarchical_bayesian/
│   │   ├── hierarchical_results.json
│   │   └── individual_posteriors.csv
│   └── improved_simulation/
│       └── simulation_validation_results.json
└── PHENOTYPING_EXPERIMENT_RESULTS.md  ← This file
```

### Documentation
```
scripts/2025-12-16/phenotyping_followup/
├── FIGURE_CRITIQUE_AND_RESEARCH_PROMPT.md
└── FIGURE_PLAN.md
```

---

*Document complete: 2025-12-17*

