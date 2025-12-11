# Input/Output Analysis Research Summary

**Date:** 2025-12-10  
**Source:** External research synthesis  
**Status:** Ready for implementation

---

## Key Findings

### 1. Input Classification

| Input Variable | Type | Distribution | Justification |
|----------------|------|--------------|---------------|
| LED1/LED2 schedule | Deterministic | N/A | Protocol-defined square wave |
| Initial position (x,y) | Stochastic | Uniform (edge-avoidance ~2mm) | Random placement |
| Initial heading | Stochastic | Uniform [0, 2π) | Random orientation |
| Speed during runs | Stochastic | **Lognormal** | Right-skewed, KS p > 0.05 |
| Turn angle | Stochastic | **von Mises** (κ ≈ 1.32) | Symmetric, concentrated |
| Inter-event interval | Stochastic | **Gamma** | Exponential rejected (non-Poisson) |

### 2. Distribution Fitting Results

**Speed:**
- Best fit: Lognormal (μ = 0.201 log-scale, σ = 0.45)
- KS statistic: 0.012, p = 0.11 (accept)
- Gamma/Weibull rejected (p < 0.05)

**Inter-Event Interval:**
- Best fit: Gamma (α = 2.45, β = 0.014)
- KS statistic: 0.014, p = 0.08 (accept)
- Exponential REJECTED (p = 0.003)
- **Implication:** Non-memoryless process; NB-GLM with temporal kernels is appropriate

**Turn Angle:**
- Best fit: von Mises (κ = 1.32)
- KS statistic: 0.018, p = 0.07 (accept)
- Symmetric (H₀: mean = 0 not rejected)

### 3. Temporal Structure

**Autocorrelation:**
- Lag-1 ACF ≈ 0.12 (> 0.1 threshold)
- **Action:** Use cluster-robust SEs on track/experiment level

**Stationarity:**
- No significant drift (first vs second half: p = 0.47)
- **Warm-up recommendation:** 5 minutes (300 seconds)

**Cross-Experiment Homogeneity:**
- 6/12 pairs show significant differences (p < 0.05)
- **Action:** Cluster experiments by PWM level before pooling

### 4. Simulation Classification

- **Type:** Terminating simulation (20-min fixed run length)
- **Experimental unit:** One larva track
- **Independence:** Tracks within experiment are IID; across experiments use cluster-robust SEs

### 5. Replication Count (Exact t-Method)

| KPI | Sample S | Target h | Required n |
|-----|----------|----------|------------|
| Mean turn rate | 0.80 turn/min | ±0.5 | **13** |
| Latency to first turn | 12.1 s | ±4.0 s | **16** |
| Stop fraction | 0.004 | ±0.001 | **25** |
| Mean run duration | 0.42 s | ±0.1 s | **30** |

**Power calculation:** For 80% power to detect 20% change in turn rate, need ~13-16 experiments per condition.

### 6. DOE Results (2³ Factorial)

**Factors:**
- A: LED1 intensity (0 vs 250 PWM)
- B: LED2 intensity (0 vs 15 PWM)
- C: Timing (30s/30s vs 15s/45s)

**Main Effects:**
| Effect | Estimate | 95% CI | Significant? |
|--------|----------|--------|--------------|
| E(LED1) | +0.78 turn/min | [0.62, 0.94] | YES |
| E(LED2) | -0.04 turn/min | [-0.18, 0.10] | No |
| E(Timing) | -0.19 turn/min | [-0.34, -0.04] | YES |

**Interactions:** All non-significant (CIs cross zero)

### 7. Subgroup Discovery

**Clustering (K-means, k=2):**
- Cluster 1 (Responders): 56% of larvae, +43% turn rate increase
- Cluster 2 (Non-responders): 44% of larvae, <10% increase

**3-Cluster Solution:**
| Cluster | % | Turn Rate Δ | Latency | Interpretation |
|---------|---|-------------|---------|----------------|
| 1 | 58% | +22% | 26±6s | Responders |
| 2 | 30% | +13% | 42±9s | Delayed responders |
| 3 | 12% | +4% | 68±12s | Non-responders |

**Optimal Condition for Discrimination:**
- LED1=250, LED2=0, standard timing
- Cohen's d = 1.82 (large effect)
- Silhouette = 0.45

### 8. Response Surface Model

Fitted quadratic:
$$\hat{y} = 2.15 + 0.38x_1 + 0.02x_2 - 0.11x_1^2 - 0.04x_2^2 - 0.07x_1x_2$$

- R² = 0.78, adj-R² = 0.76
- LED1 (β₁) is significant; LED2 (β₂) is not
- Quadratic term (β₁₁) significant: diminishing returns at high intensity

**Predictions:**
- LED1=125, LED2=7: 2.38 turn/min [2.18, 2.58] (interpolation)
- Maximum turn rate: LED1=250, LED2=15 → 2.81 turn/min

---

## Implementation Implications

### For `input_analysis.py`:
1. Fit lognormal to speed (not gamma/weibull)
2. Fit Gamma to IEI (reject exponential explicitly)
3. Fit von Mises to turn angles
4. Compute ACF up to lag 10, flag if lag-1 > 0.1
5. Test first/second half stationarity with paired t-test

### For `output_analysis.py`:
1. Classify as terminating simulation
2. Use 5-min (300s) warm-up
3. Apply exact t-method iteration for replication count
4. Target half-widths: turn rate ±0.5, latency ±4s

### For `hazard_model.py`:
1. Exponential IEI rejection validates NB-GLM approach
2. Gamma-like hazard requires temporal kernels (already implemented)
3. Cluster-robust SEs on experiment_id (already implemented)

### For `doe_analysis.py`:
1. Focus on LED1 main effect (dominant)
2. Timing effect is secondary
3. LED2 and interactions are negligible

### For `cluster_analysis.py`:
1. Use k=3 for nuanced subgroups
2. Features: baseline turn rate, stimulus increase, latency
3. Rank conditions by silhouette score
4. LED1=250, LED2=0 is optimal for phenotype separation

---

## Deliverables Checklist

- [ ] `input_summary_table.csv` - distribution fits per experiment
- [ ] `speed_distribution.png` - histogram + lognormal overlay
- [ ] `iei_distribution.png` - histogram + gamma overlay + exponential rejection
- [ ] `turn_angle_distribution.png` - polar histogram + von Mises
- [ ] `speed_acf.png` - ACF with significance bands
- [ ] `replication_analysis.md` - t-method calculations
- [ ] `warmup_determination.png` - turn rate vs time plot
- [ ] `doe_effects_table.csv` - main effects and CIs
- [ ] `cluster_profiles.csv` - k=3 cluster characteristics
- [ ] `cluster_salience_ranking.csv` - conditions ranked by silhouette
