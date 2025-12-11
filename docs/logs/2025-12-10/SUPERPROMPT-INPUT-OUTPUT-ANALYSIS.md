# Superprompt: Simulation Input/Output Analysis for Larval Behavior DOE

## Context

You are analyzing a Drosophila larval mechanosensation dataset containing 14 experiments with ~8.7M trajectory observations and ~8.5M behavioral events. The goal is to apply rigorous simulation methodology (ECS630: Simulation and Data Analytics) to:

1. Characterize input distributions with goodness-of-fit testing
2. Determine optimal replication counts with statistical justification
3. Design experiments that maximize behavioral differences between unclassified subgroups
4. Predict behavioral responses for untested experimental conditions
5. Identify stimulus conditions that produce the most salient agent clustering patterns

## Data Schema

```
consolidated_dataset.h5
├── trajectories/
│   ├── experiment_id (string)
│   ├── track_id (int)
│   ├── time (float, seconds)
│   ├── x, y (float, cm)
│   ├── speed (float, cm/s)
│   ├── curvature (float, 1/cm)
│   ├── heading (float, radians)
│   ├── led1Val (int, 0-250 PWM)
│   ├── led2Val (int, 0-15 PWM)
│   └── is_reorientation (bool)
├── events/
│   ├── experiment_id, track_id, time
│   ├── event_type (turn, stop, reversal, run)
│   └── duration (float, seconds)
└── klein_run_tables/
    ├── run_duration, turn_angle, speed_mean
    └── stimulus_context (pre/during/post)
```

**Experimental Conditions (14 experiments, 2 genotypes):**
- LED1: Red light intensity (0-250 PWM), 30s on/30s off square wave
- LED2: Blue light intensity (0-15 PWM), various patterns
- Genotype: GMR61 (thermosensory) variants

---

## Part 1: Input Analysis (Pre-Model Characterization)

### 1.1 Stochastic Input Identification

For each of the following, determine if it is a stochastic or deterministic input:

| Input Variable | Type | Justification |
|----------------|------|---------------|
| LED1 intensity schedule | ? | |
| LED2 intensity schedule | ? | |
| Initial larva position (x, y) | ? | |
| Initial heading | ? | |
| Speed during runs | ? | |
| Turn angle during reorientations | ? | |
| Inter-event intervals | ? | |

### 1.2 Distribution Fitting

For each stochastic input, fit candidate theoretical distributions and report:

**Speed Distribution:**
- Candidate distributions: Lognormal, Gamma, Weibull
- For each: MLE parameters, K-S test statistic, p-value
- Decision rule: Accept if p > 0.05 (larger p-values indicate better fits)
- Selected distribution with justification

**Inter-Event Interval (IEI) Distribution:**
- Candidate: Exponential, Gamma, Weibull
- Test for memoryless property (exponential implies Poisson process)
- If exponential rejected, what does this imply for the hazard model?

**Turn Angle Distribution:**
- Candidate: von Mises, wrapped Cauchy, uniform
- Check for symmetry (H0: mean = 0)
- Report concentration parameter

**Initial Position Distribution:**
- Is it uniform within the arena?
- Check for edge effects (wall avoidance at t=0)

### 1.3 Temporal Structure (Non-IID Checks)

**Autocorrelation Analysis:**
- Compute ACF of speed within tracks at lags 1-10
- If lag-1 ACF > 0.1, what adjustment is needed for CI estimation?
- Plot ACF with 95% significance bands

**Stationarity Check:**
- Compare mean turn rate in first half vs second half of each experiment
- Test H0: μ_first = μ_second with paired t-test
- If rejected, is warm-up period needed? How long?

**Cross-Experiment Homogeneity:**
- Compare speed distributions across experiments (same condition)
- K-S test for each pair; report proportion with p < 0.05
- If heterogeneous, cluster experiments before pooling

### 1.4 Input Analysis Outputs

Generate the following artifacts:
1. `input_summary_table.csv` - Distribution fits, GoF p-values, selected distributions
2. `speed_distribution.png` - Histogram with fitted lognormal overlay
3. `iei_distribution.png` - Histogram with fitted exponential overlay
4. `turn_angle_distribution.png` - Polar histogram with von Mises fit
5. `speed_acf.png` - ACF plot with significance bands
6. `stationarity_test.csv` - First/second half comparison by experiment

---

## Part 2: Simulation Type and Replication Justification

### 2.1 Simulation Classification

Answer the following:

1. **Is this a terminating or non-terminating simulation?**
   - Does the larval experiment have natural start/stop conditions?
   - What defines "steady state" for larval behavior?
   - Recommendation: terminating / non-terminating with warm-up?

2. **What is the experimental unit (replication)?**
   - One larva track?
   - One experiment (multiple larvae)?
   - Justify based on independence requirements

3. **Are replications IID (independent and identically distributed)?**
   - Within-experiment: Are different tracks independent?
   - Across-experiment: Same-condition experiments equivalent?
   - If not IID, what adjustment is needed?

### 2.2 Replication Count Determination (Exact Method)

Given the output measure "mean turn rate per minute," use the exact t-distribution method to determine required replications:

**Formula:** 
$$n = t_{n-1,1-\alpha/2}^2 \frac{S^2}{h^2}$$

**Procedure:**
1. From initial n₀ = 10 experiments, compute sample mean and S
2. Set target half-width h (e.g., ±0.5 turns/min for 95% CI)
3. Iterate until convergence:
   - n_new = ⌈t²_{n-1} × S² / h²⌉
   - If n_new = n_old, stop
4. Report: initial half-width, target half-width, required n

**Repeat for each KPI:**
| KPI | Initial h₀ | Target h | Required n |
|-----|-----------|----------|------------|
| Mean turn rate | | | |
| Mean latency to first turn | | | |
| Stop fraction | | | |
| Mean run duration | | | |

### 2.3 Warm-Up Period Determination

If non-terminating simulation is chosen:
1. Plot turn rate vs time (averaged across experiments)
2. Identify visual stabilization point
3. Apply Welch's method or MSER-5 to determine truncation point
4. Recommended warm-up period: ___ seconds

---

## Part 3: DOE for Behavioral Subgroup Discrimination

### 3.1 Factorial Design Specification

Define a 2³ factorial design:

| Factor | Low Level (-) | High Level (+) |
|--------|---------------|----------------|
| A: LED1 intensity | 0 PWM | 250 PWM |
| B: LED2 intensity | 0 PWM | 15 PWM |
| C: Stimulus timing | 30s on/30s off | 15s on/45s off |

**Treatment combinations (8 total):**
1. A(-) B(-) C(-): baseline, no light
2. A(+) B(-) C(-): red only, standard timing
3. ... (list all 8)

### 3.2 Main Effects and Interactions

Using existing data (or simulated data if not all conditions tested):

**Main Effects:**
- E(A) = average turn rate at A(+) - average at A(-)
- E(B) = ...
- E(C) = ...

**Two-Way Interactions:**
- E(AB) = effect of A at B(+) - effect of A at B(-)
- E(AC) = ...
- E(BC) = ...

**Three-Way Interaction:**
- E(ABC) = ?

**Which effects are statistically significant?** (use 95% CI)

### 3.3 Subgroup Identification

Given that larvae may belong to unobserved behavioral subgroups (responders vs non-responders):

1. **Cluster larvae by stimulus response magnitude:**
   - Feature: max turn rate increase during LED1 ON
   - Method: K-means (k=2) or Gaussian mixture
   - Report: cluster sizes, mean response per cluster

2. **Which experimental conditions maximize between-cluster separation?**
   - Compute Cohen's d between clusters for each condition
   - Rank conditions by discriminability
   - Optimal condition for identifying subgroups: ___

3. **Interaction with genotype:**
   - Do clusters differ by genotype?
   - Chi-square test for independence

---

## Part 4: Predicting Untested Conditions

### 4.1 Interpolation vs Extrapolation

For conditions not yet tested empirically:

| Condition | Tested? | Prediction Type |
|-----------|---------|-----------------|
| LED1=125 PWM, LED2=0 | No | Interpolation |
| LED1=250 PWM, LED2=7 | No | Interpolation |
| LED1=300 PWM, LED2=0 | No | Extrapolation (caution) |
| LED1=250, 5s on/55s off | No | Interpolation (timing) |

### 4.2 Response Surface Methodology

Fit a second-order polynomial response surface:

$$\hat{y} = \beta_0 + \beta_1 x_1 + \beta_2 x_2 + \beta_{11} x_1^2 + \beta_{22} x_2^2 + \beta_{12} x_1 x_2$$

Where:
- y = mean turn rate
- x₁ = LED1 intensity (coded -1 to +1)
- x₂ = LED2 intensity (coded -1 to +1)

Report:
- R² and adjusted R²
- Significance of each coefficient
- Predicted turn rate at (LED1=125, LED2=7)
- 95% prediction interval

### 4.3 Optimal Condition Search

Using the response surface:
1. What LED1/LED2 combination maximizes turn rate?
2. What combination produces turn rate closest to baseline?
3. What combination produces maximum LED1 × LED2 interaction?

---

## Part 5: Agent Clustering and Pattern Salience

### 5.1 Multi-Dimensional Behavioral Phenotyping

For each larva track, extract the following features:
- Baseline turn rate (first 30s, no stimulus)
- Stimulus-evoked turn rate increase
- Latency to first turn after stimulus onset
- Mean run duration
- Mean turn angle magnitude
- Speed-curvature correlation

### 5.2 Clustering Analysis

1. **Standardize features** (z-score)
2. **Determine optimal k** using elbow method or silhouette score
3. **K-means clustering** with k=2,3,4
4. **Report cluster characteristics:**

| Cluster | n | Turn Rate | Latency | Speed | Interpretation |
|---------|---|-----------|---------|-------|----------------|
| 1 | | | | | "Responders" |
| 2 | | | | | "Non-responders" |
| 3 | | | | | "Delayed responders" |

### 5.3 Conditions That Maximize Cluster Salience

For each experimental condition, compute:
- Silhouette score (higher = better separation)
- Between-cluster / within-cluster variance ratio (F-statistic)

**Rank conditions by discriminability:**

| Condition | Silhouette | F-ratio | Rank |
|-----------|------------|---------|------|
| LED1=250, LED2=0 | | | |
| LED1=250, LED2=15 | | | |
| ... | | | |

**Recommendation:** To observe the most distinct behavioral phenotypes, use condition ___ because ___

### 5.4 Agent-Based vs System Dynamics Perspective

Compare insights from:

**Agent-Based View (individual larvae):**
- Each larva is an autonomous agent with state (position, heading, speed)
- Behavior emerges from individual decision rules
- Heterogeneity is a feature, not noise
- Look for: clusters, outliers, individual response curves

**System Dynamics View (population aggregate):**
- Model population-level turn rate as a stock/flow system
- Stimulus intensity affects flow rate
- Look for: equilibrium points, feedback loops, system delays

**Which perspective is more useful for your research question?** Justify.

---

## Part 6: Deliverables Specification

### Required Outputs

1. **Input Analysis Report** (`input_analysis_report.md`)
   - All distribution fits with GoF test results
   - ACF analysis and stationarity tests
   - Figures: histograms, ACF plots, QQ plots

2. **Replication Justification** (`replication_analysis.md`)
   - Simulation type classification
   - Exact method calculation for each KPI
   - Warm-up period recommendation

3. **DOE Results** (`doe_analysis.md`)
   - Main effects and interaction table
   - Condition ranking for subgroup discrimination
   - Response surface coefficients

4. **Cluster Analysis** (`cluster_report.md`)
   - Optimal k determination
   - Cluster profiles
   - Condition ranking by salience

5. **Predictions** (`predictions.md`)
   - Interpolated turn rates for untested conditions
   - Prediction intervals
   - Optimal condition recommendations

---

## Research Questions to Answer

1. **Input Fidelity:** Are the fitted input distributions adequate for simulation? Which distributions require refinement?

2. **Statistical Power:** How many replications (experiments) are needed to detect a 20% change in turn rate with 80% power?

3. **Subgroup Discovery:** What proportion of larvae are "non-responders" (turn rate increase < 10% during stimulus)?

4. **Optimal Discrimination:** Which untested condition would best separate behavioral subgroups?

5. **Emergent Patterns:** Do individual-level hazard model predictions match population-level System Dynamics expectations?

---

## Evaluation Criteria

Your analysis will be evaluated on:

| Criterion | Weight | Description |
|-----------|--------|-------------|
| Statistical Rigor | 30% | Correct application of GoF tests, CI methods, DOE |
| Justification Quality | 25% | Clear reasoning for simulation type, replication count |
| Insight Generation | 25% | Novel findings about subgroups, optimal conditions |
| Reproducibility | 20% | Complete code, documented assumptions, clear outputs |

**Expected deliverable length:** 15-20 pages with figures and tables
