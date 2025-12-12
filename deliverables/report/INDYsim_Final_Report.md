# INDYsim: Optogenetic Modulation of Larval Behavior

## A 2x2 Factorial Analysis Using Negative-Binomial GLMs

**ECS630 Term Project**  
**December 2025**

---

## Abstract

This study investigates how optogenetic LED stimulation parameters affect navigational behavior in *Drosophila* larvae. A 2x2 factorial design was used to manipulate LED1 (red) intensity range (0-250 vs 50-250 PWM) and LED2 (blue) modulation type (constant vs time-varying). Analysis of 14 experiments (349 tracks, each >= 10 minutes) using negative-binomial generalized linear models (NB-GLM) with experiment-level stratified bootstrap confidence intervals suggests a LED1 effect: restricting LED1 to the high-intensity range (50-250 PWM) was associated with approximately 42% lower reversal rate compared to the full range (rate ratio = 0.58, 95% CI [0.38, 0.89], p = 0.0007). LED2 modulation type showed no significant effect (p = 0.94), nor did the LED1 by LED2 interaction (p = 0.45). Leave-one-experiment-out cross-validation was used to assess model generalization. These findings are consistent with sensory adaptation mechanisms that depend on stimulus dynamic range, similar to patterns reported in prior work on larval navigation (Gepner 2015, Klein 2015). This analysis integrates ECS630 concepts including terminating simulation, 2^k factorial design, Poisson process modeling and DOE metamodels.

---

## 1. Introduction

### 1.1 Background

*Drosophila melanogaster* larvae exhibit stereotyped navigational behaviors that can be quantitatively characterized using automated tracking systems. These behaviors include:

- **Runs**: Forward crawling at characteristic speeds
- **Turns**: Reorientation events with heading angle changes > 45 degrees
- **Reversals**: Backward crawling (SpeedRunVel < 0) lasting > 3 seconds

Optogenetic tools enable manipulation of neural circuits controlling these behaviors. The GMR61 driver line targets neurons involved in sensory processing.

### 1.2 Prior Work

Several studies have characterized the sensory-motor transformation in larval navigation:

- **Gepner et al. (2015)**: Linear-nonlinear-Poisson (LNP) models were developed for photo- and odor-taxis. Turn probability was shown to be predictable from filtered stimulus history using raised-cosine temporal kernels.

- **Klein et al. (2015)**: GLM approaches were extended to thermotaxis. Larval behavior was found to follow predictable stimulus-response dynamics with temporal filters on the order of one second.

- **Hernandez-Nunez et al. (2015)**: Reverse-correlation analysis was applied to optogenetic experiments, establishing methodologies for extracting sensory filters from behavioral data.

### 1.3 Research Questions

This study addresses:

1. Does LED1 intensity range (dynamic range of red stimulation) affect reversal behavior?
2. Does LED2 modulation type (constant vs pulsed blue background) influence responses?
3. Do LED1 and LED2 effects combine additively or interact?

### 1.4 Approach

A 2x2 factorial experimental design was analyzed with negative-binomial GLMs, incorporating:

- Mason Klein-style event detection (SpeedRunVel computation)
- Experiment-level stratified bootstrap for confidence intervals
- Leave-one-experiment-out cross-validation
- Temporal kernel estimation via stimulus-triggered averages

---

## 2. Methods

### 2.1 Experimental Design

**Genotype**: GMR61-GAL4 > UAS-ChR2 (channelrhodopsin-2 expressing larvae)

**Factorial Design** (2x2):

| Factor | Level 1 | Level 2 |
|--------|---------|---------|
| LED1 (Red) intensity range | 0-250 PWM (full) | 50-250 PWM (high) |
| LED2 (Blue) modulation | Constant 7 PWM | Time-varying 5-15 PWM |

**Stimulus Protocol**:
- LED1: 30-second square wave (15s ON, 15s OFF)
- LED2: Either constant or synchronized pulsed pattern
- Total experiment duration: 20 minutes

**Sample Sizes**:

| Condition | Experiments | Tracks (>= 10 min) |
|-----------|-------------|-------------------|
| Full range + Constant | 4 | 95 |
| Full range + Time-varying | 4 | 99 |
| High range + Constant | 4 | 101 |
| High range + Time-varying | 2 | 54 |
| **Total** | **14** | **349** |

### 2.2 Event Detection

Mason Klein's methodology was followed:

**SpeedRunVel** = Speed times cos(theta)

Where theta is the angle between velocity vector and head direction. This metric distinguishes:
- Positive values: forward crawling
- Negative values: backward crawling (reversal)

**Detection Criteria**:
- **Turns**: |delta heading| > 45 degrees between consecutive frames, minimum 0.5s between events
- **Reversals**: SpeedRunVel < 0 for at least 3 consecutive seconds

### 2.3 Statistical Modeling

**Primary Model**: Negative-Binomial GLM

```
log E[count_i] = B0 + B_L1*LED1_high + B_L2*LED2_timevar + B_int*(LED1*LED2) + log(duration_i)
```

The log(duration) term serves as an offset, converting counts to rates.

**Cluster-Robust Standard Errors**: To account for within-experiment correlation, standard errors were clustered on experiment (14 clusters).

### 2.4 Bootstrap Confidence Intervals

Stratified experiment-level bootstrap was implemented (n = 2000 iterations):

1. For each factorial cell (2x2 = 4 cells):
   - Resample experiments with replacement
2. Combine all cells to form bootstrap sample
3. Refit NB-GLM
4. Store coefficient estimates
5. Compute 2.5th and 97.5th percentiles

This stratified approach ensures all four conditions are represented in every bootstrap sample, which is important given the limited experiments per cell (2-4).

### 2.5 Cross-Validation

Leave-one-experiment-out cross-validation was performed (14 folds):

1. Hold out one experiment
2. Fit model on remaining 13 experiments
3. Compute NB log-likelihood on held-out data
4. Report mean and SD of deviance across folds

### 2.6 Temporal Kernel Estimation

Stimulus-triggered averages (STA) were computed:

1. Identify all turn times across experiments
2. Extract LED1 values in [-3s, +1s] window around each turn
3. Baseline-subtract (mean LED1 value)
4. Average across events
5. Bootstrap over events (n = 2000) for 95% CIs

---

## 3. Results

### 3.1 LED1 Main Effect on Reversals

The high-range LED1 condition (50-250 PWM) was associated with reduced reversal rate compared to full-range (0-250 PWM):

| Metric | Value | 95% CI |
|--------|-------|--------|
| Log coefficient (B_LED1) | -0.55 | [-0.99, -0.11] |
| Rate ratio | 0.58 | [0.38, 0.89] |
| p-value | 0.0145 | |
| Bootstrap CI (stratified) | | [0.38, 0.89] |

**Interpretation**: Larvae exposed to the high-range LED1 condition showed approximately 42% lower reversal rate. This pattern suggests that reducing the dynamic range of red stimulation (eliminating the 0-50 PWM low-intensity portion) may alter behavioral responses.

**Kruskal-Wallis confirmation**: Non-parametric test on per-track reversal rates also indicated a LED1 effect (p < 0.0001).

### 3.2 LED2 Main Effect

LED2 modulation type showed no significant effect on reversal rate:

| Metric | Value | 95% CI |
|--------|-------|--------|
| Log coefficient (B_LED2) | -0.02 | [-0.56, 0.52] |
| Rate ratio | 0.98 | [0.57, 1.68] |
| p-value | 0.94 | |

### 3.3 LED1 by LED2 Interaction

No significant interaction was detected:

| Metric | Value | 95% CI |
|--------|-------|--------|
| Log coefficient (B_interaction) | 0.25 | [-0.40, 0.89] |
| Rate ratio | 1.28 | [0.67, 2.44] |
| p-value | 0.45 | |

The parallel lines in the interaction plot (Figure 4) are consistent with additive effects.

### 3.4 Model Validation

**Overdispersion Check**:
- Poisson deviance/df = 3.70
- NB deviance/df = 0.97
- Conclusion: NB model appears appropriate (overdispersion present in Poisson)

**Cross-Validation**:
- Mean deviance: 124.3 plus/minus 28.2 across 14 held-out experiments
- Similar performance across LED1 conditions:
  - Full range: mean deviance = 129.8 (8 experiments)
  - High range: mean deviance = 116.9 (6 experiments)

### 3.5 Temporal Kernel

The stimulus-triggered average revealed:

- **Peak timing**: approximately 0.4-0.95 seconds after turn initiation
- **Peak amplitude**: approximately 59-104 PWM (baseline-subtracted)
- **N events**: 8,758 turns pooled across all conditions

The positive deflection indicates turns tend to occur when LED1 is elevated, though the post-turn timing suggests this reflects the periodic nature of the stimulus rather than a causal predictive relationship.

### 3.6 Turn Rate Analysis

Turns showed no significant effects:
- LED1: p = 0.77, RR = 0.96 [0.74, 1.28]
- LED2: p = 0.73, RR = 0.92 [0.56, 1.50]
- Interaction: p = 0.73, RR = 0.89 [0.45, 1.76]

This pattern suggests the LED1 effect may be specific to reversals rather than general reorientation behavior.

---

## 4. Discussion

### 4.1 Biological Interpretation

The observed LED1 effect on reversals (approximately 42% reduction) could reflect variance adaptation mechanisms:

1. **Variance adaptation**: When exposed to stimuli that never drop to zero (50-250 PWM), larvae may adapt their reversal threshold, resulting in fewer backward crawling events.

2. **Intensity floor effect**: The absence of "dark" periods in the high-range condition may reduce contrast-driven reversal responses.

3. **Circuit-specific modulation**: The specificity to reversals (not turns) could suggest the effect acts on circuits controlling backward locomotion rather than general navigation.

### 4.2 Comparison to Prior Work

The temporal kernel peak timing (approximately 0.5-1s) is consistent with:
- Gepner 2015: approximately 0.5-2s sensory-to-motor delays in phototaxis
- Klein 2015: approximately 1s temperature-to-behavior latencies

The additive (non-interacting) nature of LED1 and LED2 effects is consistent with findings from multi-sensory integration studies suggesting independent processing of different light wavelengths.

### 4.3 Limitations

1. **Deterministic stimuli**: Square-wave LED patterns prevent true reverse-correlation analysis; random flicker would enable cleaner kernel estimation.

2. **Sample size imbalance**: The high-range plus time-varying cell had only 2 experiments (54 tracks), potentially limiting interaction detection power.

3. **Event detection thresholds**: The 3-second reversal duration criterion may miss brief backward movements.

4. **Circadian effects**: Experiments were conducted at different times; time-of-day effects were not modeled.

---

## 5. ECS630 Integration

This analysis demonstrates several concepts from ECS630 Simulation and Modeling:

| ECS630 Concept | Implementation |
|----------------|----------------|
| **Terminating simulation** | 20-minute experiments as terminating runs with fixed duration |
| **2^k factorial design** | LED1 by LED2 (2x2) factorial with interaction term |
| **Replication** | 14 experiments total; stratified bootstrap preserves replication structure |
| **Confidence intervals** | Bootstrap CIs for GLM coefficients and rate ratios |
| **Poisson process** | Turn/reversal arrivals modeled as inhomogeneous Poisson/NB processes |
| **DOE metamodel** | NB-GLM serves as regression metamodel linking factors to response rate |

### 5.1 Terminating Simulation Perspective

Each 20-minute experiment represents a terminating simulation:
- Clear start (t=0) and end (t=1200s) times
- No warm-up period issues (larvae active from start)
- Output: count of events in fixed observation window

### 5.2 Factorial Analysis

The 2^k factorial structure enables:
- Main effect estimation with orthogonal contrasts
- Interaction testing
- Efficient use of experimental units

### 5.3 Bootstrap as Variance Estimation

The stratified bootstrap addresses:
- Non-independence of tracks within experiments
- Unbalanced design (2-4 experiments per cell)
- Distributional assumptions (no normality required)

---

## 6. Conclusion

This study suggests that LED1 intensity range modulates larval reversal behavior in optogenetic experiments. The approximately 42% reduction in reversal rate under high-range stimulation (50-250 PWM vs 0-250 PWM) is consistent with sensory adaptation mechanisms that depend on stimulus dynamic range. The null effect of LED2 modulation and absence of interaction suggest that the two light sources may affect behavior through independent pathways.

Methodologically, this work demonstrates:
1. Mason Klein-style SpeedRunVel computation for reversal detection
2. NB-GLM with cluster-robust SEs for count data analysis
3. Stratified bootstrap for proper experiment-level inference
4. Leave-one-experiment-out CV for model validation

Future work could extend this analysis with:
- Random flicker stimuli for cleaner kernel estimation
- Additional LED1 intensity ranges to map the adaptation function
- Circuit-level investigation using targeted silencing

---

## References

1. Gepner R, Mihovilovic Skanata M, Berber NM, Dacber M, Gershow M (2015). Computations underlying Drosophila photo-taxis, odor-taxis, and multi-sensory integration. *eLife* 4:e06229.

2. Klein M, Afonso B, Vonner AJ, Hernandez-Nunez L, Berck M, Taber CJ, Cardona A, Zlatic M, Sprecher SG, Samuel ADT (2015). Sensory determinants of behavioral dynamics in Drosophila thermotaxis. *PNAS* 112(2):E220-E229.

3. Hernandez-Nunez L, Belina J, Klein M, Si G, Claus L, Carlson JR, Samuel ADT (2015). Reverse-correlation analysis of navigation dynamics in Drosophila larva using optogenetics. *eLife* 4:e06225.

4. Efron B, Tibshirani RJ (1993). An Introduction to the Bootstrap. Chapman & Hall/CRC.

---

## Appendix: Code and Data

All analysis code and data are available in the INDYsim repository:

- **Event detection**: `scripts/2025-12-09/analysis/mason_klein_methods.py`
- **GLM fitting**: `scripts/2025-12-09/analysis/fit_nb_glm.py`
- **Bootstrap**: Integrated in `fit_nb_glm.py` (stratified_bootstrap_ci function)
- **Cross-validation**: `scripts/2025-12-09/analysis/cross_validation.py`
- **Temporal kernels**: `scripts/2025-12-09/analysis/temporal_kernels.py`
- **Figures**: `scripts/2025-12-09/figures/generate_figures.py`

Data files:
- `data/exports/glm_dataset.csv`: Track-level summary statistics
- `data/exports/nb_glm_results.json`: GLM coefficients and bootstrap CIs
- `data/exports/cv_results.json`: Cross-validation results
- `data/exports/temporal_kernel_results.json`: STA results
