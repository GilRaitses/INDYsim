# Design of Experiments (DOE) Design Considerations

**Date:** 2025-11-11  
**Prepared by:** larry  
**Purpose:** Technical considerations for DOE design, replication requirements, and CI calculations

## Current Experimental Data

### Factorial Design Structure

**Factors:**
- **Factor A:** LED1 Intensity Range
  - A1: 0-250 PWM (includes zero baseline)
  - A2: 50-250 PWM (no zero baseline)
- **Factor B:** LED2 Type
  - B1: Constant (7 PWM)
  - B2: Time-varying (5-15 PWM square wave)

**Current Conditions:** 2 × 2 = 4 ESETs  
**Current Replications:** 2-4 per condition (imbalanced)

## Proposed Simulation DOE

### Target Specification
- **45 conditions**
- **30 replications per condition**
- **Total simulations:** 1,350

### Factor Space Considerations

#### Factor 1: Stimulus Intensity (LED1 PWM)

**Options:**
- **Option A:** Discrete levels
  - Levels: 0, 50, 100, 150, 200, 250 PWM (6 levels)
  - Pros: Clear comparison points, covers full range
  - Cons: May miss intermediate effects
  
- **Option B:** Continuous with key points
  - Key points: 0, 50, 100, 150, 200, 250 PWM
  - Additional points: 25, 75, 125, 175, 225 PWM (11 levels)
  - Pros: Better resolution, can fit response curves
  - Cons: More conditions needed

**Recommendation:** Start with 6 levels (0, 50, 100, 150, 200, 250), can add intermediate if needed

#### Factor 2: Pulse Duration

**Current:** 10 seconds (to be validated)

**Options:**
- **Option A:** Few levels
  - Levels: 5s, 10s, 15s (3 levels)
  - Pros: Covers reasonable range, manageable
  - Cons: May miss intermediate durations
  
- **Option B:** More levels
  - Levels: 5s, 7.5s, 10s, 12.5s, 15s (5 levels)
  - Pros: Better resolution
  - Cons: More conditions

**Recommendation:** 3-4 levels (5s, 10s, 15s, maybe 20s)

#### Factor 3: Inter-Pulse Interval (Rest Duration)

**Current:** 30 seconds

**Options:**
- **Option A:** Few levels
  - Levels: 10s, 20s, 30s, 40s (4 levels)
  - Pros: Covers reasonable range
  - Cons: May miss intermediate intervals
  
- **Option B:** More levels
  - Levels: 10s, 15s, 20s, 25s, 30s, 35s, 40s (7 levels)
  - Pros: Better resolution
  - Cons: Many conditions

**Recommendation:** 4-5 levels (10s, 20s, 30s, 40s, maybe 50s)

### DOE Structure Options

#### Option 1: Full Factorial (Too Many Conditions)

**Structure:** 6 × 4 × 5 = 120 conditions  
**Replications:** 30 each  
**Total:** 3,600 simulations  
**Verdict:** Too many, computationally expensive

#### Option 2: Fractional Factorial

**Structure:** Select subset of conditions  
**Example:** 6 × 3 × 3 = 54 conditions (close to 45 target)  
**Replications:** 30 each  
**Total:** 1,620 simulations  
**Verdict:** Reasonable, but may miss interactions

#### Option 3: Response Surface Design

**Structure:** Focus on regions of interest  
**Example:** 
- Low intensity (0, 50, 100 PWM) × All durations × All intervals
- High intensity (150, 200, 250 PWM) × Key durations × Key intervals
**Total:** ~45 conditions  
**Verdict:** Good for exploring response surface, may miss some regions

#### Option 4: Optimal Design (D-Optimal or I-Optimal)

**Structure:** Statistically optimal condition selection  
**Method:** Maximize information content  
**Total:** Exactly 45 conditions  
**Verdict:** Best statistical properties, but requires prior knowledge

**Recommendation:** Start with Option 2 (Fractional Factorial) or Option 3 (Response Surface), can refine based on initial results

## Replication Requirements

### Statistical Considerations

#### Confidence Interval Width

**Formula:** CI width = 2 × t_(α/2, n-1) × s/√n

**For 95% CI:**
- **n=10:** CI width ≈ 2.26 × s/√10 ≈ 0.715 × s
- **n=20:** CI width ≈ 2.09 × s/√20 ≈ 0.467 × s
- **n=30:** CI width ≈ 2.05 × s/√30 ≈ 0.374 × s
- **n=50:** CI width ≈ 2.01 × s/√50 ≈ 0.284 × s

**Interpretation:**
- More replications → Narrower CI → More precise estimates
- Diminishing returns: Going from 30 to 50 reduces CI width by ~24%

#### Power Analysis

**Effect Size:** Depends on what we're trying to detect

**Example:** Detect 20% difference in turn rate
- **n=10:** Power ≈ 0.5-0.6 (low)
- **n=20:** Power ≈ 0.7-0.8 (moderate)
- **n=30:** Power ≈ 0.8-0.9 (good)
- **n=50:** Power ≈ 0.9-0.95 (excellent)

**Recommendation:** n=30 provides good balance between precision and computational cost

### Computational Considerations

**Simulation Time:**
- Per simulation: ~1-5 seconds (depends on trajectory length)
- 1,350 simulations: ~22-112 minutes (single-threaded)
- With parallelization: ~5-20 minutes (8-16 cores)

**Storage:**
- Per simulation: ~1-10 MB (depends on output detail)
- 1,350 simulations: ~1.4-13.5 GB

**Recommendation:** n=30 is computationally feasible

## Condition Selection Strategy

### Priority-Based Selection

**High Priority Conditions:**
1. **Current experimental conditions** (4 ESETs)
   - Validate model against experimental data
   - Ensure model captures known responses

2. **Extreme conditions**
   - Low intensity (0-50 PWM)
   - High intensity (200-250 PWM)
   - Short pulses (5s)
   - Long pulses (15-20s)
   - Short intervals (10s)
   - Long intervals (40-50s)

3. **Intermediate conditions**
   - Fill gaps between extremes
   - Explore response surface

**Low Priority Conditions:**
- Very extreme values (may be unrealistic)
- Conditions far from experimental data

### Condition Selection Algorithm

**Step 1:** Include all 4 experimental conditions (for validation)

**Step 2:** Add extreme conditions (boundaries of parameter space)

**Step 3:** Fill intermediate conditions (systematic or optimal design)

**Step 4:** Verify total ≈ 45 conditions

**Step 5:** Run pilot simulations (fewer replications) to check feasibility

## CI Calculation for Summary Statistics

### Per-Condition Statistics

**Mean Turn Rate:**
- Calculate: Mean ± 95% CI
- CI width depends on n and standard deviation
- Report: Mean (CI_lower, CI_upper)

**Mean Latency:**
- Calculate: Mean ± 95% CI
- May need log transformation if skewed

**Stop Fraction:**
- Calculate: Proportion ± 95% CI (binomial)
- Use Wilson score interval or Clopper-Pearson

### Cross-Condition Comparisons

**Pairwise Comparisons:**
- Use t-tests or ANOVA
- Adjust for multiple comparisons (Bonferroni or FDR)

**Effect Sizes:**
- Cohen's d for mean differences
- Odds ratios for proportions

**Visualization:**
- Error bars showing CI
- Overlapping CI indicates non-significant difference

## Recommendations

### DOE Design
- **45 conditions:** Use fractional factorial or response surface design
- **Factor levels:**
  - Intensity: 6 levels (0, 50, 100, 150, 200, 250 PWM)
  - Pulse duration: 3-4 levels (5s, 10s, 15s, maybe 20s)
  - Inter-pulse interval: 4-5 levels (10s, 20s, 30s, 40s, maybe 50s)
- **Selection:** Prioritize experimental conditions, extremes, then fill gaps

### Replication Level
- **30 replications per condition:** Good balance of precision and cost
- **Rationale:**
  - Provides reasonable CI width (~0.37 × SD)
  - Good statistical power (0.8-0.9 for moderate effect sizes)
  - Computationally feasible
  - Standard for simulation studies

### CI Reporting
- **95% CI** for all summary statistics
- **Report format:** Mean (CI_lower, CI_upper)
- **Visualization:** Error bars on all plots
- **Comparisons:** Use overlapping CI as indicator of significance

## Questions for Discussion

1. **DOE Structure:**
   - Is 45 conditions appropriate, or should we adjust?
   - Which factor combinations are most important?
   - Should we use full factorial, fractional factorial, or optimal design?

2. **Replication Level:**
   - Is 30 replications sufficient, or do we need more?
   - What level of precision is required?
   - Are there computational constraints?

3. **Condition Selection:**
   - What parameter ranges are most biologically relevant?
   - Are there specific conditions we must include?
   - Should we prioritize certain regions of parameter space?

4. **CI Requirements:**
   - What CI width is acceptable?
   - Should we use 95% CI or different confidence level?
   - How should we handle multiple comparisons?

---

**Status:** Technical considerations documented, ready for discussion  
**Last Updated:** 2025-11-11

