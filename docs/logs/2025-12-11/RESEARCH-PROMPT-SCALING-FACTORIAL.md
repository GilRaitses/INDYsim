# Research Prompt: Scaling INDYsim to Full 2×2 Factorial Design

## Context

We have developed a validated hazard model for larval reorientation, but discovered we've only used **2 of 14 available experiments (14%)**. The full dataset represents a **2×2 factorial design** that we should leverage.

---

## VERIFIED EXPERIMENTAL PARAMETERS

### LED Timing (CONFIRMED IDENTICAL ACROSS ALL CONDITIONS)
- **LED1 ON duration**: 10.0 seconds
- **LED1 OFF duration**: 20.0 seconds
- **Cycle**: 10s ON / 20s OFF (30s total period)
- **Frame rate**: 20 Hz (dt = 0.05s)

### Condition Definitions
| Factor | Level 1 | Level 2 |
|--------|---------|---------|
| **LED1 Intensity** | 0→250 PWM (full range) | 50→250 PWM (partial) |
| **LED2/Background** | Control (7 PWM constant) | Temp (5-15 PWM cycling) |

---

## CURRENT STATE

### What We Built
- 6-parameter gamma-difference kernel (τ₁=0.29s, τ₂=3.81s)
- Calibrated NB-GLM hazard model
- Basic RUN/TURN trajectory simulator
- Validation: rate ratio 0.97, PSTH correlation 0.84

### What We Used
- **2 experiments** from condition: 0→250 PWM | Control (7 PWM)
- **55 tracks**, **1,407 events**
- Dates: 202510301228, 202510301408

### What We Have Available (VERIFIED)

**IMPORTANT: Event count anomaly detected**
- Two experiments (202510291652, 202510291713) have 10-20x more events than others
- This suggests different annotation density or event definition
- We correctly used the consistent subset

| Condition | LED1 | LED2 | Experiments | Tracks | Events | Notes |
|-----------|------|------|-------------|--------|--------|-------|
| 0→250 \| Control | 0→250 | 7 PWM | 4 | 177 | 25,238 | 2 anomalous, 2 usable |
| 0→250 \| Temp | 0→250 | 5-15 PWM | 4 | 214 | 2,410 | All consistent |
| 50→250 \| Control | 50→250 | 7 PWM | 4 | 187 | 2,440 | All consistent |
| 50→250 \| Temp | 50→250 | 5-15 PWM | 2 | 123 | 1,031 | All consistent |
| **TOTAL** | | | **14** | **701** | **31,119** | |

### Adjusted Totals (excluding 2 anomalous experiments)
| Condition | Experiments | Tracks | Events |
|-----------|-------------|--------|--------|
| 0→250 \| Control | 2 (USED) | 99 | 1,407 |
| 0→250 \| Temp | 4 | 214 | 2,410 |
| 50→250 \| Control | 4 | 187 | 2,440 |
| 50→250 \| Temp | 2 | 123 | 1,031 |
| **USABLE TOTAL** | **12** | **623** | **7,288** |

---

## OPPORTUNITY

### 2×2 Factorial Analysis

The experimental design naturally supports:

1. **Main effect of LED1 intensity**: 0→250 vs 50→250 PWM
   - Does suppression scale with starting intensity?
   - Is the kernel shape (timescales) intensity-dependent?

2. **Main effect of temperature/background**: Control vs Temp+5-15
   - Does temperature modulate reorientation rate?
   - Does it interact with optogenetic suppression?

3. **Interaction**: Does the intensity effect differ by temperature condition?

### Potential Paper Scope Expansion

Instead of a single-condition methods paper, we could write:

**Option 1: Condition-general hazard model**
- Fit hierarchical model with condition-specific parameters
- Report main effects and interactions
- Stronger biological contribution

**Option 2: Intensity-scaling analysis**
- Compare 0→250 vs 50→250 kernels
- Test linear vs nonlinear intensity scaling
- Validate on held-out conditions

**Option 3: Keep current scope, report full data in supplement**
- Main text: single condition (current)
- Supplement: show model applies across conditions
- Lower risk, still comprehensive

---

## QUESTIONS FOR RESEARCH AGENT

### Prioritization

1. **Should we expand scope to the full factorial design?**
   - What is the effort vs. impact trade-off?
   - Would this change the paper from "methods" to "biology"?

2. **If we expand, what's the recommended approach?**
   - Separate models per condition?
   - Hierarchical/mixed-effects with condition as covariate?
   - Simple pooling with condition indicators?

### Statistical Approach

3. **How to model the 2×2 factorial structure?**
   - Should LED1 intensity be a continuous covariate or categorical?
   - How to handle the temperature condition (different mechanism)?

4. **Is intensity scaling biologically plausible?**
   - Does 50→250 PWM produce weaker suppression than 0→250?
   - Is the relationship linear or saturating?

### Practical Considerations

5. **We already used ALL usable experiments from 0→250 | Control**
   - Currently: 2 experiments (202510301228, 202510301408), 99 tracks, 1,407 events
   - The other 2 experiments in this condition are anomalous (10-20x event density)
   - Cannot add more data from same condition without addressing anomaly

6. **LED timing is VERIFIED identical across all conditions**
   - ✅ All use 10s ON / 20s OFF
   - ✅ Frame rate 20 Hz throughout
   - No timing confounds

7. **Is the temperature condition fundamentally different?**
   - LED2 cycles 5-15 PWM instead of constant 7 PWM
   - May create thermal stimulation (5-15 PWM in "T_Bl_Sq" = Temperature Block Square?)
   - Should it be analyzed as factorial or treated as separate experiment?

8. **What to do about the 2 anomalous experiments?**
   - Option A: Exclude them (as we have done)
   - Option B: Investigate different annotation/event definition
   - Option C: Reprocess with consistent event definition

---

## PROPOSED APPROACH

### Phase 1: Cross-Condition Comparison (1-2 days)
- We've already used all usable data from 0→250 | Control
- Fit same model structure to each of the other 3 conditions:
  - 0→250 | Temp (4 experiments, 2,410 events)
  - 50→250 | Control (4 experiments, 2,440 events)
  - 50→250 | Temp (2 experiments, 1,031 events)
- Compare kernel parameters across conditions

### Phase 2: Main Effects Analysis (1 day)
- Test main effect of LED1 intensity (0→250 vs 50→250)
- Test main effect of temperature (Control vs Temp)
- Check for interaction

### Phase 3: Hierarchical Model (optional, 2-3 days)
- Fit condition-specific intercepts and kernels
- Shared kernel shape, condition-specific amplitude?
- Full factorial analysis with proper error structure

### Phase 4: Paper with Full Data
- Main text: present general model structure
- Results: show works across conditions
- Discussion: interpret intensity and temperature effects

### Alternative: Minimal Extension
- Keep current paper scope (single condition)
- Add one cross-validation: fit on 0→250 | Control, test on 50→250 | Control
- Report generalization in discussion

---

## SPECIFIC UNCERTAINTIES

1. **LED timing: RESOLVED**
   - ✅ All conditions use identical 10s ON / 20s OFF timing
   - ✅ Frame rate is 20 Hz across all experiments
   - ✅ No timing differences between conditions

2. **50→250 PWM may be qualitatively different**
   - Starting at 50 PWM means larva is already partially stimulated
   - May show different adaptation dynamics
   - The "delta" is 200 PWM vs 250 PWM - is this meaningful?

3. **Temperature condition interpretation**
   - LED2 cycles between 5-15 PWM (vs constant 7 PWM in Control)
   - Is this meant to create thermal stimulation?
   - How does it interact with optogenetic activation?

4. **Event count anomaly in first two experiments**
   - 202510291652: 14,617 events (20x higher than others)
   - 202510291713: 9,214 events (13x higher than others)
   - All other experiments: 446-822 events
   - Should these be excluded or investigated?

5. **Paper identity question**
   - Currently: methods paper about hazard kernel
   - With factorial: biology paper about intensity/temperature effects?

6. **Power implications**
   - 0→250 | Control: only 2 usable experiments (we used both)
   - Other conditions: 2-4 experiments each
   - Is the factorial balanced enough for inference?

---

## REQUESTED OUTPUTS

1. **Recommendation on scope expansion**: Yes/No and why

2. **If yes, which approach**: Hierarchical vs separate fits vs pooled

3. **Priority order**: What to do first

4. **Statistical guidance**: How to analyze 2×2 factorial with hazard model

5. **Paper framing**: Does this change from methods to biology paper?

6. **Risk assessment**: What could go wrong with expanding scope?

7. **What to do about the 2 anomalous experiments**: Exclude, investigate, or reprocess?

---

## APPENDIX: VERIFIED DATA SUMMARY

### Per-Experiment Details (for reference)

```
CONDITION: 0→250 | Control
  202510291652 | 51 tracks | 14,617 events | ANOMALY - exclude?
  202510291713 | 27 tracks |  9,214 events | ANOMALY - exclude?
  202510301228 | 44 tracks |    737 events | USED ✓
  202510301408 | 55 tracks |    670 events | USED ✓

CONDITION: 0→250 | Temp
  202510301513 | 64 tracks |    555 events | Available
  202510311441 | 40 tracks |    488 events | Available
  202510311510 | 52 tracks |    822 events | Available
  202510311634 | 58 tracks |    545 events | Available

CONDITION: 50→250 | Control
  202510291435 | 48 tracks |    766 events | Available
  202510291502 | 70 tracks |    571 events | Available
  202510291532 | 45 tracks |    657 events | Available
  202510291601 | 24 tracks |    446 events | Available

CONDITION: 50→250 | Temp
  202511051636 | 58 tracks |    477 events | Available
  202511051713 | 65 tracks |    554 events | Available
```

### LED Protocol Summary
- **LED1 (optogenetic)**: Square wave, 10s ON at 250 PWM / 20s OFF
- **LED1 baseline**: 0 PWM (0→250 condition) or 50 PWM (50→250 condition)
- **LED2 (background)**: 7 PWM constant (Control) or 5-15 PWM cycling (Temp)
- **Frame rate**: 20 Hz (dt = 0.05s)
- **Genotype**: GMR61 (all experiments)

