# Supplementary Tables

## Table S1: Leave-One-Experiment-Out Cross-Validation Results

| Experiment | Condition | Empirical Events | Predicted Events | Rate Ratio | Status |
|------------|-----------|------------------|------------------|------------|--------|
| GMR61_T_Re_Sq_0to250PWM_30#C_Bl_7PWM | 0→250 | Control | 737 | 785 | 1.066 | + |
| GMR61_T_Re_Sq_0to250PWM_30#C_Bl_7PWM | 0→250 | Control | 670 | 629 | 0.938 | + |
| GMR61_T_Re_Sq_0to250PWM_30#T_Bl_Sq_5to15PWM_30 | 0→250 | Temp | 555 | 921 | 1.659 | - |
| GMR61_T_Re_Sq_0to250PWM_30#T_Bl_Sq_5to15PWM_30 | 0→250 | Temp | 488 | 395 | 0.809 | + |
| GMR61_T_Re_Sq_0to250PWM_30#T_Bl_Sq_5to15PWM_30 | 0→250 | Temp | 822 | 603 | 0.733 | - |
| GMR61_T_Re_Sq_0to250PWM_30#T_Bl_Sq_5to15PWM_30 | 0→250 | Temp | 545 | 534 | 0.981 | + |
| GMR61_T_Re_Sq_50to250PWM_30#C_Bl_7PWM | 50→250 | Control | 766 | 603 | 0.787 | - |
| GMR61_T_Re_Sq_50to250PWM_30#C_Bl_7PWM | 50→250 | Control | 571 | 937 | 1.641 | - |
| GMR61_T_Re_Sq_50to250PWM_30#C_Bl_7PWM | 50→250 | Control | 657 | 655 | 0.997 | + |
| GMR61_T_Re_Sq_50to250PWM_30#C_Bl_7PWM | 50→250 | Control | 446 | 305 | 0.684 | - |
| GMR61_T_Re_Sq_50to250PWM_30#T_Bl_Sq_5to15PWM_30 | 50→250 | Temp | 477 | 553 | 1.160 | + |
| GMR61_T_Re_Sq_50to250PWM_30#T_Bl_Sq_5to15PWM_30 | 50→250 | Temp | 554 | 478 | 0.862 | + |

**Summary:** 12 experiments, mean rate ratio = 1.026 +/- 0.309, pass rate = 58.3% (7/12)

---

## Table S2: Model Comparison

| Model | Parameters | AIC | Deviance | Notes |
|-------|------------|-----|----------|-------|
| Fixed-effects NB-GLM | 8 | 114814.4 | 94591.8 | Primary model |
| NB-GLMM (1|track) | 9 + 623 RE | -- | -- | Random intercepts |

---

## Table S3: Factorial Model Coefficients

| Parameter | Estimate | SE | 95% CI | p-value | Interpretation |
|-----------|----------|----|---------|---------:|----------------|
| beta_0 | -6.6444 | 0.0273 | [-6.698, -6.591] | 0.00e+00* | Baseline log-hazard (reference condition) |
| beta_I | -0.1991 | 0.0341 | [-0.266, -0.132] | 5.23e-09* | Effect of 50->250 intensity on baseline |
| beta_T | -0.1078 | 0.0336 | [-0.174, -0.042] | 0.0014* | Effect of cycling background on baseline |
| beta_IT | -0.1185 | 0.0506 | [-0.218, -0.019] | 0.0191* | Intensity x Cycling interaction (baseline) |
| alpha | 1.0048 | 0.0538 | [0.899, 1.110] | 9.37e-78* | Reference suppression amplitude |
| alpha_I | -0.6648 | 0.0550 | [-0.773, -0.557] | 1.12e-33* | Intensity effect on suppression (66% weaker) |
| alpha_T | 0.1518 | 0.0520 | [0.050, 0.254] | 0.0035* | Cycling effect on suppression (15% stronger) |
| gamma | 1.6694 | 0.6120 | [0.470, 2.869] | 0.0064* | LED-OFF rebound coefficient |

*p < 0.05

---

## Table S4: Condition-Specific Suppression Amplitudes

| Condition | Amplitude | Events | Tracks | Interpretation |
|-----------|-----------|--------|--------|----------------|
| 0→250 | Control | 1.005 | 1,407 | 99 | Reduced (partial adaptation) |
| 0→250 | Temp | 1.157 | 2,410 | 214 | Reduced (partial adaptation) |
| 50→250 | Control | 0.340 | 2,440 | 187 | Reduced (partial adaptation) |
| 50→250 | Temp | 0.492 | 1,031 | 123 | Reduced (partial adaptation) |

---

