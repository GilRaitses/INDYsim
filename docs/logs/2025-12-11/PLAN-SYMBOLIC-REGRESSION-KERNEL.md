# Plan: Symbolic Regression for Analytic Kernel Discovery

**Created**: 2025-12-11  
**Status**: Planning  
**Goal**: Extract interpretable closed-form expression for the LNP temporal kernel

---

## Context

The hybrid model achieves excellent validation metrics:
- Rate ratio: 1.16x (target ≤1.2x) ✅
- Early suppression: 0.53 vs 0.55 (2% off) ✅
- Late suppression: 0.31 vs 0.32 (1% off) ✅

However, the kernel is represented as a sum of 12 raised-cosine basis functions:
- 4 early bases (0.2, 0.63, 1.07, 1.5s)
- 2 intermediate bases (2.0, 2.5s)
- 6 late bases (3.0, 4.2, 5.4, 6.6, 7.8, 9.0s)

**Problem**: This representation lacks biological interpretability. We want a compact analytic form like:

```
K(t) = A·exp(-t/τ_fast) - B·exp(-t/τ_slow) + C·exp(-t/τ_recovery)
```

Where parameters (τ_fast, τ_slow, τ_recovery) have direct physiological meaning.

---

## Phase 1: Evaluate Kernel on Dense Grid (30 min)

### Tasks
1. Load hybrid model coefficients from `data/model/hybrid_model_results.json`
2. Evaluate kernel K(t) on dense grid t ∈ [0, 10]s at 0.01s resolution
3. Save kernel values to `data/model/kernel_dense.csv`

### Output
- `data/model/kernel_dense.csv` - columns: [time, kernel_value]
- `data/validation/kernel_shape.png` - visualization

### Code Location
- Extend `scripts/fit_analytic_kernel.py`

---

## Phase 2: Fit Parametric Models (1 hour)

### Candidate Functional Forms

| Model | Formula | Parameters | Interpretation |
|-------|---------|------------|----------------|
| **Double-exponential** | A·exp(-t/τ₁) - B·exp(-t/τ₂) | 4 | Fast activation, slow suppression |
| **Triple-exponential** | A·exp(-t/τ₁) - B·exp(-t/τ₂) + C·exp(-t/τ₃) | 6 | +Recovery phase |
| **Gamma-difference** | A·Γ(t;α₁,β₁) - B·Γ(t;α₂,β₂) | 6 | Smooth rise/fall |
| **Alpha function** | A·t·exp(-t/τ₁) - B·exp(-t/τ₂) | 4 | Delayed peak |
| **DoG (Difference of Gaussians)** | A·exp(-t²/2σ₁²) - B·exp(-t²/2σ₂²) | 4 | Symmetric bump |

### Tasks
1. Implement each model as scipy.optimize target
2. Fit each to kernel_dense.csv using curve_fit with bounds
3. Compute AIC, BIC, R² for model comparison
4. Select best model by AIC

### Output
- `data/model/parametric_fits.json` - all fit results
- `data/validation/parametric_comparison.png` - overlay of fits

---

## Phase 3: Symbolic Regression with PySR (2 hours)

### Setup
```bash
pip install pysr
```

### Configuration
```python
from pysr import PySRRegressor

model = PySRRegressor(
    niterations=100,
    binary_operators=["+", "-", "*", "/"],
    unary_operators=["exp", "neg"],
    populations=20,
    population_size=50,
    maxsize=20,  # Max expression complexity
    timeout_in_seconds=3600,
    equation_file="data/model/pysr_equations.csv"
)
```

### Constraints
- **Complexity penalty**: Prefer simpler expressions
- **Physical constraints**: 
  - K(0) should be positive (early bump)
  - K(t→∞) should approach 0
  - K should have at least one sign change (suppression)

### Tasks
1. Run PySR on kernel_dense.csv
2. Extract Pareto-optimal equations (accuracy vs complexity)
3. Validate top 5 candidates against bootstrap threshold

### Output
- `data/model/pysr_equations.csv` - discovered expressions
- `data/model/symbolic_kernel.json` - best expression with parameters

---

## Phase 4: SINDy Alternative (1 hour)

### Motivation
SINDy (Sparse Identification of Nonlinear Dynamics) works by finding sparse combinations of a library of candidate functions.

### Library
```python
library = [
    1,                    # constant
    exp(-t/0.3),          # fast decay
    exp(-t/0.8),          # medium decay (matches IEI)
    exp(-t/2.0),          # slow decay
    exp(-t/5.0),          # very slow decay
    t * exp(-t/0.5),      # alpha function
    t * exp(-t/2.0),      # slow alpha
]
```

### Tasks
1. Construct library matrix Θ
2. Solve Θξ = K for sparse ξ using LASSO
3. Select non-zero terms
4. Refit with least squares

### Output
- `data/model/sindy_kernel.json` - sparse representation

---

## Phase 5: Physiological Interpretation (1 hour)

### Map Parameters to Biology

| Parameter | Expected Range | Biological Meaning |
|-----------|---------------|-------------------|
| τ_fast | 0.2-0.5s | Sensory transduction + early circuit |
| τ_slow | 1-3s | Synaptic depression / adaptation |
| τ_recovery | 3-8s | Network state recovery |
| A (amplitude) | 0.5-3 | Strength of optogenetic drive |
| B (suppression) | 0.3-2 | Strength of inhibition |

### Validation Against Literature
- Compare τ values to Gepner et al. 2015, 2018 (larval kernel timescales)
- Compare to Klein et al. 2015 (thermotaxis decision timescales)

### Output
- `docs/KERNEL_INTERPRETATION.md` - biological interpretation document

---

## Phase 6: Simulation with Analytic Kernel (1 hour)

### Tasks
1. Replace raised-cosine kernel with analytic form in simulation
2. Run validation to confirm metrics preserved
3. Compare computational efficiency

### Acceptance Criteria
- Rate ratio within 5% of hybrid model
- PSTH correlation within 0.05 of hybrid model
- Suppression magnitude within 5% of hybrid model

### Output
- `data/simulated/analytic_kernel_events.csv`
- `data/validation/analytic_vs_hybrid.png`

---

## Deliverables Summary

| Deliverable | File | Description |
|-------------|------|-------------|
| Dense kernel | `data/model/kernel_dense.csv` | 1000-point kernel evaluation |
| Parametric fits | `data/model/parametric_fits.json` | 5 candidate models |
| PySR equations | `data/model/pysr_equations.csv` | Discovered symbolic forms |
| Best kernel | `data/model/analytic_kernel_final.json` | Selected expression |
| Interpretation | `docs/KERNEL_INTERPRETATION.md` | Biological meaning |
| Validation | `data/validation/analytic_validation.json` | Simulation results |

---

## Timeline

| Phase | Duration | Dependencies |
|-------|----------|--------------|
| 1. Dense evaluation | 30 min | Hybrid model complete |
| 2. Parametric fits | 1 hr | Phase 1 |
| 3. PySR symbolic | 2 hrs | Phase 1 |
| 4. SINDy alternative | 1 hr | Phase 1 |
| 5. Interpretation | 1 hr | Phases 2-4 |
| 6. Validation | 1 hr | Phase 5 |
| **Total** | **6.5 hrs** | |

---

## Risks and Mitigations

| Risk | Likelihood | Impact | Mitigation |
|------|------------|--------|------------|
| PySR finds complex expressions | Medium | Medium | Enforce maxsize=15, prioritize Pareto front |
| No closed form fits well | Low | High | Fall back to piecewise or spline representation |
| Analytic kernel loses accuracy | Medium | Medium | Accept 5% degradation if interpretable |
| Installation issues with PySR | Medium | Low | Use SINDy as backup |

---

## Success Criteria

1. **Interpretability**: Kernel expressed in ≤6 parameters
2. **Accuracy**: R² ≥ 0.95 vs learned kernel
3. **Validation**: Rate ratio within 1.25x
4. **Publication-ready**: Parameters have biological interpretation

---

## Questions for Research Agent

1. What is the standard functional form for sensory adaptation kernels in Drosophila?
2. Are there known timescales (τ values) for optogenetic response in larval circuits?
3. Should the kernel include a refractory component or is that handled separately?
4. Is difference-of-exponentials or alpha-function more common in sensorimotor models?
5. How do we handle the LED-off rebound term in the analytic form?

