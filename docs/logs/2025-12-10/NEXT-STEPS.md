# Next Steps: INDYsim Hazard Modeling

## Phase 1: Data Validation (Immediate)

1. **Validate consolidated HDF5 structure**
   - Confirm all columns present and correctly typed
   - Check for NaN/Inf values in key covariates
   - Verify time_since_stimulus bounded [0, 30]

2. **Generate summary statistics**
   - Turn rate per experiment (events/minute)
   - Heading change distribution (reo_dtheta)
   - Inter-event interval distribution
   - Speed/curvature distributions

3. **Stimulus-locked analysis**
   - PSTH of turn rate aligned to LED1 onset
   - Compare ON vs OFF period turn rates
   - Latency to first turn after stimulus onset

## Phase 2: Hazard Model Fitting

1. **Prepare event data for modeling**
   - Bin frames into 0.5s or 1s windows
   - Count reorientations per bin
   - Attach mean covariates (speed, LED intensity)

2. **Fit baseline model**
   - Poisson GLM with LED1 only
   - Check for overdispersion (Pearson chi2 / df)
   - If overdispersed, switch to NB

3. **Fit full NB-GLM**
   - Add temporal kernel (4 bases, -3s to 0)
   - Add phase covariates
   - Add speed and curvature
   - Report coefficient table

4. **Cross-validate kernel hyperparameters**
   - Leave-one-experiment-out CV
   - Compare 3/4/5 bases
   - Compare -2s/-3s/-4s windows

## Phase 3: Model Interpretation

1. **Extract temporal kernel shape**
   - Plot weighted sum of basis functions
   - Identify peak response latency
   - Quantify response duration

2. **Compute effect sizes**
   - LED1 ON vs OFF: relative risk ratio
   - Speed effect: turn probability at low/high speed
   - Curvature effect: turn probability at low/high curvature

3. **Model diagnostics**
   - Residual vs fitted plot
   - Q-Q plot of deviance residuals
   - Dispersion ratio (target: ~1.0)

## Phase 4: Simulation

1. **Implement hazard-based event generator**
   - Use inversion sampling (already scaffolded)
   - Generate synthetic reorientation times
   - Sample heading changes from empirical distribution

2. **Simulate full trajectories**
   - Run segments: constant speed + Gaussian noise
   - Reorientation segments: heading change + head swings
   - Concatenate to form full track

3. **Validation**
   - Compare simulated vs empirical turn rate
   - Compare heading change distributions
   - Compare stimulus-locked PSTHs

## Timeline Estimate

| Phase | Duration | Depends On |
|-------|----------|------------|
| 1. Validation | 1-2 hours | Data complete |
| 2. Model fitting | 2-4 hours | Validation |
| 3. Interpretation | 1-2 hours | Model fitting |
| 4. Simulation | 2-4 hours | Interpretation |

Total: 6-12 hours of focused work

## Dependencies

Required packages:
- statsmodels (NB-GLM)
- pyarrow (parquet I/O)
- h5py (HDF5 I/O)
- matplotlib (plotting)
- scipy (statistics)

All currently installed.

## Key Files

- `scripts/hazard_model.py` - Model fitting (650 lines, ready to run)
- `scripts/event_generator.py` - Simulation (579 lines, scaffolded)
- `data/processed/consolidated_dataset.h5` - Input data (1.92 GB)

