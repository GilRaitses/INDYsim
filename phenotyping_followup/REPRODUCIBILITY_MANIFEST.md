# Reproducibility Manifest
## Phenotyping Follow-up Study

**Generated**: 2025-12-17  
**Authors**: Gil Raitses, Devindi Goonawardhana, Mirna Mihovilovic-Skanata  
**Institution**: Syracuse University

---

## 1. Primary Data Source

| Property | Value |
|----------|-------|
| **File** | `data/processed/consolidated_dataset.h5` |
| **Size** | 5,927 MB |
| **Format** | HDF5 |
| **Experiments** | 14 |

### 1.1 Data Tables

| Table | Rows | Description |
|-------|------|-------------|
| `/events` | 8,510,608 | Frame-level event annotations |
| `/trajectories` | 8,741,825 | Frame-level trajectory data |
| `/klein_run_table` | 8,822 | Run-level summaries (MAGAT segmented) |

### 1.2 Klein Run Table (Primary Event Source)

The `klein_run_table` contains run-level summaries from MAGAT segmentation. Each row represents one run (inter-reorientation interval).

| Field | Type | Description |
|-------|------|-------------|
| `experiment_id` | string | Experiment identifier |
| `track` | int64 | Track number within experiment |
| `expt` | int64 | Experiment index |
| `set` | int64 | Experimental set |
| `time0` | float64 | Run start time (s) |
| `runT` | float64 | Run duration (s) |
| `runL` | float64 | Run path length (mm) |
| `run_speed` | float64 | Mean run speed (mm/s) |
| `run_displacement` | float64 | Net displacement (mm) |
| `run_efficiency` | float64 | Path efficiency |
| `reoYN` | int64 | Reorientation occurred (0/1) |
| `reo#HS` | int64 | Number of head sweeps |
| `reoHS1` | float64 | First head sweep magnitude |
| `reoQ1`, `reoQ2` | float64 | Reorientation quantiles |
| `turn_magnitude` | float64 | Turn angle (degrees) |
| `turn_direction` | string | Turn direction (L/R) |

### 1.3 Data Filtering Pipeline

```
┌─────────────────────────────────────────────────────────────────┐
│ consolidated_dataset.h5                                        │
│ └─ klein_run_table: 8,822 rows (runs, not tracks)             │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│ Unique (experiment_id, track) pairs: 424 tracks                │
│ Total reorientation events: 8,822                              │
│ Mean events/track: 20.8 | Median: 18.0                         │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼ Filter: duration ≥ 10 min
┌─────────────────────────────────────────────────────────────────┐
│ Tracks ≥ 10 min: 299 tracks                                    │
│ Mean events/track: 22.7                                        │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼ Filter: ≥ 10 reorientation events
┌─────────────────────────────────────────────────────────────────┐
│ Final analyzed set: 260 tracks                                 │
│ Mean events/track: 25.2 | Median: 22                           │
└─────────────────────────────────────────────────────────────────┘
```

---

## 2. Model Specification

### 2.1 Gamma-Difference Kernel (6 Parameters)

$$K_{on}(t) = A \cdot \Gamma(t; \alpha_1, \beta_1) - B \cdot \Gamma(t; \alpha_2, \beta_2)$$

| Parameter | Symbol | Description | Bounds |
|-----------|--------|-------------|--------|
| Excitatory amplitude | A | Peak excitation | [0.1, 5.0] |
| Excitatory shape | α₁ | Gamma shape | [1.5, 5.0] |
| Excitatory scale | β₁ | Gamma scale | [0.05, 0.5] |
| Suppressive amplitude | B | Peak suppression | [0.1, 20.0] |
| Suppressive shape | α₂ | Gamma shape | [2.0, 8.0] |
| Suppressive scale | β₂ | Gamma scale | [0.3, 2.0] |

**Derived parameters**:
- τ₁ = α₁ × β₁ (excitatory time constant)
- τ₂ = α₂ × β₂ (suppressive time constant)

### 2.2 Data-to-Parameter Ratio

| Metric | Value |
|--------|-------|
| Parameters per track | 6 |
| Median events per track | 18 |
| **Data:parameter ratio** | **3:1** |
| Recommended ratio | ≥ 10:1 |
| Events needed for reliability | ~100 |

---

## 3. Analysis Pipelines

### 3.1 Pipeline Inventory

| Script | Purpose | Input | Output |
|--------|---------|-------|--------|
| `01_quick_test_fewer_clusters.py` | Test clustering stability on simulated data | Simulated tracks (N=300) | `quick_test_fewer_clusters.csv` |
| `02_empirical_10min_hypothesis.py` | Fit kernels to empirical tracks ≥10 min | Klein run table | `empirical_10min_kernel_fits_v2.csv` |
| `03_cluster_validation.py` | Permutation test, gap statistic, reproducibility | Kernel fits | `validation/` |
| `04_cluster_characterization.py` | ANOVA, LDA, centroid analysis | Kernel fits + clusters | `characterization/` |
| `05_external_validation.py` | Association with experiment, condition | Kernel fits + metadata | `external_validation/` |
| `06_improved_simulation.py` | Round-trip validation | Cluster centroids | `improved_simulation/` |
| `07_deep_eda.py` | PSTH vs kernel clustering comparison | Event data | `deep_eda/` |
| `08_fno_phenotyping.py` | Fourier Neural Operator kernel learning | Synthetic + empirical | `fno_phenotyping/` |
| `09_hierarchical_bayesian.py` | Hierarchical Bayesian kernel estimation | Event data | `hierarchical_bayesian/` |
| `10_outlier_characterization.py` | Characterize candidate fast responders | Bayesian posteriors | `outlier_analysis/` |
| `generate_core_figures.py` | Generate manuscript figures | All results | `figures/` |

### 3.2 Execution Order

```
01 → 02 → 03 → 04 → 05 → 06 → 07 → 08 → 09 → 10 → figures
```

---

## 4. Output Files

### 4.1 Primary Results

| File | Rows | Fields | Description |
|------|------|--------|-------------|
| `empirical_10min_kernel_fits_v2.csv` | 260 | A, α₁, β₁, B, α₂, β₂, τ₁, τ₂, n_events, converged, experiment_id, track_id, duration | Individual kernel fits |
| `empirical_10min_clustering_results_v2.csv` | 4 | k, silhouette, stability_mean, stability_std, n_tracks, cluster_sizes | Clustering metrics for k=2,3,4,5 |

### 4.2 Validation Results

| Directory | Key Files | Key Metrics |
|-----------|-----------|-------------|
| `validation/` | `cluster_validation_results.json` | Permutation p-value, Gap statistic, Train/test ARI |
| `characterization/` | `cluster_characterization.json` | ANOVA η², LDA accuracy, centroid coordinates |
| `external_validation/` | `external_validation_results.json` | χ² tests, Kruskal-Wallis H |
| `improved_simulation/` | `round_trip_validation.json` | ARI = 0.13, parameter correlations ≈ 0 |
| `deep_eda/` | `deep_eda_results.json` | PSTH dimensionality, PSTH-kernel ARI = 0.01 |
| `fno_phenotyping/` | `fno_results.json`, `fno_model.pt` | FNO validation r > 0.92, empirical clustering |
| `hierarchical_bayesian/` | `hierarchical_results.json`, `individual_posteriors.csv` | Population τ₁ = 0.63s, outlier rate = 8.6% |

---

## 5. Figures

### 5.1 Main Manuscript Figures

| Figure | File | Panels | Manuscript Section |
|--------|------|--------|-------------------|
| Fig 1 | `fig1_clustering_null.pdf` | A: PCA density, B: t-SNE density, C: Silhouette scores, D: ARI validation | §3.1 Clustering Results |
| Fig 2 | `fig2_data_sparsity.pdf` | A: Events histogram, B: MLE τ₁ histogram, C: Data:parameter schematic | §3.2 Data Sparsity |
| Fig 3 | `fig3_hierarchical_shrinkage.pdf` | A: Caterpillar plot, B: MLE vs Bayes scatter, C: Population K(t) ribbon | §3.3 Hierarchical Model |
| Fig 4 | `fig4_fast_responders.pdf` | A: τ₁ violin plot, B: Example K(t) curves | §3.4 Candidate Outliers |

### 5.2 Figure-to-Claim Mapping

| Figure | Primary Claim | Evidence |
|--------|---------------|----------|
| Fig 1 | Clustering is an artifact | ARI ≈ 0.1, unimodal density |
| Fig 2 | Data are too sparse for individual estimation | 3:1 ratio, MLE over-dispersion |
| Fig 3 | 91% of larvae match population kernel | Posterior shrinkage |
| Fig 4 | 8.6% are candidate fast responders | Outlier CIs exclude population mean |

---

## 6. Manuscript Sections

### 6.1 Section-to-Data Mapping

| Section | Data Sources | Key Statistics |
|---------|--------------|----------------|
| §2.1 Data Quality | Klein run table | 424 tracks, 8,822 events |
| §2.2 Track Selection | Filtering pipeline | 260 tracks after filters |
| §2.3 Kernel Model | Model spec | 6 parameters, bounds |
| §3.1 Clustering | `clustering_results_v2.csv` | Silhouette = 0.57, Stability = 0.94 |
| §3.2 Validation | `validation/` | Permutation p < 0.01, Gap optimal k = 1 |
| §3.3 Round-trip | `improved_simulation/` | ARI = 0.13 |
| §3.4 Bayesian | `hierarchical_bayesian/` | τ₁ = 0.63s, outliers = 22/256 |
| §4 Discussion | All | Null result interpretation |

---

## 7. Software Environment

### 7.1 Python Dependencies

```
python>=3.10
numpy>=1.24
pandas>=2.0
scipy>=1.10
scikit-learn>=1.2
matplotlib>=3.7
h5py>=3.8
torch>=2.0 (FNO only)
numpyro>=0.12 (Bayesian only)
jax>=0.4 (Bayesian only)
plotly>=5.15 (interactive figures)
```

### 7.2 Key Algorithms

| Analysis | Library | Function/Class |
|----------|---------|----------------|
| Kernel fitting | scipy | `minimize` (L-BFGS-B) |
| Clustering | sklearn | `KMeans`, `silhouette_score` |
| Dimensionality | sklearn | `PCA`, `TSNE` |
| Permutation test | custom | Bootstrap ARI comparison |
| Gap statistic | custom | Per Tibshirani et al. 2001 |
| Hierarchical Bayes | numpyro | `NUTS`, `Predictive` |
| FNO | torch | Custom `FNO1d` module |

---

## 8. Reproducibility Checklist

- [ ] `consolidated_dataset.h5` available
- [ ] Python environment matches `requirements.txt`
- [ ] Run scripts 01-10 in order
- [ ] Run `generate_core_figures.py`
- [ ] Compile `main.tex` with pdflatex (2 passes)
- [ ] Verify figure checksums (optional)

### 8.1 Expected Runtime

| Script | Approx. Time |
|--------|--------------|
| 01-06 | < 5 min each |
| 07 (Deep EDA) | ~10 min |
| 08 (FNO) | ~30 min (GPU) / ~2 hr (CPU) |
| 09 (Bayesian) | ~20 min |
| Figures | ~5 min |

---

## 9. Version History

| Date | Version | Changes |
|------|---------|---------|
| 2025-12-17 | 1.0 | Initial manifest |
| 2025-12-17 | 1.1 | Corrected 4-param → 6-param |
| 2025-12-17 | 1.2 | Added Klein run table event counts |

---

## Appendix A: Data Field Definitions

### A.1 Events Table (`/events`)

| Field | Type | Description |
|-------|------|-------------|
| `experiment_id` | string | Experiment identifier |
| `track_id` | int64 | Track number |
| `time` | float64 | Time since experiment start (s) |
| `is_reorientation_start` | bool | Frame is start of reorientation |
| `led1Val` | float64 | LED1 intensity (PWM) |
| `led1Val_ton` | bool | LED1 just turned on |
| `led1Val_toff` | bool | LED1 just turned off |
| `speed` | float64 | Instantaneous speed (mm/s) |
| `curvature` | float64 | Path curvature |
| `heading` | float64 | Heading direction (rad) |

### A.2 Trajectories Table (`/trajectories`)

| Field | Type | Description |
|-------|------|-------------|
| `experiment_id` | string | Experiment identifier |
| `frame` | int64 | Video frame number |
| `time` | float64 | Time (s) |
| `x`, `y` | float64 | Centroid position (mm) |
| `head_x`, `head_y` | float64 | Head position (mm) |
| `tail_x`, `tail_y` | float64 | Tail position (mm) |
| `spine_x_0` ... `spine_x_10` | float64 | Spine points x |
| `spine_y_0` ... `spine_y_10` | float64 | Spine points y |
| `speed` | float64 | Speed (mm/s) |
| `heading` | float64 | Heading (rad) |
| `is_run` | bool | Currently in run state |
| `is_turn` | bool | Currently turning |
| `is_reorientation` | bool | In reorientation maneuver |
| `stimulus_on` | bool | LED stimulus active |

---

*This manifest is designed for inclusion as Supplementary Material or Appendix.*

