# Phenotyping Follow-up: Analysis Code

## Pipeline Overview

This directory contains all analysis scripts for the phenotyping follow-up study. Scripts are numbered sequentially to show the analytical progression.

## Script Organization

### Phase 1: Initial Phenotyping (01-10)
Exploratory analysis and initial phenotype detection.

| Script | Purpose |
|--------|---------|
| `01_quick_test_fewer_clusters.py` | Initial test with reduced cluster count |
| `02_empirical_10min_hypothesis.py` | Core hypothesis testing on empirical data |
| `03_cluster_validation.py` | Statistical validation of detected clusters |
| `04_cluster_characterization.py` | Characterize phenotype properties |
| `05_external_validation.py` | Cross-validation against external measures |
| `06_improved_simulation.py` | Round-trip simulation validation |
| `07_deep_eda.py` | Deep exploratory data analysis |
| `08_fno_phenotyping.py` | Fourier Neural Operator approach |
| `09_hierarchical_bayesian.py` | Hierarchical Bayesian modeling (NumPyro) |
| `10_outlier_characterization.py` | Characterize the 8.6% genuine outliers |

### Phase 2: Statistical Enhancements (11-14)
Power analysis and model validation methods.

| Script | Purpose |
|--------|---------|
| `11_power_analysis.py` | How many events needed for 80% power? |
| `12_posterior_predictive.py` | Posterior predictive checks |
| `13_model_comparison.py` | 4-param vs 2-param model (BIC/AIC) |
| `14_loeo_validation.py` | Leave-one-experiment-out CV |

### Phase 3: Identifiability Analysis (15-16)
Critical discovery: structural identifiability failure and design optimization.

| Script | Purpose |
|--------|---------|
| `15_composite_phenotype_validation.py` | Validate Precision/Burstiness as alternative phenotypes |
| `16_design_kernel_sweep.py` | Systematic sweep of designs × kernel regimes |

**Note:** The identifiability analysis also produced intermediate results (on_off_ratio, first_event_latency, hazard_deviation, etc.) that are documented in the manuscript. These analyses were consolidated into scripts 15-16.

## Utility Scripts

### Cross-Validation
- `cv_kernel_fits.py` - Leave-one-track-out CV for kernel fitting
- `cv_clustering.py` - Bootstrap stability for clustering

### Data Generation
- `generate_simulated_tracks_for_phenotyping.py` - Generate synthetic tracks
- `generate_extended_phenotyping_tracks.py` - Extended dataset generation

### Figure Generation
- `generate_validation_figures.py` - Main manuscript figures (Fig 1-4)
- `generate_enhancement_figures.py` - Statistical enhancement figures (Fig 5-8)
- `generate_identifiability_figure.py` - Identifiability problem (Fig 2 v3)
- `generate_summary_figure.py` - Design comparison summary
- `generate_stimulation_schematic.py` - Stimulation protocol schematic
- `generate_stimulation_protocol_figure.py` - Full protocol visualization

### Main Pipeline
- `phenotyping_analysis_pipeline.py` - End-to-end pipeline (runs all analyses)

## Key Findings

1. **Phenotypes are artifacts** - Initial 4-cluster solution fails round-trip validation (ARI = 0.13)
2. **Data sparsity** - 25 events with 6 parameters is fundamentally underdetermined
3. **Hierarchical homogeneity** - 91% of larvae are consistent with population mean
4. **Design matters** - Burst stimulation provides 10× higher Fisher Information
5. **Composite phenotypes work** - Precision is recoverable; Burstiness requires higher baseline

## Reproducibility

All scripts can be run independently. Expected runtime:
- Scripts 01-10: ~5-15 min each (CPU)
- Scripts 11-14: ~10-30 min each (CPU)
- Scripts 15-16: ~5-20 min (GPU recommended, CPU works but slower)

## Dependencies

```
numpy, scipy, pandas, matplotlib, seaborn
scikit-learn, statsmodels
numpyro, jax (for hierarchical Bayesian and GPU acceleration)
```

See `requirements.txt` in root directory.





