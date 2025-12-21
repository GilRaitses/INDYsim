# Cross-Validation Pipeline

## Overview

Comprehensive cross-validation scripts to assess statistical rigor of phenotyping analysis.

## Scripts

### 1. `cv_kernel_fits.py`
**Purpose:** Validate kernel fitting

**Tests:**
- Leave-one-track-out cross-validation
- Bootstrap confidence intervals
- Parameter recovery (if ground truth available)

**Usage:**
```bash
python3 cv_kernel_fits.py \
    --data-dir data/simulated_phenotyping \
    --kernel-fits results/phenotyping_analysis_v2/track_kernel_fits.csv \
    --output-dir results/phenotyping_analysis_v2/cv \
    --n-bootstrap 100
```

**Outputs:**
- `cv_kernel_fits.csv` - LOOCV results
- `bootstrap_confidence_intervals.csv` - Bootstrap CIs
- `ground_truth_validation.csv` - Parameter recovery (if ground truth provided)

### 2. `cv_clustering.py`
**Purpose:** Validate clustering stability

**Tests:**
- Bootstrap stability (agreement matrix)
- Seed sensitivity (ARI across seeds)
- Per-cluster silhouette scores

**Usage:**
```bash
python3 cv_clustering.py \
    --features results/phenotyping_analysis_v2/phenotype_features.csv \
    --output-dir results/phenotyping_analysis_v2/cv \
    --n-clusters 5 \
    --n-bootstrap 100 \
    --n-seeds 20
```

**Outputs:**
- `cluster_stability.csv` - Bootstrap agreement metrics
- `seed_sensitivity.csv` - ARI across seeds
- `per_cluster_silhouette.csv` - Silhouette per cluster

### 3. `run_cv_pipeline.command`
**Purpose:** Run entire CV pipeline (double-clickable)

**Usage:** Double-click the file or:
```bash
bash run_cv_pipeline.command
```

## Expected Results

### Kernel Fitting CV
- **LOOCV:** Should show consistent fits across tracks
- **Bootstrap CIs:** 95% CI width indicates uncertainty
- **Ground truth:** Correlation > 0.7 indicates good recovery

### Clustering CV
- **Stability:** Mean agreement > 0.7 indicates stable clusters
- **Seed sensitivity:** Mean ARI > 0.8 indicates robust to seed
- **Silhouette:** Per-cluster scores > 0.2 indicate good separation

## Interpretation

### Good Results
- ✅ Bootstrap agreement > 0.7
- ✅ Seed ARI > 0.8
- ✅ Per-cluster silhouette > 0.2
- ✅ Parameter recovery r > 0.7

### Concerning Results
- ⚠️ Bootstrap agreement < 0.5 (clusters unstable)
- ⚠️ Seed ARI < 0.6 (sensitive to initialization)
- ⚠️ Per-cluster silhouette < 0.1 (poor separation)
- ⚠️ Parameter recovery r < 0.5 (poor fitting)

## Next Steps After CV

1. **If results are good:** Proceed to real data application
2. **If results are concerning:** 
   - Investigate unstable clusters
   - Consider different clustering methods
   - Check feature scaling/normalization
   - Increase bootstrap samples

---

**Ready to run!** Double-click `run_cv_pipeline.command` to start validation.

