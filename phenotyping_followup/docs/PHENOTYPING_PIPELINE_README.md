# Phenotyping Analysis Pipeline

## Overview

End-to-end script for phenotyping analysis with **Bambi-style progress monitoring**. Performs complete analysis pipeline:

1. ✅ **Load simulated tracks** - Loads all 300 tracks from simulation
2. ✅ **Fit track-level kernels** - Fits gamma-difference kernel to each track
3. ✅ **Extract phenotype features** - Computes kernel parameters, event statistics, behavioral allocation
4. ✅ **Clustering analysis** - K-means and hierarchical clustering with silhouette scores
5. ✅ **Generate visualizations** - Parameter distributions, clustering plots, PCA visualization

## Features

### Bambi-Style Progress Monitoring
- **Real-time progress bars** with ETA estimates
- **Phase indicators** (LOAD, FIT, EXTRACT, CLUSTER, VIZ)
- **Status messages** with timestamps
- **Color-coded output** (INFO, SUCCESS, WARNING, ERROR)
- **Detailed timing** for each step

### Example Output

```
======================================================================
PHENOTYPING ANALYSIS PIPELINE
======================================================================

[22:45:12] [INFO] INIT: Data directory: /Users/gilraitses/InDySim/data/simulated_phenotyping
[22:45:12] [INFO] INIT: Output directory: /Users/gilraitses/InDySim/results/phenotyping_analysis
[22:45:12] [INFO] INIT: Start time: 2025-12-16 22:45:12

======================================================================
STEP 1: Fitting Track-Level Kernels
======================================================================

[22:45:12] [INFO] FIT: Fitting kernels to 300 tracks...
[FIT] ████████████████████████████████████████████████████ 150/300 (50.0%) ETA: 45s | 0-250_Constant/track_0075
✓ Fitting kernels Complete! (90.2s)

[22:46:42] [SUCCESS] FIT: Successfully fitted 287/300 tracks
```

## Quick Start

### Option 1: Double-Click (Easiest)

1. **Double-click:** `run_phenotyping_analysis.command`
2. Terminal opens and shows progress
3. Wait for completion
4. Results saved to: `/Users/gilraitses/InDySim/results/phenotyping_analysis/`

### Option 2: Terminal

```bash
cd /Users/gilraitses/InDySim
python3 scripts/2025-12-16/poster/phenotyping_analysis_pipeline.py \
    --data-dir data/simulated_phenotyping \
    --output-dir results/phenotyping_analysis
```

### Option 3: With Options

```bash
# Skip clustering (faster)
python3 phenotyping_analysis_pipeline.py --skip-clustering

# Skip visualizations (faster)
python3 phenotyping_analysis_pipeline.py --skip-viz

# Custom directories
python3 phenotyping_analysis_pipeline.py \
    --data-dir /path/to/tracks \
    --output-dir /path/to/output
```

## Output Files

```
results/phenotyping_analysis/
├── track_kernel_fits.csv          # Kernel parameters for each track
├── track_kernel_fits.parquet
├── phenotype_features.csv          # All extracted features
├── phenotype_features.parquet
├── clustering_results.csv          # Clustering assignments
├── clustering_results.parquet
├── cluster_summary.csv             # Mean features per cluster
└── visualizations/
    ├── parameter_distributions.png
    └── clustering_visualization.png
```

## Progress Monitoring Details

### Phase Indicators

- **[LOAD]** - Loading track files
- **[FIT]** - Fitting kernels to tracks
- **[EXTRACT]** - Extracting phenotype features
- **[CLUSTER]** - Performing clustering analysis
- **[VIZ]** - Generating visualizations

### Status Colors

- **INFO** (cyan) - Normal progress updates
- **SUCCESS** (green) - Completed steps
- **WARNING** (yellow) - Non-critical issues
- **ERROR** (red) - Fatal errors

### Progress Bar Format

```
[PHASE] ████████████████████████████████████████████████████ 150/300 (50.0%) ETA: 45s | message
```

- **Bar:** Visual progress indicator
- **Count:** Current/total items
- **Percentage:** Completion percentage
- **ETA:** Estimated time remaining
- **Message:** Current item being processed

## Requirements

### Python Packages

```bash
pip install numpy pandas scipy scikit-learn matplotlib seaborn tqdm
```

### Optional (for better progress bars)

```bash
pip install tqdm
```

If `tqdm` is not installed, the script will use a simplified progress bar.

## Troubleshooting

### "No track files found"
- Check that simulated tracks were generated successfully
- Verify path: `/Users/gilraitses/InDySim/data/simulated_phenotyping/`

### "Could not import AnalyticHazardModel"
- Make sure you're running from `/Users/gilraitses/InDySim` directory
- Check that `code/analytic_hazard.py` exists

### Progress bars not showing
- Install tqdm: `pip install tqdm`
- Or use simplified progress bars (works without tqdm)

### Clustering fails
- Need at least 10 tracks with valid kernel fits
- Try `--skip-clustering` to continue without clustering

## Performance

- **Loading tracks:** ~10-30 seconds (300 tracks)
- **Fitting kernels:** ~2-5 minutes (300 tracks, ~0.5s per track)
- **Extracting features:** ~30-60 seconds
- **Clustering:** ~10-30 seconds
- **Visualizations:** ~10-20 seconds

**Total:** ~5-10 minutes for complete analysis

## Next Steps

After running the pipeline:

1. **Review cluster assignments** in `clustering_results.csv`
2. **Check cluster summaries** in `cluster_summary.csv`
3. **Examine visualizations** in `visualizations/` directory
4. **Validate clustering** - Compare to known ground truth (simulated tracks)
5. **Apply to real data** - Use same pipeline on 79 real complete tracks

---

**Ready to run!** Just double-click `run_phenotyping_analysis.command` 🚀

