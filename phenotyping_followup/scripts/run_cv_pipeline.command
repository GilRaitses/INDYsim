#!/bin/bash
# Double-clickable script to run cross-validation pipeline

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
INDYSIM_DIR="/Users/gilraitses/InDySim"

cd "$INDYSIM_DIR" || {
    echo "Error: Could not change to $INDYSIM_DIR"
    exit 1
}

# Activate virtual environment if it exists
if [ -d "venv" ]; then
    source venv/bin/activate
elif [ -d ".venv" ]; then
    source .venv/bin/activate
fi

echo "=========================================="
echo "Cross-Validation Pipeline"
echo "=========================================="
echo ""
echo "This will:"
echo "  1. Validate kernel fits (LOOCV, bootstrap CIs)"
echo "  2. Test cluster stability (bootstrap, seed sensitivity)"
echo "  3. Analyze per-cluster silhouette scores"
echo ""
echo "Starting CV..."
echo ""

# Run kernel CV
python3 "$SCRIPT_DIR/cv_kernel_fits.py" \
    --data-dir data/simulated_phenotyping \
    --kernel-fits results/phenotyping_analysis_v2/track_kernel_fits.csv \
    --output-dir results/phenotyping_analysis_v2/cv

# Run clustering CV
python3 "$SCRIPT_DIR/cv_clustering.py" \
    --features results/phenotyping_analysis_v2/phenotype_features.csv \
    --output-dir results/phenotyping_analysis_v2/cv \
    --n-clusters 5

echo ""
echo "=========================================="
echo "CV Complete!"
echo "=========================================="
echo ""
echo "Results saved to: $INDYSIM_DIR/results/phenotyping_analysis_v2/cv/"
echo ""
read -n 1

