#!/bin/bash
# Double-clickable script to run phenotyping analysis pipeline
# Features Bambi-style progress monitoring

# Get the directory where this script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
SCRIPT_PATH="$SCRIPT_DIR/phenotyping_analysis_pipeline.py"

# Change to InDySim directory (where the code is)
INDYSIM_DIR="/Users/gilraitses/InDySim"
cd "$INDYSIM_DIR" || {
    echo "Error: Could not change to $INDYSIM_DIR"
    echo "Please check that the InDySim directory exists"
    exit 1
}

# Activate virtual environment if it exists
if [ -d "venv" ]; then
    source venv/bin/activate
elif [ -d ".venv" ]; then
    source .venv/bin/activate
fi

# Check if tqdm is installed (for better progress bars)
python3 -c "import tqdm" 2>/dev/null || {
    echo "Installing tqdm for better progress bars..."
    pip install tqdm --quiet
}

# Run the analysis pipeline
echo "=========================================="
echo "Phenotyping Analysis Pipeline"
echo "=========================================="
echo ""
echo "This will:"
echo "  1. Load simulated tracks"
echo "  2. Fit track-level kernels"
echo "  3. Extract phenotype features"
echo "  4. Perform clustering analysis"
echo "  5. Generate visualizations"
echo ""
echo "Starting analysis..."
echo ""

python3 "$SCRIPT_PATH" \
    --data-dir data/simulated_phenotyping \
    --output-dir results/phenotyping_analysis_v2

EXIT_CODE=$?

echo ""
echo "=========================================="
if [ $EXIT_CODE -eq 0 ]; then
    echo "Analysis Complete!"
else
    echo "Analysis Failed (exit code: $EXIT_CODE)"
fi
echo "=========================================="
echo ""
echo "Results saved to: $INDYSIM_DIR/results/phenotyping_analysis/"
echo ""
echo "Press any key to close this window..."
read -n 1

