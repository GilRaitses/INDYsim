#!/bin/bash
# Double-clickable script to generate extended phenotyping dataset

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
echo "Generate Extended Phenotyping Dataset"
echo "=========================================="
echo ""
echo "This will generate 1000 complete tracks:"
echo "  - 250 tracks per condition"
echo "  - 4 conditions"
echo "  - 20-minute duration each"
echo ""
echo "Estimated time: ~8-10 minutes"
echo ""
echo "Starting generation..."
echo ""

python3 "$SCRIPT_DIR/generate_extended_phenotyping_tracks.py" \
    --n-tracks-per-condition 250 \
    --output-dir data/simulated_phenotyping_extended

EXIT_CODE=$?

echo ""
echo "=========================================="
if [ $EXIT_CODE -eq 0 ]; then
    echo "Generation Complete!"
else
    echo "Generation Failed (exit code: $EXIT_CODE)"
fi
echo "=========================================="
echo ""
echo "Results saved to: $INDYSIM_DIR/data/simulated_phenotyping_extended/"
echo ""
read -n 1

