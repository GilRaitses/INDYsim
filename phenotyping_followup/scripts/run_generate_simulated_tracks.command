#!/bin/bash
# Double-clickable script to generate simulated tracks for phenotyping
# Generated tracks will be saved to: data/simulated_phenotyping/

# Get the directory where this script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$( cd "$SCRIPT_DIR/../../.." && pwd )"

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

# Run the script
echo "=========================================="
echo "Generating Simulated Tracks for Phenotyping"
echo "=========================================="
echo ""
echo "This will generate 300 complete tracks (75 per condition)"
echo "Output directory: data/simulated_phenotyping/"
echo ""
echo "Starting generation..."
echo ""

python3 "$SCRIPT_DIR/../code/generate_simulated_tracks_for_phenotyping.py" \
    --n-tracks 300 \
    --per-condition \
    --output-dir data/simulated_phenotyping

echo ""
echo "=========================================="
echo "Generation Complete!"
echo "=========================================="
echo ""
echo "Files saved to: $INDYSIM_DIR/data/simulated_phenotyping/"
echo ""
echo "Press any key to close this window..."
read -n 1

