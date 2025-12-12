#!/bin/bash
# Run GLMM fitting in external terminal
# This script must be run outside of Cursor IDE

cd /Users/gilraitses/INDYsim
source .venv-glmm/bin/activate

echo "Python version: $(python --version)"
echo "Starting GLMM fitting..."
echo "This may take 10-30 minutes depending on hardware."
echo ""

python scripts/fit_glmm.py

echo ""
echo "Done. Results saved to data/model/glmm_results.json"
