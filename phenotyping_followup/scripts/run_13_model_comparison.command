#!/bin/bash
# Run Model Comparison (2-param vs 6-param)
# Estimated runtime: ~40 minutes

cd /Users/gilraitses/INDYsim_project
source venv/bin/activate

echo "=========================================="
echo "  MODEL COMPARISON: 2-PARAM vs 6-PARAM"
echo "=========================================="
echo ""
echo "Goal: Determine if 6-param model is"
echo "justified or over-parameterized"
echo ""
echo "Starting in 3 seconds..."
sleep 3

python3 scripts/2025-12-17/phenotyping_experiments/13_model_comparison.py

if [ $? -eq 0 ]; then
    echo ""
    echo "=========================================="
    echo "  ✓ MODEL COMPARISON COMPLETE"
    echo "=========================================="
    echo "Results saved to: results/model_comparison/"
else
    echo ""
    echo "=========================================="
    echo "  ⚠ MODEL COMPARISON FAILED (exit code: $?)"
    echo "=========================================="
fi

echo ""
read -p "Press any key to continue..."

