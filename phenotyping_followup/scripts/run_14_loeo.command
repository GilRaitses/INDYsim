#!/bin/bash
# Run Leave-One-Experiment-Out Cross-Validation
# Estimated runtime: ~2 hours

cd /Users/gilraitses/INDYsim_project
source venv/bin/activate

echo "=========================================="
echo "  LEAVE-ONE-EXPERIMENT-OUT CV"
echo "=========================================="
echo ""
echo "Goal: Test if phenotype structure"
echo "generalizes across experiments"
echo ""
echo "This will take approximately 2 hours"
echo "Starting in 3 seconds..."
sleep 3

python3 scripts/2025-12-17/phenotyping_experiments/14_loeo_validation.py

if [ $? -eq 0 ]; then
    echo ""
    echo "=========================================="
    echo "  ✓ LOEO-CV COMPLETE"
    echo "=========================================="
    echo "Results saved to: results/loeo_validation/"
else
    echo ""
    echo "=========================================="
    echo "  ⚠ LOEO-CV FAILED (exit code: $?)"
    echo "=========================================="
fi

echo ""
read -p "Press any key to continue..."

