#!/bin/bash
# Run Posterior Predictive Checks
# Estimated runtime: ~20 minutes

cd /Users/gilraitses/INDYsim_project
source venv/bin/activate

echo "=========================================="
echo "  POSTERIOR PREDICTIVE CHECKS"
echo "=========================================="
echo ""
echo "Goal: Validate that model generates"
echo "event patterns consistent with data"
echo ""
echo "Starting in 3 seconds..."
sleep 3

python3 scripts/2025-12-17/phenotyping_experiments/12_posterior_predictive.py

if [ $? -eq 0 ]; then
    echo ""
    echo "=========================================="
    echo "  ✓ PPC COMPLETE"
    echo "=========================================="
    echo "Results saved to: results/posterior_predictive/"
else
    echo ""
    echo "=========================================="
    echo "  ⚠ PPC FAILED (exit code: $?)"
    echo "=========================================="
fi

echo ""
read -p "Press any key to continue..."

