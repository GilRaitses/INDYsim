#!/bin/bash
# Run Power Analysis
# Estimated runtime: ~30 minutes

cd /Users/gilraitses/INDYsim_project
source venv/bin/activate

echo "=========================================="
echo "  POWER ANALYSIS"
echo "=========================================="
echo ""
echo "Goal: Determine events needed for 80% power"
echo "to detect τ₁ difference of 0.2s"
echo ""
echo "Starting in 3 seconds..."
sleep 3

python3 scripts/2025-12-17/phenotyping_experiments/11_power_analysis.py

if [ $? -eq 0 ]; then
    echo ""
    echo "=========================================="
    echo "  ✓ POWER ANALYSIS COMPLETE"
    echo "=========================================="
    echo "Results saved to: results/power_analysis/"
else
    echo ""
    echo "=========================================="
    echo "  ⚠ POWER ANALYSIS FAILED (exit code: $?)"
    echo "=========================================="
fi

echo ""
read -p "Press any key to continue..."

