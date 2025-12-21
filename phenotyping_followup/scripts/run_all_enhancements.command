#!/bin/bash
# Run All Statistical Enhancement Pipelines
# Estimated total runtime: ~4 hours

cd /Users/gilraitses/INDYsim_project
source venv/bin/activate

echo "=========================================="
echo "  STATISTICAL RIGOR ENHANCEMENT SUITE"
echo "=========================================="
echo ""
echo "This will run all 4 enhancement pipelines:"
echo "  1. Power Analysis (~30 min)"
echo "  2. Posterior Predictive Checks (~20 min)"
echo "  3. Model Comparison (~40 min)"
echo "  4. Leave-One-Experiment-Out CV (~2 hr)"
echo ""
echo "Total estimated time: ~4 hours"
echo ""
echo "Starting in 5 seconds..."
sleep 5

START_TIME=$(date +%s)

echo ""
echo "=========================================="
echo "  [1/4] POWER ANALYSIS"
echo "=========================================="
python3 scripts/2025-12-17/phenotyping_experiments/11_power_analysis.py
POWER_STATUS=$?

echo ""
echo "=========================================="
echo "  [2/4] POSTERIOR PREDICTIVE CHECKS"
echo "=========================================="
python3 scripts/2025-12-17/phenotyping_experiments/12_posterior_predictive.py
PPC_STATUS=$?

echo ""
echo "=========================================="
echo "  [3/4] MODEL COMPARISON"
echo "=========================================="
python3 scripts/2025-12-17/phenotyping_experiments/13_model_comparison.py
MODEL_STATUS=$?

echo ""
echo "=========================================="
echo "  [4/4] LEAVE-ONE-EXPERIMENT-OUT CV"
echo "=========================================="
python3 scripts/2025-12-17/phenotyping_experiments/14_loeo_validation.py
LOEO_STATUS=$?

END_TIME=$(date +%s)
ELAPSED=$((END_TIME - START_TIME))
ELAPSED_MIN=$((ELAPSED / 60))

echo ""
echo "=========================================="
echo "  ENHANCEMENT SUITE COMPLETE"
echo "=========================================="
echo ""
echo "Status:"
echo "  Power Analysis:    $([ $POWER_STATUS -eq 0 ] && echo '✓ PASS' || echo '✗ FAIL')"
echo "  PPC:               $([ $PPC_STATUS -eq 0 ] && echo '✓ PASS' || echo '✗ FAIL')"
echo "  Model Comparison:  $([ $MODEL_STATUS -eq 0 ] && echo '✓ PASS' || echo '✗ FAIL')"
echo "  LOEO-CV:           $([ $LOEO_STATUS -eq 0 ] && echo '✓ PASS' || echo '✗ FAIL')"
echo ""
echo "Total runtime: ${ELAPSED_MIN} minutes"
echo ""
echo "Results saved to:"
echo "  - results/power_analysis/"
echo "  - results/posterior_predictive/"
echo "  - results/model_comparison/"
echo "  - results/loeo_validation/"
echo ""
echo "Next step: Run generate_enhancement_figures.py"
echo ""
read -p "Press any key to continue..."

