#!/bin/bash
# Compile manuscript to PDF

cd "$(dirname "$0")"

echo "========================================"
echo "Compiling Phenotyping Follow-up Manuscript"
echo "========================================"

# Run pdflatex twice for references
echo ""
echo "First pass..."
pdflatex -interaction=nonstopmode main.tex > /dev/null 2>&1

echo "Second pass (for references)..."
pdflatex -interaction=nonstopmode main.tex > /dev/null 2>&1

# Check if PDF was created
if [ -f "main.pdf" ]; then
    echo ""
    echo "========================================"
    echo "✓ SUCCESS: main.pdf created"
    echo "========================================"
    echo ""
    echo "Opening PDF..."
    open main.pdf
else
    echo ""
    echo "========================================"
    echo "✗ FAILED: Check main.log for errors"
    echo "========================================"
    echo ""
    echo "Last 30 lines of log:"
    tail -30 main.log
fi

echo ""
echo "Press any key to close..."
read -n 1

