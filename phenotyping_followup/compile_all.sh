#!/bin/bash
# Compile main and supplement to both PDF and HTML

set -e  # Exit on error

cd "$(dirname "$0")"

echo "========================================"
echo "Compiling Phenotyping Follow-up Manuscript"
echo "========================================"
echo ""

# Compile main.tex to PDF
echo "1. Compiling main.tex to PDF..."
pdflatex -interaction=nonstopmode main.tex > /dev/null 2>&1
pdflatex -interaction=nonstopmode main.tex > /dev/null 2>&1

if [ -f "main.pdf" ]; then
    echo "   ✓ main.pdf created"
else
    echo "   ✗ FAILED: Check main.log"
    exit 1
fi

# Compile main.tex to HTML
echo "2. Compiling main.tex to HTML..."
pandoc main.tex -o main.html --standalone --mathjax --css=https://cdn.jsdelivr.net/npm/katex@0.16.9/dist/katex.min.css 2>/dev/null || {
    echo "   Warning: pandoc HTML conversion may have issues, but continuing..."
}

if [ -f "main.html" ]; then
    echo "   ✓ main.html created"
else
    echo "   ✗ FAILED: HTML not created"
fi

# Compile supplement.tex to PDF
echo "3. Compiling supplement.tex to PDF..."
pdflatex -interaction=nonstopmode supplement.tex > /dev/null 2>&1
pdflatex -interaction=nonstopmode supplement.tex > /dev/null 2>&1

if [ -f "supplement.pdf" ]; then
    echo "   ✓ supplement.pdf created"
else
    echo "   ✗ FAILED: Check supplement.log"
    exit 1
fi

# Compile supplement.tex to HTML
echo "4. Compiling supplement.tex to HTML..."
pandoc supplement.tex -o supplement.html --standalone --mathjax --css=https://cdn.jsdelivr.net/npm/katex@0.16.9/dist/katex.min.css 2>/dev/null || {
    echo "   Warning: pandoc HTML conversion may have issues, but continuing..."
}

if [ -f "supplement.html" ]; then
    echo "   ✓ supplement.html created"
else
    echo "   ✗ FAILED: HTML not created"
fi

echo ""
echo "========================================"
echo "✓ Compilation Complete"
echo "========================================"
echo ""
echo "Generated files:"
ls -lh main.pdf main.html supplement.pdf supplement.html 2>/dev/null | awk '{print "  " $9 " (" $5 ")"}'

