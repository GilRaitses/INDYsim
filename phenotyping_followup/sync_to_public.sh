#!/bin/bash
# Copy phenotyping_followup to public repository and push

set -e

PROJECT_DIR="/Users/gilraitses/INDYsim_project/phenotyping_followup"
PUBLIC_DIR="/Users/gilraitses/InDySim/phenotyping_followup"

echo "========================================"
echo "Syncing to Public Repository"
echo "========================================"
echo ""

# Create destination directory if it doesn't exist
mkdir -p "$PUBLIC_DIR"

# Copy files, excluding build artifacts and cache
echo "Copying files to public repository..."
rsync -av \
    --exclude='__pycache__' \
    --exclude='*.pyc' \
    --exclude='*.log' \
    --exclude='*.aux' \
    --exclude='*.out' \
    --exclude='*.fdb_latexmk' \
    --exclude='*.fls' \
    --exclude='*.synctex.gz' \
    --exclude='.DS_Store' \
    --exclude='*.swp' \
    --exclude='*.swo' \
    "$PROJECT_DIR/" "$PUBLIC_DIR/"

echo ""
echo "✓ Files copied to: $PUBLIC_DIR"
echo ""

# Check git status in public repo
cd "$PUBLIC_DIR"
echo "Git status in public repository:"
git status --short | head -20

echo ""
echo "Ready to commit and push."
echo "Run the following commands:"
echo "  cd $PUBLIC_DIR"
echo "  git add ."
echo "  git commit -m 'Update phenotyping manuscript with corrected simulation parameters'"
echo "  git push"

