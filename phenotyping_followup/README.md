# Phenotyping Follow-Up Study

## Overview

This folder contains all materials for the follow-up phenotyping study extending the Interface Dynamics Simulation Modeling Study to individual-level behavioral analysis.

## Structure

```
phenotyping_followup/
├── main.tex                 # Main LaTeX document
├── sections/                # LaTeX section files
│   ├── 01_introduction.tex
│   ├── 02_methods.tex
│   ├── 03_results.tex
│   ├── 04_discussion.tex
│   ├── 05_conclusions.tex
│   ├── 06_acknowledgments.tex
│   ├── 07_data_availability.tex
│   └── 08_references.tex
├── code/                    # Analysis scripts
│   ├── phenotyping_analysis_pipeline.py
│   ├── cv_kernel_fits.py
│   ├── cv_clustering.py
│   ├── generate_simulated_tracks_for_phenotyping.py
│   └── generate_extended_phenotyping_tracks.py
├── data/                    # Simulated track data
│   └── simulated_phenotyping/
├── results/                 # Analysis results
│   └── phenotyping_analysis_v2/
├── figures/                 # Generated figures
├── docs/                    # Documentation
└── scripts/                 # Executable command scripts
```

## Compiling the LaTeX Document

```bash
cd scripts/2025-12-16/phenotyping_followup
pdflatex main.tex
bibtex main
pdflatex main.tex
pdflatex main.tex
```

## Key Files

- **main.tex**: Main manuscript document
- **sections/**: Individual section files (modular structure)
- **code/**: All analysis scripts
- **results/**: Analysis outputs (CSV files, visualizations)
- **figures/**: Generated figures for manuscript

## Status

- [x] LaTeX document structure created
- [ ] Results sections populated
- [ ] Figures integrated
- [ ] References added
- [ ] Cross-validation results incorporated


