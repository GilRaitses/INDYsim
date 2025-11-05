# INDYsim

Behavioral simulation term project for ECS630. The repository bundles the proposal, presentation, configuration, and scripts for modeling Drosophila larval behavior with stimulus-locked event-hazard methods. The current execution plan follows a 12-day window (November 6–20, 2025).

## Repository Structure

```
INDYsim/
├── README.md
├── TermProject_Proposal.qmd        # Main proposal (Quarto → PDF)
├── TermProject_Report.qmd          # Report scaffold
├── Simulation_Presentation.qmd     # Reveal.js presentation deck
├── scripts/                        # Analysis and simulation scripts
│   ├── fit_hazard_model.py         # GLM hazard model fitting
│   ├── simulate_trajectories.py    # Trajectory simulation engine
│   ├── run_doe.py                  # DOE execution
│   └── export_arena_format.py      # Arena-style CSV export
├── config/
│   ├── doe_table.csv               # 3 × 5 × 3 factorial design (45 conditions)
│   └── model_config.json           # Model hyperparameters, CI targets, paths
├── data/                           # Links or copies of mechanosensation data
├── docs/                           # Supporting documentation (Markdown)
├── output/
│   ├── fitted_models/              # Saved model objects
│   ├── simulation_results/         # DOE simulation outputs
│   └── arena_csvs/                 # Arena-format summary CSVs
└── styles.css                      # Shared presentation/report styles
```

## Quick Start

1. **Review Proposal** – open `TermProject_Proposal.qmd` in RStudio/Quarto.
2. **Install Dependencies** (if needed)
   ```bash
   pip install -r requirements.txt
   ```
3. **Run Example Analysis**
   ```bash
   python scripts/fit_hazard_model.py --help
   ```
4. **Render Proposal or Presentation**
   ```bash
   quarto render TermProject_Proposal.qmd
   quarto render Simulation_Presentation.qmd
   ```

## Key Components

### Proposal & Report
- `TermProject_Proposal.qmd` renders to PDF with the full narrative.
- `TermProject_Report.qmd` tracks results and appendices.

### Design of Experiments
- `config/doe_table.csv` encodes a full-factorial design (Intensity × Pulse Duration × Inter-Pulse Interval = 3 × 5 × 3 = 45 conditions).
- Each condition runs 30 replications (1,350 simulations total).

### Model Implementation
- Event-hazard GLMs with raised-cosine temporal kernels.
- Events: turn starts, stop starts, reversal starts.
- Features: stimulus history, speed, orientation, wall distance, interaction terms.

### Output Format
- Arena-style CSVs: `AcrossReplicationsSummary.csv`, `ContinuousTimeStatsByRep.csv`, `DiscreteTimeStatsByRep.csv`.
- Compatible with the ECS630 lab analysis workflow.

## Data Requirements

Primary data: H5 files in `/Users/gilraitses/mechanosensation/h5tests/` (for example `GMR61_202509051201_tier1.h5`). Backup CSVs live in `output/spatial_analysis/`. Experiment metadata is embedded in the H5 files. Data paths and confidence-interval targets are configured in `config/model_config.json`. See `DATA_SOURCES.md` for an inventory of available datasets.

## Current Timeline (12 Days)

- **Week 1: November 6 – 13** – Data preparation, feature engineering, model development, and hazard model fitting with validation (KS tests, PSTHs).
- **Week 2: November 13 – 20** – Simulation engine completion, DOE execution across 45 conditions, results analysis, presentation/report updates, and final Quarto renders.

## Contact

Questions or updates: Gil Raitses (`gilraitses@gmail.com`).
