# Scripts Queue for src Integration

This folder contains scripts that need to be adapted and integrated into the `src/` directory structure.

## Purpose

These scripts are currently functional but need to be:
1. Refactored to follow the `src/` module structure
2. Updated with proper imports and dependencies
3. Organized into appropriate subdirectories (e.g., `@analysis/`, `@simulation/`, `@visualization/`)
4. Documented with proper docstrings and usage examples

## Scripts by Category

### Core Data Processing
- `engineer_dataset_from_h5.py` - **CRITICAL** - Extracts trajectory and stimulus data from H5 files
- `prepare_simulation_dataset.py` - Prepares datasets for simulation
- `detect_events.py` - Detects behavioral events in trajectories

### Analysis Scripts
- `analyze_doe_results.py` - Analyzes DOE simulation results
- `analyze_h5_stimulus.py` - Analyzes H5 stimulus data
- `analyze_main_effects.py` - Main effects analysis (ANOVA)
- `analyze_pause_patterns.py` - Pause pattern analysis
- `analyze_turn_durations.py` - Turn duration analysis

### Model Fitting
- `fit_hazard_model.py` - Fits event-hazard GLM models
- `fit_all_event_models.py` - Fits multiple event models
- `learn_event_parameters.py` - Learns event parameters
- `learn_magat_parameters.py` - Learns MAGAT parameters

### Simulation
- `simulate_trajectories.py` - Simulates larval trajectories using fitted models
- `run_doe.py` - Executes full factorial DOE
- `run_stimulus_locked_analysis.py` - Stimulus-locked analysis
- `run_stimulus_locked_analysis_production.py` - Production version

### Visualization
- `visualize_behavioral_events.py` - Visualizes behavioral events
- `visualize_behavioral_events_stepwise.py` - Stepwise visualization
- `visualize_kernel.py` - Kernel visualization
- `visualize_stimulus_cycles.py` - Stimulus cycle visualization
- `generate_experiment_visualizations.py` - Experiment visualizations
- `generate_report_tables.py` - Report table generation
- `create_eda_figures.py` - EDA figure creation

### Utilities
- `inspect_h5_files.py` - Inspects H5 file structure
- `inspect_h5_stimulus.py` - Inspects H5 stimulus data
- `inspect_h5_structure.py` - Inspects H5 structure
- `validate_simulation.py` - Validates simulation results
- `export_arena_format.py` - Exports to Arena CSV format
- `magat_segmentation.py` - MAGAT segmentation
- `magat_speed_analysis.py` - MAGAT speed analysis
- `magat_spine_analysis.py` - MAGAT spine analysis
- `klein_run_table.py` - Klein run table generation
- `find_ton_toff_fields.py` - Finds ton/toff fields
- `cinnamoroll_palette.py` - Color palette definitions

## Integration Priority

### P0 (Critical - Needed Immediately)
1. `engineer_dataset_from_h5.py` - Core data extraction pipeline

### P1 (High Priority - Needed Soon)
2. `run_stimulus_locked_analysis_production.py` - Production analysis
3. `simulate_trajectories.py` - Simulation engine
4. `fit_hazard_model.py` - Model fitting

### P2 (Medium Priority)
5. Analysis scripts (`analyze_*.py`)
6. Visualization scripts (`visualize_*.py`, `generate_*.py`)
7. Utility scripts (`inspect_*.py`, `validate_*.py`)

## Next Steps

1. Review each script's dependencies
2. Create appropriate `src/@module/` structure
3. Refactor scripts to use proper imports
4. Update documentation
5. Create unit tests where appropriate
6. Update references in other scripts/docs

## Notes

- Scripts in this folder are still functional and can be run directly
- Before moving to `src/`, ensure all dependencies are documented
- Some scripts may need significant refactoring to fit the module structure
- Check `src/@matlab_conversion/` for examples of proper module structure









