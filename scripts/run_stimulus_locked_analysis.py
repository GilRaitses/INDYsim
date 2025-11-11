#!/usr/bin/env python3
"""
Run stimulus-locked turn rate analysis on an H5 file.
"""

import sys
from pathlib import Path
import subprocess

def main():
    h5_file = Path(r"D:\INDYsim\data\GMR61@GMR61_T_Re_Sq_0to100PWM_30#C_Bl_7PWM_202508221246.h5")
    
    if not h5_file.exists():
        print(f"ERROR: H5 file not found: {h5_file}")
        sys.exit(1)
    
    # Extract experiment ID from filename
    experiment_id = h5_file.stem.replace(' ', '_')
    output_dir = Path("data/engineered")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Processing H5 file: {h5_file.name}")
    print(f"Experiment ID: {experiment_id}")
    
    # Step 1: Generate events CSV from H5 file
    print("\n=== Step 1: Generating events CSV ===")
    import sys
    sys.path.insert(0, str(Path(__file__).parent / "scripts"))
    
    from engineer_dataset_from_h5 import process_h5_file
    process_h5_file(h5_file, output_dir, experiment_id)
    
    # Step 2: Run stimulus-locked turn rate analysis
    print("\n=== Step 2: Running stimulus-locked turn rate analysis ===")
    events_file = output_dir / f"{experiment_id}_events.csv"
    trajectories_file = output_dir / f"{experiment_id}_trajectories.csv"
    
    if not events_file.exists():
        print(f"ERROR: Events file not found: {events_file}")
        print("Please run engineer_dataset_from_h5.py first")
        sys.exit(1)
    
    # Create output directory for figures
    figures_dir = Path("output/figures/eda")
    figures_dir.mkdir(parents=True, exist_ok=True)
    output_path = figures_dir / f"{experiment_id}_stimulus_locked_turn_rate.png"
    
    from create_eda_figures import create_stimulus_locked_turn_rate_analysis
    create_stimulus_locked_turn_rate_analysis(
        str(trajectories_file),
        str(events_file),
        str(h5_file),
        output_path
    )
    
    print(f"\nAnalysis complete!")
    print(f"  Events CSV: {events_file}")
    print(f"  Output figure: {output_path}")

if __name__ == '__main__':
    main()

