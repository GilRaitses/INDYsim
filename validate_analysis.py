#!/usr/bin/env python3
"""Validate the stimulus-locked analysis results."""

import json
from pathlib import Path
import pandas as pd

def main():
    experiment_id = "GMR61@GMR61_T_Re_Sq_0to100PWM_30#C_Bl_7PWM_202508221246"
    
    print("=" * 70)
    print("STIMULUS-LOCKED ANALYSIS VALIDATION")
    print("=" * 70)
    print()
    
    # Check progress file
    progress_file = Path(f"data/engineered/{experiment_id}_progress.json")
    if progress_file.exists():
        with open(progress_file) as f:
            progress = json.load(f)
        
        print("PROGRESS STATUS:")
        print(f"  Status: {progress.get('status', 'unknown')}")
        print(f"  Stage: {progress.get('stage', 'unknown')}")
        print(f"  Progress: {progress.get('progress_pct', 0)}%")
        print(f"  Tracks: {progress.get('current_track', 0)}/{progress.get('total_tracks', 0)}")
        print(f"  Elapsed: {progress.get('elapsed_time', 0):.1f}s")
        print(f"  Messages: {len(progress.get('messages', []))}")
        if progress.get('messages'):
            print("  Last 3 messages:")
            for msg in progress.get('messages', [])[-3:]:
                ts = msg.get('time', '')[-8:] if len(msg.get('time', '')) >= 8 else msg.get('time', '')
                print(f"    [{ts}] {msg.get('text', '')}")
        print()
    else:
        print("WARNING: Progress file not found")
        print()
    
    # Check output files
    print("OUTPUT FILES:")
    events_file = Path(f"data/engineered/{experiment_id}_events.csv")
    trajectories_file = Path(f"data/engineered/{experiment_id}_trajectories.csv")
    figure_file = Path(f"output/figures/eda/{experiment_id}_stimulus_locked_turn_rate.png")
    
    all_exist = True
    
    if events_file.exists():
        size_mb = events_file.stat().st_size / (1024 * 1024)
        print(f"  Events CSV: EXISTS ({size_mb:.1f} MB)")
        
        # Validate CSV structure
        try:
            df_sample = pd.read_csv(events_file, nrows=1000)
            print(f"    Rows (sample): {len(df_sample):,}")
            print(f"    Columns: {len(df_sample.columns)}")
            print(f"    Has 'is_turn': {'is_turn' in df_sample.columns}")
            print(f"    Has 'is_reorientation': {'is_reorientation' in df_sample.columns}")
            if 'is_turn' in df_sample.columns:
                print(f"    Turns (sample): {df_sample['is_turn'].sum():,}")
        except Exception as e:
            print(f"    ERROR reading CSV: {e}")
            all_exist = False
    else:
        print(f"  Events CSV: NOT FOUND")
        all_exist = False
    
    if trajectories_file.exists():
        size_mb = trajectories_file.stat().st_size / (1024 * 1024)
        print(f"  Trajectories CSV: EXISTS ({size_mb:.1f} MB)")
    else:
        print(f"  Trajectories CSV: NOT FOUND")
        all_exist = False
    
    if figure_file.exists():
        size_kb = figure_file.stat().st_size / 1024
        print(f"  Figure PNG: EXISTS ({size_kb:.1f} KB)")
    else:
        print(f"  Figure PNG: NOT FOUND")
        all_exist = False
    
    print()
    
    # Final validation
    print("VALIDATION RESULT:")
    print("=" * 70)
    if progress_file.exists() and progress.get('status') == 'complete' and all_exist:
        print("SUCCESS: Analysis completed successfully!")
        print(f"  All output files exist")
        print(f"  Progress: {progress.get('progress_pct')}%")
        if 'output_files' in progress:
            print(f"  Output files registered: {len(progress['output_files'])}")
    elif all_exist:
        print("PARTIAL: Output files exist but analysis may still be running")
        print(f"  Status: {progress.get('status', 'unknown')}")
    else:
        print("INCOMPLETE: Some output files are missing")
    print("=" * 70)

if __name__ == '__main__':
    main()

