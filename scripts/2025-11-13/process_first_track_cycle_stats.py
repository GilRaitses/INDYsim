#!/usr/bin/env python3
"""
Process first track of first H5 file and display cycle-by-cycle statistics.
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd

# Add scripts directory to path
script_dir = Path(__file__).parent
project_root = script_dir.parent.parent.parent  # scripts/2025-11-13 -> scripts/ -> project root
scripts_dir = project_root / "scripts"
sys.path.insert(0, str(scripts_dir))

from engineer_dataset_from_h5 import load_h5_file, extract_stimulus_timing, extract_trajectory_features, align_trajectory_with_stimulus, create_event_records

# Import extract_cycles_from_h5 from queue directory
sys.path.insert(0, str(scripts_dir / "queue"))
from create_eda_figures import extract_cycles_from_h5

def main():
    # First H5 file
    h5_file = Path("D:/INDYsim/data/h5_files/GMR61@GMR61_T_Re_Sq_0to250PWM_30#C_Bl_7PWM_202510291652.h5")
    
    print("="*80)
    print("PROCESSING FIRST TRACK - CYCLE-BY-CYCLE STATISTICS")
    print("="*80)
    print(f"File: {h5_file.name}")
    print("="*80)
    
    # Load H5 file
    print("\n1. Loading H5 file...")
    h5_data = load_h5_file(h5_file)
    
    # Get metadata
    fps = h5_data.get('metadata', {}).get('fps', 10.0)
    print(f"   Frame rate: {fps} Hz")
    
    # Get first track
    track_keys = list(h5_data['tracks'].keys())
    if not track_keys:
        print("ERROR: No tracks found in H5 file")
        return
    
    first_track_key = track_keys[0]
    track_data = h5_data['tracks'][first_track_key]
    print(f"\n2. Processing track: {first_track_key}")
    print(f"   Total tracks in file: {len(track_keys)}")
    
    # Extract stimulus timing
    print("\n3. Extracting stimulus timing...")
    stimulus_df = extract_stimulus_timing(h5_data, fps)
    print(f"   Stimulus data points: {len(stimulus_df)}")
    
    # Extract trajectory features
    print("\n4. Extracting trajectory features...")
    traj_df = extract_trajectory_features(track_data, fps)
    print(f"   Trajectory frames: {len(traj_df)}")
    print(f"   Time range: {traj_df['time'].min():.1f}s to {traj_df['time'].max():.1f}s")
    
    # Align trajectory with stimulus
    print("\n5. Aligning trajectory with stimulus...")
    aligned_df = align_trajectory_with_stimulus(traj_df, stimulus_df)
    print(f"   Aligned frames: {len(aligned_df)}")
    
    # Extract cycles
    print("\n6. Extracting cycles from H5 file...")
    cycles, _ = extract_cycles_from_h5(h5_file)
    print(f"   Total cycles found: {len(cycles)}")
    
    # Get reorientation and turn start times
    print("\n7. Detecting event start times...")
    
    # Detect reorientation START events (False->True transitions)
    if 'is_reorientation' in aligned_df.columns:
        aligned_df_sorted = aligned_df.sort_values('time').reset_index(drop=True)
        is_reo = aligned_df_sorted['is_reorientation'].values
        if len(is_reo) > 1:
            is_reo_padded = np.concatenate([[False], is_reo])
            reo_start_mask = (~is_reo_padded[:-1]) & is_reo_padded[1:]
            reo_starts = aligned_df_sorted[reo_start_mask].copy()
        else:
            reo_starts = pd.DataFrame()
    else:
        reo_starts = pd.DataFrame()
    
    # Detect turn START events
    if 'is_turn' in aligned_df.columns:
        aligned_df_sorted = aligned_df.sort_values('time').reset_index(drop=True)
        is_turn = aligned_df_sorted['is_turn'].values
        if len(is_turn) > 1:
            is_turn_padded = np.concatenate([[False], is_turn])
            turn_start_mask = (~is_turn_padded[:-1]) & is_turn_padded[1:]
            turn_starts = aligned_df_sorted[turn_start_mask].copy()
        else:
            turn_starts = pd.DataFrame()
    else:
        turn_starts = pd.DataFrame()
    
    print(f"   Reorientation starts: {len(reo_starts)}")
    print(f"   Turn starts: {len(turn_starts)}")
    
    # Calculate per-cycle statistics
    print("\n" + "="*80)
    print("CYCLE-BY-CYCLE STATISTICS")
    print("="*80)
    
    BIN_SIZE = 0.5  # 0.5 second bins for per-bin analysis
    
    total_reos = 0
    total_turns = 0
    
    for cycle in cycles:
        cycle_num = cycle['cycle_num']
        cycle_start = cycle['cycle_start_time']
        cycle_end = cycle['cycle_end_time']
        onset_time = cycle['onset_time']
        pulse_dur = cycle['pulse_duration']
        
        # Count reorientations in this cycle
        cycle_reos = reo_starts[
            (reo_starts['time'] >= cycle_start) & 
            (reo_starts['time'] <= cycle_end)
        ] if len(reo_starts) > 0 else pd.DataFrame()
        n_cycle_reos = len(cycle_reos)
        
        # Count turns in this cycle
        cycle_turns = turn_starts[
            (turn_starts['time'] >= cycle_start) & 
            (turn_starts['time'] <= cycle_end)
        ] if len(turn_starts) > 0 else pd.DataFrame()
        n_cycle_turns = len(cycle_turns)
        
        # Calculate turn rate (reorientations per minute) for entire cycle
        cycle_duration_min = (cycle_end - cycle_start) / 60.0
        if cycle_duration_min > 0:
            turn_rate = (n_cycle_reos / cycle_duration_min) if n_cycle_reos > 0 else 0.0
        else:
            turn_rate = 0.0
        
        total_reos += n_cycle_reos
        total_turns += n_cycle_turns
        
        # Print cycle summary header
        print(f"\n{'='*80}")
        print(f"CYCLE {cycle_num} (Pulse Duration: {pulse_dur:.1f}s)")
        print(f"{'='*80}")
        print(f"Cycle Time Range: {cycle_start:.1f}s to {cycle_end:.1f}s")
        print(f"Onset Time (t=0): {onset_time:.1f}s")
        print(f"Cycle Duration: {cycle_end - cycle_start:.1f}s ({cycle_duration_min:.2f} minutes)")
        print(f"\nCycle Totals:")
        print(f"  Reorientations: {n_cycle_reos}")
        print(f"  Turns: {n_cycle_turns}")
        print(f"  Turn Rate: {turn_rate:.2f} min⁻¹")
        print(f"\n{'-'*80}")
        print(f"Per-Bin Statistics (0.5s bins):")
        print(f"{'-'*80}")
        print(f"{'Bin':<6} {'Time (rel)':<12} {'Reos':<6} {'Turns':<6} {'Rate (min⁻¹)':<12}")
        print(f"{'-'*80}")
        
        # Calculate per-bin statistics
        n_bins = int(np.ceil((cycle_end - cycle_start) / BIN_SIZE))
        bin_rates = []
        
        for bin_idx in range(n_bins):
            bin_start = cycle_start + (bin_idx * BIN_SIZE)
            bin_end = min(cycle_start + ((bin_idx + 1) * BIN_SIZE), cycle_end)
            
            # Count reorientations in this bin
            bin_reos = reo_starts[
                (reo_starts['time'] >= bin_start) & 
                (reo_starts['time'] < bin_end)
            ] if len(reo_starts) > 0 else pd.DataFrame()
            n_bin_reos = len(bin_reos)
            
            # Count turns in this bin
            bin_turns = turn_starts[
                (turn_starts['time'] >= bin_start) & 
                (turn_starts['time'] < bin_end)
            ] if len(turn_starts) > 0 else pd.DataFrame()
            n_bin_turns = len(bin_turns)
            
            # Calculate turn rate for this bin (reorientations per minute)
            bin_duration_min = (bin_end - bin_start) / 60.0
            if bin_duration_min > 0:
                bin_rate = (n_bin_reos / bin_duration_min) if n_bin_reos > 0 else 0.0
            else:
                bin_rate = 0.0
            
            bin_rates.append(bin_rate)
            
            # Time relative to onset (t=0 is onset)
            time_rel_onset = (bin_start + bin_end) / 2.0 - onset_time
            
            print(f"{bin_idx+1:<6} {time_rel_onset:>11.1f}s {n_bin_reos:<6} {n_bin_turns:<6} {bin_rate:>11.2f}")
    
    # Overall summary
    print(f"\n{'='*80}")
    print(f"TRACK SUMMARY: {first_track_key}")
    print(f"{'='*80}")
    print(f"Total Cycles: {len(cycles)}")
    print(f"Total Reorientations: {total_reos}")
    print(f"Total Turns: {total_turns}")
    
    # Overall turn rate
    track_duration_min = (aligned_df['time'].max() - aligned_df['time'].min()) / 60.0
    overall_rate = (total_reos / track_duration_min) if track_duration_min > 0 else 0.0
    print(f"Overall Turn Rate: {overall_rate:.2f} reorientations/min")
    print(f"Track Duration: {track_duration_min:.1f} minutes")
    print(f"{'='*80}\n")

if __name__ == "__main__":
    main()

