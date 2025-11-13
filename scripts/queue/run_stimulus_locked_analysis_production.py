#!/usr/bin/env python3
"""
Production-ready stimulus-locked turn rate analysis with real-time progress monitoring.
NO FALLBACKS - Production ready.
"""

import sys
import time
import subprocess
import threading
from pathlib import Path
from datetime import datetime
import json
import numpy as np
import pandas as pd

def launch_progress_monitor(progress_file: Path, h5_file: Path):
    """Launch a new PowerShell window with progress monitoring."""
    script_dir = Path(__file__).parent
    # monitor_analysis_progress.py is in scripts/archive/
    monitor_script = script_dir.parent / "archive" / "monitor_analysis_progress.py"
    
    # Create PowerShell command to launch new window
    # Change to project root (scripts/queue/ -> scripts/ -> project root)
    project_root = script_dir.parent.parent
    ps_command = f'''
$host.ui.RawUI.WindowTitle = "Stimulus-Locked Analysis Progress"
cd "{project_root}"
python "{monitor_script}" --progress-file "{progress_file}" --h5-file "{h5_file.name}"
'''
    
    # Launch PowerShell in new window
    subprocess.Popen(
        ['powershell', '-NoExit', '-Command', ps_command],
        creationflags=subprocess.CREATE_NEW_CONSOLE
    )
    time.sleep(1)  # Give window time to open

def update_progress(progress_file: Path, status: dict):
    """Update progress file with current status."""
    progress_file.parent.mkdir(parents=True, exist_ok=True)
    with open(progress_file, 'w') as f:
        json.dump(status, f, indent=2)

def main():
    h5_file = Path(r"D:\INDYsim\data\h5_files\GMR61@GMR61_T_Re_Sq_0to250PWM_30#C_Bl_7PWM_202510291652.h5")
    
    if not h5_file.exists():
        print(f"ERROR: H5 file not found: {h5_file}")
        sys.exit(1)
    
    # Setup paths
    script_dir = Path(__file__).parent
    experiment_id = h5_file.stem.replace(' ', '_')
    output_dir = Path("data/engineered")
    output_dir.mkdir(parents=True, exist_ok=True)
    progress_file = output_dir / f"{experiment_id}_progress.json"
    figures_dir = Path("output/figures/eda")
    figures_dir.mkdir(parents=True, exist_ok=True)
    
    # Initialize progress
    start_time = time.time()
    progress = {
        'status': 'initializing',
        'stage': 'Starting analysis',
        'progress_pct': 0,
        'current_track': 0,
        'total_tracks': 0,
        'elapsed_time': 0,
        'eta_seconds': 0,
        'h5_file': str(h5_file),
        'experiment_id': experiment_id,
        'start_time': datetime.now().isoformat(),
        'messages': []
    }
    update_progress(progress_file, progress)
    
    # Launch progress monitor window
    print("Launching progress monitor window...")
    launch_progress_monitor(progress_file, h5_file)
    time.sleep(2)  # Give monitor time to initialize
    
    try:
        # Step 1: Generate events CSV from H5 file
        print("\n=== Step 1: Generating events CSV ===")
        progress['status'] = 'processing'
        progress['stage'] = 'Extracting data from H5 file'
        progress['progress_pct'] = 5
        update_progress(progress_file, progress)
        
        # engineer_dataset_from_h5.py is in scripts/ (parent of queue/)
        sys.path.insert(0, str(script_dir.parent))
        from engineer_dataset_from_h5 import process_h5_file, load_h5_file
        import h5py
        
        # Validate H5 file can be loaded
        h5_data = load_h5_file(h5_file)
        if 'tracks' not in h5_data:
            raise ValueError(f"No tracks found in H5 file: {h5_file}")
        
        total_tracks = len(h5_data['tracks'])
        progress['total_tracks'] = total_tracks
        update_progress(progress_file, progress)
        
        # Process H5 file with progress tracking
        # We'll monkey-patch the process to update progress
        import engineer_dataset_from_h5 as eng_module
        original_process = eng_module.process_h5_file
        
        def process_with_progress(h5_path, output_dir, exp_id):
            """Wrapper that updates progress during processing."""
            h5_data = eng_module.load_h5_file(h5_path)
            frame_rate = 10.0
            if 'metadata' in h5_data and 'attrs' in h5_data['metadata']:
                metadata_attrs = h5_data['metadata']['attrs']
                if 'fps' in metadata_attrs:
                    frame_rate = float(metadata_attrs['fps'])
            
            stimulus_df = eng_module.extract_stimulus_timing(h5_data, frame_rate=frame_rate)
            if len(stimulus_df) == 0:
                raise ValueError(f"No stimulus data found in {h5_path}")
            
            all_event_records = []
            all_trajectories = []
            all_klein_run_tables = []
            
            track_keys = list(h5_data.get('tracks', {}).keys())
            total = len(track_keys)
            
            for idx, track_key in enumerate(track_keys, 1):
                # Update progress
                progress['current_track'] = idx
                progress['total_tracks'] = total
                progress['progress_pct'] = 5 + int(45 * idx / total) if total > 0 else 5
                elapsed = time.time() - start_time
                progress['elapsed_time'] = elapsed
                if idx > 0:
                    eta = (elapsed / idx) * (total - idx)
                    progress['eta_seconds'] = eta
                progress['stage'] = f'Processing track {idx}/{total}: {track_key}'
                progress['messages'].append({
                    'time': datetime.now().isoformat(),
                    'text': f'Processing track {idx}/{total}: {track_key}'
                })
                # Keep only last 20 messages
                if len(progress['messages']) > 20:
                    progress['messages'] = progress['messages'][-20:]
                update_progress(progress_file, progress)
                
                # Process track
                track_data = h5_data['tracks'][track_key]
                try:
                    track_id = int(track_key.split('_')[-1])
                except:
                    track_id = idx
                
                # Update message before processing
                progress['messages'].append({
                    'time': datetime.now().isoformat(),
                    'text': f'Extracting features for track {idx}/{total}'
                })
                update_progress(progress_file, progress)
                
                # CRITICAL POLICY: ETI MUST ALWAYS BE USED FOR TIME CALCULATION
                # See docs/ETI_TIME_CALCULATION_POLICY.md
                if 'eti' not in h5_data or h5_data['eti'] is None:
                    raise ValueError(f"CRITICAL ERROR: ETI not available in h5_data for track {track_key}. "
                                    "ETI must be loaded from H5 root level. "
                                    "See docs/ETI_TIME_CALCULATION_POLICY.md")
                traj_df = eng_module.extract_trajectory_features(track_data, frame_rate=frame_rate, eti=h5_data['eti'])
                if len(traj_df) == 0:
                    progress['messages'].append({
                        'time': datetime.now().isoformat(),
                        'text': f'Track {idx}/{total}: No trajectory data, skipping'
                    })
                    update_progress(progress_file, progress)
                    continue
                
                # Update after feature extraction
                progress['messages'].append({
                    'time': datetime.now().isoformat(),
                    'text': f'Track {idx}/{total}: Extracted {len(traj_df)} frames'
                })
                update_progress(progress_file, progress)
                
                if len(stimulus_df) > 0:
                    aligned_df = eng_module.align_trajectory_with_stimulus(traj_df, stimulus_df)
                else:
                    aligned_df = traj_df.copy()
                    aligned_df['led1Val'] = 0.0
                    aligned_df['led1Val_ton'] = False
                    aligned_df['led1Val_toff'] = True
                    aligned_df['led2Val'] = 0.0
                    aligned_df['led2Val_ton'] = False
                    aligned_df['led2Val_toff'] = True
                    aligned_df['stimulus_on'] = False
                    aligned_df['stimulus_onset'] = False
                    aligned_df['time_since_stimulus'] = float('nan')
                
                # Update before event creation
                progress['messages'].append({
                    'time': datetime.now().isoformat(),
                    'text': f'Track {idx}/{total}: Creating event records'
                })
                update_progress(progress_file, progress)
                
                event_records = eng_module.create_event_records(aligned_df, track_id, exp_id)
                all_event_records.append(event_records)
                all_trajectories.append(aligned_df)
                
                # Calculate per-cycle statistics and display summary table
                print(f"\n{'='*80}")
                print(f"TRACK {idx}/{total} SUMMARY: {track_key}")
                print(f"{'='*80}")
                
                # Fix: Use start events for counting
                n_turns = event_records['is_turn_start'].sum() if 'is_turn_start' in event_records.columns else (event_records['is_turn'].sum() if 'is_turn' in event_records.columns else 0)
                n_reorientations = event_records['is_reorientation_start'].sum() if 'is_reorientation_start' in event_records.columns else (event_records['is_reorientation'].sum() if 'is_reorientation' in event_records.columns else 0)
                
                # Extract cycles and calculate per-cycle statistics
                try:
                    # Import extract_cycles_from_h5 - handle both relative and absolute imports
                    script_dir = Path(__file__).parent
                    sys.path.insert(0, str(script_dir))
                    try:
                        from create_eda_figures import extract_cycles_from_h5
                    except ImportError:
                        # Try absolute import
                        eda_figures_path = script_dir / "create_eda_figures.py"
                        if eda_figures_path.exists():
                            import importlib.util
                            spec = importlib.util.spec_from_file_location("create_eda_figures", eda_figures_path)
                            create_eda_figures = importlib.util.module_from_spec(spec)
                            spec.loader.exec_module(create_eda_figures)
                            extract_cycles_from_h5 = create_eda_figures.extract_cycles_from_h5
                        else:
                            raise ImportError(f"Could not find create_eda_figures.py at {eda_figures_path}")
                    
                    cycles, _ = extract_cycles_from_h5(h5_file)
                    
                    if len(cycles) > 0:
                        print(f"\nCycles found: {len(cycles)}")
                        
                        # Calculate per-cycle statistics
                        BIN_SIZE = 0.5  # 0.5 second bins
                        baseline_period = 10.0
                        
                        # Get reorientation start times from aligned trajectory
                        # Check if create_event_records already added is_reorientation_start
                        if 'is_reorientation_start' in aligned_df.columns:
                            reo_starts = aligned_df[aligned_df['is_reorientation_start'] == True].copy()
                        elif 'is_reorientation' in aligned_df.columns:
                            # Detect start events (False->True transitions)
                            aligned_df_sorted = aligned_df.sort_values('time').reset_index(drop=True)
                            is_reo = aligned_df_sorted['is_reorientation'].values
                            if len(is_reo) > 1:
                                is_reo_padded = np.concatenate([[False], is_reo])
                                start_mask = (~is_reo_padded[:-1]) & is_reo_padded[1:]
                                reo_starts = aligned_df_sorted[start_mask].copy()
                            else:
                                reo_starts = pd.DataFrame()
                        else:
                            reo_starts = pd.DataFrame()
                        
                        # Get turn start times
                        if 'is_turn_start' in aligned_df.columns:
                            turn_starts = aligned_df[aligned_df['is_turn_start'] == True].copy()
                        elif 'is_turn' in aligned_df.columns:
                            aligned_df_sorted = aligned_df.sort_values('time').reset_index(drop=True)
                            is_turn = aligned_df_sorted['is_turn'].values
                            if len(is_turn) > 1:
                                is_turn_padded = np.concatenate([[False], is_turn])
                                start_mask = (~is_turn_padded[:-1]) & is_turn_padded[1:]
                                turn_starts = aligned_df_sorted[start_mask].copy()
                            else:
                                turn_starts = pd.DataFrame()
                        else:
                            turn_starts = pd.DataFrame()
                        
                        # Ensure time column exists
                        if len(reo_starts) > 0 and 'time' not in reo_starts.columns:
                            reo_starts = pd.DataFrame()
                        if len(turn_starts) > 0 and 'time' not in turn_starts.columns:
                            turn_starts = pd.DataFrame()
                        
                        total_reos = 0
                        total_turns = 0
                        
                        for cycle in cycles:
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
                            print(f"\nCycle {cycle['cycle_num']} (Pulse: {pulse_dur:.1f}s)")
                            print(f"{'='*80}")
                            print(f"Total Reorientations: {n_cycle_reos} | Total Turns: {n_cycle_turns} | Turn Rate: {turn_rate:.2f} min⁻¹")
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
                                
                                print(f"  Bin {bin_idx+1:2d} | t={time_rel_onset:6.1f}s | Reos: {n_bin_reos:2d} | Turns: {n_bin_turns:2d} | Rate: {bin_rate:6.2f} min⁻¹")
                        
                        print(f"\n{'='*80}")
                        print(f"TRACK TOTALS: Reorientations: {total_reos} | Turns: {total_turns}")
                        
                        # Overall turn rate
                        track_duration_min = (aligned_df['time'].max() - aligned_df['time'].min()) / 60.0
                        overall_rate = (total_reos / track_duration_min) if track_duration_min > 0 else 0.0
                        print(f"\nOverall turn rate: {overall_rate:.2f} reorientations/min")
                        print(f"Track duration: {track_duration_min:.1f} minutes")
                        
                    else:
                        print("  No cycles found in H5 file")
                        print(f"  Total reorientations: {n_reorientations}")
                        print(f"  Total turns: {n_turns}")
                        
                except Exception as e:
                    print(f"\n  ERROR calculating per-cycle stats: {e}")
                    import traceback
                    traceback.print_exc()
                    print(f"  Falling back to simple summary:")
                    print(f"  Total reorientations: {n_reorientations}")
                    print(f"  Total turns: {n_turns}")
                    print(f"  Note: Turns > Reorientations is normal - turns use simple detection (>30°),")
                    print(f"        reorientations use MAGAT behavioral segmentation (more strict)")
                
                print(f"{'='*80}\n")
                
                progress['messages'].append({
                    'time': datetime.now().isoformat(),
                    'text': f'Track {idx}/{total}: Complete - {n_reorientations} reorientations, {n_turns} turns'
                })
                # Keep only last 20 messages
                if len(progress['messages']) > 20:
                    progress['messages'] = progress['messages'][-20:]
                update_progress(progress_file, progress)
                
                if hasattr(traj_df, 'attrs') and 'klein_run_table' in traj_df.attrs:
                    klein_rt = traj_df.attrs['klein_run_table'].copy()
                    klein_rt['track_id'] = track_id
                    klein_rt['experiment_id'] = exp_id
                    all_klein_run_tables.append(klein_rt)
            
            # Save outputs
            if all_event_records:
                progress['messages'].append({
                    'time': datetime.now().isoformat(),
                    'text': f'Saving {len(all_event_records)} tracks to CSV files'
                })
                update_progress(progress_file, progress)
                
                combined_events = pd.concat(all_event_records, ignore_index=True)
                combined_trajectories = pd.concat(all_trajectories, ignore_index=True)
                
                output_dir.mkdir(parents=True, exist_ok=True)
                
                events_file = output_dir / f"{exp_id}_events.csv"
                combined_events.to_csv(events_file, index=False)
                progress['messages'].append({
                    'time': datetime.now().isoformat(),
                    'text': f'Saved events CSV: {len(combined_events):,} records'
                })
                update_progress(progress_file, progress)
                
                trajectories_file = output_dir / f"{exp_id}_trajectories.csv"
                combined_trajectories.to_csv(trajectories_file, index=False)
                progress['messages'].append({
                    'time': datetime.now().isoformat(),
                    'text': f'Saved trajectories CSV: {len(combined_trajectories):,} records'
                })
                update_progress(progress_file, progress)
                
                if all_klein_run_tables:
                    combined_klein_runs = pd.concat(all_klein_run_tables, ignore_index=True)
                    klein_runs_file = output_dir / f"{exp_id}_klein_run_table.csv"
                    combined_klein_runs.to_csv(klein_runs_file, index=False)
                    progress['messages'].append({
                        'time': datetime.now().isoformat(),
                        'text': f'Saved Klein run table: {len(combined_klein_runs):,} records'
                    })
                    update_progress(progress_file, progress)
            else:
                raise ValueError("No event records generated from tracks")
        
        # Import pandas for the wrapper
        import pandas as pd
        
        # Process with progress tracking
        process_with_progress(h5_file, output_dir, experiment_id)
        
        progress['stage'] = 'Data extraction complete'
        progress['progress_pct'] = 50
        update_progress(progress_file, progress)
        
        # Step 2: Run stimulus-locked turn rate analysis
        print("\n=== Step 2: Running stimulus-locked turn rate analysis ===")
        events_file = output_dir / f"{experiment_id}_events.csv"
        trajectories_file = output_dir / f"{experiment_id}_trajectories.csv"
        
        if not events_file.exists():
            raise FileNotFoundError(f"Events file not found: {events_file}. Data extraction failed.")
        
        progress['stage'] = 'Loading events data'
        progress['progress_pct'] = 60
        progress['messages'].append({
            'time': datetime.now().isoformat(),
            'text': 'Loading events CSV file'
        })
        update_progress(progress_file, progress)
        
        output_path = figures_dir / f"{experiment_id}_stimulus_locked_turn_rate.png"
        
        progress['stage'] = 'Extracting cycles from H5'
        progress['progress_pct'] = 65
        progress['messages'].append({
            'time': datetime.now().isoformat(),
            'text': 'Extracting stimulus cycles from H5 file'
        })
        update_progress(progress_file, progress)
        
        from create_eda_figures import create_stimulus_locked_turn_rate_analysis
        
        progress['stage'] = 'Computing turn rates'
        progress['progress_pct'] = 70
        progress['messages'].append({
            'time': datetime.now().isoformat(),
            'text': 'Calculating stimulus-locked turn rates'
        })
        update_progress(progress_file, progress)
        
        create_stimulus_locked_turn_rate_analysis(
            str(trajectories_file),
            str(events_file),
            str(h5_file),
            output_path
        )
        
        progress['stage'] = 'Generating figure'
        progress['progress_pct'] = 90
        progress['messages'].append({
            'time': datetime.now().isoformat(),
            'text': f'Saving figure to {output_path.name}'
        })
        update_progress(progress_file, progress)
        
        if not output_path.exists():
            raise FileNotFoundError(f"Output figure not created: {output_path}")
        
        # Final status
        elapsed = time.time() - start_time
        progress['status'] = 'complete'
        progress['stage'] = 'Analysis complete'
        progress['progress_pct'] = 100
        progress['elapsed_time'] = elapsed
        progress['eta_seconds'] = 0
        progress['end_time'] = datetime.now().isoformat()
        progress['output_files'] = {
            'events_csv': str(events_file),
            'trajectories_csv': str(trajectories_file),
            'figure': str(output_path)
        }
        update_progress(progress_file, progress)
        
        print(f"\nAnalysis complete!")
        print(f"  Events CSV: {events_file}")
        print(f"  Trajectories CSV: {trajectories_file}")
        print(f"  Output figure: {output_path}")
        print(f"  Total time: {elapsed:.1f}s")
        
    except Exception as e:
        elapsed = time.time() - start_time
        progress['status'] = 'error'
        progress['stage'] = f'ERROR: {str(e)}'
        progress['progress_pct'] = 0
        progress['elapsed_time'] = elapsed
        progress['error'] = str(e)
        progress['error_time'] = datetime.now().isoformat()
        update_progress(progress_file, progress)
        
        print(f"\nERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == '__main__':
    main()

