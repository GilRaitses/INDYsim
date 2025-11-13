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
    h5_file = Path(r"D:\INDYsim\data\h5_files\GMR61@GMR61_T_Re_Sq_0to250PWM_30#T_Bl_Sq_5to15PWM_30_202510301513.h5")
    
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
        
        sys.path.insert(0, str(Path(__file__).parent.parent))
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
                
                traj_df = eng_module.extract_trajectory_features(track_data, frame_rate=frame_rate)
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
                
                # Update after completion
                n_turns = event_records['is_turn'].sum()
                n_reorientations = event_records['is_reorientation'].sum() if 'is_reorientation' in event_records.columns else 0
                progress['messages'].append({
                    'time': datetime.now().isoformat(),
                    'text': f'Track {idx}/{total}: Complete - {n_turns} turns, {n_reorientations} reorientations'
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

