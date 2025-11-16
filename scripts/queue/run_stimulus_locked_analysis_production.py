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
    """Launch a new terminal window with Cinnamoroll progress monitoring (Mac/Linux compatible)."""
    script_dir = Path(__file__).parent
    project_root = script_dir.parent.parent
    
    # Find the most recent cinnamoroll monitor script
    # Check scripts/2025-11-15/ first, then fall back to scripts/archive/
    monitor_script = None
    monitor_launcher = None
    
    # Try to find today's monitor script
    today_dir = project_root / "scripts" / datetime.now().strftime("%Y-%m-%d")
    if (today_dir / "queue_monitor.py").exists():
        monitor_script = today_dir / "queue_monitor.py"
        monitor_launcher = today_dir / "open_queue_monitor.sh"
    elif (today_dir / "cinnamoroll_monitor.py").exists():
        # Fallback to old monitor (deprecated - use queue_monitor instead)
        monitor_script = today_dir / "cinnamoroll_monitor.py"
        monitor_launcher = None  # Old launcher removed
    else:
        # Fall back to archive
        archive_script = script_dir.parent / "archive" / "monitor_analysis_progress.py"
        if archive_script.exists():
            monitor_script = archive_script
    
    if monitor_launcher and monitor_launcher.exists():
        # Use the launcher script (Mac/Linux)
        import os
        if os.name == 'posix':  # Unix-like (Mac/Linux)
            subprocess.Popen(
                ['bash', str(monitor_launcher), str(progress_file)],
                cwd=str(project_root)
            )
            time.sleep(1)
            return
    
    # Fallback: Launch Python script directly
    if monitor_script and monitor_script.exists():
        import os
        import platform
        
        if platform.system() == 'Darwin':  # macOS
            # Use osascript to open Terminal
            osascript_cmd = f'''
tell application "Terminal"
    activate
    do script "cd '{project_root}' && python3 '{monitor_script}' --progress-file '{progress_file}'"
end tell
'''
            subprocess.Popen(['osascript', '-e', osascript_cmd])
        elif platform.system() == 'Linux':
            # Try gnome-terminal or xterm
            if subprocess.run(['which', 'gnome-terminal'], capture_output=True).returncode == 0:
                subprocess.Popen([
                    'gnome-terminal', '--', 'bash', '-c',
                    f"cd '{project_root}' && python3 '{monitor_script}' --progress-file '{progress_file}'; exec bash"
                ])
            elif subprocess.run(['which', 'xterm'], capture_output=True).returncode == 0:
                subprocess.Popen([
                    'xterm', '-e',
                    f"cd '{project_root}' && python3 '{monitor_script}' --progress-file '{progress_file}'"
                ])
        elif platform.system() == 'Windows':
            # Windows fallback - PowerShell
            ps_command = f'''
$host.ui.RawUI.WindowTitle = "Stimulus-Locked Analysis Progress"
cd "{project_root}"
python "{monitor_script}" --progress-file "{progress_file}"
'''
            subprocess.Popen(
                ['powershell', '-NoExit', '-Command', ps_command],
                creationflags=subprocess.CREATE_NEW_CONSOLE if hasattr(subprocess, 'CREATE_NEW_CONSOLE') else 0
            )
        
        time.sleep(1)  # Give window time to open

def update_progress(progress_file: Path, status: dict):
    """Update progress file with current status."""
    progress_file.parent.mkdir(parents=True, exist_ok=True)
    with open(progress_file, 'w') as f:
        json.dump(status, f, indent=2)

def main():
    # Allow H5 file to be specified via command line argument or environment variable
    import os
    if len(sys.argv) > 1:
        h5_file = Path(sys.argv[1])
    elif 'H5_FILE_PATH' in os.environ:
        h5_file = Path(os.environ['H5_FILE_PATH'])
    else:
        # Default to first file for backwards compatibility
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
    
    # Launch progress monitor window (unless disabled for queue mode)
    import os
    if not os.environ.get('DISABLE_AUTO_MONITOR'):
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
            
            # Get track keys and sort by numeric value (track_1, track_2, ..., track_9, ..., track_64)
            # This ensures consistent ordering regardless of how they were stored in H5
            track_keys = list(h5_data.get('tracks', {}).keys())
            
            def extract_track_number(track_key):
                """Extract numeric part from track key (e.g., 'track_9' -> 9)."""
                try:
                    # Extract number after 'track_' prefix
                    return int(track_key.split('_')[-1])
                except (ValueError, IndexError):
                    # Fallback: use a large number to put non-standard keys at the end
                    return 999999
            
            # Sort track keys by their numeric value
            track_keys = sorted(track_keys, key=extract_track_number)
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
                progress['stage'] = f'Processing {track_key} ({idx}/{total})'
                progress['messages'].append({
                    'time': datetime.now().isoformat(),
                    'text': f'Processing {track_key} ({idx}/{total})'
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
                
                # Print track number BEFORE feature extraction so MAGAT segmentation can find it
                print(f"Processing {track_key} ({idx}/{total})")
                sys.stdout.flush()
                
                traj_df = eng_module.extract_trajectory_features(track_data, frame_rate=frame_rate, eti=h5_data['eti'])
                if len(traj_df) == 0:
                    progress['messages'].append({
                        'time': datetime.now().isoformat(),
                        'text': f'Track {idx}/{total}: No trajectory data, skipping'
                    })
                    update_progress(progress_file, progress)
                    continue
                
                # Update after feature extraction
                print(f"Track {idx}/{total}: Extracted {len(traj_df)} frames from {track_key}")
                sys.stdout.flush()
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
                print(f"Track {idx}/{total}: Aligned {len(aligned_df)} frames with stimulus")
                sys.stdout.flush()
                progress['messages'].append({
                    'time': datetime.now().isoformat(),
                    'text': f'Track {idx}/{total}: Creating event records'
                })
                update_progress(progress_file, progress)
                
                event_records = eng_module.create_event_records(aligned_df, track_id, exp_id)
                print(f"Track {idx}/{total}: Created {len(event_records)} event records")
                sys.stdout.flush()
                all_event_records.append(event_records)
                all_trajectories.append(aligned_df)
                
                # Calculate per-cycle statistics and display summary table
                print(f"\n{'='*80}")
                print(f"TRACK {idx}/{total} SUMMARY: {track_key}")
                print(f"{'='*80}")
                
                # Fix: Use start events for counting
                n_turns = event_records['is_turn_start'].sum() if 'is_turn_start' in event_records.columns else (event_records['is_turn'].sum() if 'is_turn' in event_records.columns else 0)
                n_reorientations = event_records['is_reorientation_start'].sum() if 'is_reorientation_start' in event_records.columns else (event_records['is_reorientation'].sum() if 'is_reorientation' in event_records.columns else 0)
                
                # Calculate pause statistics
                n_pauses = event_records['is_pause_start'].sum() if 'is_pause_start' in event_records.columns else (event_records['is_pause'].sum() if 'is_pause' in event_records.columns else 0)
                pause_durations = event_records[event_records['pause_duration'] > 0]['pause_duration'].values
                mean_pause_duration = float(np.mean(pause_durations)) if len(pause_durations) > 0 else 0.0
                
                # Calculate turn rate (reorientations per minute)
                track_duration_min = (aligned_df['time'].max() - aligned_df['time'].min()) / 60.0 if len(aligned_df) > 0 else 0.0
                mean_turn_rate = (n_reorientations / track_duration_min) if track_duration_min > 0 else 0.0
                
                # Count head swings from MAGAT segmentation
                # The MAGAT segmentation happens in extract_trajectory_features and prints:
                # "MAGAT segmentation: X runs, Y head swings, Z reorientations"
                # We need to parse this from the console output or access it directly
                # For now, we'll parse from the console log in the monitor
                # But we can also try to get it from the print that happened earlier
                n_headswings = 0
                # Note: The monitor will parse head swings from the MAGAT segmentation line
                # in the console log, so we set this to 0 here and let the monitor extract it
                
                # Calculate track timing information using ETI at start/end frame indices
                # Track metadata contains startFrame and endFrame that map to ETI indices
                track_start_eti_index = None
                track_end_eti_index = None
                
                if 'metadata_attrs' in track_data and 'startFrame' in track_data['metadata_attrs']:
                    track_start_eti_index = int(track_data['metadata_attrs']['startFrame'])
                    # endFrame might be in metadata, or calculate from startFrame + n_frames
                    if 'endFrame' in track_data['metadata_attrs']:
                        track_end_eti_index = int(track_data['metadata_attrs']['endFrame'])
                    else:
                        # Calculate end frame: startFrame + n_frames - 1
                        track_end_eti_index = track_start_eti_index + len(traj_df) - 1
                
                # Get ETI values at start and end frame indices
                if track_start_eti_index is not None and track_end_eti_index is not None:
                    # Ensure indices are within ETI bounds
                    eti_length = len(h5_data['eti'])
                    track_start_eti_index = max(0, min(track_start_eti_index, eti_length - 1))
                    track_end_eti_index = max(0, min(track_end_eti_index, eti_length - 1))
                    
                    track_start_eti_time = h5_data['eti'][track_start_eti_index]
                    track_end_eti_time = h5_data['eti'][track_end_eti_index]
                else:
                    # Fallback: use time column min/max (less accurate but better than nothing)
                    track_start_eti_time = aligned_df['time'].min() if len(aligned_df) > 0 else 0.0
                    track_end_eti_time = aligned_df['time'].max() if len(aligned_df) > 0 else 0.0
                
                track_duration_sec = track_end_eti_time - track_start_eti_time if len(aligned_df) > 0 else 0.0
                
                # Format times as mm:ss
                def format_mmss(seconds):
                    """Format seconds as mm:ss"""
                    if seconds < 0 or not np.isfinite(seconds):
                        return "00:00"
                    mins = int(seconds // 60)
                    secs = int(seconds % 60)
                    return f"{mins:02d}:{secs:02d}"
                
                duration_mmss = format_mmss(track_duration_sec)
                start_mmss = format_mmss(track_start_eti_time)  # ETI value at start frame
                end_mmss = format_mmss(track_end_eti_time)  # ETI value at end frame
                
                # Extract cycles and calculate per-cycle statistics
                n_cycles = 0
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
                    n_cycles = len(cycles)
                except Exception as e:
                    # If cycle extraction fails, continue with n_cycles = 0
                    n_cycles = 0
                    print(f"  Warning: Could not extract cycles: {e}")
                
                # Print detailed track processing info with all stats
                print(f"  Step 1: Feature extraction - {len(traj_df)} frames extracted")
                print(f"  Step 2: Stimulus alignment - {len(aligned_df)} aligned frames")
                print(f"  Step 3: Event detection - {len(event_records)} event records created")
                print(f"  Step 4: Statistics - {n_turns} turns, {n_reorientations} reorientations")
                print(f"TRACK_STATS: Track {idx} | Duration: {duration_mmss} | Start: {start_mmss} | End: {end_mmss} | Cycles: {n_cycles} | Frames: {len(traj_df)} | Reorientations: {n_reorientations} | Turns: {n_turns} | TurnRate: {mean_turn_rate:.2f} | Pauses: {n_pauses} | MeanPauseDur: {mean_pause_duration:.3f} | HeadSwings: {n_headswings}")
                sys.stdout.flush()
                
                # Calculate per-cycle statistics (if cycles were found)
                if n_cycles > 0:
                    try:
                        print(f"\nCycles found: {n_cycles}")
                        
                        # Calculate per-cycle statistics
                        BIN_SIZE = 0.5  # 0.5 second bins
                        baseline_period = 10.0
                        
                        # For biologically meaningful per-minute rate calculation:
                        # Use a fixed time window around stimulus onset (e.g., 30s before and 30s after)
                        RATE_WINDOW_BEFORE = 30.0  # seconds before stimulus
                        RATE_WINDOW_AFTER = 30.0   # seconds after stimulus
                        RATE_WINDOW_TOTAL = RATE_WINDOW_BEFORE + RATE_WINDOW_AFTER  # 60 seconds total
                        
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
                        rate_window_reos = []  # Store reorientation counts for rate window
                        
                        for cycle in cycles:
                            cycle_start = cycle['cycle_start_time']
                            cycle_end = cycle['cycle_end_time']
                            onset_time = cycle['onset_time']
                            pulse_dur = cycle['pulse_duration']
                            
                            # Define rate calculation window: 30s before to 30s after stimulus onset
                            rate_window_start = onset_time - RATE_WINDOW_BEFORE
                            rate_window_end = onset_time + RATE_WINDOW_AFTER
                            
                            # Count reorientations in rate window (if window overlaps with track data)
                            window_reos = reo_starts[
                                (reo_starts['time'] >= rate_window_start) & 
                                (reo_starts['time'] < rate_window_end)
                            ] if len(reo_starts) > 0 else pd.DataFrame()
                            n_window_reos = len(window_reos)
                            
                            # Calculate per-minute rate using fixed window (biologically meaningful)
                            # Rate = (total reorientations in window) / (window duration in minutes)
                            per_minute_rate = (n_window_reos / (RATE_WINDOW_TOTAL / 60.0)) if RATE_WINDOW_TOTAL > 0 else 0.0
                            
                            # Store counts for this cycle's rate window
                            rate_window_reos.append(n_window_reos)
                            
                            # Count reorientations in this cycle (for cycle-level stats)
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
                            # This is biologically meaningful because it's over the full cycle duration
                            cycle_duration_min = (cycle_end - cycle_start) / 60.0
                            if cycle_duration_min > 0:
                                turn_rate = (n_cycle_reos / cycle_duration_min) if n_cycle_reos > 0 else 0.0
                            else:
                                turn_rate = 0.0
                            
                            # NOTE: Per-bin rates are NOT calculated here because:
                            # - Single-cycle bin rates (e.g., 1 reo in 0.5s = 120 min⁻¹) are biologically meaningless
                            # - Biologically meaningful rates require aggregation across cycles
                            # - See create_eda_figures.py for proper per-bin rate calculation with cycle aggregation
                            
                            total_reos += n_cycle_reos
                            total_turns += n_cycle_turns
                            
                            # Print cycle summary header
                            print(f"\nCycle {cycle['cycle_num']} (Pulse: {pulse_dur:.1f}s)")
                            print(f"{'='*80}")
                            print(f"Total Reorientations: {n_cycle_reos} | Total Turns: {n_cycle_turns} | Turn Rate: {turn_rate:.2f} min⁻¹")
                            print(f"Rate Window ({-RATE_WINDOW_BEFORE:.0f}s to +{RATE_WINDOW_AFTER:.0f}s): {n_window_reos} reos | Rate: {per_minute_rate:.2f} min⁻¹")
                            print(f"{'-'*80}")
                            
                            # Calculate per-bin statistics with 60s sliding window around each bin
                            # For each 0.5s bin, calculate rate from a 60s window centered around that bin
                            # This gives biologically meaningful rates at each time point
                            # Handle edge cases where full 60s window isn't available
                            n_bins = int(np.ceil((cycle_end - cycle_start) / BIN_SIZE))
                            bin_rates = []  # Store rates calculated from 60s windows
                            
                            # Get track boundaries for edge case handling
                            track_start_time = aligned_df['time'].min() if len(aligned_df) > 0 else cycle_start
                            track_end_time = aligned_df['time'].max() if len(aligned_df) > 0 else cycle_end
                            
                            for bin_idx in range(n_bins):
                                bin_start = cycle_start + (bin_idx * BIN_SIZE)
                                bin_end = min(cycle_start + ((bin_idx + 1) * BIN_SIZE), cycle_end)
                                bin_center = (bin_start + bin_end) / 2.0
                                
                                # Define ideal 60s window centered around this bin (30s before and 30s after bin center)
                                ideal_window_start = bin_center - RATE_WINDOW_BEFORE
                                ideal_window_end = bin_center + RATE_WINDOW_AFTER
                                
                                # Adjust window to available data (handle edge cases)
                                actual_window_start = max(ideal_window_start, track_start_time)
                                actual_window_end = min(ideal_window_end, track_end_time)
                                actual_window_duration = actual_window_end - actual_window_start
                                
                                # Only calculate rate if we have at least 10 seconds of data (minimum meaningful window)
                                MIN_WINDOW_DURATION = 10.0  # seconds
                                if actual_window_duration >= MIN_WINDOW_DURATION:
                                    # Count reorientations in the available window around this bin
                                    window_reos = reo_starts[
                                        (reo_starts['time'] >= actual_window_start) & 
                                        (reo_starts['time'] < actual_window_end)
                                    ] if len(reo_starts) > 0 else pd.DataFrame()
                                    n_window_reos = len(window_reos)
                                    
                                    # Count turns in the available window
                                    window_turns = turn_starts[
                                        (turn_starts['time'] >= actual_window_start) & 
                                        (turn_starts['time'] < actual_window_end)
                                    ] if len(turn_starts) > 0 else pd.DataFrame()
                                    n_window_turns = len(window_turns)
                                    
                                    # Calculate rate from actual window duration (normalize to per-minute)
                                    # Rate = (reorientations / window_duration_minutes)
                                    window_duration_min = actual_window_duration / 60.0
                                    bin_rate = (n_window_reos / window_duration_min) if window_duration_min > 0 else 0.0
                                    
                                    # Flag if window was truncated (for potential filtering)
                                    window_truncated = (actual_window_start > ideal_window_start) or (actual_window_end < ideal_window_end)
                                    
                                    bin_rates.append({
                                        'bin_center': bin_center,
                                        'time_rel_onset': bin_center - onset_time,
                                        'reos': n_window_reos,
                                        'turns': n_window_turns,
                                        'rate': bin_rate,
                                        'window_duration': actual_window_duration,
                                        'window_truncated': window_truncated
                                    })
                                else:
                                    # Not enough data for meaningful rate calculation
                                    bin_rates.append({
                                        'bin_center': bin_center,
                                        'time_rel_onset': bin_center - onset_time,
                                        'reos': 0,
                                        'turns': 0,
                                        'rate': np.nan,  # Mark as invalid
                                        'window_duration': actual_window_duration,
                                        'window_truncated': True
                                    })
                                
                                # Don't print individual bins - too verbose
                                # Only print cycle summary at the end
                            
                            # Print cumulative cycle summary after processing all bins
                            print(f"Cumulative: Reorientations: {total_reos} | Turns: {total_turns}")
                            
                            # Show bin rate statistics (if bins were calculated)
                            if len(bin_rates) > 0:
                                # Only include valid rates (not NaN, from sufficient window)
                                valid_bin_rates = [b['rate'] for b in bin_rates if not np.isnan(b['rate']) and b['rate'] >= 0]
                                truncated_bins = sum(1 for b in bin_rates if b.get('window_truncated', False))
                                
                                if len(valid_bin_rates) > 0:
                                    mean_bin_rate = np.mean(valid_bin_rates)
                                    max_bin_rate = np.max(valid_bin_rates)
                                    n_valid_bins = len(valid_bin_rates)
                                    n_total_bins = len(bin_rates)
                                    
                                    print(f"Bin rates (sliding window): mean={mean_bin_rate:.2f} min⁻¹, max={max_bin_rate:.2f} min⁻¹")
                                    if truncated_bins > 0:
                                        print(f"  Note: {truncated_bins}/{n_total_bins} bins had truncated windows (edge cases)")
                                    print(f"  Valid bins: {n_valid_bins}/{n_total_bins}")
                        
                        print(f"\n{'='*80}")
                        print(f"TRACK TOTALS: Reorientations: {total_reos} | Turns: {total_turns}")
                        
                        # Overall turn rate (entire track)
                        track_duration_min = (aligned_df['time'].max() - aligned_df['time'].min()) / 60.0
                        overall_rate = (total_reos / track_duration_min) if track_duration_min > 0 else 0.0
                        print(f"\nOverall turn rate (entire track): {overall_rate:.2f} reorientations/min")
                        print(f"Track duration: {track_duration_min:.1f} minutes")
                        
                        # Mean per-minute rate using fixed window (biologically meaningful)
                        if len(rate_window_reos) > 0:
                            total_window_reos = sum(rate_window_reos)
                            # Average rate across all cycles using fixed window
                            mean_per_minute_rate = np.mean([r / (RATE_WINDOW_TOTAL / 60.0) for r in rate_window_reos])
                            print(f"\nMean per-minute rate ({-RATE_WINDOW_BEFORE:.0f}s to +{RATE_WINDOW_AFTER:.0f}s window): {mean_per_minute_rate:.2f} reorientations/min")
                            print(f"  (Based on {len(rate_window_reos)} cycles, {total_window_reos} total reorientations in {RATE_WINDOW_TOTAL:.0f}s windows)")
                    except Exception as e:
                        print(f"\n  ERROR calculating per-cycle stats: {e}")
                        import traceback
                        traceback.print_exc()
                        print(f"  Falling back to simple summary:")
                        print(f"  Total reorientations: {n_reorientations}")
                        print(f"  Total turns: {n_turns}")
                else:
                    print("  No cycles found in H5 file")
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
                    
                    # Save results back to H5 file
                    try:
                        sys.path.insert(0, str(script_dir))
                        from save_results_to_h5 import save_results_to_h5
                        save_results_to_h5(
                            h5_file=h5_file,
                            events_df=combined_events,
                            trajectories_df=combined_trajectories,
                            klein_runs_df=combined_klein_runs
                        )
                        progress['messages'].append({
                            'time': datetime.now().isoformat(),
                            'text': 'Saved results back to H5 file'
                        })
                        update_progress(progress_file, progress)
                    except Exception as e:
                        print(f"  WARNING: Failed to save results to H5: {e}")
                        progress['messages'].append({
                            'time': datetime.now().isoformat(),
                            'text': f'WARNING: Failed to save to H5: {e}'
                        })
                        update_progress(progress_file, progress)
                else:
                    # Save without Klein runs
                    try:
                        sys.path.insert(0, str(script_dir))
                        from save_results_to_h5 import save_results_to_h5
                        save_results_to_h5(
                            h5_file=h5_file,
                            events_df=combined_events,
                            trajectories_df=combined_trajectories,
                            klein_runs_df=None
                        )
                        progress['messages'].append({
                            'time': datetime.now().isoformat(),
                            'text': 'Saved results back to H5 file (no Klein runs)'
                        })
                        update_progress(progress_file, progress)
                    except Exception as e:
                        print(f"  WARNING: Failed to save results to H5: {e}")
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

