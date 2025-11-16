#!/usr/bin/env python3
"""Queue monitor that tracks all analyses in one window."""

import json
import time
from pathlib import Path
from datetime import datetime
from zoneinfo import ZoneInfo
import sys

# Import the monitor components
sys.path.insert(0, str(Path(__file__).parent))
from cinnamoroll_monitor import (
    Colors, FloatingCloud, create_clouds, render_clouds_on_canvas,
    SpeckSystem, CinnamorollSystem, load_cloud_art
)

QUEUE_STATUS_FILE = Path("data/engineered/analysis_queue_status.json")
PROJECT_ROOT = Path(__file__).parent.parent.parent

def load_queue_status():
    """Load queue status."""
    if QUEUE_STATUS_FILE.exists():
        try:
            with open(QUEUE_STATUS_FILE) as f:
                data = json.load(f)
                # Ensure all required keys exist
                if "current" not in data:
                    data["current"] = None
                if "queue" not in data:
                    data["queue"] = []
                if "completed" not in data:
                    data["completed"] = []
                if "failed" not in data:
                    data["failed"] = []
                if "total_files" not in data:
                    data["total_files"] = len(data.get("queue", []))
                if "processed" not in data:
                    data["processed"] = 0
                return data
        except (json.JSONDecodeError, IOError, ValueError) as e:
            # If file is corrupted or being written, return empty status
            return {
                "queue": [],
                "current": None,
                "completed": [],
                "failed": [],
                "total_files": 0,
                "processed": 0
            }
    return {
        "queue": [],
        "current": None,
        "completed": [],
        "failed": [],
        "total_files": 0,
        "processed": 0
    }

def get_current_progress():
    """Get progress from current analysis if running."""
    status = load_queue_status()
    if not status.get("current"):
        return 0.0
    
    current_file = status["current"]
    filename = current_file["filename"]
    progress_file = PROJECT_ROOT / "data" / "engineered" / f"{filename}_progress.json"
    
    if progress_file.exists():
        try:
            with open(progress_file) as f:
                data = json.load(f)
            return data.get("progress_pct", 0.0)
        except:
            pass
    
    return 0.0

def monitor_queue(refresh_interval: float = 0.1):
    """Monitor the analysis queue."""
    terminal_width = 120  # Wider for console output
    terminal_height = 35  # Reduced height for more compact display
    
    # Load clouds
    assets_dir = PROJECT_ROOT / "src" / "indysim" / "monitors" / "assets"
    clouds = create_clouds(assets_dir)
    last_cloud_update = time.time()
    
    # Cinnamoroll system
    cinnamoroll_system = CinnamorollSystem(terminal_width, terminal_height)
    
    # Speck system for swirling background
    speck_system = SpeckSystem(terminal_width, terminal_height)
    
    start_time = time.time()
    last_update = start_time
    
    try:
        while True:
            current_time = time.time()
            dt = current_time - last_update
            last_update = current_time
            
            # Load queue status
            status = load_queue_status()
            
            # Calculate overall progress
            total = status.get("total_files", 1)
            processed = len(status.get("completed", []))
            current_progress = get_current_progress()
            
            # Overall progress: completed files + current file progress
            if total > 0:
                overall_progress = ((processed + (current_progress / 100.0)) / total) * 100.0
            else:
                overall_progress = 0.0
            
            # Update systems
            cinnamoroll_system.update(dt, overall_progress)
            speck_system.update(dt, overall_progress)
            
            # Clear screen
            print(Colors.CLEAR, end='')
            
            # Header with preamble
            print(Colors.BOLD + Colors.BLUE + "=" * terminal_width + Colors.RESET)
            print(Colors.BOLD + Colors.CREAM + "  ANALYSIS QUEUE MONITOR" + Colors.RESET)
            print(Colors.BOLD + Colors.BLUE + "=" * terminal_width + Colors.RESET)
            print(Colors.CREAM + "Press Ctrl+C to stop" + Colors.RESET)
            print(Colors.BOLD + Colors.BLUE + "=" * terminal_width + Colors.RESET)
            
            # Time and Queue status (compact)
            time_str = datetime.now(ZoneInfo("America/New_York")).strftime("%Y-%m-%d %H:%M:%S %Z")
            print(Colors.CREAM + f"Time: {time_str}" + Colors.RESET + " | " + 
                  Colors.MINT + f"Overall Progress: {overall_progress:.1f}%" + Colors.RESET + " | " +
                  Colors.PINK + f"Processed: {processed}/{total} files" + Colors.RESET)
            
            # Current analysis with track matrix
            current = status.get("current")
            if current:
                filename = current["filename"]
                progress_file = PROJECT_ROOT / "data" / "engineered" / f"{filename}_progress.json"
                stage = "Unknown"
                if progress_file.exists():
                    try:
                        with open(progress_file) as f:
                            prog_data = json.load(f)
                        stage = prog_data.get('stage', 'Processing...')
                    except:
                        pass
                
                print(Colors.BOLD + Colors.BLUE + "Current Experiment:" + Colors.RESET + " " +
                      Colors.CREAM + f"{Path(current['h5_file']).name}" + Colors.RESET)
                print(Colors.CREAM + f"  Progress: {current_progress:.1f}% | Stage: {stage}" + Colors.RESET)
                
                # Get detailed track info
                if progress_file.exists():
                    try:
                        with open(progress_file) as f:
                            prog_data = json.load(f)
                        
                        current_track = prog_data.get('current_track', 0)
                        total_tracks = prog_data.get('total_tracks', 0)
                        
                        # Track statistics matrix - sliding window of 12 tracks
                        if total_tracks > 0:
                            print(Colors.BOLD + Colors.BLUE + "Track Progress Matrix:" + Colors.RESET)
                            
                            # Parse console log for track-level stats
                            log_file = PROJECT_ROOT / "data" / "engineered" / f"{filename}_console.log"
                            track_stats = {}  # track_num -> stats dict
                            
                            # Track which experiment we're monitoring (for reset detection)
                            if not hasattr(monitor_queue, 'last_experiment'):
                                monitor_queue.last_experiment = None
                            
                            # Reset window if new experiment started
                            if monitor_queue.last_experiment != filename:
                                monitor_queue.last_experiment = filename
                                monitor_queue.track_window_start = 1  # Reset to track 1
                            
                            if log_file.exists():
                                try:
                                    import re
                                    with open(log_file) as f:
                                        log_lines = f.readlines()
                                    
                                    # Parse TRACK_STATS lines for comprehensive stats
                                    for line in log_lines:
                                        if 'TRACK_STATS:' in line:
                                            # Parse: TRACK_STATS: Track X | Frames: Y | Reorientations: Z | ...
                                            track_match = re.search(r'Track\s+(\d+)', line)
                                            if track_match:
                                                track_num = int(track_match.group(1))
                                                
                                                # Initialize if not exists
                                                if track_num not in track_stats:
                                                    track_stats[track_num] = {
                                                        'status': '[PENDING]',
                                                        'duration_mmss': '00:00',
                                                        'start_mmss': '00:00',
                                                        'end_mmss': '00:00',
                                                        'cycles': 0,
                                                        'frames': 0,
                                                        'reorientations': 0,
                                                        'turns': 0,
                                                        'turn_rate': 0.0,
                                                        'pauses': 0,
                                                        'mean_pause_dur': 0.0,
                                                        'headswings': 0
                                                    }
                                                
                                                # Extract all stats from TRACK_STATS line
                                                # Parse duration, start, end, cycles first (new fields)
                                                duration_match = re.search(r'Duration:\s+([\d:]+)', line)
                                                if duration_match:
                                                    track_stats[track_num]['duration_mmss'] = duration_match.group(1)
                                                
                                                start_match = re.search(r'Start:\s+([\d:]+)', line)
                                                if start_match:
                                                    track_stats[track_num]['start_mmss'] = start_match.group(1)
                                                
                                                end_match = re.search(r'End:\s+([\d:]+)', line)
                                                if end_match:
                                                    track_stats[track_num]['end_mmss'] = end_match.group(1)
                                                
                                                cycles_match = re.search(r'Cycles:\s+(\d+)', line)
                                                if cycles_match:
                                                    track_stats[track_num]['cycles'] = int(cycles_match.group(1))
                                                
                                                frames_match = re.search(r'Frames:\s+(\d+)', line)
                                                if frames_match:
                                                    track_stats[track_num]['frames'] = int(frames_match.group(1))
                                                
                                                reo_match = re.search(r'Reorientations:\s+(\d+)', line)
                                                if reo_match:
                                                    track_stats[track_num]['reorientations'] = int(reo_match.group(1))
                                                
                                                turns_match = re.search(r'Turns:\s+(\d+)', line)
                                                if turns_match:
                                                    track_stats[track_num]['turns'] = int(turns_match.group(1))
                                                
                                                turnrate_match = re.search(r'TurnRate:\s+([\d.]+)', line)
                                                if turnrate_match:
                                                    track_stats[track_num]['turn_rate'] = float(turnrate_match.group(1))
                                                
                                                pauses_match = re.search(r'Pauses:\s+(\d+)', line)
                                                if pauses_match:
                                                    track_stats[track_num]['pauses'] = int(pauses_match.group(1))
                                                
                                                pausedur_match = re.search(r'MeanPauseDur:\s+([\d.]+)', line)
                                                if pausedur_match:
                                                    track_stats[track_num]['mean_pause_dur'] = float(pausedur_match.group(1))
                                                
                                                # Don't overwrite headswings from TRACK_STATS (it's always 0)
                                                # MAGAT parsing happens separately and has the correct value
                                                # We preserve any existing headswings value from MAGAT
                                                # (TRACK_STATS line has HeadSwings: 0 because analysis script doesn't populate it)
                                                # So we skip updating headswings from TRACK_STATS
                                                
                                                # Mark as complete if we have stats
                                                track_stats[track_num]['status'] = '[OK]'
                                        
                                        # Also parse MAGAT segmentation line for head swings
                                        # Format: "MAGAT segmentation: X runs, Y head swings, Z reorientations"
                                        # MAGAT segmentation happens during feature extraction
                                        # Track number appears AFTER MAGAT in "Track X/Y: Extracted..." line
                                        if 'MAGAT segmentation' in line and 'head swing' in line.lower():
                                            # Find the track number from nearby lines
                                            line_idx = log_lines.index(line)
                                            track_num = None
                                            
                                            # Strategy 1: Look FORWARD (track number comes after MAGAT)
                                            for forward_idx in range(line_idx + 1, min(len(log_lines), line_idx + 10)):
                                                forward_line = log_lines[forward_idx]
                                                track_match = re.search(r'Track\s+(\d+)[/\s]', forward_line)
                                                if track_match:
                                                    track_num = int(track_match.group(1))
                                                    break
                                            
                                            # Strategy 2: Look BACKWARD (in case "Processing Track" is there)
                                            if not track_num:
                                                for back_idx in range(max(0, line_idx - 50), line_idx):
                                                    back_line = log_lines[back_idx]
                                                    # Try "Processing Track" first
                                                    track_match = re.search(r'Processing Track\s+(\d+)', back_line)
                                                    if track_match:
                                                        track_num = int(track_match.group(1))
                                                        break
                                                    # Try "Track X/Y:"
                                                    if not track_match:
                                                        track_match = re.search(r'Track\s+(\d+)[/\s]', back_line)
                                                    if track_match:
                                                        track_num = int(track_match.group(1))
                                                        break
                                            
                                            if track_num:
                                                # Initialize if not exists
                                                if track_num not in track_stats:
                                                    track_stats[track_num] = {
                                                        'status': '[PENDING]',
                                                        'frames': 0,
                                                        'reorientations': 0,
                                                        'turns': 0,
                                                        'turn_rate': 0.0,
                                                        'pauses': 0,
                                                        'mean_pause_dur': 0.0,
                                                        'headswings': 0
                                                    }
                                                # Extract head swings count
                                                hs_match = re.search(r'(\d+)\s+head\s+swings?', line, re.IGNORECASE)
                                                if hs_match:
                                                    track_stats[track_num]['headswings'] = int(hs_match.group(1))
                                        
                                        # Also check for processing status
                                        track_match = re.search(r'track[_\s]*(\d+)[/\s]', line, re.IGNORECASE)
                                        if track_match:
                                            track_num = int(track_match.group(1))
                                            if track_num not in track_stats:
                                                track_stats[track_num] = {
                                                    'status': '[PENDING]',
                                                    'frames': 0,
                                                    'duration_mmss': '00:00',
                                                    'start_mmss': '00:00',
                                                    'end_mmss': '00:00',
                                                    'cycles': 0,
                                                    'reorientations': 0,
                                                    'turns': 0,
                                                    'turn_rate': 0.0,
                                                    'pauses': 0,
                                                    'mean_pause_dur': 0.0,
                                                    'headswings': 0
                                                }
                                            
                                            if 'extracted' in line.lower() or 'aligned' in line.lower() or 'created' in line.lower():
                                                track_stats[track_num]['status'] = '[RUNNING]'
                                            elif 'error' in line.lower() or 'fail' in line.lower():
                                                track_stats[track_num]['status'] = '[FAIL]'
                                
                                except Exception as e:
                                    pass
                            
                            # Sliding window: show max 12 tracks, centered around current track
                            MAX_TRACKS_DISPLAY = 12
                            
                            # Determine window start
                            if not hasattr(monitor_queue, 'track_window_start'):
                                monitor_queue.track_window_start = 1
                            
                            # Adjust window to keep current track visible
                            if current_track > 0:
                                # Center window around current track, but don't go below 1
                                ideal_start = max(1, current_track - MAX_TRACKS_DISPLAY // 2)
                                # But if we're near the end, show the last MAX_TRACKS_DISPLAY tracks
                                if current_track + MAX_TRACKS_DISPLAY // 2 > total_tracks:
                                    ideal_start = max(1, total_tracks - MAX_TRACKS_DISPLAY + 1)
                                
                                # Slide window forward as processing progresses
                                if current_track >= monitor_queue.track_window_start + MAX_TRACKS_DISPLAY:
                                    monitor_queue.track_window_start = max(1, current_track - MAX_TRACKS_DISPLAY + 1)
                                elif current_track < monitor_queue.track_window_start:
                                    monitor_queue.track_window_start = max(1, current_track)
                            
                            window_start = monitor_queue.track_window_start
                            window_end = min(window_start + MAX_TRACKS_DISPLAY, total_tracks + 1)
                            tracks_to_show = list(range(window_start, window_end))
                            
                            if len(tracks_to_show) > 0:
                                # Fixed column width for alignment (7 chars per column)
                                COL_WIDTH = 7
                                # Header width for consistent vertical alignment (match longest header)
                                HEADER_WIDTH = 14
                                
                                # Track numbers header
                                track_nums = [f"{i:>{COL_WIDTH}d}" for i in tracks_to_show]
                                print(Colors.MINT + f"  {'Track:':<{HEADER_WIDTH}s}" + " ".join(track_nums) + Colors.RESET)
                                
                                # Duration (mm:ss) - replaces frames at top
                                duration_row = []
                                for i in tracks_to_show:
                                    if i in track_stats and track_stats[i].get('duration_mmss'):
                                        duration_row.append(f"{track_stats[i]['duration_mmss']:>{COL_WIDTH}s}")
                                    else:
                                        duration_row.append(f"{'-':>{COL_WIDTH}s}")
                                print(Colors.CREAM + f"  {'Duration:':<{HEADER_WIDTH}s}" + " ".join(duration_row) + Colors.RESET)
                                
                                # Start time (mm:ss)
                                start_row = []
                                for i in tracks_to_show:
                                    if i in track_stats and track_stats[i].get('start_mmss'):
                                        start_row.append(f"{track_stats[i]['start_mmss']:>{COL_WIDTH}s}")
                                    else:
                                        start_row.append(f"{'-':>{COL_WIDTH}s}")
                                print(Colors.MINT + f"  {'Start:':<{HEADER_WIDTH}s}" + " ".join(start_row) + Colors.RESET)
                                
                                # End time (mm:ss)
                                end_row = []
                                for i in tracks_to_show:
                                    if i in track_stats and track_stats[i].get('end_mmss'):
                                        end_row.append(f"{track_stats[i]['end_mmss']:>{COL_WIDTH}s}")
                                    else:
                                        end_row.append(f"{'-':>{COL_WIDTH}s}")
                                print(Colors.MINT + f"  {'End:':<{HEADER_WIDTH}s}" + " ".join(end_row) + Colors.RESET)
                                
                                # Cycles count
                                cycles_row = []
                                for i in tracks_to_show:
                                    if i in track_stats and track_stats[i].get('cycles', 0) >= 0:
                                        cycles_row.append(f"{track_stats[i]['cycles']:>{COL_WIDTH}d}")
                                    else:
                                        cycles_row.append(f"{'-':>{COL_WIDTH}s}")
                                print(Colors.BLUE + f"  {'Cycles:':<{HEADER_WIDTH}s}" + " ".join(cycles_row) + Colors.RESET)
                                
                                # Frames
                                frames_row = []
                                for i in tracks_to_show:
                                    if i in track_stats and track_stats[i]['frames'] >= 0:
                                        frames_row.append(f"{track_stats[i]['frames']:>{COL_WIDTH}d}")
                                    else:
                                        frames_row.append(f"{'-':>{COL_WIDTH}s}")
                                print(Colors.MINT + f"  {'Frames:':<{HEADER_WIDTH}s}" + " ".join(frames_row) + Colors.RESET)
                                
                                # Reorientations
                                reo_row = []
                                for i in tracks_to_show:
                                    if i in track_stats and track_stats[i]['reorientations'] >= 0:
                                        reo_row.append(f"{track_stats[i]['reorientations']:>{COL_WIDTH}d}")
                                    else:
                                        reo_row.append(f"{'-':>{COL_WIDTH}s}")
                                print(Colors.BLUE + f"  {'Reos:':<{HEADER_WIDTH}s}" + " ".join(reo_row) + Colors.RESET)
                                
                                # Num Turns
                                turns_row = []
                                for i in tracks_to_show:
                                    if i in track_stats and track_stats[i]['turns'] >= 0:
                                        turns_row.append(f"{track_stats[i]['turns']:>{COL_WIDTH}d}")
                                    else:
                                        turns_row.append(f"{'-':>{COL_WIDTH}s}")
                                print(Colors.PINK + f"  {'Turns:':<{HEADER_WIDTH}s}" + " ".join(turns_row) + Colors.RESET)
                                
                                # Mean Turn Rate
                                turnrate_row = []
                                for i in tracks_to_show:
                                    if i in track_stats and track_stats[i]['turn_rate'] >= 0:
                                        turnrate_row.append(f"{track_stats[i]['turn_rate']:>{COL_WIDTH}.2f}")
                                    else:
                                        turnrate_row.append(f"{'-':>{COL_WIDTH}s}")
                                print(Colors.CREAM + f"  {'TurnRate:':<{HEADER_WIDTH}s}" + " ".join(turnrate_row) + Colors.RESET)
                                
                                # Mean Pause Duration
                                pausedur_row = []
                                for i in tracks_to_show:
                                    if i in track_stats and track_stats[i]['mean_pause_dur'] >= 0:
                                        pausedur_row.append(f"{track_stats[i]['mean_pause_dur']:>{COL_WIDTH}.3f}")
                                    else:
                                        pausedur_row.append(f"{'-':>{COL_WIDTH}s}")
                                print(Colors.MINT + f"  {'MeanPauseDur:':<{HEADER_WIDTH}s}" + " ".join(pausedur_row) + Colors.RESET)
                                
                                # Num Pauses
                                pauses_row = []
                                for i in tracks_to_show:
                                    if i in track_stats and track_stats[i]['pauses'] >= 0:
                                        pauses_row.append(f"{track_stats[i]['pauses']:>{COL_WIDTH}d}")
                                    else:
                                        pauses_row.append(f"{'-':>{COL_WIDTH}s}")
                                print(Colors.BLUE + f"  {'Pauses:':<{HEADER_WIDTH}s}" + " ".join(pauses_row) + Colors.RESET)
                                
                                # Num HeadSwings
                                hs_row = []
                                for i in tracks_to_show:
                                    if i in track_stats and track_stats[i]['headswings'] >= 0:
                                        hs_row.append(f"{track_stats[i]['headswings']:>{COL_WIDTH}d}")
                                    else:
                                        hs_row.append(f"{'-':>{COL_WIDTH}s}")
                                print(Colors.PINK + f"  {'HeadSwings:':<{HEADER_WIDTH}s}" + " ".join(hs_row) + Colors.RESET)
                                
                                # Show window info if not showing all tracks
                                if total_tracks > MAX_TRACKS_DISPLAY:
                                    print(Colors.CREAM + f"  Showing tracks {window_start}-{window_end-1} of {total_tracks} (sliding window)" + Colors.RESET)
                    except Exception as e:
                        print(Colors.CREAM + f"  Error loading progress: {e}" + Colors.RESET)
            else:
                if processed >= total:
                    print(Colors.BOLD + Colors.GREEN + "[OK] All analyses complete!" + Colors.RESET)
                else:
                    print(Colors.CREAM + "Waiting for next analysis..." + Colors.RESET)
            
            # Console output feed (9 lines)
            console_lines = []
            if current:
                filename = current["filename"]
                log_file = PROJECT_ROOT / "data" / "engineered" / f"{filename}_console.log"
                if log_file.exists():
                    try:
                        with open(log_file) as f:
                            all_lines = f.readlines()
                        # Get last 9 lines for display
                        console_lines = [line.rstrip('\n') for line in all_lines[-9:]]
                    except:
                        pass
            
            # Update clouds
            dt_clouds = current_time - last_cloud_update
            last_cloud_update = current_time
            for cloud in clouds:
                cloud.update(dt_clouds, terminal_width, terminal_height)
            
            # Render specks (swirling background)
            bg_speck_canvas, progress_speck_canvas = speck_system.render(overall_progress)
            
            # Render clouds
            cloud_canvas = [[' ' for _ in range(terminal_width)] for _ in range(terminal_height)]
            render_clouds_on_canvas(clouds, cloud_canvas, terminal_width, terminal_height)
            
            # Render Cinnamoroll
            cinnamoroll_canvas = cinnamoroll_system.render()
            
            # Combine layers
            combined_canvas = [[' ' for _ in range(terminal_width)] 
                              for _ in range(terminal_height)]
            
            # Render clouds
            render_clouds_on_canvas(clouds, combined_canvas, terminal_width, terminal_height)
            
            # Render background specks (swirling pattern)
            progress_width = int((overall_progress / 100.0) * terminal_width)
            for y in range(min(len(bg_speck_canvas), terminal_height)):
                line = bg_speck_canvas[y]
                x_pos = 0
                i = 0
                current_color = ''
                while i < len(line) and x_pos < terminal_width:
                    if line[i] == '\033':
                        ansi_start = i
                        while i < len(line) and line[i] != 'm':
                            i += 1
                        if i < len(line):
                            current_color = line[ansi_start:i+1]
                            i += 1
                        continue
                    
                    char = line[i]
                    if char != ' ' and x_pos >= progress_width:
                        combined_canvas[y][x_pos] = current_color + char + Colors.RESET
                        x_pos += 1
                    elif char == ' ':
                        x_pos += 1
                    else:
                        x_pos += 1
                    i += 1
            
            # Render progress specks
            for y in range(min(len(progress_speck_canvas), terminal_height)):
                line = progress_speck_canvas[y]
                x_pos = 0
                i = 0
                current_color = ''
                while i < len(line) and x_pos < progress_width:
                    if line[i] == '\033':
                        ansi_start = i
                        while i < len(line) and line[i] != 'm':
                            i += 1
                        if i < len(line):
                            current_color = line[ansi_start:i+1]
                            i += 1
                        continue
                    
                    char = line[i]
                    if char != ' ' and x_pos < progress_width:
                        combined_canvas[y][x_pos] = current_color + char + Colors.RESET
                        x_pos += 1
                    elif char == ' ':
                        if x_pos < progress_width:
                            x_pos += 1
                    i += 1
            
            # Render Cinnamoroll
            for y in range(min(len(cinnamoroll_canvas), terminal_height)):
                line = cinnamoroll_canvas[y]
                x_pos = 0
                i = 0
                current_color = ''
                
                cinnamoroll_chars = []
                while i < len(line):
                    if line[i] == '\033':
                        ansi_start = i
                        while i < len(line) and line[i] != 'm':
                            i += 1
                        if i < len(line):
                            current_color = line[ansi_start:i+1]
                            i += 1
                        continue
                    
                    char = line[i]
                    if char != ' ':
                        cinnamoroll_chars.append((x_pos, char, current_color))
                    x_pos += 1
                    i += 1
                
                for x_pos, char, color in cinnamoroll_chars:
                    if 0 <= x_pos < terminal_width:
                        combined_canvas[y][x_pos] = color + char + Colors.RESET
            
            # Calculate how much space we have for animation vs console
            # Reserve bottom 13 lines for console output (9 lines + 4 for separators)
            animation_height = terminal_height - 13
            
            # Print animation canvas (top portion)
            for y in range(min(animation_height, len(combined_canvas))):
                line = ''.join(combined_canvas[y])
                print(line)
            
            # Console output feed below animation (9 lines)
            if console_lines:
                print(Colors.BOLD + Colors.BLUE + "-" * terminal_width + Colors.RESET)
                print(Colors.BOLD + Colors.CREAM + "Console Output:" + Colors.RESET)
                for line in console_lines[-9:]:  # Show last 9 lines
                    # Truncate long lines to fit terminal width
                    if len(line) > terminal_width - 2:
                        line = line[:terminal_width - 5] + "..."
                    print(Colors.CREAM + line + Colors.RESET)
                print(Colors.BLUE + "-" * terminal_width + Colors.RESET)
            
            # Footer
            print(Colors.BOLD + Colors.BLUE + "=" * terminal_width + Colors.RESET)
            print(Colors.CREAM + "Press Ctrl+C to stop" + Colors.RESET)
            print(Colors.BOLD + Colors.BLUE + "=" * terminal_width + Colors.RESET)
            
            # Check if all complete
            if status.get("current") is None and processed >= total and total > 0:
                print()
                print(Colors.BOLD + Colors.GREEN + "[DONE] All analyses complete!" + Colors.RESET)
                time.sleep(5)
                break
            
            time.sleep(refresh_interval)
            
    except KeyboardInterrupt:
        print("\n\n" + Colors.BOLD + Colors.PINK + "Monitoring stopped!" + Colors.RESET)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Queue monitor for analysis pipeline")
    parser.add_argument("--refresh-interval", type=float, default=0.1, help="Refresh interval")
    args = parser.parse_args()
    monitor_queue(args.refresh_interval)

