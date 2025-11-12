#!/usr/bin/env python3
"""
Real-time progress monitor for stimulus-locked analysis.
Updates every 2 seconds with detailed progress information.
"""

import sys
import json
import time
import os
from pathlib import Path
from datetime import datetime, timedelta

# Unicode box drawing characters for graphics
BOX_CHARS = {
    'h': '─', 'v': '│', 'tl': '┌', 'tr': '┐', 'bl': '└', 'br': '┘',
    't': '┬', 'b': '┴', 'l': '├', 'r': '┤', 'c': '┼'
}

def clear_screen():
    """Clear terminal screen."""
    os.system('cls' if os.name == 'nt' else 'clear')

def format_time(seconds):
    """Format seconds as HH:MM:SS."""
    return str(timedelta(seconds=int(seconds)))

def draw_progress_bar(width, pct):
    """Draw a progress bar."""
    filled = int(width * pct / 100)
    bar = '█' * filled + '░' * (width - filled)
    return f"[{bar}] {pct:3.0f}%"

def format_number(num, width=8):
    """Format number with commas and fixed width."""
    return f"{num:>{width},}"

def main():
    import argparse
    parser = argparse.ArgumentParser(description='Monitor analysis progress')
    parser.add_argument('--progress-file', type=str, required=True)
    parser.add_argument('--h5-file', type=str, required=True)
    args = parser.parse_args()
    
    progress_file = Path(args.progress_file)
    h5_file_name = args.h5_file
    
    print("Initializing progress monitor...")
    time.sleep(1)
    
    last_status = None
    while True:
        clear_screen()
        
        # Read progress file
        if not progress_file.exists():
            print("Waiting for analysis to start...")
            time.sleep(2)
            continue
        
        try:
            with open(progress_file, 'r') as f:
                progress = json.load(f)
        except (json.JSONDecodeError, IOError):
            print("Reading progress file...")
            time.sleep(2)
            continue
        
        status = progress.get('status', 'unknown')
        stage = progress.get('stage', 'Unknown stage')
        pct = progress.get('progress_pct', 0)
        elapsed = progress.get('elapsed_time', 0)
        eta = progress.get('eta_seconds', 0)
        current_track = progress.get('current_track', 0)
        total_tracks = progress.get('total_tracks', 0)
        messages = progress.get('messages', [])
        
        # Header
        print("=" * 80)
        print(" " * 25 + "STIMULUS-LOCKED TURN RATE ANALYSIS")
        print("=" * 80)
        print()
        
        # File info
        print(f"File: {h5_file_name}")
        print(f"Experiment ID: {progress.get('experiment_id', 'N/A')}")
        print()
        
        # Status box
        print("┌" + "─" * 78 + "┐")
        status_text = f"Status: {status.upper()}"
        print(f"│ {status_text:<76} │")
        print("├" + "─" * 78 + "┤")
        print(f"│ Stage: {stage:<70} │")
        print("└" + "─" * 78 + "┘")
        print()
        
        # Progress bar
        print("Progress:")
        print(draw_progress_bar(70, pct))
        print()
        
        # Statistics
        print("┌" + "─" * 78 + "┐")
        print(f"│ Elapsed Time:  {format_time(elapsed):<20} │ ETA: {format_time(eta) if eta > 0 else 'N/A':<20} │")
        if total_tracks > 0:
            track_pct = int(100 * current_track / total_tracks) if total_tracks > 0 else 0
            print(f"│ Tracks:        {format_number(current_track)} / {format_number(total_tracks):<20} ({track_pct}%) │")
        print("└" + "─" * 78 + "┘")
        print()
        
        # Additional status info
        if status == 'processing':
            print("Current Operation:")
            print("─" * 80)
            print(f"  {stage}")
            print()
        
        # Messages (last 10)
        if messages:
            print("Recent Activity Log:")
            print("─" * 80)
            for msg in messages[-10:]:
                timestamp = msg.get('time', '')[:19] if 'time' in msg else ''
                text = msg.get('text', '')
                # Format timestamp more compactly
                if timestamp:
                    try:
                        dt = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
                        ts = dt.strftime('%H:%M:%S')
                    except:
                        ts = timestamp[-8:] if len(timestamp) >= 8 else timestamp
                else:
                    ts = ''
                print(f"  [{ts}] {text}")
            print()
        else:
            print("Activity Log:")
            print("─" * 80)
            print("  Waiting for updates...")
            print()
        
        # Output files (if complete)
        if status == 'complete' and 'output_files' in progress:
            print("Output Files:")
            print("─" * 80)
            for key, path in progress['output_files'].items():
                print(f"  {key}: {Path(path).name}")
            print()
        
        # Error (if any)
        if status == 'error' and 'error' in progress:
            print("ERROR:")
            print("─" * 80)
            print(f"  {progress['error']}")
            print()
        
        # Footer
        print("─" * 80)
        print(f"Last update: {datetime.now().strftime('%H:%M:%S')} | Press Ctrl+C to exit")
        
        # Check if complete or error
        if status in ['complete', 'error']:
            if status == 'complete':
                print("\n" + "=" * 80)
                print(" " * 30 + "ANALYSIS COMPLETE!")
                print("=" * 80)
            else:
                print("\n" + "=" * 80)
                print(" " * 30 + "ANALYSIS FAILED!")
                print("=" * 80)
            print("\nThis window will remain open. Press Ctrl+C to close.")
            while True:
                time.sleep(10)
        
        time.sleep(2)  # Update every 2 seconds

if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nProgress monitor closed.")
        sys.exit(0)

