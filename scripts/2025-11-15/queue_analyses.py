#!/usr/bin/env python3
"""
Queue system to run H5 analyses sequentially, one at a time.
Tracks all analyses in a single status file for monitoring.
"""

import json
import os
import subprocess
import time
from pathlib import Path
from datetime import datetime
import sys

QUEUE_STATUS_FILE = Path("data/engineered/analysis_queue_status.json")
PROJECT_ROOT = Path(__file__).parent.parent.parent

def initialize_queue_status():
    """Initialize the queue status file."""
    status = {
        "queue": [],
        "current": None,
        "completed": [],
        "failed": [],
        "start_time": datetime.now().isoformat(),
        "total_files": 0,
        "processed": 0
    }
    QUEUE_STATUS_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(QUEUE_STATUS_FILE, 'w') as f:
        json.dump(status, f, indent=2)
    return status

def load_queue_status():
    """Load the queue status."""
    if QUEUE_STATUS_FILE.exists():
        with open(QUEUE_STATUS_FILE) as f:
            return json.load(f)
    return initialize_queue_status()

def save_queue_status(status):
    """Save the queue status."""
    with open(QUEUE_STATUS_FILE, 'w') as f:
        json.dump(status, f, indent=2)

def get_h5_files(only_failed=False, force_rerun=None):
    """Get all H5 files that need processing.
    
    Args:
        only_failed: If True, only return files that previously failed
        force_rerun: List of filenames (stems) to force rerun even if complete
    """
    h5_dir = PROJECT_ROOT / "data" / "h5_files"
    h5_files = sorted(h5_dir.glob("*.h5"))
    
    # Exclude these experiments (user requested deletion)
    EXCLUDED_EXPERIMENTS = {
        "GMR61_202509051201",
        "GMR61_test",
        "GMR61_tier2_complete"
    }
    
    # If only_failed, load previous queue status to find failed ones
    failed_filenames = set()
    if only_failed:
        status = load_queue_status()
        failed_items = status.get("failed", [])
        failed_filenames = {item.get("filename") for item in failed_items}
        if not failed_filenames:
            return []  # No failed files to requeue
    
    # Filter out already completed ones and excluded experiments
    queue = []
    for h5_file in h5_files:
        filename = h5_file.stem
        
        # Skip excluded experiments
        if filename in EXCLUDED_EXPERIMENTS:
            continue
        
        # If only_failed mode, skip files that didn't fail
        if only_failed and filename not in failed_filenames:
            continue
        
        progress_file = PROJECT_ROOT / "data" / "engineered" / f"{filename}_progress.json"
        
        # Check if already complete (skip if complete and not in failed list, unless force_rerun)
        if progress_file.exists() and (force_rerun is None or filename not in force_rerun):
            try:
                with open(progress_file) as f:
                    progress_data = json.load(f)
                if progress_data.get('status') == 'complete':
                    if filename not in failed_filenames:
                        continue  # Skip completed files (unless they're in failed list or force_rerun)
            except:
                pass
        
        queue.append({
            "h5_file": str(h5_file.relative_to(PROJECT_ROOT)),
            "filename": filename,
            "status": "pending",
            "started_at": None,
            "completed_at": None
        })
    
    return queue

def run_analysis(h5_file_path):
    """Run a single analysis."""
    script_path = PROJECT_ROOT / "scripts" / "queue" / "run_stimulus_locked_analysis_production.py"
    
    # Run analysis (without launching monitor - we'll use the queue monitor)
    cmd = [
        sys.executable,
        str(script_path),
        str(h5_file_path)
    ]
    
    # Set environment variable to disable auto-launch of monitor
    env = dict(os.environ) if 'os' in dir() else {}
    env['DISABLE_AUTO_MONITOR'] = '1'
    
    process = subprocess.Popen(
        cmd,
        cwd=str(PROJECT_ROOT),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        env=env
    )
    
    return process

def main():
    """Main queue processing loop."""
    
    # Launch queue monitor first
    monitor_script = Path(__file__).parent / "queue_monitor.py"
    if monitor_script.exists():
        import subprocess
        import platform
        project_root = Path(__file__).parent.parent.parent
        
        if platform.system() == 'Darwin':
            osascript_cmd = f'''
tell application "Terminal"
    activate
    do script "cd '{project_root}' && python3 '{monitor_script}' --refresh-interval 0.5"
end tell
'''
            subprocess.Popen(['osascript', '-e', osascript_cmd])
            time.sleep(2)  # Give monitor time to launch
    
    # Initialize queue
    status = initialize_queue_status()
    
    # Check if we should only requeue failed ones
    import sys
    only_failed = "--failed-only" in sys.argv or "-f" in sys.argv
    
    # Check for force-rerun argument
    force_rerun = None
    if "--force-rerun" in sys.argv:
        force_rerun_idx = sys.argv.index("--force-rerun")
        if force_rerun_idx + 1 < len(sys.argv):
            force_rerun = set(sys.argv[force_rerun_idx + 1:])
            # Stop at next argument or end
            next_arg_idx = len(sys.argv)
            for i in range(force_rerun_idx + 1, len(sys.argv)):
                if sys.argv[i].startswith("--"):
                    next_arg_idx = i
                    break
            force_rerun = set(sys.argv[force_rerun_idx + 1:next_arg_idx])
    
    h5_files = get_h5_files(only_failed=only_failed, force_rerun=force_rerun)
    
    if not h5_files:
        if only_failed:
            print("No failed files to requeue")
        else:
            print("No H5 files to process (all may be complete)")
        return
    
    if only_failed:
        print(f"Requeuing {len(h5_files)} failed analyses...")
    
    status["queue"] = h5_files
    status["total_files"] = len(h5_files)
    status["failed"] = []  # Clear failed list since we're requeuing them
    save_queue_status(status)
    
    print(f"Queued {len(h5_files)} H5 files for processing")
    print("Starting sequential processing...")
    print("Monitor the queue status in the Cinnamoroll monitor window")
    print()
    
    # Process each file sequentially
    for i, file_info in enumerate(h5_files, 1):
        h5_file = PROJECT_ROOT / file_info["h5_file"]
        filename = file_info["filename"]
        
        print(f"[{i}/{len(h5_files)}] Processing: {h5_file.name}")
        
        # Update status
        status["current"] = {
            "h5_file": file_info["h5_file"],
            "filename": filename,
            "started_at": datetime.now().isoformat(),
            "index": i,
            "total": len(h5_files)
        }
        file_info["status"] = "processing"
        file_info["started_at"] = datetime.now().isoformat()
        save_queue_status(status)
        
        # Run analysis and capture output
        script_path = PROJECT_ROOT / "scripts" / "queue" / "run_stimulus_locked_analysis_production.py"
        cmd = [sys.executable, str(script_path), str(h5_file)]
        
        # Disable auto-monitor launch
        env = os.environ.copy()
        env['DISABLE_AUTO_MONITOR'] = '1'
        
        # Log file for console output
        log_file = PROJECT_ROOT / "data" / "engineered" / f"{filename}_console.log"
        
        # Run with output capture
        with open(log_file, 'w') as log:
            process = subprocess.run(
                cmd,
                cwd=str(PROJECT_ROOT),
                env=env,
                stdout=log,
                stderr=subprocess.STDOUT,
                text=True
            )
        
        # Update status based on result
        if process.returncode == 0:
            file_info["status"] = "complete"
            file_info["completed_at"] = datetime.now().isoformat()
            status["completed"].append(file_info)
            status["processed"] += 1
            print(f"  [OK] Completed successfully")
        else:
            file_info["status"] = "failed"
            file_info["completed_at"] = datetime.now().isoformat()
            status["failed"].append(file_info)
            print(f"  [FAIL] Failed with return code {process.returncode}")
        
        status["current"] = None
        save_queue_status(status)
        print()
    
    print("═══════════════════════════════════════════════════════════════")
    print(f"Queue processing complete!")
    print(f"  Processed: {status['processed']}/{status['total_files']}")
    print(f"  Completed: {len(status['completed'])}")
    print(f"  Failed: {len(status['failed'])}")
    print("═══════════════════════════════════════════════════════════════")

if __name__ == "__main__":
    main()

