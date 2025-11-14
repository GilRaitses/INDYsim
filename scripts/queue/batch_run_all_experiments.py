#!/usr/bin/env python3
"""
Batch process all H5 files through the stimulus-locked analysis pipeline.
"""

import sys
import subprocess
from pathlib import Path
from datetime import datetime

def main():
    # Get all H5 files
    h5_dir = Path("data/h5_files")
    if not h5_dir.exists():
        print(f"ERROR: H5 directory not found: {h5_dir}")
        sys.exit(1)
    
    h5_files = sorted(h5_dir.glob("*.h5"))
    
    if len(h5_files) == 0:
        print(f"ERROR: No H5 files found in {h5_dir}")
        sys.exit(1)
    
    print(f"\n{'='*80}")
    print(f"BATCH PROCESSING: {len(h5_files)} H5 FILES")
    print(f"{'='*80}\n")
    
    script_dir = Path(__file__).parent
    analysis_script = script_dir / "run_stimulus_locked_analysis_production.py"
    
    successful = []
    failed = []
    
    for idx, h5_file in enumerate(h5_files, 1):
        print(f"\n{'='*80}")
        print(f"PROCESSING FILE {idx}/{len(h5_files)}: {h5_file.name}")
        print(f"{'='*80}\n")
        
        start_time = datetime.now()
        
        try:
            # Modify the script to accept H5 file as argument
            # We'll use a temporary modified version or pass via environment variable
            import os
            os.environ['H5_FILE_PATH'] = str(h5_file.absolute())
            
            # Run the analysis script with H5 file as argument
            result = subprocess.run(
                [sys.executable, str(analysis_script), str(h5_file.absolute())],
                cwd=script_dir.parent.parent,  # Project root
                capture_output=False,  # Show output in real-time
                text=True
            )
            
            elapsed = datetime.now() - start_time
            
            if result.returncode == 0:
                successful.append((h5_file.name, elapsed))
                print(f"\n✅ SUCCESS: {h5_file.name} completed in {elapsed}")
            else:
                failed.append((h5_file.name, elapsed, result.returncode))
                print(f"\n❌ FAILED: {h5_file.name} (exit code: {result.returncode})")
        
        except Exception as e:
            elapsed = datetime.now() - start_time
            failed.append((h5_file.name, elapsed, str(e)))
            print(f"\n❌ ERROR: {h5_file.name} - {e}")
    
    # Summary
    print(f"\n{'='*80}")
    print(f"BATCH PROCESSING COMPLETE")
    print(f"{'='*80}\n")
    print(f"Successful: {len(successful)}/{len(h5_files)}")
    print(f"Failed: {len(failed)}/{len(h5_files)}\n")
    
    if successful:
        print("Successful files:")
        for name, elapsed in successful:
            print(f"  ✅ {name} ({elapsed})")
    
    if failed:
        print("\nFailed files:")
        for name, elapsed, error in failed:
            print(f"  ❌ {name} ({elapsed}) - {error}")

if __name__ == "__main__":
    main()

