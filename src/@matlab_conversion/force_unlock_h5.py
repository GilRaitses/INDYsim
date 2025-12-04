#!/usr/bin/env python3
"""
Force unlock H5 file by closing Python/MATLAB processes and attempting deletion.
"""

import sys
import subprocess
import os
import time
from pathlib import Path

def find_locking_processes(file_path: Path):
    """Find processes that might be locking the file."""
    try:
        # Use handle.exe or similar tools if available
        # For now, just check common processes
        result = subprocess.run(
            ['tasklist', '/FI', 'IMAGENAME eq python.exe', '/FO', 'CSV'],
            capture_output=True,
            text=True
        )
        python_procs = []
        if 'python.exe' in result.stdout:
            for line in result.stdout.split('\n')[1:]:
                if line.strip():
                    parts = line.split(',')
                    if len(parts) > 1:
                        pid = parts[1].strip('"')
                        python_procs.append(pid)
        
        result = subprocess.run(
            ['tasklist', '/FI', 'IMAGENAME eq MATLAB.exe', '/FO', 'CSV'],
            capture_output=True,
            text=True
        )
        matlab_procs = []
        if 'MATLAB.exe' in result.stdout:
            for line in result.stdout.split('\n')[1:]:
                if line.strip():
                    parts = line.split(',')
                    if len(parts) > 1:
                        pid = parts[1].strip('"')
                        matlab_procs.append(pid)
        
        return python_procs, matlab_procs
    except:
        return [], []

def force_unlock(file_path: Path):
    """Attempt to force unlock a file."""
    print(f"Attempting to force unlock: {file_path.name}")
    print()
    
    # Find locking processes
    python_procs, matlab_procs = find_locking_processes(file_path)
    
    if python_procs:
        print(f"Found {len(python_procs)} Python processes:")
        for pid in python_procs:
            print(f"  PID: {pid}")
    
    if matlab_procs:
        print(f"Found {len(matlab_procs)} MATLAB processes:")
        for pid in matlab_procs:
            print(f"  PID: {pid}")
    
    print()
    print("Attempting to close processes...")
    print("(This may interrupt running scripts)")
    
    # Try to close Python processes
    for pid in python_procs:
        try:
            subprocess.run(['taskkill', '/F', '/PID', pid], 
                         capture_output=True, timeout=5)
            print(f"  Closed Python process {pid}")
        except:
            pass
    
    # Try to close MATLAB processes (be more careful)
    print()
    print("MATLAB processes found - please close MATLAB manually if needed")
    print("(Closing MATLAB may cause data loss)")
    
    # Wait a moment
    time.sleep(2)
    
    # Try to delete/move the file
    print()
    print("Attempting to move file...")
    try:
        target = file_path.parent / "h5_files" / file_path.name
        target.parent.mkdir(exist_ok=True)
        
        # Try multiple times
        for attempt in range(5):
            try:
                file_path.rename(target)
                print(f"[SUCCESS] File moved to: {target}")
                return True
            except PermissionError:
                print(f"  Attempt {attempt + 1}/5: Still locked, waiting...")
                time.sleep(2)
            except Exception as e:
                print(f"  Error: {e}")
                break
        
        print("[ERROR] File still locked after multiple attempts")
        print()
        print("Manual steps:")
        print("1. Close all MATLAB windows")
        print("2. Close all Python scripts/terminals")
        print("3. Close File Explorer if previewing the file")
        print("4. Try moving manually:")
        print(f"   Move-Item '{file_path}' 'h5_files\\' -Force")
        
        return False
        
    except Exception as e:
        print(f"[ERROR] {e}")
        return False

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python force_unlock_h5.py <file_path>")
        sys.exit(1)
    
    file_path = Path(sys.argv[1])
    if not file_path.exists():
        print(f"[ERROR] File does not exist: {file_path}")
        sys.exit(1)
    
    success = force_unlock(file_path)
    sys.exit(0 if success else 1)









