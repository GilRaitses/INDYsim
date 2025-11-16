#!/usr/bin/env python3
"""
Convert MATLAB ESET folder data to H5 format.

Reads native ESET folder structure:
- .mat files from btdfiles/ subdirectory (track data)
- .bin files from root and * sup data dir/ (LED values)

Creates H5 files compatible with engineer_dataset_from_h5.py

Usage:
    python convert_matlab_to_h5.py \
        --eset-dir "data/matlab_data/GMR61@GMR61/T_Re_Sq_0to250PWM_30#T_Bl_Sq_5to15PWM_30" \
        --output-dir "data/h5_files"
"""

import sys
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple

try:
    import h5py
    HAS_H5PY = True
except ImportError:
    HAS_H5PY = False
    print("ERROR: h5py required. Install with: pip install h5py")
    sys.exit(1)

try:
    import scipy.io
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False
    print("ERROR: scipy required. Install with: pip install scipy")
    sys.exit(1)


def read_bin_file(bin_path: Path) -> np.ndarray:
    """
    Read LED values from .bin file.
    
    Binary format: uint16 values (PWM values 0-255)
    """
    try:
        with open(bin_path, 'rb') as f:
            data = np.frombuffer(f.read(), dtype=np.uint16)
        return data
    except Exception as e:
        print(f"  ERROR reading bin file {bin_path}: {e}")
        return np.array([])


def read_mat_file(mat_path: Path) -> Dict:
    """
    Read MATLAB .mat file containing track data.
    
    Returns dictionary with track data structure.
    """
    try:
        mat_data = scipy.io.loadmat(str(mat_path), simplify_cells=True)
        return mat_data
    except Exception as e:
        print(f"  ERROR reading mat file {mat_path}: {e}")
        return {}


def find_led_bin_files(eset_dir: Path) -> Tuple[List[Path], List[Path]]:
    """
    Find LED1 and LED2 .bin files in ESET folder.
    
    Looks in:
    - Root directory: *led1 values.bin, *led2 values.bin
    - * sup data dir/ subdirectories: *led1 values.bin, *led2 values.bin
    
    Returns:
        (led1_files, led2_files) - Lists of Path objects
    """
    led1_files = []
    led2_files = []
    
    # Search root directory
    for bin_file in eset_dir.glob("*led1 values.bin"):
        led1_files.append(bin_file)
    for bin_file in eset_dir.glob("*led2 values.bin"):
        led2_files.append(bin_file)
    
    # Search sup data dir subdirectories
    for sup_dir in eset_dir.glob("* sup data dir"):
        if sup_dir.is_dir():
            for bin_file in sup_dir.glob("*led1 values.bin"):
                led1_files.append(bin_file)
            for bin_file in sup_dir.glob("*led2 values.bin"):
                led2_files.append(bin_file)
    
    return led1_files, led2_files


def find_mat_files(eset_dir: Path) -> List[Path]:
    """
    Find .mat files in btdfiles/ subdirectory.
    
    Returns list of .mat file paths.
    """
    btdfiles_dir = eset_dir / "btdfiles"
    if not btdfiles_dir.exists():
        print(f"  WARNING: btdfiles/ directory not found in {eset_dir}")
        return []
    
    mat_files = list(btdfiles_dir.glob("*.mat"))
    return mat_files


def extract_track_data_from_mat(mat_data: Dict, track_id: int) -> Optional[Dict]:
    """
    Extract track data from MATLAB .mat file structure.
    
    Expected structure (varies by MATLAB version):
    - btd structure with track data
    - Points: head, mid, tail positions
    - Derived quantities: speed, theta, curvature
    
    Returns dictionary with track data or None if not found.
    """
    # Try to find btd structure
    btd = None
    for key in ['btd', 'BTD', 'track_data']:
        if key in mat_data:
            btd = mat_data[key]
            break
    
    if btd is None:
        # Try to find any structure that might contain tracks
        for key, value in mat_data.items():
            if isinstance(value, dict) and 'tracks' in str(value).lower():
                btd = value
                break
    
    if btd is None:
        return None
    
    # Extract track data (structure varies)
    track_data = {}
    
    # Try to get positions (head, mid, tail)
    # MATLAB structures may have different field names
    for pos_name in ['head', 'mid', 'tail', 'Head', 'Mid', 'Tail']:
        if isinstance(btd, dict) and pos_name in btd:
            pos_data = btd[pos_name]
            if isinstance(pos_data, np.ndarray):
                track_data[pos_name.lower()] = pos_data
    
    # Try to get derived quantities
    for deriv_name in ['speed', 'theta', 'curv', 'Speed', 'Theta', 'Curv']:
        if isinstance(btd, dict) and deriv_name in btd:
            deriv_data = btd[deriv_name]
            if isinstance(deriv_data, np.ndarray):
                track_data[deriv_name.lower()] = deriv_data
    
    return track_data if track_data else None


def create_h5_from_eset(eset_dir: Path, output_dir: Path, eset_name: str) -> Optional[Path]:
    """
    Convert ESET folder to H5 file.
    
    Args:
        eset_dir: Path to ESET folder
        output_dir: Path to output directory for H5 files
        eset_name: Name of ESET (for output filename)
    
    Returns:
        Path to created H5 file or None if failed
    """
    print(f"\nProcessing ESET: {eset_name}")
    print(f"  Source: {eset_dir}")
    
    # Find LED bin files
    led1_files, led2_files = find_led_bin_files(eset_dir)
    print(f"  Found {len(led1_files)} LED1 bin files, {len(led2_files)} LED2 bin files")
    
    if not led1_files:
        print(f"  ERROR: No LED1 bin files found")
        return None
    
    # Read LED values (use first file found, or combine if multiple)
    led1_values = None
    if led1_files:
        led1_values = read_bin_file(led1_files[0])
        if len(led1_values) == 0:
            print(f"  ERROR: Failed to read LED1 values")
            return None
        print(f"  LED1 values: {len(led1_values)} samples")
    
    led2_values = None
    if led2_files:
        led2_values = read_bin_file(led2_files[0])
        if len(led2_values) > 0:
            print(f"  LED2 values: {len(led2_values)} samples")
        else:
            print(f"  WARNING: Failed to read LED2 values (may not be present)")
    
    # Find and process .mat files
    mat_files = find_mat_files(eset_dir)
    print(f"  Found {len(mat_files)} .mat files")
    
    if not mat_files:
        print(f"  ERROR: No .mat files found in btdfiles/")
        return None
    
    # Create output H5 file
    output_dir.mkdir(parents=True, exist_ok=True)
    h5_filename = f"{eset_name}.h5"
    h5_path = output_dir / h5_filename
    
    print(f"  Creating H5 file: {h5_path}")
    
    try:
        with h5py.File(h5_path, 'w') as h5f:
            # Create global quantities group
            gq_group = h5f.create_group('global_quantities')
            
            # Add LED1 values
            if led1_values is not None:
                led1_group = gq_group.create_group('led1Val')
                led1_group.create_dataset('yData', data=led1_values.astype(np.float32))
            
            # Add LED2 values (if available)
            if led2_values is not None and len(led2_values) > 0:
                led2_group = gq_group.create_group('led2Val')
                led2_group.create_dataset('yData', data=led2_values.astype(np.float32))
            
            # Process tracks from .mat files
            tracks_group = h5f.create_group('tracks')
            
            # First pass: collect all track data with their original indices
            track_data_list = []
            for mat_file in mat_files:
                print(f"    Processing {mat_file.name}...")
                mat_data = read_mat_file(mat_file)
                
                if not mat_data:
                    continue
                
                # Extract track data - preserve original track count for key
                track_count = len(track_data_list)
                track_data = extract_track_data_from_mat(mat_data, track_count)
                
                if track_data:
                    track_data_list.append((track_count, track_data))
            
            # Second pass: create tracks in sorted order by track number
            # This ensures tracks are stored in H5 file in numeric order (track_0, track_1, ..., track_N)
            for track_count, track_data in sorted(track_data_list, key=lambda x: x[0]):
                track_key = f'track_{track_count}'
                track_group = tracks_group.create_group(track_key)
                
                # Create points group
                points_group = track_group.create_group('points')
                
                # Add position data
                for pos_name in ['head', 'mid', 'tail']:
                    if pos_name in track_data:
                        pos_data = np.array(track_data[pos_name])
                        if pos_data.ndim == 2 and pos_data.shape[1] == 2:
                            points_group.create_dataset(pos_name, data=pos_data.astype(np.float32))
                
                # Create derived_quantities group
                if any(k in track_data for k in ['speed', 'theta', 'curv']):
                    deriv_group = track_group.create_group('derived_quantities')
                    for deriv_name in ['speed', 'theta', 'curv']:
                        if deriv_name in track_data:
                            deriv_data = np.array(track_data[deriv_name])
                            if deriv_data.ndim == 1:
                                deriv_group.create_dataset(deriv_name, data=deriv_data.astype(np.float32))
            
            # Add metadata
            metadata_group = h5f.create_group('metadata')
            metadata_group.attrs['eset_name'] = eset_name
            metadata_group.attrs['source_dir'] = str(eset_dir)
            metadata_group.attrs['frame_rate'] = 10.0  # Default, may need adjustment
            metadata_group.attrs['num_tracks'] = track_count
            
            print(f"  Created H5 file with {track_count} tracks")
        
        return h5_path
    
    except Exception as e:
        print(f"  ERROR creating H5 file: {e}")
        import traceback
        traceback.print_exc()
        return None


def main():
    parser = argparse.ArgumentParser(description='Convert MATLAB ESET folder to H5 format')
    parser.add_argument('--eset-dir', type=str, required=True,
                        help='Path to ESET folder (e.g., data/matlab_data/GMR61@GMR61/T_Re_Sq_0to250PWM_30#T_Bl_Sq_5to15PWM_30)')
    parser.add_argument('--output-dir', type=str, default='data/h5_files',
                        help='Output directory for H5 files (default: data/h5_files)')
    
    args = parser.parse_args()
    
    eset_dir = Path(args.eset_dir)
    if not eset_dir.exists():
        print(f"ERROR: ESET directory not found: {eset_dir}")
        sys.exit(1)
    
    eset_name = eset_dir.name
    output_dir = Path(args.output_dir)
    
    h5_path = create_h5_from_eset(eset_dir, output_dir, eset_name)
    
    if h5_path:
        print(f"\n✅ Successfully created: {h5_path}")
        sys.exit(0)
    else:
        print(f"\n❌ Failed to create H5 file")
        sys.exit(1)


if __name__ == '__main__':
    main()

