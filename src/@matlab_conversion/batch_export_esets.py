#!/usr/bin/env python3
"""
Batch Export Script for INDYsim MATLAB Data Conversion

Processes all ESET folders in a genotype directory using native folder structure.
Handles dynamic genotype parsing, file discovery, and H5 export.

Source: Adapted from D:\mechanosensation\scripts\2025-11-11\batch_export_indysim.py

Author: mechanobro (adapted for INDYsim)
Date: 2025-11-11
"""

import sys
import subprocess
from pathlib import Path
import time
import re
from typing import Dict, List, Optional

# Paths
INDYSIM_ROOT = Path(__file__).parent.parent.parent  # Go up from src/@matlab_conversion
MATLAB_DATA_DIR = INDYSIM_ROOT / "data" / "matlab_data"
CONVERT_SCRIPT = Path(__file__).parent / "convert_matlab_to_h5.py"
OUTPUT_DIR = INDYSIM_ROOT / "data" / "h5_files"  # Output to h5_files subdirectory


def parse_genotype_from_path(eset_path: Path, mat_filename: str) -> Optional[str]:
    """Parse genotype from mat filename or parent folder path"""
    match = re.search(r'^([A-Za-z0-9]+@[A-Za-z0-9]+)_', mat_filename)
    if match:
        return match.group(1)
    
    parent = eset_path.parent
    if parent.name and '@' in parent.name:
        parts = parent.name.split('@')
        if len(parts) == 2 and parts[0] == parts[1]:
            return parent.name
    
    return None


def extract_timestamp_from_mat(mat_filename: str) -> Optional[str]:
    """Extract 12-digit timestamp from mat filename"""
    match = re.search(r'_(\d{12})\.mat$', mat_filename)
    return match.group(1) if match else None


def detect_experiments_in_eset(eset_dir: Path) -> List[Dict]:
    """
    Detect all experiments in an ESET folder.
    
    Uses matfiles/ directory for .mat files (MATLAB expects these, not btdfiles/).
    """
    matfiles_dir = eset_dir / "matfiles"
    
    if not matfiles_dir.exists():
        print(f"  [ERROR] matfiles directory not found: {matfiles_dir}")
        return []
    
    # Find all .mat files in matfiles/ (MATLAB expects these)
    mat_files = list(matfiles_dir.glob("*.mat"))
    
    if not mat_files:
        print(f"  [WARNING] No .mat files found in {matfiles_dir}")
        return []
    
    print(f"  Found {len(mat_files)} .mat files in matfiles/")
    
    experiments = []
    
    for mat_file in sorted(mat_files):
        timestamp = extract_timestamp_from_mat(mat_file.name)
        if not timestamp:
            print(f"  [SKIP] Could not extract timestamp from {mat_file.name}")
            continue
        
        base_name = mat_file.stem
        genotype = parse_genotype_from_path(eset_dir, mat_file.name)
        
        if not genotype:
            print(f"  [SKIP] Could not parse genotype from {mat_file.name}")
            continue
        
        # Tracks directory: in matfiles/ subdirectory
        tracks_name = f"{genotype}_{timestamp} - tracks"
        tracks_dir = matfiles_dir / tracks_name
        
        # FID .bin file: Root level of ESET
        bin_file = eset_dir / f"{base_name}.bin"
        
        # Sup data dir: Root level of ESET
        sup_data_name = f"{base_name} sup data dir"
        sup_data_dir = eset_dir / sup_data_name
        
        # LED bin files in sup data dir
        led1_bin = sup_data_dir / f"{base_name} led1 values.bin"
        led2_bin = sup_data_dir / f"{base_name} led2 values.bin"
        
        # Validate ALL required files exist
        missing_files = []
        
        if not mat_file.exists():
            missing_files.append(f"MAT file: {mat_file}")
        if not tracks_dir.exists():
            missing_files.append(f"Tracks directory: {tracks_dir}")
        if not bin_file.exists():
            missing_files.append(f"FID .bin file: {bin_file}")
        if not sup_data_dir.exists():
            missing_files.append(f"Sup data directory: {sup_data_dir}")
        if not led1_bin.exists():
            missing_files.append(f"LED1 values bin: {led1_bin}")
        
        has_led2 = led2_bin.exists()
        
        if missing_files:
            print(f"  [SKIP] {mat_file.name} - missing files:")
            for f in missing_files:
                print(f"    - {f}")
            continue
        
        experiments.append({
            'mat_file': mat_file,
            'tracks_dir': tracks_dir,
            'bin_file': bin_file,
            'sup_data_dir': sup_data_dir,
            'led1_bin': led1_bin,
            'led2_bin': led2_bin,
            'has_led2': has_led2,
            'base_name': base_name,
            'timestamp': timestamp,
            'genotype': genotype
        })
    
    return experiments


def export_experiment(file_info: Dict, output_dir: Path) -> Dict:
    """Export a single experiment using convert_matlab_to_h5.py"""
    base_name = file_info['base_name']
    
    print(f"\n{'='*80}")
    print(f"EXPORTING: {base_name}")
    print(f"{'='*80}")
    print(f"  Genotype: {file_info['genotype']}")
    print(f"  Timestamp: {file_info['timestamp']}")
    print(f"  MAT file: {file_info['mat_file'].name}")
    print(f"  Tracks: {file_info['tracks_dir'].name}")
    print(f"  BIN file: {file_info['bin_file'].name}")
    print()
    
    # Output filename matches base_name with .h5 extension
    output_file = output_dir / f"{base_name}.h5"
    
    # Build command
    cmd = [
        sys.executable,
        str(CONVERT_SCRIPT),
        '--mat', str(file_info['mat_file']),
        '--tracks', str(file_info['tracks_dir']),
        '--bin', str(file_info['bin_file']),
        '--output', str(output_file)
    ]
    
    # Run export
    start_time = time.time()
    try:
        result = subprocess.run(
            cmd,
            cwd=str(CONVERT_SCRIPT.parent),
            check=True,
            capture_output=False,
            text=True
        )
        
        elapsed = time.time() - start_time
        file_size = output_file.stat().st_size / (1024 * 1024) if output_file.exists() else 0
        
        print(f"\n  [SUCCESS] Export complete: {output_file.name}")
        print(f"     Size: {file_size:.1f} MB")
        print(f"     Time: {elapsed/60:.1f} minutes")
        
        return {
            'success': True,
            'output_file': output_file,
            'file_size_mb': file_size,
            'time_min': elapsed / 60,
            'base_name': base_name,
            'timestamp': file_info['timestamp']
        }
        
    except subprocess.CalledProcessError as e:
        elapsed = time.time() - start_time
        print(f"\n  [ERROR] Export failed after {elapsed/60:.1f} minutes")
        print(f"     Error code: {e.returncode}")
        return {
            'success': False,
            'error': f"Exit code {e.returncode}",
            'base_name': base_name,
            'timestamp': file_info['timestamp'],
            'time_min': elapsed / 60
        }
    except Exception as e:
        elapsed = time.time() - start_time
        print(f"\n  [ERROR] Export failed: {e}")
        return {
            'success': False,
            'error': str(e),
            'base_name': base_name,
            'timestamp': file_info['timestamp'],
            'time_min': elapsed / 60
        }


def process_genotype(genotype_dir: Path, output_dir: Path) -> List[Dict]:
    """
    Process all ESET folders in a genotype directory.
    
    Args:
        genotype_dir: Path to genotype directory (e.g., GMR61@GMR61)
        output_dir: Path to output directory for H5 files
    
    Returns:
        List of export results
    """
    print("="*80)
    print(f"PROCESSING GENOTYPE: {genotype_dir.name}")
    print("="*80)
    
    # Find all ESET folders
    eset_folders = [d for d in genotype_dir.iterdir() if d.is_dir() and (d / "matfiles").exists()]
    
    if not eset_folders:
        print(f"[WARNING] No ESET folders found in {genotype_dir.name}")
        return []
    
    print(f"[OK] Found {len(eset_folders)} ESET folders")
    
    all_results = []
    
    for eset_idx, eset_dir in enumerate(sorted(eset_folders), 1):
        print(f"\n{'='*80}")
        print(f"ESET {eset_idx}/{len(eset_folders)}: {eset_dir.name}")
        print(f"{'='*80}")
        
        experiments = detect_experiments_in_eset(eset_dir)
        
        if not experiments:
            print(f"[WARNING] No complete experiments found in {eset_dir.name}")
            continue
        
        print(f"[OK] Found {len(experiments)} complete experiments")
        
        for exp_idx, file_info in enumerate(experiments, 1):
            print(f"\n[{exp_idx}/{len(experiments)}] Processing experiment...")
            result = export_experiment(file_info, output_dir)
            all_results.append(result)
            
            if exp_idx < len(experiments):
                print("\n" + "-"*80)
                time.sleep(2)
    
    return all_results


def main():
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Batch export INDYsim MATLAB experiments to H5 format',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Process all ESETs in a genotype folder
  python batch_export_esets.py --genotype "GMR61@GMR61"
  
  # Process specific ESET folder
  python batch_export_esets.py --genotype "GMR61@GMR61" --eset "T_Re_Sq_0to250PWM_30#C_Bl_7PWM"
  
  # Custom output directory
  python batch_export_esets.py --genotype "GMR61@GMR61" --output-dir "D:\\output"
        """
    )
    parser.add_argument('--genotype', type=str, required=True,
                       help='Genotype folder name (e.g., GMR61@GMR61)')
    parser.add_argument('--eset', type=str, default=None,
                       help='Specific ESET folder name to process (optional)')
    parser.add_argument('--output-dir', type=str, default=None,
                       help=f'Output directory (default: {OUTPUT_DIR})')
    
    args = parser.parse_args()
    
    # Verify dependencies
    if not CONVERT_SCRIPT.exists():
        print(f"[ERROR] Convert script not found: {CONVERT_SCRIPT}")
        return 1
    
    if not MATLAB_DATA_DIR.exists():
        print(f"[ERROR] MATLAB data directory not found: {MATLAB_DATA_DIR}")
        return 1
    
    # Find genotype directory
    genotype_dir = MATLAB_DATA_DIR / args.genotype
    if not genotype_dir.exists():
        print(f"[ERROR] Genotype directory not found: {genotype_dir}")
        return 1
    
    # Set output directory
    output_dir = Path(args.output_dir) if args.output_dir else OUTPUT_DIR
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("="*80)
    print("BATCH EXPORT: INDYSIM MATLAB DATA")
    print("="*80)
    print(f"Genotype: {args.genotype}")
    print(f"MATLAB data directory: {MATLAB_DATA_DIR}")
    print(f"Output directory: {output_dir}")
    print(f"Convert script: {CONVERT_SCRIPT}")
    print()
    
    # Process ESETs
    if args.eset:
        # Process specific ESET
        eset_dir = genotype_dir / args.eset
        if not eset_dir.exists():
            print(f"[ERROR] ESET folder not found: {eset_dir}")
            return 1
        
        experiments = detect_experiments_in_eset(eset_dir)
        if not experiments:
            print(f"[ERROR] No complete experiments found in {args.eset}")
            return 1
        
        all_results = []
        for exp_idx, file_info in enumerate(experiments, 1):
            print(f"\n[{exp_idx}/{len(experiments)}] Processing experiment...")
            result = export_experiment(file_info, output_dir)
            all_results.append(result)
    else:
        # Process all ESETs in genotype
        all_results = process_genotype(genotype_dir, output_dir)
    
    # Summary
    total_time = time.time()
    successful = [r for r in all_results if r['success']]
    failed = [r for r in all_results if not r['success']]
    
    print("\n" + "="*80)
    print("BATCH EXPORT SUMMARY")
    print("="*80)
    print(f"Total experiments: {len(all_results)}")
    print(f"Successful: {len(successful)}")
    print(f"Failed: {len(failed)}")
    print()
    
    if successful:
        print("[SUCCESS] Successful exports:")
        total_size = 0
        for r in successful:
            size = r.get('file_size_mb', 0)
            total_size += size
            print(f"   {r['base_name']}: {size:.1f} MB")
        print(f"\n   Total size: {total_size:.1f} MB")
    
    if failed:
        print("\n[ERROR] Failed exports:")
        for r in failed:
            error = r.get('error', 'Unknown error')
            print(f"   {r['base_name']}: {error}")
    
    print(f"\nOutput directory: {output_dir}")
    print("="*80)
    
    return 0 if len(failed) == 0 else 1


if __name__ == "__main__":
    exit(main())

