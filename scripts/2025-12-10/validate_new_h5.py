#!/usr/bin/env python3
"""
Quick validation script for new MagatFairy H5 exports.

Run this on a newly exported H5 file to verify:
1. File is complete (not still being written)
2. derivation_rules present with all required attrs
3. All required fields exist
4. Pipeline runs without errors/warnings

Usage:
    python scripts/2025-12-10/validate_new_h5.py /path/to/file.h5
    
    # Or for the test file:
    python scripts/2025-12-10/validate_new_h5.py scripts/2025-12-10/new_h5_expt_validate/*.h5
"""

import sys
import os
from pathlib import Path
import h5py
import numpy as np

# Minimum expected file size (bytes) - complete H5 should be at least 1MB
MIN_FILE_SIZE = 1_000_000

# Required fields
REQUIRED_ROOT = ['eti', 'tracks', 'global_quantities']
REQUIRED_DERIVATION_RULES = ['smoothTime', 'derivTime', 'interpTime']
REQUIRED_TRACK_DQ = ['sloc', 'shead', 'smid', 'speed', 'sspineTheta', 'vel_dp', 'eti']


def validate_h5_complete(h5_path: Path) -> dict:
    """
    Validate H5 file is complete and ready for pipeline.
    
    Returns dict with 'passed', 'errors', 'warnings', 'info'.
    """
    result = {
        'passed': True,
        'errors': [],
        'warnings': [],
        'info': []
    }
    
    # Check file exists
    if not h5_path.exists():
        result['errors'].append(f"File not found: {h5_path}")
        result['passed'] = False
        return result
    
    # Check file size
    file_size = h5_path.stat().st_size
    result['info'].append(f"File size: {file_size:,} bytes ({file_size/1e6:.1f} MB)")
    
    if file_size < MIN_FILE_SIZE:
        result['errors'].append(f"File too small ({file_size:,} bytes). Export may be incomplete.")
        result['passed'] = False
        return result
    
    # Check for lock file
    lock_path = h5_path.with_suffix('.h5.lock')
    if lock_path.exists():
        result['errors'].append(f"Lock file exists: {lock_path}. Export still in progress.")
        result['passed'] = False
        return result
    
    # Open and validate structure
    try:
        with h5py.File(h5_path, 'r') as f:
            # Check derivation_rules
            if 'derivation_rules' not in f:
                result['errors'].append("Missing /derivation_rules group")
                result['passed'] = False
            else:
                dr = f['derivation_rules']
                for attr in REQUIRED_DERIVATION_RULES:
                    if attr not in dr.attrs:
                        result['errors'].append(f"Missing derivation_rules.{attr}")
                        result['passed'] = False
                    else:
                        result['info'].append(f"derivation_rules.{attr} = {dr.attrs[attr]}")
            
            # Check required root groups/datasets
            for field in REQUIRED_ROOT:
                if field not in f:
                    result['errors'].append(f"Missing /{field}")
                    result['passed'] = False
                else:
                    if isinstance(f[field], h5py.Dataset):
                        result['info'].append(f"/{field}: {f[field].shape}")
                    else:
                        result['info'].append(f"/{field}/ (group)")
            
            # Check tracks structure
            if 'tracks' in f:
                track_keys = [k for k in f['tracks'].keys() if k.startswith('track_')]
                result['info'].append(f"Found {len(track_keys)} tracks")
                
                if len(track_keys) == 0:
                    result['errors'].append("No tracks found in /tracks")
                    result['passed'] = False
                else:
                    # Check first track
                    first_track = f['tracks'][track_keys[0]]
                    
                    if 'derived_quantities' not in first_track:
                        result['errors'].append(f"Missing derived_quantities in {track_keys[0]}")
                        result['passed'] = False
                    else:
                        dq = first_track['derived_quantities']
                        for field in REQUIRED_TRACK_DQ:
                            if field not in dq:
                                result['warnings'].append(f"Missing {track_keys[0]}/derived_quantities/{field}")
                        
                        # Sanity checks
                        if 'sloc' in dq:
                            sloc = dq['sloc'][:]
                            sloc_range = sloc.max() - sloc.min()
                            result['info'].append(f"sloc range: {sloc_range:.2f} (expect ~10 cm)")
                            if sloc_range > 100:
                                result['warnings'].append(f"sloc range {sloc_range:.1f} suggests pixels, not cm")
                        
                        if 'speed' in dq:
                            speed = dq['speed'][:]
                            if speed.ndim > 1:
                                speed = speed.flatten()
                            mean_speed = np.nanmean(speed)
                            result['info'].append(f"mean speed: {mean_speed:.4f} cm/s ({mean_speed*10:.2f} mm/s)")
                            if mean_speed > 1.0:
                                result['warnings'].append(f"mean speed {mean_speed:.3f} suggests mm/s units, not cm/s")
            
            # Check global quantities (LED data)
            if 'global_quantities' in f:
                gq = f['global_quantities']
                for led in ['led1Val', 'led2Val']:
                    if led in gq:
                        if isinstance(gq[led], h5py.Group) and 'yData' in gq[led]:
                            led_data = gq[led]['yData'][:]
                            result['info'].append(f"{led}: {len(led_data)} frames, range [{led_data.min():.0f}, {led_data.max():.0f}]")
                        elif isinstance(gq[led], h5py.Dataset):
                            led_data = gq[led][:]
                            result['info'].append(f"{led}: {len(led_data)} frames")
                    else:
                        result['warnings'].append(f"Missing global_quantities/{led}")
            
            # Check ETI
            if 'eti' in f:
                eti = f['eti'][:]
                duration = eti.max() - eti.min()
                result['info'].append(f"ETI: {len(eti)} frames, duration {duration:.1f}s ({duration/60:.1f} min)")
                if duration > 1300 or duration < 1100:
                    result['warnings'].append(f"ETI duration {duration:.0f}s unusual (expect ~1200s = 20min)")
    
    except Exception as e:
        result['errors'].append(f"Failed to read H5: {e}")
        result['passed'] = False
    
    return result


def run_pipeline_test(h5_path: Path) -> dict:
    """
    Run the engineer_dataset_from_h5.py pipeline on a single file.
    
    Returns dict with 'success', 'output', 'errors'.
    """
    import subprocess
    import tempfile
    
    result = {
        'success': False,
        'output': '',
        'errors': []
    }
    
    # Create temp output directory
    with tempfile.TemporaryDirectory() as tmpdir:
        cmd = [
            sys.executable,
            'scripts/engineer_dataset_from_h5.py',
            '--file', str(h5_path),
            '--output-dir', tmpdir
        ]
        
        try:
            proc = subprocess.run(
                cmd,
                cwd=str(Path(__file__).parent.parent.parent),  # INDYsim root
                capture_output=True,
                text=True,
                timeout=300  # 5 min timeout
            )
            
            result['output'] = proc.stdout + proc.stderr
            result['success'] = proc.returncode == 0
            
            if proc.returncode != 0:
                result['errors'].append(f"Pipeline exited with code {proc.returncode}")
            
            # Check for warnings in output
            for line in result['output'].split('\n'):
                if 'WARNING' in line.upper() or 'WARN' in line.upper():
                    result['errors'].append(f"Warning in output: {line.strip()}")
                if 'ERROR' in line.upper():
                    result['errors'].append(f"Error in output: {line.strip()}")
            
            # Check output files exist
            output_files = list(Path(tmpdir).glob('*'))
            if len(output_files) == 0:
                result['errors'].append("No output files generated")
                result['success'] = False
            else:
                result['output'] += f"\n\nGenerated {len(output_files)} output files"
                
        except subprocess.TimeoutExpired:
            result['errors'].append("Pipeline timed out after 5 minutes")
        except Exception as e:
            result['errors'].append(f"Failed to run pipeline: {e}")
    
    return result


def main():
    if len(sys.argv) < 2:
        print("Usage: python validate_new_h5.py <h5_file>")
        print("       python validate_new_h5.py scripts/2025-12-10/new_h5_expt_validate/*.h5")
        sys.exit(1)
    
    h5_path = Path(sys.argv[1])
    
    print("=" * 70)
    print(f"VALIDATING: {h5_path.name}")
    print("=" * 70)
    
    # Phase 1: Structure validation
    print("\n[Phase 1] Checking H5 structure...")
    result = validate_h5_complete(h5_path)
    
    print("\nInfo:")
    for info in result['info']:
        print(f"  ✓ {info}")
    
    if result['warnings']:
        print("\nWarnings:")
        for warn in result['warnings']:
            print(f"  ⚠ {warn}")
    
    if result['errors']:
        print("\nErrors:")
        for err in result['errors']:
            print(f"  ✗ {err}")
    
    if not result['passed']:
        print("\n" + "=" * 70)
        print("VALIDATION FAILED - H5 file is incomplete or malformed")
        print("=" * 70)
        sys.exit(1)
    
    print("\n[Phase 1] PASSED ✓")
    
    # Phase 2: Pipeline test
    print("\n[Phase 2] Running pipeline test...")
    pipeline_result = run_pipeline_test(h5_path)
    
    if pipeline_result['errors']:
        print("\nPipeline issues:")
        for err in pipeline_result['errors']:
            print(f"  ⚠ {err}")
    
    if not pipeline_result['success']:
        print("\n" + "=" * 70)
        print("PIPELINE TEST FAILED")
        print("=" * 70)
        print("\nOutput:")
        print(pipeline_result['output'][-2000:])  # Last 2000 chars
        sys.exit(1)
    
    print("\n[Phase 2] PASSED ✓")
    
    # Summary
    print("\n" + "=" * 70)
    print("ALL VALIDATION PASSED ✓")
    print("=" * 70)
    print(f"\nH5 file is ready for full pipeline processing:")
    print(f"  python scripts/engineer_dataset_from_h5.py \\")
    print(f"      --file {h5_path} \\")
    print(f"      --output-dir data/engineered_validated")


if __name__ == '__main__':
    main()
