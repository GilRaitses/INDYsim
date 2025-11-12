#!/usr/bin/env python3
"""
Test LED alignment implementation with converted H5 files.

Tests period-relative timing calculations and period detection.
"""

import h5py
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, Tuple, Optional
import sys

# Add scripts directory to path
scripts_dir = Path(__file__).parent.parent
if str(scripts_dir) not in sys.path:
    sys.path.insert(0, str(scripts_dir))

# Try to import MATLAB method functions
try:
    # Try from current directory first
    from matlab_method_functions import compute_led12Val, add_ton_toff_matlab
    HAS_MATLAB_METHOD = True
except ImportError:
    try:
        # Try from 2025-11-11 directory
        matlab_method_dir = scripts_dir / '2025-11-11'
        if str(matlab_method_dir) not in sys.path:
            sys.path.insert(0, str(matlab_method_dir))
        from adapt_matlab_turnrate_method import compute_led12Val, add_ton_toff_matlab
        HAS_MATLAB_METHOD = True
    except ImportError:
        # Try from current script directory
        script_dir = Path(__file__).parent
        if str(script_dir) not in sys.path:
            sys.path.insert(0, str(script_dir))
        from matlab_method_functions import compute_led12Val, add_ton_toff_matlab
        HAS_MATLAB_METHOD = True


def load_h5_led_data(h5_file: Path) -> Dict:
    """Load LED data from H5 file."""
    data = {}
    with h5py.File(h5_file, 'r') as f:
        # Load LED data from global_quantities
        if 'global_quantities' in f:
            gq = f['global_quantities']
            for key in ['led1Val', 'led2Val']:
                if key in gq:
                    gq_item = gq[key]
                    if isinstance(gq_item, h5py.Group) and 'yData' in gq_item:
                        data[key] = gq_item['yData'][:]
                    elif isinstance(gq_item, h5py.Dataset):
                        data[key] = gq_item[:]
        
        # Load ETI (elapsed time index)
        if 'eti' in f:
            data['eti'] = f['eti'][:]
        elif 'elapsedTime' in f:
            data['eti'] = f['elapsedTime'][:]
    
    return data


def detect_period_from_led(led_values: np.ndarray, time_axis: np.ndarray) -> float:
    """
    Detect period length from LED ON/OFF pattern.
    
    Returns detected period in seconds.
    """
    # Detect ON/OFF transitions
    threshold = np.max(led_values) * 0.1
    is_on = led_values > threshold
    on_transitions = np.where(np.diff(is_on.astype(int)) > 0)[0]
    
    if len(on_transitions) < 2:
        return None
    
    # Calculate time differences between consecutive ON transitions
    transition_times = time_axis[on_transitions]
    periods = np.diff(transition_times)
    
    # Filter reasonable periods (5-60 seconds)
    valid_periods = periods[(periods >= 5.0) & (periods <= 60.0)]
    
    if len(valid_periods) == 0:
        return None
    
    # Use median to be robust to outliers
    detected_period = np.median(valid_periods)
    return float(detected_period)


def test_h5_file(h5_file: Path, frame_rate: float = 10.0) -> Dict:
    """Test LED alignment for a single H5 file."""
    print(f"\n{'='*70}")
    print(f"Testing: {h5_file.name}")
    print(f"{'='*70}")
    
    results = {
        'file': str(h5_file),
        'success': False,
        'errors': [],
        'warnings': [],
        'period_detected': None,
        'led12Val_computed': False,
        'led12Val_ton_range': None,
        'led12Val_toff_range': None,
        'led12Val_ton_sample': None,
        'led12Val_toff_sample': None
    }
    
    try:
        # Load LED data
        led_data = load_h5_led_data(h5_file)
        
        if 'led1Val' not in led_data:
            results['errors'].append("led1Val not found in H5 file")
            return results
        
        led1_values = led_data['led1Val']
        n_frames = len(led1_values)
        
        # Create time axis
        if 'eti' in led_data:
            times = led_data['eti']
        else:
            times = np.arange(n_frames) / frame_rate
        
        print(f"LED1 data: {n_frames} frames")
        print(f"Time range: {times[0]:.2f} to {times[-1]:.2f} seconds")
        
        # Detect period from LED1 pattern
        detected_period = detect_period_from_led(led1_values, times)
        results['period_detected'] = detected_period
        
        if detected_period is None:
            results['warnings'].append("Could not detect period from LED pattern")
        else:
            print(f"Detected period: {detected_period:.2f} seconds")
        
        # Check LED2 availability
        if 'led2Val' not in led_data:
            results['warnings'].append("led2Val not found - period-relative timing will be skipped")
            return results
        
        led2_values = led_data['led2Val']
        
        if len(led2_values) != n_frames:
            results['errors'].append(f"LED2 length mismatch: {len(led2_values)} vs {n_frames}")
            return results
        
        # Compute led12Val and period-relative timing
        if not HAS_MATLAB_METHOD:
            results['errors'].append("MATLAB method functions not available")
            return results
        
        try:
            # Compute combined LED values
            led12Val = compute_led12Val(led1_values, led2_values)
            results['led12Val_computed'] = True
            
            # Use detected period or default
            tperiod = detected_period if detected_period is not None else 10.0
            
            # Compute period-relative timing
            led12Val_ton, led12Val_toff = add_ton_toff_matlab(
                led12Val, times, tperiod=tperiod, method='square'
            )
            
            # Check value ranges
            ton_min, ton_max = float(np.nanmin(led12Val_ton)), float(np.nanmax(led12Val_ton))
            toff_min, toff_max = float(np.nanmin(led12Val_toff)), float(np.nanmax(led12Val_toff))
            
            results['led12Val_ton_range'] = [ton_min, ton_max]
            results['led12Val_toff_range'] = [toff_min, toff_max]
            
            # Get sample values (first 10 non-NaN)
            ton_valid = led12Val_ton[~np.isnan(led12Val_ton)][:10]
            toff_valid = led12Val_toff[~np.isnan(led12Val_toff)][:10]
            
            results['led12Val_ton_sample'] = ton_valid.tolist()
            results['led12Val_toff_sample'] = toff_valid.tolist()
            
            print(f"\nPeriod-relative timing results:")
            print(f"  led12Val_ton range: [{ton_min:.2f}, {ton_max:.2f}]")
            print(f"  led12Val_toff range: [{toff_min:.2f}, {toff_max:.2f}]")
            print(f"  Sample led12Val_ton: {ton_valid[:5].tolist()}")
            print(f"  Sample led12Val_toff: {toff_valid[:5].tolist()}")
            
            # Validate ranges
            # Expected: period is 40s (10s ON + 30s OFF)
            # led12Val_ton should cycle [0, 40] but during ON phase [0, 10]
            # led12Val_toff should cycle [0, 40] but during OFF phase [0, 30]
            if ton_max > 40.0:
                results['warnings'].append(f"led12Val_ton max ({ton_max:.2f}) exceeds expected period (40s)")
            if toff_max > 40.0:
                results['warnings'].append(f"led12Val_toff max ({toff_max:.2f}) exceeds expected period (40s)")
            
            results['success'] = True
            
        except Exception as e:
            results['errors'].append(f"Error computing period-relative timing: {e}")
            import traceback
            results['errors'].append(traceback.format_exc())
    
    except Exception as e:
        results['errors'].append(f"Error loading H5 file: {e}")
        import traceback
        results['errors'].append(traceback.format_exc())
    
    return results


def main():
    """Test LED alignment on all test files."""
    # Get project root (two levels up from scripts/2025-11-12)
    project_root = Path(__file__).parent.parent.parent
    h5_dir = project_root / 'data' / 'h5_files'
    
    # Test files from handoff
    test_files = [
        'GMR61@GMR61_T_Re_Sq_0to250PWM_30#C_Bl_7PWM_202510291652.h5',
        'GMR61@GMR61_T_Re_Sq_0to250PWM_30#T_Bl_Sq_5to15PWM_30_202510301513.h5',
        'GMR61@GMR61_T_Re_Sq_50to250PWM_30#C_Bl_7PWM_202510291435.h5'
    ]
    
    all_results = []
    
    for test_file in test_files:
        h5_file = h5_dir / test_file
        if not h5_file.exists():
            print(f"WARNING: Test file not found: {h5_file}")
            continue
        
        results = test_h5_file(h5_file)
        all_results.append(results)
    
    # Print summary
    print(f"\n{'='*70}")
    print("TEST SUMMARY")
    print(f"{'='*70}")
    
    for results in all_results:
        print(f"\n{Path(results['file']).name}:")
        if results['success']:
            print(f"  [OK] Period-relative timing computed successfully")
            print(f"  Period detected: {results['period_detected']:.2f}s" if results['period_detected'] else "  Period: Not detected")
            print(f"  led12Val_ton range: {results['led12Val_ton_range']}")
            print(f"  led12Val_toff range: {results['led12Val_toff_range']}")
        else:
            print(f"  [FAILED]")
        
        if results['errors']:
            print(f"  Errors: {len(results['errors'])}")
            for err in results['errors'][:3]:  # Show first 3
                print(f"    - {err}")
        
        if results['warnings']:
            print(f"  Warnings: {len(results['warnings'])}")
            for warn in results['warnings'][:3]:  # Show first 3
                print(f"    - {warn}")
    
    # Save results
    import json
    output_file = Path('scripts/2025-11-12/led_alignment_test_results.json')
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, 'w') as f:
        json.dump(all_results, f, indent=2)
    
    print(f"\nResults saved to: {output_file}")


if __name__ == '__main__':
    main()

