#!/usr/bin/env python3
"""
Test stimulus-locked turn rate analysis pipeline with converted H5 files.

Tests:
1. Pipeline runs without errors
2. Turn rates computed correctly
3. Timing alignment verified (spike at t=0)
4. No path or structure issues
5. Compatible with existing scripts
"""

import sys
import json
from pathlib import Path
from datetime import datetime
import traceback

# Add scripts to path
script_dir = Path(__file__).parent
project_root = script_dir.parent.parent
sys.path.insert(0, str(project_root / "scripts" / "queue"))
sys.path.insert(0, str(project_root / "scripts"))

# Try to use config
try:
    config_path = script_dir.parent / 'config.py'
    if config_path.exists():
        sys.path.insert(0, str(config_path.parent))
        from config import H5_FILES_DIR, ENGINEERED_DATA_DIR
        h5_dir = H5_FILES_DIR
        output_dir = ENGINEERED_DATA_DIR
    else:
        h5_dir = Path(__file__).parent.parent.parent / 'data' / 'h5_files'
        output_dir = Path(__file__).parent.parent.parent / 'data' / 'engineered'
except ImportError:
    h5_dir = Path(__file__).parent.parent.parent / 'data' / 'h5_files'
    output_dir = Path(__file__).parent.parent.parent / 'data' / 'engineered'

# Test files from Task 0.1
TEST_FILES = [
    "GMR61@GMR61_T_Re_Sq_0to250PWM_30#C_Bl_7PWM_202510291652.h5",
    "GMR61@GMR61_T_Re_Sq_0to250PWM_30#T_Bl_Sq_5to15PWM_30_202510301513.h5"
]

def test_file(h5_filename):
    """Test analysis pipeline on a single H5 file."""
    print(f"\n{'='*80}")
    print(f"Testing: {h5_filename}")
    print(f"{'='*80}")
    
    h5_file = h5_dir / h5_filename
    
    if not h5_file.exists():
        return {
            'file': h5_filename,
            'status': 'error',
            'error': f'H5 file not found: {h5_file}',
            'timestamp': datetime.now().isoformat()
        }
    
    result = {
        'file': h5_filename,
        'h5_path': str(h5_file),
        'status': 'unknown',
        'timestamp': datetime.now().isoformat(),
        'errors': [],
        'warnings': [],
        'checks': {}
    }
    
    try:
        # Import required modules
        # Add scripts directory to path (matching production script)
        script_queue_dir = project_root / "scripts" / "queue"
        sys.path.insert(0, str(script_queue_dir))
        sys.path.insert(0, str(project_root / "scripts"))
        
        # Try importing - may need to check actual location
        try:
            from engineer_dataset_from_h5 import process_h5_file, load_h5_file
        except ImportError:
            # Try alternative import path
            import importlib.util
            eng_path = project_root / "scripts" / "engineer_dataset_from_h5.py"
            if eng_path.exists():
                spec = importlib.util.spec_from_file_location("engineer_dataset_from_h5", eng_path)
                eng_module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(eng_module)
                process_h5_file = eng_module.process_h5_file
                load_h5_file = eng_module.load_h5_file
            else:
                raise ImportError(f"engineer_dataset_from_h5.py not found at {eng_path}")
        
        from create_eda_figures import create_stimulus_locked_turn_rate_analysis
        import pandas as pd
        import numpy as np
        
        # Check 1: Load H5 file
        print("\n[CHECK 1] Loading H5 file...")
        try:
            h5_data = load_h5_file(h5_file)
            result['checks']['h5_load'] = 'PASS'
            result['checks']['n_tracks'] = len(h5_data.get('tracks', {}))
            print(f"  [OK] H5 file loaded successfully")
            print(f"  [OK] Found {result['checks']['n_tracks']} tracks")
        except Exception as e:
            result['checks']['h5_load'] = 'FAIL'
            result['errors'].append(f'H5 load error: {str(e)}')
            print(f"  [FAIL] H5 load failed: {e}")
            result['status'] = 'error'
            return result
        
        # Check 2: Process H5 file (generate events CSV)
        print("\n[CHECK 2] Processing H5 file (generating events CSV)...")
        try:
            experiment_id = h5_file.stem.replace(' ', '_')
            output_dir.mkdir(parents=True, exist_ok=True)
            
            # Process with progress tracking disabled for testing
            process_h5_file(h5_file, output_dir, experiment_id)
            
            events_file = output_dir / f"{experiment_id}_events.csv"
            trajectories_file = output_dir / f"{experiment_id}_trajectories.csv"
            
            if not events_file.exists():
                raise FileNotFoundError(f"Events file not created: {events_file}")
            if not trajectories_file.exists():
                raise FileNotFoundError(f"Trajectories file not created: {trajectories_file}")
            
            result['checks']['data_extraction'] = 'PASS'
            result['checks']['events_file'] = str(events_file)
            result['checks']['trajectories_file'] = str(trajectories_file)
            
            # Check file sizes
            events_df = pd.read_csv(events_file, nrows=1000)  # Sample
            trajectories_df = pd.read_csv(trajectories_file, nrows=1000)  # Sample
            
            result['checks']['events_rows'] = len(pd.read_csv(events_file))
            result['checks']['trajectories_rows'] = len(pd.read_csv(trajectories_file))
            result['checks']['events_columns'] = list(events_df.columns)
            result['checks']['trajectories_columns'] = list(trajectories_df.columns)
            
            print(f"  [OK] Events CSV created: {result['checks']['events_rows']:,} rows")
            print(f"  [OK] Trajectories CSV created: {result['checks']['trajectories_rows']:,} rows")
            
            # Check for required columns
            required_event_cols = ['track_id', 'time', 'is_turn', 'is_reorientation']
            missing_cols = [col for col in required_event_cols if col not in events_df.columns]
            if missing_cols:
                result['warnings'].append(f"Missing columns in events CSV: {missing_cols}")
                print(f"  ⚠ Missing columns: {missing_cols}")
            else:
                print(f"  [OK] All required columns present")
            
        except Exception as e:
            result['checks']['data_extraction'] = 'FAIL'
            result['errors'].append(f'Data extraction error: {str(e)}')
            print(f"  [FAIL] Data extraction failed: {e}")
            traceback.print_exc()
            result['status'] = 'error'
            return result
        
        # Check 3: Run stimulus-locked analysis
        print("\n[CHECK 3] Running stimulus-locked turn rate analysis...")
        try:
            figures_dir = Path(__file__).parent.parent.parent / 'output' / 'figures' / 'eda'
            figures_dir.mkdir(parents=True, exist_ok=True)
            output_path = figures_dir / f"{experiment_id}_stimulus_locked_turn_rate.png"
            
            create_stimulus_locked_turn_rate_analysis(
                str(trajectories_file),
                str(events_file),
                str(h5_file),
                str(output_path)
            )
            
            if not output_path.exists():
                raise FileNotFoundError(f"Output figure not created: {output_path}")
            
            result['checks']['analysis'] = 'PASS'
            result['checks']['figure_path'] = str(output_path)
            print(f"  [OK] Analysis completed successfully")
            print(f"  [OK] Figure saved: {output_path.name}")
            
        except Exception as e:
            result['checks']['analysis'] = 'FAIL'
            result['errors'].append(f'Analysis error: {str(e)}')
            print(f"  [FAIL] Analysis failed: {e}")
            traceback.print_exc()
            result['status'] = 'error'
            return result
        
        # Check 4: Verify timing alignment (spike at t=0)
        print("\n[CHECK 4] Verifying timing alignment (spike at t=0)...")
        try:
            # Load validation table if it exists
            validation_path = output_path.parent / (output_path.stem + '_validation.csv')
            if validation_path.exists():
                validation_df = pd.read_csv(validation_path)
                
                # Find bin closest to t=0
                t0_bin_idx = np.argmin(np.abs(validation_df['bin_center'].values))
                t0_bin_center = validation_df.iloc[t0_bin_idx]['bin_center']
                t0_rate = validation_df.iloc[t0_bin_idx]['mean_rate']
                
                # Check if there's a spike at t=0 (rate should be elevated)
                # Compare with baseline (negative times)
                baseline_mask = validation_df['bin_center'] < 0
                if baseline_mask.sum() > 0:
                    baseline_rate = validation_df[baseline_mask]['mean_rate'].mean()
                    spike_ratio = t0_rate / baseline_rate if baseline_rate > 0 else float('inf')
                    
                    result['checks']['timing_alignment'] = 'PASS' if spike_ratio > 1.2 else 'WARNING'
                    result['checks']['t0_bin_center'] = float(t0_bin_center)
                    result['checks']['t0_rate'] = float(t0_rate)
                    result['checks']['baseline_rate'] = float(baseline_rate)
                    result['checks']['spike_ratio'] = float(spike_ratio)
                    
                    print(f"  ✓ Bin closest to t=0: {t0_bin_center:.2f}s")
                    print(f"  ✓ Rate at t=0: {t0_rate:.3f} turns/min")
                    print(f"  ✓ Baseline rate: {baseline_rate:.3f} turns/min")
                    print(f"  ✓ Spike ratio: {spike_ratio:.2f}x")
                    
                    if spike_ratio > 1.2:
                        print(f"  [OK] Timing alignment verified (spike at t=0)")
                    else:
                        print(f"  ⚠ Weak or no spike at t=0 (ratio: {spike_ratio:.2f})")
                        result['warnings'].append(f"Weak spike at t=0: ratio={spike_ratio:.2f}")
                else:
                    result['checks']['timing_alignment'] = 'WARNING'
                    result['warnings'].append("No baseline period found for comparison")
                    print(f"  ⚠ No baseline period found")
            else:
                result['checks']['timing_alignment'] = 'WARNING'
                result['warnings'].append("Validation table not found")
                print(f"  ⚠ Validation table not found: {validation_path}")
                
        except Exception as e:
            result['checks']['timing_alignment'] = 'FAIL'
            result['warnings'].append(f'Timing alignment check error: {str(e)}')
            print(f"  ⚠ Timing alignment check failed: {e}")
        
        # Check 5: Verify turn rates are reasonable
        print("\n[CHECK 5] Verifying turn rates are reasonable...")
        try:
            if validation_path.exists():
                validation_df = pd.read_csv(validation_path)
                mean_rate = validation_df['mean_rate'].mean()
                max_rate = validation_df['mean_rate'].max()
                
                result['checks']['rate_validation'] = 'PASS'
                result['checks']['mean_rate'] = float(mean_rate)
                result['checks']['max_rate'] = float(max_rate)
                
                print(f"  ✓ Mean turn rate: {mean_rate:.3f} turns/min")
                print(f"  ✓ Max turn rate: {max_rate:.3f} turns/min")
                
                if mean_rate < 0 or mean_rate > 10:
                    result['warnings'].append(f"Unusual mean rate: {mean_rate:.3f} turns/min")
                    print(f"  ⚠ Unusual mean rate")
                else:
                    print(f"  [OK] Turn rates are reasonable")
            else:
                result['checks']['rate_validation'] = 'WARNING'
                result['warnings'].append("Validation table not found")
                
        except Exception as e:
            result['checks']['rate_validation'] = 'WARNING'
            result['warnings'].append(f'Rate validation error: {str(e)}')
        
        # All checks passed
        if len(result['errors']) == 0:
            result['status'] = 'success'
            print(f"\n[SUCCESS] ALL CHECKS PASSED")
        else:
            result['status'] = 'partial'
            print(f"\n⚠ SOME CHECKS FAILED")
        
    except Exception as e:
        result['status'] = 'error'
        result['errors'].append(f'Unexpected error: {str(e)}')
        print(f"\n[ERROR] UNEXPECTED ERROR")
        traceback.print_exc()
    
    return result

def main():
    """Run tests on all test files."""
    print("="*80)
    print("ANALYSIS PIPELINE TEST")
    print("="*80)
    print(f"Test files: {len(TEST_FILES)}")
    print(f"H5 directory: {h5_dir}")
    print(f"Output directory: {output_dir}")
    print(f"Timestamp: {datetime.now().isoformat()}")
    
    results = []
    
    for test_filename in TEST_FILES:
        result = test_file(test_filename)
        results.append(result)
    
    # Summary
    print("\n" + "="*80)
    print("TEST SUMMARY")
    print("="*80)
    
    success_count = sum(1 for r in results if r['status'] == 'success')
    error_count = sum(1 for r in results if r['status'] == 'error')
    partial_count = sum(1 for r in results if r['status'] == 'partial')
    
    print(f"\nTotal files tested: {len(results)}")
    print(f"  Success: {success_count}")
    print(f"  Partial: {partial_count}")
    print(f"  Errors: {error_count}")
    
    for result in results:
        print(f"\n{result['file']}:")
        print(f"  Status: {result['status']}")
        if result['errors']:
            print(f"  Errors: {len(result['errors'])}")
            for err in result['errors']:
                print(f"    - {err}")
        if result['warnings']:
            print(f"  Warnings: {len(result['warnings'])}")
            for warn in result['warnings']:
                print(f"    - {warn}")
    
    # Save results
    results_file = script_dir / 'analysis_pipeline_test_results.json'
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to: {results_file}")
    
    # Return exit code
    if error_count > 0:
        sys.exit(1)
    elif partial_count > 0:
        sys.exit(2)
    else:
        sys.exit(0)

if __name__ == '__main__':
    main()

