#!/usr/bin/env python3
"""
Simple test script that uses the production analysis script directly.
Tests the analysis pipeline by running it on test files.
"""

import sys
import subprocess
import json
from pathlib import Path
from datetime import datetime

# Test files from Task 0.1
TEST_FILES = [
    "GMR61@GMR61_T_Re_Sq_0to250PWM_30#C_Bl_7PWM_202510291652.h5",
    "GMR61@GMR61_T_Re_Sq_0to250PWM_30#T_Bl_Sq_5to15PWM_30_202510301513.h5"
]

# Try to use config
script_dir = Path(__file__).parent
project_root = script_dir.parent.parent

try:
    config_path = script_dir.parent / 'config.py'
    if config_path.exists():
        sys.path.insert(0, str(config_path.parent))
        from config import H5_FILES_DIR
        h5_dir = H5_FILES_DIR
    else:
        h5_dir = project_root / 'data' / 'h5_files'
except ImportError:
    h5_dir = project_root / 'data' / 'h5_files'

def test_with_production_script(h5_filename):
    """Test using the production script directly."""
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
        # Modify production script temporarily to use this file
        production_script = project_root / "scripts" / "queue" / "run_stimulus_locked_analysis_production.py"
        
        if not production_script.exists():
            result['status'] = 'error'
            result['errors'].append(f'Production script not found: {production_script}')
            return result
        
        # Read production script
        with open(production_script, 'r') as f:
            script_content = f.read()
        
        # Fix path setup - production script uses script_dir / "scripts" which is wrong
        # It should use project_root / "scripts" instead
        import re
        
        # Fix the path setup line (line 86)
        # Change: sys.path.insert(0, str(script_dir / "scripts"))
        # To: sys.path.insert(0, str(Path(__file__).parent.parent))
        path_pattern = r'sys\.path\.insert\(0, str\(script_dir / "scripts"\)\)'
        path_replacement = 'sys.path.insert(0, str(Path(__file__).parent.parent))'
        script_content = re.sub(path_pattern, path_replacement, script_content)
        
        # Always modify h5_file path if it exists
        if 'h5_file = Path(r"' in script_content:
            # Temporarily modify to use test file
            pattern = r'h5_file = Path\(r"[^"]+"\)'
            # Escape backslashes in replacement
            escaped_path = str(h5_file).replace('\\', '\\\\')
            replacement = f'h5_file = Path(r"{escaped_path}")'
            modified_content = re.sub(pattern, replacement, script_content)
        elif 'h5_file = Path(' in script_content:
            # Try alternative pattern
            pattern = r'h5_file = Path\([^)]+\)'
            escaped_path = str(h5_file).replace('\\', '\\\\')
            replacement = f'h5_file = Path(r"{escaped_path}")'
            modified_content = re.sub(pattern, replacement, script_content)
        else:
            # If no h5_file assignment found, add it after script_dir definition
            modified_content = script_content
            if 'script_dir = Path(__file__).parent' in script_content:
                # Insert h5_file assignment after script_dir
                insert_pos = script_content.find('script_dir = Path(__file__).parent')
                next_line = script_content.find('\n', insert_pos)
                escaped_path = str(h5_file).replace('\\', '\\\\')
                h5_line = f'\n    h5_file = Path(r"{escaped_path}")'
                modified_content = script_content[:next_line] + h5_line + script_content[next_line:]
        
        # Write to temp file
        temp_script = script_dir / f"temp_analysis_{h5_filename.replace('.h5', '').replace('@', '_').replace('#', '_')}.py"
        with open(temp_script, 'w') as f:
            f.write(modified_content)
        
        # Run modified script
        print(f"\n[RUNNING] Production analysis script...")
        print(f"  Script: {temp_script.name}")
        
        try:
            proc = subprocess.run(
                [sys.executable, str(temp_script)],
                capture_output=True,
                text=True,
                timeout=600  # 10 minute timeout
            )
            
            if proc.returncode == 0:
                result['checks']['script_execution'] = 'PASS'
                result['status'] = 'success'
                print(f"  [OK] Script executed successfully")
                
                # Check for output files
                experiment_id = h5_file.stem.replace(' ', '_')
                output_dir = project_root / 'data' / 'engineered'
                events_file = output_dir / f"{experiment_id}_events.csv"
                trajectories_file = output_dir / f"{experiment_id}_trajectories.csv"
                figures_dir = project_root / 'output' / 'figures' / 'eda'
                figure_file = figures_dir / f"{experiment_id}_stimulus_locked_turn_rate.png"
                
                if events_file.exists():
                    result['checks']['events_file'] = 'PASS'
                    result['checks']['events_path'] = str(events_file)
                    print(f"  [OK] Events CSV created")
                else:
                    result['warnings'].append("Events CSV not found")
                
                if trajectories_file.exists():
                    result['checks']['trajectories_file'] = 'PASS'
                    result['checks']['trajectories_path'] = str(trajectories_file)
                    print(f"  [OK] Trajectories CSV created")
                else:
                    result['warnings'].append("Trajectories CSV not found")
                
                if figure_file.exists():
                    result['checks']['figure_file'] = 'PASS'
                    result['checks']['figure_path'] = str(figure_file)
                    print(f"  [OK] Figure created")
                else:
                    result['warnings'].append("Figure not found")
                
            else:
                result['checks']['script_execution'] = 'FAIL'
                result['status'] = 'error'
                result['errors'].append(f'Script failed with return code {proc.returncode}')
                result['errors'].append(f'STDOUT: {proc.stdout[:500]}')
                result['errors'].append(f'STDERR: {proc.stderr[:500]}')
                print(f"  [FAIL] Script execution failed")
                print(f"  Return code: {proc.returncode}")
                if proc.stderr:
                    print(f"  Error: {proc.stderr[:500]}")
            
        except subprocess.TimeoutExpired:
            result['checks']['script_execution'] = 'FAIL'
            result['status'] = 'error'
            result['errors'].append('Script execution timed out (>10 minutes)')
            print(f"  [FAIL] Script execution timed out")
        
        finally:
            # Clean up temp script
            if temp_script.exists():
                temp_script.unlink()
    
    except Exception as e:
        result['status'] = 'error'
        result['errors'].append(f'Unexpected error: {str(e)}')
        print(f"  [ERROR] Unexpected error: {e}")
        import traceback
        traceback.print_exc()
    
    return result

def main():
    """Run tests on all test files."""
    print("="*80)
    print("ANALYSIS PIPELINE TEST")
    print("="*80)
    print(f"Test files: {len(TEST_FILES)}")
    print(f"H5 directory: {h5_dir}")
    print(f"Timestamp: {datetime.now().isoformat()}")
    
    results = []
    
    for test_filename in TEST_FILES:
        result = test_with_production_script(test_filename)
        results.append(result)
    
    # Summary
    print("\n" + "="*80)
    print("TEST SUMMARY")
    print("="*80)
    
    success_count = sum(1 for r in results if r['status'] == 'success')
    error_count = sum(1 for r in results if r['status'] == 'error')
    
    print(f"\nTotal files tested: {len(results)}")
    print(f"  Success: {success_count}")
    print(f"  Errors: {error_count}")
    
    for result in results:
        print(f"\n{result['file']}:")
        print(f"  Status: {result['status']}")
        if result['errors']:
            print(f"  Errors: {len(result['errors'])}")
            for err in result['errors'][:3]:  # Show first 3 errors
                print(f"    - {err}")
        if result['warnings']:
            print(f"  Warnings: {len(result['warnings'])}")
            for warn in result['warnings'][:3]:  # Show first 3 warnings
                print(f"    - {warn}")
    
    # Save results
    results_file = script_dir / 'analysis_pipeline_test_results.json'
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to: {results_file}")
    
    # Return exit code
    if error_count > 0:
        sys.exit(1)
    else:
        sys.exit(0)

if __name__ == '__main__':
    main()

