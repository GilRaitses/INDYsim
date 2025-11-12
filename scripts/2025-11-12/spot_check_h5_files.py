"""
Quick spot-check validation of H5 files from batch conversion.
Checks one file from each ESET to verify completeness.
"""

import h5py
from pathlib import Path

# Spot-check files: one from each ESET
spot_check_files = [
    r"D:\INDYsim\data\h5_files\GMR61@GMR61_T_Re_Sq_0to250PWM_30#C_Bl_7PWM_202510291652.h5",
    r"D:\INDYsim\data\h5_files\GMR61@GMR61_T_Re_Sq_0to250PWM_30#T_Bl_Sq_5to15PWM_30_202510301513.h5",
    r"D:\INDYsim\data\h5_files\GMR61@GMR61_T_Re_Sq_50to250PWM_30#C_Bl_7PWM_202510291435.h5",
    r"D:\INDYsim\data\h5_files\GMR61@GMR61_T_Re_Sq_50to250PWM_30#T_Bl_Sq_5to15PWM_30_202511051636.h5"
]

print("=" * 80)
print("Spot-Check Validation: One File Per ESET")
print("=" * 80)
print()

all_pass = True

for filepath in spot_check_files:
    filepath = Path(filepath)
    print(f"\n{'='*80}")
    print(f"Checking: {filepath.name}")
    print(f"{'='*80}")
    
    try:
        with h5py.File(filepath, 'r') as f:
            # Critical checks
            checks = {
                'ETI at root': 'eti' in f,
                'Root structure': all(k in f for k in ['eti', 'experiment_info', 'global_quantities', 'tracks', 'metadata']),
                'Global quantities': 'global_quantities' in f and len(list(f['global_quantities'].keys())) > 0,
                'Tracks': 'tracks' in f and len(list(f['tracks'].keys())) > 0,
                'Metadata': 'metadata' in f and 'has_eti' in f['metadata'].attrs
            }
            
            for check_name, result in checks.items():
                status = "✅" if result else "❌"
                print(f"  {status} {check_name}")
                if not result:
                    all_pass = False
            
            # Details
            if 'eti' in f:
                print(f"    ETI shape: {f['eti'].shape}")
            if 'tracks' in f:
                print(f"    Number of tracks: {len(list(f['tracks'].keys()))}")
            if 'global_quantities' in f:
                gq_keys = list(f['global_quantities'].keys())
                print(f"    Global quantities: {', '.join(gq_keys)}")
            
            file_size_mb = filepath.stat().st_size / (1024 * 1024)
            print(f"    File size: {file_size_mb:.1f} MB")
            
    except Exception as e:
        print(f"  ❌ ERROR: {e}")
        all_pass = False

print(f"\n{'='*80}")
print("Spot-Check Summary")
print(f"{'='*80}")
if all_pass:
    print("✅ All spot-check files PASS validation")
else:
    print("❌ Some files FAILED validation")

