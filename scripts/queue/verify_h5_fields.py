#!/usr/bin/env python3
"""Verify H5 files have all required fields for segmentation"""

import h5py
from pathlib import Path

h5_file = Path(r"D:\rawdata\GMR61@GMR61\T_Re_Sq_0to250PWM_30#C_Bl_7PWM\h5_exports\GMR61@GMR61_T_Re_Sq_0to250PWM_30#C_Bl_7PWM_202509051201.h5")

print("=" * 70)
print("VERIFYING H5 FILE STRUCTURE")
print("=" * 70)
print(f"File: {h5_file.name}\n")

with h5py.File(h5_file, 'r') as f:
    track = f['tracks/track_1']
    
    # Check derivation_rules
    print("DERIVATION RULES:")
    if 'derivation_rules' in track:
        dr = track['derivation_rules']
        print("  [OK] derivation_rules group found")
        print(f"     interpTime: {dr.attrs.get('interpTime', 'MISSING')}")
        print(f"     smoothTime: {dr.attrs.get('smoothTime', 'MISSING')}")
        print(f"     derivTime: {dr.attrs.get('derivTime', 'MISSING')}")
    else:
        print("  [MISSING] derivation_rules group MISSING")
    
    print()
    
    # Check segment_options
    print("SEGMENT OPTIONS:")
    if 'segment_options' in track:
        so = track['segment_options']
        print("  [OK] segment_options group found")
        print(f"     curv_cut: {so.attrs.get('curv_cut', 'MISSING')}")
        print(f"     theta_cut: {so.attrs.get('theta_cut', 'MISSING')}")
        print(f"     minRunTime: {so.attrs.get('minRunTime', 'MISSING')}")
        print(f"     minRunLength: {so.attrs.get('minRunLength', 'MISSING')}")
    else:
        print("  [MISSING] segment_options group MISSING")
    
    print()
    
    # Check derived quantities
    print("DERIVED QUANTITIES (required for segmentation):")
    dq = track['derived_quantities']
    required_fields = ['curv', 'vel_dp', 'speed', 'sspineTheta', 'spineTheta', 'spineLength', 'theta', 'eti']
    for field in required_fields:
        status = "[OK]" if field in dq else "[MISSING]"
        print(f"  {status} {field}: {'present' if field in dq else 'MISSING'}")
    
    print()
    
    # Check ETI at root
    print("ETI AT ROOT:")
    if 'eti' in f:
        eti = f['eti']
        print(f"  [OK] ETI found: {len(eti)} frames")
        print(f"     Range: {eti[0]:.3f} to {eti[-1]:.3f} seconds")
    else:
        print("  [MISSING] ETI MISSING at root")

print("\n" + "=" * 70)

