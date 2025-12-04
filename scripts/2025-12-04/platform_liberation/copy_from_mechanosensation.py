"""
copy_from_mechanosensation.py - Copy all validation and export scripts

This script copies the complete platform liberation prototype from
mechanosensation to INDYsim for consolidation.

Run: python copy_from_mechanosensation.py
"""

import shutil
from pathlib import Path

# Source directories
MECHANO_VALIDATION = Path(r"D:\mechanosensation\scripts\2025-12-04\validation")
MATLAB_CONVERSION = Path(r"D:\INDYsim\src\@matlab_conversion")

# Destination
DEST_BASE = Path(r"D:\INDYsim\scripts\2025-12-04\platform_liberation")

def main():
    print("=" * 70)
    print("COPYING PLATFORM LIBERATION PROTOTYPE TO INDYsim")
    print("=" * 70)
    
    # Create directories
    (DEST_BASE / "validation" / "matlab").mkdir(parents=True, exist_ok=True)
    (DEST_BASE / "validation" / "python").mkdir(parents=True, exist_ok=True)
    (DEST_BASE / "h5_export").mkdir(parents=True, exist_ok=True)
    
    # Copy validation MATLAB scripts
    print("\n1. Copying MATLAB validation scripts...")
    matlab_src = MECHANO_VALIDATION / "matlab"
    matlab_dest = DEST_BASE / "validation" / "matlab"
    for f in matlab_src.glob("*.m"):
        shutil.copy2(f, matlab_dest / f.name)
        print(f"   {f.name}")
    
    # Copy validation Python scripts
    print("\n2. Copying Python validation scripts...")
    python_src = MECHANO_VALIDATION / "python"
    python_dest = DEST_BASE / "validation" / "python"
    for f in python_src.glob("*.py"):
        shutil.copy2(f, python_dest / f.name)
        print(f"   {f.name}")
    
    # Copy top-level validation scripts
    print("\n3. Copying top-level validation scripts...")
    validation_dest = DEST_BASE / "validation"
    top_level_scripts = [
        "run_full_validation.py",
        "run_matlab_validation.py",
        "compare_real_data.py",
        "copy_validated_h5s.py",
        "check_camcal_fields.py",
        "inspect_h5_structure.py",
        "batch_process_all_esets.py",
    ]
    for script_name in top_level_scripts:
        src = MECHANO_VALIDATION / script_name
        if src.exists():
            shutil.copy2(src, validation_dest / script_name)
            print(f"   {script_name}")
    
    # Copy documentation
    print("\n4. Copying documentation...")
    docs = ["README.md", "FIELD_MAPPING.md", "VALIDATION_REPORT.md", "DISCREPANCY_REPORT.md"]
    for doc in docs:
        src = MECHANO_VALIDATION / doc
        if src.exists():
            shutil.copy2(src, validation_dest / doc)
            print(f"   {doc}")
    
    # Copy H5 export scripts from @matlab_conversion
    print("\n5. Copying H5 export scripts...")
    h5_export_dest = DEST_BASE / "h5_export"
    export_scripts = [
        "convert_matlab_to_h5.py",
        "append_camcal_to_h5.py",
        "batch_export_esets.py",
        "unlock_h5_file.py",
        "README.md",
        "QUICK_START.md",
    ]
    for script_name in export_scripts:
        src = MATLAB_CONVERSION / script_name
        if src.exists():
            shutil.copy2(src, h5_export_dest / script_name)
            print(f"   {script_name}")
    
    # Copy engineer_data.py
    print("\n6. Copying analysis script...")
    engineer_src = Path(r"D:\mechanosensation\scripts\2025-12-04\engineer_data.py")
    if engineer_src.exists():
        shutil.copy2(engineer_src, DEST_BASE / "engineer_data.py")
        print(f"   engineer_data.py")
    
    print("\n" + "=" * 70)
    print("COPY COMPLETE")
    print("=" * 70)
    print(f"\nDestination: {DEST_BASE}")
    print("\nStructure:")
    print("  platform_liberation/")
    print("  ├── engineer_data.py          # Main analysis script")
    print("  ├── h5_export/                 # MATLAB→H5 conversion")
    print("  │   ├── convert_matlab_to_h5.py")
    print("  │   ├── append_camcal_to_h5.py")
    print("  │   └── ...")
    print("  └── validation/                # Validation framework")
    print("      ├── matlab/                # MATLAB validation scripts")
    print("      ├── python/                # Python validation scripts")
    print("      ├── run_full_validation.py")
    print("      └── FIELD_MAPPING.md")


if __name__ == "__main__":
    main()

