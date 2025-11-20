# MATLAB Engine Setup for H5 Export

## Issue
The H5 export script requires MATLAB Engine API for Python, which must be installed separately from MATLAB.

## Installation

The MATLAB Engine API needs to be installed from MATLAB's installation directory:

```bash
# Find your MATLAB installation (usually one of these):
# C:\Program Files\MATLAB\R2024a\extern\engines\python
# C:\Apps-SU\MathWorks\MATLAB\R2024a\extern\engines\python

# Navigate to MATLAB's Python engine directory
cd "C:\Program Files\MATLAB\R2024a\extern\engines\python"
# OR
cd "C:\Apps-SU\MathWorks\MATLAB\R2024a\extern\engines\python"

# Install for current user
python setup.py install --user

# Or install system-wide (requires admin)
python setup.py install
```

## Verify Installation

```bash
python -c "import matlab.engine; print('MATLAB engine available')"
```

## Running the Export

Once MATLAB engine is installed, run:

```bash
cd D:\mechanosensation\scripts\2025-11-10
python batch_export_h5_reusable.py D:\rawdata\GMR61@GMR61\T_Re_Sq_0to250PWM_30#C_Bl_7PWM
```

## Alternative: Use MATLAB's Python

If MATLAB engine isn't available in your default Python, MATLAB may have its own Python installation:

```bash
# MATLAB's Python is usually at:
# C:\Program Files\MATLAB\R2024a\sys\python\win64\python.exe

# Use that Python to run the export:
"C:\Program Files\MATLAB\R2024a\sys\python\win64\python.exe" D:\mechanosensation\scripts\2025-11-10\batch_export_h5_reusable.py D:\rawdata\GMR61@GMR61\T_Re_Sq_0to250PWM_30#C_Bl_7PWM
```

