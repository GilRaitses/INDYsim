@echo off
REM Process a single ESET folder from MATLAB data to H5 format
REM Usage: process_single_eset.bat [eset_name] [output_dir]
REM Example: process_single_eset.bat T_Re_Sq_0to250PWM_30#T_Bl_Sq_5to15PWM_30 data\h5_files

setlocal enabledelayedexpansion

REM Check arguments
if "%~1"=="" (
    echo ERROR: ESET name required
    echo Usage: process_single_eset.bat [eset_name] [output_dir]
    echo Example: process_single_eset.bat T_Re_Sq_0to250PWM_30#T_Bl_Sq_5to15PWM_30 data\h5_files
    pause
    exit /b 1
)

set ESET_NAME=%~1
set OUTPUT_DIR=data\h5_files
if not "%~2"=="" set OUTPUT_DIR=%~2

echo ========================================
echo MATLAB to H5 Conversion - Single ESET
echo ========================================
echo.
echo ESET: %ESET_NAME%
echo Output directory: %OUTPUT_DIR%
echo.

REM Base directory for MATLAB data
set BASE_DIR=data\matlab_data\GMR61@GMR61
set ESET_DIR=%BASE_DIR%\%ESET_NAME%

REM Check if ESET directory exists
if not exist "%ESET_DIR%" (
    echo ERROR: ESET directory not found: %ESET_DIR%
    echo.
    echo Available ESET folders:
    dir /b /ad "%BASE_DIR%"
    pause
    exit /b 1
)

echo Processing: %ESET_NAME%
echo.

python scripts\2025-11-11\convert_matlab_to_h5.py --eset-dir "%ESET_DIR%" --output-dir "%OUTPUT_DIR%"

if errorlevel 1 (
    echo.
    echo ERROR: Failed to process %ESET_NAME%
    pause
    exit /b 1
) else (
    echo.
    echo SUCCESS: Processed %ESET_NAME%
    echo H5 file saved to: %OUTPUT_DIR%
    pause
    exit /b 0
)

