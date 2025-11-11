@echo off
REM Process all 4 ESET folders from MATLAB data to H5 format
REM Usage: process_all_esets.bat [output_dir]
REM Default output: data/h5_files

setlocal enabledelayedexpansion

REM Set default output directory
set OUTPUT_DIR=data\h5_files
if not "%~1"=="" set OUTPUT_DIR=%~1

echo ========================================
echo MATLAB to H5 Conversion - All ESETs
echo ========================================
echo.
echo Output directory: %OUTPUT_DIR%
echo.

REM Base directory for MATLAB data
set BASE_DIR=data\matlab_data\GMR61@GMR61

REM Check if base directory exists
if not exist "%BASE_DIR%" (
    echo ERROR: Base directory not found: %BASE_DIR%
    pause
    exit /b 1
)

REM List of ESET folders to process
set ESET1=T_Re_Sq_0to250PWM_30#T_Bl_Sq_5to15PWM_30
set ESET2=T_Re_Sq_0to250PWM_30#C_Bl_7PWM
set ESET3=T_Re_Sq_50to250PWM_30#C_Bl_7PWM
set ESET4=T_Re_Sq_50to250PWM_30#T_Bl_Sq_5to15PWM_30

echo Processing 4 ESET folders...
echo.

REM Process each ESET
call :process_eset "%BASE_DIR%\%ESET1%" "%OUTPUT_DIR%"
call :process_eset "%BASE_DIR%\%ESET2%" "%OUTPUT_DIR%"
call :process_eset "%BASE_DIR%\%ESET3%" "%OUTPUT_DIR%"
call :process_eset "%BASE_DIR%\%ESET4%" "%OUTPUT_DIR%"

echo.
echo ========================================
echo Conversion complete!
echo ========================================
echo.
echo H5 files saved to: %OUTPUT_DIR%
echo.
pause
exit /b 0

:process_eset
set ESET_DIR=%~1
set OUT_DIR=%~2
set ESET_NAME=%~nx1

echo ----------------------------------------
echo Processing: %ESET_NAME%
echo ----------------------------------------

if not exist "%ESET_DIR%" (
    echo ERROR: ESET directory not found: %ESET_DIR%
    goto :eof
)

python scripts\2025-11-11\convert_matlab_to_h5.py --eset-dir "%ESET_DIR%" --output-dir "%OUT_DIR%"

if errorlevel 1 (
    echo ERROR: Failed to process %ESET_NAME%
) else (
    echo SUCCESS: Processed %ESET_NAME%
)

goto :eof

