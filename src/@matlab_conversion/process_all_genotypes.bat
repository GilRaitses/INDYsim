@echo off
REM Batch script to process all genotype folders in matlab_data
REM Usage: process_all_genotypes.bat

setlocal

echo ================================================================================
echo BATCH PROCESSING: ALL GENOTYPES
echo ================================================================================
echo.

REM Change to script directory
cd /d "%~dp0"

REM Find all genotype folders
set DATA_DIR=..\..\data\matlab_data

if not exist "%DATA_DIR%" (
    echo [ERROR] MATLAB data directory not found: %DATA_DIR%
    exit /b 1
)

echo Scanning for genotype folders in %DATA_DIR%...
echo.

REM Process each genotype folder
for /d %%G in ("%DATA_DIR%\*@*") do (
    setlocal enabledelayedexpansion
    set GENOTYPE=%%~nG
    echo ================================================================================
    echo Processing genotype: !GENOTYPE!
    echo ================================================================================
    echo.
    
    python batch_export_esets.py --genotype "!GENOTYPE!"
    
    if !ERRORLEVEL! NEQ 0 (
        echo.
        echo [WARNING] Failed to process !GENOTYPE!
    )
    
    echo.
    echo --------------------------------------------------------------------------------
    echo.
    endlocal
)

echo ================================================================================
echo [COMPLETE] Batch processing finished for all genotypes
echo ================================================================================

endlocal






