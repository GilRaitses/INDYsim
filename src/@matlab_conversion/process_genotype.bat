@echo off
REM Batch script to process all ESETs in a genotype folder
REM Usage: process_genotype.bat [GENOTYPE]
REM Example: process_genotype.bat GMR61@GMR61

setlocal

REM Get genotype from argument or use default
if "%~1"=="" (
    set GENOTYPE=GMR61@GMR61
    echo [INFO] No genotype specified, using default: %GENOTYPE%
) else (
    set GENOTYPE=%~1
)

echo ================================================================================
echo BATCH PROCESSING: %GENOTYPE%
echo ================================================================================
echo.

REM Change to script directory
cd /d "%~dp0"

REM Run batch export
python batch_export_esets.py --genotype "%GENOTYPE%"

if %ERRORLEVEL% EQU 0 (
    echo.
    echo ================================================================================
    echo [SUCCESS] Batch processing complete for %GENOTYPE%
    echo ================================================================================
) else (
    echo.
    echo ================================================================================
    echo [ERROR] Batch processing failed for %GENOTYPE%
    echo ================================================================================
    exit /b %ERRORLEVEL%
)

endlocal


