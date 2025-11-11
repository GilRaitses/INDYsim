@echo off
REM Open ESET batch processing monitor
REM Usage: open_eset_monitor.bat [GENOTYPE]

setlocal

if "%~1"=="" (
    set GENOTYPE=GMR61@GMR61
    echo [INFO] No genotype specified, using default: %GENOTYPE%
) else (
    set GENOTYPE=%~1
)

cd /d "%~dp0"

python monitor_eset_batch.py --genotype-dir "..\..\data\matlab_data\%GENOTYPE%" --output-dir "..\..\data\h5_files"

endlocal

