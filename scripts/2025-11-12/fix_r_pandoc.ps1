# PowerShell script to fix R/pandoc configuration
# Run with: Start-Process powershell -Verb RunAs -ArgumentList "-File scripts/2025-11-12/fix_r_pandoc.ps1"

# Set RSTUDIO_PANDOC system-wide to use Quarto's pandoc
[System.Environment]::SetEnvironmentVariable("RSTUDIO_PANDOC", "C:\Apps-SU\quarto\bin\pandoc.exe", [System.EnvironmentVariableTarget]::Machine)

Write-Host "RSTUDIO_PANDOC set to: C:\Apps-SU\quarto\bin\pandoc.exe"
Write-Host "You may need to restart R/RStudio/Quarto for changes to take effect"


