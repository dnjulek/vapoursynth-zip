@echo OFF
:: This batch file exists to run RunWithBAT.ps1 without hassle

:: Check if pwsh is in the system's PATH
where pwsh >nul 2>nul
if %errorlevel% equ 0 (
    :: pwsh is in PATH, so run the script using Windows Powershell
    pwsh -NoProfile -NoLogo -ExecutionPolicy Bypass -File powershell\RunWithBAT.ps1
) else (
    :: pwsh is not in PATH, run the script using PowerShell Core
    powershell -NoProfile -NoLogo -ExecutionPolicy Bypass -File powershell\RunWithBAT.ps1
)

pause