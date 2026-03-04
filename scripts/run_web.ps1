<#
.SYNOPSIS
    Launch the FastAPI + React web UI for the Synthetic Data Generator.

.DESCRIPTION
    Starts uvicorn serving web.api:app on port 8502.
    Automatically installs fastapi and uvicorn if missing.

.PARAMETER Reload
    Pass -Reload to enable auto-reload during development.

.EXAMPLE
    .\scripts\run_web.ps1
    .\scripts\run_web.ps1 -Reload
#>
param(
    [switch]$Reload
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

# Resolve repo root (script lives in <repo>/scripts/)
$ScriptDir  = Split-Path -Parent $MyInvocation.MyCommand.Definition
$RepoRoot   = (Resolve-Path (Join-Path $ScriptDir "..")).Path

Push-Location $RepoRoot
try {
    # Activate venv if present
    $VenvActivate = Join-Path $RepoRoot ".venv\Scripts\Activate.ps1"
    if (Test-Path $VenvActivate) {
        Write-Host "Activating virtual environment..." -ForegroundColor DarkGray
        & $VenvActivate
    }

    # Ensure dependencies
    $Missing = @()
    python -c "import fastapi" 2>$null
    if ($LASTEXITCODE -ne 0) { $Missing += "fastapi" }
    python -c "import uvicorn" 2>$null
    if ($LASTEXITCODE -ne 0) { $Missing += "uvicorn[standard]" }

    if ($Missing.Count -gt 0) {
        Write-Host "Installing missing packages: $($Missing -join ', ')" -ForegroundColor Yellow
        pip install @Missing --quiet
    }

    # Launch
    $Port = 8502
    $Host_ = "127.0.0.1"
    $Url = "http://${Host_}:${Port}"
    $Args_ = @("--host", $Host_, "--port", $Port)
    if ($Reload) { $Args_ += "--reload" }

    Write-Host ""
    Write-Host "Starting web UI at $Url" -ForegroundColor Cyan
    Write-Host "Press Ctrl+C to stop." -ForegroundColor DarkGray
    Write-Host ""

    # Open browser after a short delay (non-blocking)
    Start-Job -ScriptBlock {
        param($u)
        Start-Sleep -Milliseconds 1500
        Start-Process $u
    } -ArgumentList $Url | Out-Null

    python -m uvicorn web.api:app @Args_
}
finally {
    Pop-Location
}
