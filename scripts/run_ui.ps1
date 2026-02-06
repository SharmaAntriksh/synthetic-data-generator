# ---------------------------------------------
# Run Streamlit UI (Windows / PowerShell)
# ---------------------------------------------
# Examples:
#   .\scripts\run_ui.ps1
#   .\scripts\run_ui.ps1 -Port 8502 -Headless
#   .\scripts\run_ui.ps1 -Sync
#   .\scripts\run_ui.ps1 -Config ".\config.yaml" -ModelsConfig ".\models.yaml"
#   .\scripts\run_ui.ps1 -- --some-app-arg foo
#
# Notes:
# - Remaining args are forwarded to the Streamlit script (ui/app.py) after `--`.
# - Uses python -m streamlit (more reliable than streamlit.exe on PATH).

[CmdletBinding(PositionalBinding=$false)]
param(
    [string] $AppPath = "ui/app.py",

    # Passed to the Streamlit app as script args
    [string] $Config = "config.yaml",
    [string] $ModelsConfig = "models.yaml",

    # Streamlit server flags
    [int]    $Port = 8501,
    [string] $ServerHost = "localhost",
    [switch] $Headless,

    # Environment / dependency control
    [string] $VenvPath = ".venv",
    [switch] $Sync,          # install/sync requirements.txt before launching
    [switch] $NoVenv,         # skip venv activation (assume current env is correct)

    # Any remaining args are forwarded to the Streamlit app
    [Parameter(ValueFromRemainingArguments=$true)]
    [string[]] $AppArgs
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

function Resolve-ProjectRoot {
    return (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
}

function Ensure-Python {
    $py = Get-Command python -ErrorAction SilentlyContinue
    if (-not $py) {
        throw "Python was not found in PATH. Install Python 3.10+ and reopen the terminal."
    }
}

function Ensure-Venv {
    param(
        [string] $ProjectRoot,
        [string] $VenvDir
    )

    $venvFull = Join-Path $ProjectRoot $VenvDir
    if (Test-Path $venvFull) {
        return $venvFull
    }

    Write-Host "Virtual environment not found. Creating: $venvFull" -ForegroundColor Yellow
    python -m venv $venvFull

    if (-not (Test-Path $venvFull)) {
        throw "Failed to create venv at: $venvFull"
    }
    return $venvFull
}

function Activate-Venv {
    param([string] $VenvFullPath)

    $activate = Join-Path $VenvFullPath "Scripts\Activate.ps1"
    if (-not (Test-Path $activate)) {
        throw "Activate script not found: $activate"
    }

    . $activate

    if (-not $env:VIRTUAL_ENV) {
        throw "Venv activation failed (VIRTUAL_ENV not set)."
    }
}

function Sync-Requirements {
    param([string] $ProjectRoot)

    $req = Join-Path $ProjectRoot "requirements.txt"
    if (-not (Test-Path $req)) {
        Write-Host "requirements.txt not found; skipping dependency install." -ForegroundColor DarkYellow
        return
    }

    Write-Host "Syncing Python dependencies from requirements.txt..." -ForegroundColor Yellow
    python -m pip install --upgrade pip
    python -m pip install -r $req
}

# ----------------- Main -----------------

$PROJECT_ROOT = Resolve-ProjectRoot

Write-Host "Starting Streamlit UI" -ForegroundColor Cyan
Write-Host "  Project root : $PROJECT_ROOT"
Write-Host "  App          : $AppPath"
Write-Host "  Config       : $Config"
Write-Host "  Models       : $ModelsConfig"
Write-Host "  Host:Port    : $ServerHost`:$Port"
Write-Host "  Headless     : $Headless"
Write-Host "  Use venv     : $(-not $NoVenv)"
Write-Host "  Sync deps    : $Sync"

Push-Location $PROJECT_ROOT
try {
    Ensure-Python

    if (-not $NoVenv) {
        $venv = Ensure-Venv -ProjectRoot $PROJECT_ROOT -VenvDir $VenvPath
        Activate-Venv -VenvFullPath $venv
    }

    if ($Sync) {
        Sync-Requirements -ProjectRoot $PROJECT_ROOT
    }

    $appFull = Join-Path $PROJECT_ROOT $AppPath
    if (-not (Test-Path $appFull)) {
        throw "Streamlit app not found: $appFull"
    }

    # Streamlit CLI flags
    $stFlags = @(
        "run", $appFull,
        "--server.port=$Port",
        "--server.address=$ServerHost"
    )
    if ($Headless) {
        $stFlags += "--server.headless=true"
    }

    # Pass config paths + any extra app args to the Streamlit script via `--`
    $scriptArgs = @("--config", $Config, "--models-config", $ModelsConfig)

    # If the caller used `--` separator, drop it.
    if ($AppArgs -and $AppArgs.Count -gt 0 -and $AppArgs[0] -eq "--") {
        if ($AppArgs.Count -gt 1) { $AppArgs = $AppArgs[1..($AppArgs.Count-1)] } else { $AppArgs = @() }
    }

    if ($AppArgs -and $AppArgs.Count -gt 0) {
        $scriptArgs += $AppArgs
    }

    Write-Host "Launching Streamlit..." -ForegroundColor Green
    python -m streamlit @stFlags -- @scriptArgs
}
finally {
    Pop-Location
}
