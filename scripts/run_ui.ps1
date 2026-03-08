# Run Streamlit UI (Windows / PowerShell)
# Examples:
#   .\scripts\run_ui.ps1
#   .\scripts\run_ui.ps1 -Port 8502 -Headless
#   .\scripts\run_ui.ps1 -Sync

[CmdletBinding(PositionalBinding=$false)]
param(
    [string] $AppPath = "ui/app.py",
    [string] $Config = "config.yaml",
    [string] $ModelsConfig = "models.yaml",
    [int]    $Port = 8501,
    [string] $ServerHost = "localhost",
    [switch] $Headless,
    [string] $VenvPath = ".venv",
    [switch] $Sync,
    [switch] $NoVenv,
    [Parameter(ValueFromRemainingArguments=$true)]
    [string[]] $AppArgs
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

. (Join-Path $PSScriptRoot "_common.ps1")

function Ensure-Venv {
    param([string] $ProjectRoot, [string] $VenvDir)
    $venvFull = Join-Path $ProjectRoot $VenvDir
    if (Test-Path $venvFull) { return $venvFull }
    Write-Step "Virtual environment not found. Creating: $venvFull" -Level warn
    Invoke-Checked -Exe python -Arguments @("-m", "venv", $venvFull) -ErrorMessage "Failed to create venv at: $venvFull"
    if (-not (Test-Path $venvFull)) { throw "Venv directory was not created: $venvFull" }
    return $venvFull
}

function Activate-Venv {
    param([string] $VenvFullPath)
    $activate = Join-Path $VenvFullPath "Scripts\Activate.ps1"
    if (-not (Test-Path $activate)) { throw "Activate script not found: $activate" }
    . $activate
    if (-not $env:VIRTUAL_ENV) { throw "Venv activation failed (VIRTUAL_ENV not set)." }
}

function Sync-Requirements {
    param([string] $ProjectRoot)
    $req = Join-Path $ProjectRoot "requirements.txt"
    if (-not (Test-Path $req)) {
        Write-Step "requirements.txt not found; skipping dependency install." -Level warn
        return
    }
    Write-Step "Syncing Python dependencies from requirements.txt..." -Level cmd
    Invoke-Checked -Exe python -Arguments @("-m", "pip", "install", "--upgrade", "pip", "--disable-pip-version-check") -ErrorMessage "pip upgrade failed."
    Invoke-Checked -Exe python -Arguments @("-m", "pip", "install", "-r", $req, "--disable-pip-version-check") -ErrorMessage "Dependency install from requirements.txt failed."
}

# ---- Main ----

$PROJECT_ROOT = Resolve-ProjectRoot -StartDir $PSScriptRoot

try {
    $hostPort = "{0}:{1}" -f $ServerHost, $Port

    Write-Step "Starting Streamlit UI" -Level cmd
    Write-Step "  Project root : $PROJECT_ROOT"
    Write-Step "  App          : $AppPath"
    Write-Step "  Config       : $Config"
    Write-Step "  Models       : $ModelsConfig"
    Write-Step "  Host:Port    : $hostPort"
    Write-Step "  Headless     : $Headless"
    Write-Step "  Use venv     : $(-not $NoVenv)"
    Write-Step "  Sync deps    : $Sync"

    Push-Location $PROJECT_ROOT

    $pyCheck = Get-PythonRunner -MinVersion "3.10"
    if (-not $pyCheck) { throw "Python not found. Install Python 3.10+ and reopen the terminal." }
    $pyVer = Get-PythonVersion -Runner $pyCheck
    $pyLabel = Format-RunnerLabel -Runner $pyCheck
    Write-Step "  Python       : $pyVer  [$pyLabel]"

    if (-not $NoVenv) {
        $venv = Ensure-Venv -ProjectRoot $PROJECT_ROOT -VenvDir $VenvPath
        Activate-Venv -VenvFullPath $venv
    }

    if ($Sync) { Sync-Requirements -ProjectRoot $PROJECT_ROOT }

    $appFull = Join-Path $PROJECT_ROOT $AppPath
    if (-not (Test-Path $appFull)) { throw "Streamlit app not found: $appFull" }

    $stFlags = @("run", $appFull, "--server.port=$Port", "--server.address=$ServerHost")
    if ($Headless) { $stFlags += "--server.headless=true" }

    $scriptArgs = @("--config", $Config, "--models-config", $ModelsConfig)

    if ($AppArgs -and $AppArgs.Count -gt 0 -and $AppArgs[0] -eq "--") {
        if ($AppArgs.Count -gt 1) { $AppArgs = $AppArgs[1..($AppArgs.Count-1)] } else { $AppArgs = @() }
    }
    if ($AppArgs -and $AppArgs.Count -gt 0) { $scriptArgs += $AppArgs }

    # Use the detected Python runner (not bare 'python') to ensure correct executable
    $pyCmd = $pyCheck.Cmd
    $pyArgs = @($pyCheck.Args)

    Write-Step "Launching Streamlit..." -Level ok
    & $pyCmd @pyArgs -m streamlit @stFlags -- @scriptArgs
    $ec = $LASTEXITCODE
    if ($ec -ne 0) { Write-Step "Streamlit exited with code $ec." -Level err }
    exit $ec
}
catch {
    Write-Step "run_ui failed: $($_.Exception.Message)" -Level err
    exit 1
}
finally {
    Pop-Location
}
