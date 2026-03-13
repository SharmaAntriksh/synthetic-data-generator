# ------------------------------------------------------------
# Sync (create/update) a project virtual environment
#
# Behavior:
#   - Uses uv (preferred) when available for locked, reproducible installs
#   - Falls back to pip + requirements.txt if uv is not installed
#   - Resolves project root (walks upward for .git / pyproject.toml / requirements.txt)
#   - Creates the venv if missing (or recreates with -Force)
#   - Installs dependencies only when inputs change (hash stamp or uv lockfile)
#
# Exit codes:
#   0 = OK
#   1 = failure (missing python, venv create failed, install failed)
#
# Usage:
#   .\scripts\sync_venv.ps1
#   .\scripts\sync_venv.ps1 -Force
#   .\scripts\sync_venv.ps1 -Dev
#   .\scripts\sync_venv.ps1 -Requirements requirements-dev.txt
#   .\scripts\sync_venv.ps1 -Constraints constraints.txt
#   .\scripts\sync_venv.ps1 -Quiet
# ------------------------------------------------------------

[CmdletBinding()]
param(
    [string]$VenvDir = ".venv",
    [string]$Requirements = "requirements.txt",
    [string]$Constraints = "",
    [switch]$Force,
    [switch]$Dev,
    [switch]$NoUpgradePip,
    [switch]$Quiet,
    [version]$MinPythonVersion = "3.11",
    [string]$ProjectRoot
)

$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest

. (Join-Path $PSScriptRoot "_common.ps1")

# ---- Stamp helpers (local to this script, pip fallback only) ----

function Read-Stamp {
    param([string]$Path)
    if (-not (Test-Path -LiteralPath $Path)) { return $null }
    try {
        $raw = Get-Content -LiteralPath $Path -Raw
        if ([string]::IsNullOrWhiteSpace($raw)) { return $null }
        return ($raw | ConvertFrom-Json)
    } catch {
        Write-Warning "Stamp file is corrupted ($Path): $($_.Exception.Message). Will re-sync."
        return $null
    }
}

function Write-Stamp {
    param(
        [string]$Path,
        [string]$ReqHash,
        [string]$ConHash,
        [string]$PyVer
    )
    $obj = [ordered]@{
        requirements_sha256 = $ReqHash
        constraints_sha256  = $ConHash
        python_version      = $PyVer
        stamped_utc         = (Get-Date).ToUniversalTime().ToString("o")
    }
    ($obj | ConvertTo-Json -Depth 3) | Set-Content -LiteralPath $Path -Encoding UTF8
}

function Get-FileHashOrEmpty {
    param([string]$Path)
    if ([string]::IsNullOrWhiteSpace($Path)) { return "" }
    if (-not (Test-Path -LiteralPath $Path)) { return "" }
    return (Get-FileHash -LiteralPath $Path -Algorithm SHA256).Hash
}

# Helper to conditionally log
function Log {
    param([string]$Msg, [string]$Level = "info")
    if (-not $Quiet) { Write-Step $Msg -Level $Level }
}

# ---- Main ----

try {
    $RootPath = if ($ProjectRoot) { (Resolve-Path $ProjectRoot).Path } else { Resolve-ProjectRoot -StartDir $PSScriptRoot }
    $VenvPath = Join-Path $RootPath $VenvDir

    Log "Project root  : $RootPath"
    Log "Venv path     : $VenvPath"

    # Handle -Force: remove existing venv
    if ($Force -and (Test-Path -LiteralPath $VenvPath)) {
        Log "Removing existing venv (-Force)..." -Level warn
        Remove-Item -LiteralPath $VenvPath -Recurse -Force
    }

    # --- Try uv first ---
    $useUv = $false
    if (Get-Command uv -ErrorAction SilentlyContinue) {
        $useUv = $true
    }

    if ($useUv) {
        Log "Using uv sync (locked dependencies)" -Level ok

        $uvArgs = @("sync", "--project", $RootPath)
        if ($Dev) {
            $uvArgs += "--dev"
        } else {
            $uvArgs += "--no-dev"
        }
        if ($Quiet) {
            $uvArgs += "--quiet"
        }

        Invoke-Checked -Exe "uv" -Arguments $uvArgs -ErrorMessage "uv sync failed."
        Log "Virtual environment synced successfully (uv, locked)." -Level ok
        exit 0
    }

    # --- Fallback: pip + requirements.txt ---
    Log "uv not found — falling back to pip" -Level warn
    Log "  Tip: install uv for locked, reproducible installs: pip install uv" -Level info

    $ReqFile    = Join-Path $RootPath $Requirements
    $ConFile    = if ([string]::IsNullOrWhiteSpace($Constraints)) { "" } else { (Join-Path $RootPath $Constraints) }
    $VenvPython = Join-Path $VenvPath "Scripts\python.exe"
    $StampFile  = Join-Path $VenvPath ".requirements.stamp.json"

    Log "Requirements  : $ReqFile"
    if ($ConFile) { Log "Constraints   : $ConFile" }
    Log "Stamp file    : $StampFile"

    # Locate system python for venv creation / version check
    $runner = Get-PythonRunner -MinVersion $MinPythonVersion
    if (-not $runner) {
        Log "Python not found. Install Python $MinPythonVersion+ and ensure 'py' or 'python' is available." -Level err
        exit 1
    }

    $sysPyVer = Get-PythonVersion -Runner $runner
    if ($sysPyVer -lt $MinPythonVersion) {
        Log "Python version too old: $sysPyVer (required: $MinPythonVersion+)" -Level err
        exit 1
    }

    # Create venv if needed
    if (-not (Test-Path -LiteralPath $VenvPython)) {
        Log "Creating virtual environment..." -Level cmd
        $venvArgs = @() + $runner.Args + @("-m", "venv", $VenvPath)
        Invoke-Checked -Exe $runner.Cmd -Arguments $venvArgs -ErrorMessage "Failed to create venv at: $VenvPath"

        if (-not (Test-Path -LiteralPath $VenvPython)) {
            throw "Venv python not found after creation: $VenvPython"
        }
    }

    # No requirements file means nothing to sync (but venv is ensured)
    if (-not (Test-Path -LiteralPath $ReqFile)) {
        Log "Requirements file not found; skipping dependency sync." -Level warn
        exit 0
    }

    if ($ConFile -and (-not (Test-Path -LiteralPath $ConFile))) {
        throw "Constraints file not found: $ConFile"
    }

    # Decide whether sync is needed
    $reqHash = Get-FileHashOrEmpty -Path $ReqFile
    $conHash = Get-FileHashOrEmpty -Path $ConFile
    $venvPyVer = (& $VenvPython -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}")' 2>$null).Trim()

    $stamp = Read-Stamp -Path $StampFile
    $stampPyNorm = if ($stamp -and $stamp.python_version) { try { [version]$stamp.python_version } catch { $stamp.python_version } } else { $null }
    $venvPyNorm  = try { [version]$venvPyVer } catch { $venvPyVer }
    $needsSync =
        (-not $stamp) -or
        ($stamp.requirements_sha256 -ne $reqHash) -or
        ($stamp.constraints_sha256  -ne $conHash) -or
        ($stampPyNorm -ne $venvPyNorm)

    if (-not $needsSync) {
        Log "Virtual environment already up to date." -Level ok
        exit 0
    }

    Log "Updating virtual environment..." -Level cmd

    # Upgrade packaging tooling unless explicitly disabled
    if (-not $NoUpgradePip) {
        $pipArgs = @("-m","pip","install","--upgrade","pip","setuptools","wheel","--disable-pip-version-check")
        if ($Quiet) { $pipArgs += "--quiet" }
        Invoke-Checked -Exe $VenvPython -Arguments $pipArgs -ErrorMessage "Failed to upgrade pip tooling."
    }

    # Install requirements
    $installArgs = @("-m","pip","install","-r",$ReqFile,"--disable-pip-version-check")
    if ($ConFile) { $installArgs += @("-c",$ConFile) }
    if ($Quiet) { $installArgs += "--quiet" }

    & $VenvPython @installArgs
    $pipExit = $LASTEXITCODE
    if ($pipExit -ne 0) {
        Log "Dependency installation failed (exit $pipExit); stamp not updated." -Level err
        exit $pipExit
    }

    Write-Stamp -Path $StampFile -ReqHash $reqHash -ConHash $conHash -PyVer $venvPyVer
    Log "Virtual environment synced successfully (pip, unpinned transitive deps)." -Level ok
    exit 0
}
catch {
    Log "sync_venv failed: $($_.Exception.Message)" -Level err
    exit 1
}
