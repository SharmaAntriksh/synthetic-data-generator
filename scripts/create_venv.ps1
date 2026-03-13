# ------------------------------------------------------------
# Create virtual environment (one-time setup)
# Always creates .venv at project root by default.
# DOES NOT activate the environment.
#
# Uses uv (preferred) when available, falls back to pip.
#
# Usage:
#   .\scripts\create_venv.ps1
#   .\scripts\create_venv.ps1 -Force
#   .\scripts\create_venv.ps1 -Dev
#   .\scripts\create_venv.ps1 -Requirements requirements-dev.txt
# ------------------------------------------------------------

[CmdletBinding()]
param(
    [string]$VenvDir = ".venv",
    [string]$Requirements = "requirements.txt",
    [switch]$Force,
    [switch]$Dev,
    [version]$MinPythonVersion = "3.11"
)

$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest

. (Join-Path $PSScriptRoot "_common.ps1")

try {
    $ProjectRoot = Resolve-ProjectRoot -StartDir $PSScriptRoot
    $VenvPath    = Join-Path $ProjectRoot $VenvDir

    Write-Step "Project root : $ProjectRoot"
    Write-Step "Venv target  : $VenvPath"

    # Guard: already exists (unless -Force)
    if (Test-Path $VenvPath) {
        if (-not $Force) {
            Write-Step "Virtual environment already exists at $VenvPath" -Level warn
            Write-Step "Use -Force to recreate it." -Level info
            return
        }
        Write-Step "Removing existing venv (-Force)..." -Level warn
        Remove-Item -LiteralPath $VenvPath -Recurse -Force
    }

    # --- Try uv first (creates venv + installs locked deps in one step) ---
    $useUv = $false
    if (Get-Command uv -ErrorAction SilentlyContinue) {
        $useUv = $true
        Write-Step "Found uv — using uv sync (locked dependencies)" -Level ok
    }

    if ($useUv) {
        Write-Step "Creating venv and installing dependencies via uv..." -Level cmd
        $uvArgs = @("sync", "--project", $ProjectRoot)
        if ($Dev) {
            $uvArgs += "--dev"
        } else {
            $uvArgs += "--no-dev"
        }
        Invoke-Checked -Exe "uv" -Arguments $uvArgs -ErrorMessage "uv sync failed."
        Write-Step "Virtual environment created successfully (uv, locked)." -Level ok
    }
    else {
        # --- Fallback: pip + requirements.txt ---
        Write-Step "uv not found — falling back to pip" -Level warn
        Write-Step "  Tip: install uv for locked, reproducible installs: pip install uv" -Level info

        $ReqFile = Join-Path $ProjectRoot $Requirements

        # Locate python and verify version
        $py = Get-PythonRunner -MinVersion $MinPythonVersion
        if (-not $py) {
            throw "Python not found. Install Python $MinPythonVersion+ and ensure 'py' or 'python' is available."
        }

        $pyVer = Get-PythonVersion -Runner $py
        if ($pyVer -lt $MinPythonVersion) {
            throw "Python $MinPythonVersion+ is required. Found: $pyVer"
        }

        $label = Format-RunnerLabel -Runner $py
        Write-Step "Using Python $pyVer  [$label]"

        # Create venv
        Write-Step "Creating virtual environment..." -Level cmd
        $venvArgs = @() + $py.Args + @("-m", "venv", $VenvPath)
        Invoke-Checked -Exe $py.Cmd -Arguments $venvArgs -ErrorMessage "Failed to create virtual environment."

        $VenvPython = Join-Path $VenvPath "Scripts\python.exe"
        if (-not (Test-Path $VenvPython)) {
            throw "Venv python not found at expected path: $VenvPython"
        }

        # Upgrade packaging tooling
        Write-Step "Upgrading pip / setuptools / wheel..." -Level cmd
        $upgradeArgs = @("-m", "pip", "install", "--upgrade", "pip", "setuptools", "wheel", "--disable-pip-version-check")
        Invoke-Checked -Exe $VenvPython -Arguments $upgradeArgs -ErrorMessage "Failed to upgrade pip tooling inside the new venv."

        # Install dependencies
        if (Test-Path $ReqFile) {
            Write-Step "Installing dependencies from $Requirements..." -Level cmd
            $installArgs = @("-m", "pip", "install", "-r", $ReqFile, "--disable-pip-version-check")
            Invoke-Checked -Exe $VenvPython -Arguments $installArgs -ErrorMessage "Dependency installation from $Requirements failed."
        } else {
            Write-Step "$Requirements not found; skipping dependency install." -Level warn
        }

        Write-Step "Virtual environment created successfully (pip, unpinned transitive deps)." -Level ok
    }

    $activatePath = Join-Path $VenvPath "Scripts\Activate.ps1"
    Write-Step "  Activate: . $activatePath"
}
catch {
    Write-Step "create_venv failed: $($_.Exception.Message)" -Level err
    exit 1
}
