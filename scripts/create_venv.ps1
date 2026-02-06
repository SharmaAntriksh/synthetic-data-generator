# ------------------------------------------------------------
# Create virtual environment (one-time setup)
# Always creates .venv at project root by default
# DOES NOT activate the environment
# ------------------------------------------------------------

[CmdletBinding()]
param(
    [string]$VenvDir = ".venv",
    [string]$Requirements = "requirements.txt",
    [switch]$Force,
    [version]$MinPythonVersion = "3.9"
)

$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest

function Resolve-ProjectRoot {
    param([string]$StartDir)

    # Default behavior: parent of /scripts (your current logic)
    $candidate = Resolve-Path (Join-Path $StartDir "..")

    # Optional robustness: walk upwards for a marker
    $markers = @(".git", "pyproject.toml", "requirements.txt")
    $dir = $candidate.Path
    while ($true) {
        foreach ($m in $markers) {
            if (Test-Path (Join-Path $dir $m)) { return (Resolve-Path $dir) }
        }
        $parent = Split-Path $dir -Parent
        if ($parent -eq $dir -or [string]::IsNullOrWhiteSpace($parent)) { break }
        $dir = $parent
    }

    return $candidate
}

function Get-PythonCommand {
    # Prefer Python Launcher if present (common on Windows)
    if (Get-Command py -ErrorAction SilentlyContinue) {
        return @{ Cmd = "py"; Args = @("-3") }
    }
    if (Get-Command python -ErrorAction SilentlyContinue) {
        return @{ Cmd = "python"; Args = @() }
    }
    throw "Python not found. Install Python $MinPythonVersion+ and ensure 'py' or 'python' is available."
}

function Get-PythonVersion {
    param($py)

    $code = 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}")'
    $v = & $py.Cmd @($py.Args) -c $code
    return [version]$v.Trim()
}

# ------------------------------------------------------------
# Resolve paths
# ------------------------------------------------------------
$ProjectRoot = Resolve-ProjectRoot -StartDir $PSScriptRoot
$VenvPath    = Join-Path $ProjectRoot $VenvDir
$ReqFile     = Join-Path $ProjectRoot $Requirements

Write-Host "Project root: $ProjectRoot" -ForegroundColor DarkGray

# ------------------------------------------------------------
# Guard: already exists (unless -Force)
# ------------------------------------------------------------
if (Test-Path $VenvPath) {
    if (-not $Force) {
        Write-Host "Virtual environment already exists at $VenvPath" -ForegroundColor Yellow
        return
    }
    Write-Host "Removing existing venv at $VenvPath (Force)..." -ForegroundColor Yellow
    Remove-Item -LiteralPath $VenvPath -Recurse -Force
}

# ------------------------------------------------------------
# Python availability + version check
# ------------------------------------------------------------
$py = Get-PythonCommand
$pyVer = Get-PythonVersion -py $py

if ($pyVer -lt $MinPythonVersion) {
    throw "Python $MinPythonVersion+ is required. Found: $pyVer"
}

Write-Host "Using Python $pyVer ($($py.Cmd) $($py.Args -join ' '))" -ForegroundColor DarkGray

# ------------------------------------------------------------
# Create venv
# ------------------------------------------------------------
Write-Host "Creating virtual environment at $VenvPath" -ForegroundColor Cyan
& $py.Cmd @($py.Args) -m venv $VenvPath

$VenvPython = Join-Path $VenvPath "Scripts\python.exe"
if (-not (Test-Path $VenvPython)) {
    throw "Venv python not found at expected path: $VenvPython"
}

# ------------------------------------------------------------
# Upgrade packaging tooling + install deps
# ------------------------------------------------------------
Write-Host "Upgrading pip/setuptools/wheel..." -ForegroundColor Yellow
& $VenvPython -m pip install --upgrade pip setuptools wheel

if (Test-Path $ReqFile) {
    Write-Host "Installing dependencies from $Requirements..." -ForegroundColor Yellow
    & $VenvPython -m pip install -r $ReqFile
} else {
    Write-Host "$Requirements not found. Skipping dependency install." -ForegroundColor Yellow
}

Write-Host "Virtual environment created successfully." -ForegroundColor Green
Write-Host "To activate: `"$VenvPath\Scripts\Activate.ps1`"" -ForegroundColor DarkGray
