# ------------------------------------------------------------
# Activate virtual environment
#
# Default behavior:
#   - Auto-detects the project root (prefers repo markers like .git/pyproject.toml)
#   - Activates the venv (default: .venv)
#
# IMPORTANT:
#   This script MUST be dot-sourced, otherwise activation won't persist.
#   Example:
#     . .\scripts\activate_venv.ps1
# ------------------------------------------------------------

[CmdletBinding()]
param(
    [string]$VenvDir = ".venv",

    # Optional override. If omitted, the script auto-detects project root.
    [string]$ProjectRoot,

    # If the venv is missing, create it by calling scripts/create_venv.ps1.
    [switch]$CreateIfMissing,

    # Relative path (from project root) to the create script.
    [string]$CreateScript = "scripts/create_venv.ps1"
)

$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest

function Resolve-ProjectRoot {
    param([string]$StartDir)

    # Walk up looking for common repo/project markers.
    $markers = @(".git", "pyproject.toml", "requirements.txt")

    $dir = (Resolve-Path $StartDir).Path
    while ($true) {
        foreach ($m in $markers) {
            if (Test-Path -LiteralPath (Join-Path $dir $m)) {
                return (Resolve-Path $dir)
            }
        }

        $parent = Split-Path $dir -Parent
        if ([string]::IsNullOrWhiteSpace($parent) -or $parent -eq $dir) { break }
        $dir = $parent
    }

    # Fallback: parent of the directory containing this script (works for /scripts layout).
    return (Resolve-Path (Join-Path $StartDir ".."))
}

# ------------------------------------------------------------
# Enforce dot-sourcing
# ------------------------------------------------------------
if ($MyInvocation.InvocationName -ne '.') {
    Write-Host "ERROR: This script must be dot-sourced to activate the virtual environment." -ForegroundColor Red
    Write-Host "Run it like this:" -ForegroundColor Yellow

    if ($PSCommandPath) {
        Write-Host ("  . `"{0}`"" -f $PSCommandPath) -ForegroundColor Cyan
    } else {
        Write-Host "  . .\scripts\activate_venv.ps1" -ForegroundColor Cyan
    }
    return
}

# ------------------------------------------------------------
# Resolve project root + venv paths
# ------------------------------------------------------------
$RootPath = if ($ProjectRoot) {
    (Resolve-Path $ProjectRoot).Path
} else {
    (Resolve-ProjectRoot -StartDir $PSScriptRoot).Path
}

$VenvPath   = Join-Path $RootPath $VenvDir
$ActivatePs = Join-Path $VenvPath "Scripts\Activate.ps1"

Write-Host ("Project root: {0}" -f $RootPath) -ForegroundColor DarkGray
Write-Host ("Venv path:     {0}" -f $VenvPath) -ForegroundColor DarkGray

# ------------------------------------------------------------
# Guard / optional creation
# ------------------------------------------------------------
if (-not (Test-Path -LiteralPath $ActivatePs)) {
    if (-not $CreateIfMissing) {
        Write-Host ("Virtual environment not found (expected: {0})" -f $VenvPath) -ForegroundColor Red
        Write-Host "Create it first:" -ForegroundColor Yellow
        Write-Host "  .\scripts\create_venv.ps1" -ForegroundColor Cyan
        Write-Host "Or auto-create then activate:" -ForegroundColor Yellow
        Write-Host "  . .\scripts\activate_venv.ps1 -CreateIfMissing" -ForegroundColor Cyan
        return
    }

    $CreateScriptPath = Join-Path $RootPath $CreateScript
    if (-not (Test-Path -LiteralPath $CreateScriptPath)) {
        throw "Create script not found at: $CreateScriptPath"
    }

    Write-Host "Venv missing; creating it now..." -ForegroundColor Yellow

    # Call without parameters so it works with the simplest create_venv.ps1 implementation.
    & $CreateScriptPath

    # Re-check expected activation script.
    if (-not (Test-Path -LiteralPath $ActivatePs)) {
        # Common case: create_venv.ps1 always creates '.venv', but caller set -VenvDir to something else.
        $DefaultActivate = Join-Path (Join-Path $RootPath ".venv") "Scripts\Activate.ps1"
        if ($VenvDir -ne ".venv" -and (Test-Path -LiteralPath $DefaultActivate)) {
            throw "Venv was created at '.venv' but you requested '$VenvDir'. Re-run with -VenvDir .venv or update $CreateScript to support custom venv paths."
        }

        throw "Venv creation did not produce expected activation script: $ActivatePs"
    }
}

# ------------------------------------------------------------
# Activate (dot-source into current session)
# ------------------------------------------------------------
. $ActivatePs

# Optional: show interpreter for quick sanity check
try {
    $py = Get-Command python -ErrorAction Stop
    $pyVer = & python -c "import sys; print('%d.%d.%d' % sys.version_info[:3])"
    Write-Host ("Activated: python {0} ({1})" -f $pyVer.Trim(), $py.Source) -ForegroundColor Green
} catch {
    Write-Host "Activated venv. (python not resolved on PATH in this session.)" -ForegroundColor Green
}
