# ------------------------------------------------------------
# Activate virtual environment
#
# Default behavior:
#   - Auto-detects the project root (prefers repo markers like .git/pyproject.toml)
#   - Activates the venv (default: .venv)
#
# IMPORTANT:
#   This script MUST be dot-sourced, otherwise activation will not persist.
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

. (Join-Path $PSScriptRoot "_common.ps1")

# ------------------------------------------------------------
# Enforce dot-sourcing
#
# $MyInvocation.InvocationName is '.' when dot-sourced in a standard
# console, but some hosts (ISE, VS Code) report differently.  Fall back
# to checking CommandOrigin when the simple test is inconclusive.
# ------------------------------------------------------------
$isDotSourced = ($MyInvocation.InvocationName -eq '.')
if (-not $isDotSourced) {
    # ISE / VS Code / pwsh may report CommandOrigin differently; also check Line for dot-source pattern
    if ($MyInvocation.CommandOrigin -and $MyInvocation.CommandOrigin -eq 'Internal') {
        $isDotSourced = $true
    } elseif ($MyInvocation.Line -and $MyInvocation.Line.TrimStart() -match '^\.\s') {
        $isDotSourced = $true
    }
}

if (-not $isDotSourced) {
    Write-Step "This script must be dot-sourced to activate the virtual environment." -Level err
    Write-Step "Run it like this:" -Level warn

    if ($PSCommandPath) {
        Write-Step ("  . ""{0}""" -f $PSCommandPath) -Level cmd
    } else {
        Write-Step "  . .\scripts\activate_venv.ps1" -Level cmd
    }
    return
}

# ------------------------------------------------------------
# Resolve project root + venv paths
# ------------------------------------------------------------
$RootPath = if ($ProjectRoot) {
    (Resolve-Path $ProjectRoot).Path
} else {
    Resolve-ProjectRoot -StartDir $PSScriptRoot
}

$VenvPath   = Join-Path $RootPath $VenvDir
$ActivatePs = Join-Path $VenvPath "Scripts\Activate.ps1"

Write-Step "Project root : $RootPath"
Write-Step "Venv path    : $VenvPath"

# ------------------------------------------------------------
# Guard / optional creation
# ------------------------------------------------------------
if (-not (Test-Path -LiteralPath $ActivatePs)) {
    if (-not $CreateIfMissing) {
        Write-Step "Virtual environment not found at: $VenvPath" -Level err
        Write-Step "Create it first:" -Level warn
        Write-Step "  .\scripts\create_venv.ps1" -Level cmd
        Write-Step "Or auto-create then activate:" -Level warn
        Write-Step "  . .\scripts\activate_venv.ps1 -CreateIfMissing" -Level cmd
        return
    }

    $CreateScriptPath = Join-Path $RootPath $CreateScript
    if (-not (Test-Path -LiteralPath $CreateScriptPath)) {
        throw "Create script not found at: $CreateScriptPath"
    }

    Write-Step "Venv missing; creating via $CreateScript ..." -Level warn
    & $CreateScriptPath
    if ($LASTEXITCODE -and $LASTEXITCODE -ne 0) {
        throw "Venv creation script failed with exit code $LASTEXITCODE."
    }

    # Re-check expected activation script.
    if (-not (Test-Path -LiteralPath $ActivatePs)) {
        $DefaultActivate = Join-Path (Join-Path $RootPath ".venv") "Scripts\Activate.ps1"
        if ($VenvDir -ne ".venv" -and (Test-Path -LiteralPath $DefaultActivate)) {
            throw "Venv was created at '.venv' but you requested '$VenvDir'. Re-run with -VenvDir .venv or update create_venv.ps1."
        }

        throw "Venv creation did not produce expected activation script: $ActivatePs"
    }
}

# ------------------------------------------------------------
# Activate (dot-source into current session)
# ------------------------------------------------------------
. $ActivatePs

try {
    $py = Get-Command python -ErrorAction Stop
    $pyVer = & python -c "import sys; print('%d.%d.%d' % sys.version_info[:3])"
    Write-Step "Activated: Python $($pyVer.Trim())  [$($py.Source)]" -Level ok
} catch {
    Write-Step "Activated venv (python not resolved on PATH in this session)." -Level ok
}
