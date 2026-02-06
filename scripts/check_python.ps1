# ------------------------------------------------------------
# Check Python availability + minimum version
#
# Exit codes:
#   0 = OK
#   1 = Python missing / too old / not runnable
#
# Usage examples:
#   .\scripts\check_python.ps1
#   .\scripts\check_python.ps1 -MinPythonVersion 3.11
# ------------------------------------------------------------

[CmdletBinding()]
param(
    [version]$MinPythonVersion = '3.10',
    [switch]$Quiet
)

$ErrorActionPreference = 'Stop'
Set-StrictMode -Version Latest

function Get-PythonRunner {
    param([version]$MinV)

    $maj = $MinV.Major
    $min = $MinV.Minor

    # Prefer Windows Python Launcher if available.
    if (Get-Command py -ErrorAction SilentlyContinue) {
        # Try to target the requested minor version first (e.g., py -3.10)
        $candidates = @(
            @{ Cmd = 'py'; Args = @("-$maj.$min") },
            @{ Cmd = 'py'; Args = @('-3') }
        )

        foreach ($cand in $candidates) {
            try {
                $null = & $cand.Cmd @($cand.Args) -c "import sys" 2>$null
                if ($LASTEXITCODE -eq 0) { return $cand }
            } catch { }
        }
    }

    # Fallback to python on PATH.
    if (Get-Command python -ErrorAction SilentlyContinue) {
        return @{ Cmd = 'python'; Args = @() }
    }

    return $null
}

function Get-PythonVersion {
    param($Runner)

    $code = 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}")'
    $out = & $Runner.Cmd @($Runner.Args) -c $code 2>$null

    if ($LASTEXITCODE -ne 0 -or -not $out) {
        throw 'Python was found but could not be executed.'
    }

    return [version]$out.Trim()
}

try {
    $runner = Get-PythonRunner -MinV $MinPythonVersion
    if (-not $runner) {
        if (-not $Quiet) {
            Write-Host "Python not found." -ForegroundColor Red
            Write-Host "Install Python $MinPythonVersion+ and ensure either 'py' or 'python' is available." -ForegroundColor Yellow
            Write-Host "Windows installer: https://www.python.org/downloads/" -ForegroundColor DarkGray
        }
        exit 1
    }

    $pyVer = Get-PythonVersion -Runner $runner

    if ($pyVer -lt $MinPythonVersion) {
        if (-not $Quiet) {
            Write-Host "Python version too old: $pyVer" -ForegroundColor Red
            Write-Host "Required: $MinPythonVersion or newer" -ForegroundColor Yellow
        }
        exit 1
    }

    if (-not $Quiet) {
        $cmdPath = (Get-Command $runner.Cmd -ErrorAction SilentlyContinue).Source
        $args = ($runner.Args -join ' ')
        if ([string]::IsNullOrWhiteSpace($args)) { $args = '(none)' }

        Write-Host "Found Python $pyVer" -ForegroundColor Green
        Write-Host "Command: $($runner.Cmd)  Args: $args" -ForegroundColor DarkGray
        if ($cmdPath) { Write-Host "Resolved: $cmdPath" -ForegroundColor DarkGray }
    }

    exit 0
}
catch {
    if (-not $Quiet) {
        Write-Host "Python check failed: $($_.Exception.Message)" -ForegroundColor Red
    }
    exit 1
}
