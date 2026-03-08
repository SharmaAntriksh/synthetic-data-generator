# ------------------------------------------------------------
# Check Python availability + minimum version
#
# Exit codes:
#   0 = OK
#   1 = Python missing / too old / not runnable
#
# Usage:
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

. (Join-Path $PSScriptRoot "_common.ps1")

try {
    $runner = Get-PythonRunner -MinVersion $MinPythonVersion
    if (-not $runner) {
        if (-not $Quiet) {
            Write-Step "Python not found." -Level err
            Write-Step "Install Python $MinPythonVersion+ and ensure either 'py' or 'python' is available." -Level warn
            Write-Step "Windows installer: https://www.python.org/downloads/" -Level info
        }
        exit 1
    }

    $pyVer = Get-PythonVersion -Runner $runner

    if ($pyVer -lt $MinPythonVersion) {
        if (-not $Quiet) {
            Write-Step "Python version too old: $pyVer (required: $MinPythonVersion+)" -Level err
        }
        exit 1
    }

    if (-not $Quiet) {
        $label = Format-RunnerLabel -Runner $runner
        $cmdPath = (Get-Command $runner.Cmd -ErrorAction SilentlyContinue).Source
        Write-Step "Python $pyVer  [$label]" -Level ok
        if ($cmdPath) { Write-Step "  Path: $cmdPath" -Level info }
    }

    exit 0
}
catch {
    if (-not $Quiet) {
        $errMsg = if ($_.Exception.Message) { $_.Exception.Message } else { $_.ToString() }
        Write-Step "Python check failed: $errMsg" -Level err
        if ($_.ScriptStackTrace) { Write-Step "  Stack: $($_.ScriptStackTrace)" -Level info }
    }
    exit 1
}
