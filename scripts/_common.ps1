# ------------------------------------------------------------
# Shared helpers for project PowerShell scripts.
#
# Dot-source this file at the top of any script:
#   . (Join-Path $PSScriptRoot "_common.ps1")
#
# Provides:
#   Resolve-ProjectRoot   - walk-up search for repo markers
#   Get-PythonRunner      - locate best python (venv > py > python)
#   Get-PythonVersion     - return [version] from a runner
#   Assert-IsoDate        - validate YYYY-MM-DD strings
#   Write-Step            - consistent coloured log lines
#   Invoke-Checked        - run an external command and throw on failure
#   Format-RunnerLabel    - human-readable label for a runner hashtable
# ------------------------------------------------------------

Set-StrictMode -Version Latest

# Prevent double-loading in the same session.
if ((Get-Variable -Name '__ProjectCommonLoaded' -Scope Global -ValueOnly -ErrorAction SilentlyContinue)) { return }
$Global:__ProjectCommonLoaded = $true

function Resolve-ProjectRoot {
    <#
    .SYNOPSIS
        Walk upward from StartDir looking for repo/project markers.
        Falls back to the parent of StartDir (the /scripts to repo-root convention).
    #>
    param(
        [Parameter(Mandatory)]
        [string]$StartDir,

        [string[]]$ExtraMarkers = @()
    )

    $markers = @(".git", "pyproject.toml", "requirements.txt") + $ExtraMarkers
    $dir = (Resolve-Path $StartDir).Path

    while ($true) {
        foreach ($m in $markers) {
            if (Test-Path -LiteralPath (Join-Path $dir $m)) {
                return $dir
            }
        }

        $parent = Split-Path $dir -Parent
        if ([string]::IsNullOrWhiteSpace($parent) -or $parent -eq $dir) { break }
        $dir = $parent
    }

    return (Resolve-Path (Join-Path $StartDir "..")).Path
}

function Get-PythonRunner {
    <#
    .SYNOPSIS
        Return a hashtable @{ Cmd; Args } for the best available Python.
        Search order: venv python > py launcher (targeted, then generic) > bare python.
    #>
    param(
        [version]$MinVersion = "3.10",
        [string]$VenvPath
    )

    # 1. Venv python (highest priority when a venv exists).
    if ($VenvPath) {
        $venvPy = Join-Path $VenvPath "Scripts\python.exe"
        if (Test-Path -LiteralPath $venvPy) {
            return @{ Cmd = $venvPy; Args = @() }
        }
    }

    # 2. Windows Python Launcher (targeted minor, then generic -3).
    if (Get-Command py -ErrorAction SilentlyContinue) {
        $candidates = @(
            @{ Cmd = "py"; Args = @("-$($MinVersion.Major).$($MinVersion.Minor)") },
            @{ Cmd = "py"; Args = @("-3") }
        )

        foreach ($cand in $candidates) {
            try {
                $null = & $cand.Cmd @($cand.Args) -c "import sys" 2>$null
                if ($LASTEXITCODE -eq 0) { return $cand }
            } catch { }
        }
    }

    # 3. Bare python on PATH.
    if (Get-Command python -ErrorAction SilentlyContinue) {
        return @{ Cmd = "python"; Args = @() }
    }

    return $null
}

function Get-PythonVersion {
    <#
    .SYNOPSIS  Return [version] for a runner returned by Get-PythonRunner.
    #>
    param(
        [Parameter(Mandatory)]
        [hashtable]$Runner
    )

    $code = 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}")'
    $out = & $Runner.Cmd @($Runner.Args) -c $code 2>$null

    if ($LASTEXITCODE -ne 0 -or -not $out) {
        throw "Python was found ($($Runner.Cmd)) but could not report its version."
    }

    return [version]$out.Trim()
}

function Assert-IsoDate {
    <#
    .SYNOPSIS  Validate that a string is YYYY-MM-DD (or empty/null).
    #>
    param(
        [string]$Value,
        [string]$Name = "Date"
    )

    if ([string]::IsNullOrWhiteSpace($Value)) { return }
    try {
        [void][DateTime]::ParseExact($Value, "yyyy-MM-dd", $null)
    } catch {
        throw "$Name must be YYYY-MM-DD. Got: '$Value'."
    }
}

function Write-Step {
    <#
    .SYNOPSIS  Emit a consistently formatted status line.
    .PARAMETER Level
        info (default, DarkGray), ok (Green), warn (Yellow), err (Red), cmd (Cyan).
    #>
    param(
        [Parameter(Mandatory)]
        [string]$Message,

        [ValidateSet("info","ok","warn","err","cmd")]
        [string]$Level = "info"
    )

    $color = switch ($Level) {
        "info" { "DarkGray" }
        "ok"   { "Green"    }
        "warn" { "Yellow"   }
        "err"  { "Red"      }
        "cmd"  { "Cyan"     }
    }

    Write-Host $Message -ForegroundColor $color
}

function Invoke-Checked {
    <#
    .SYNOPSIS  Run an external command; throw when it returns a non-zero exit code.
    #>
    param(
        [Parameter(Mandatory)]
        [string]$Exe,

        [string[]]$Arguments = @(),

        [string]$ErrorMessage
    )

    & $Exe @Arguments
    if ($LASTEXITCODE -ne 0) {
        $msg = if ($ErrorMessage) { $ErrorMessage } else { "$Exe exited with code $LASTEXITCODE." }
        throw $msg
    }
}

function Format-RunnerLabel {
    <#
    .SYNOPSIS  Human-readable label for a python runner hashtable.
    #>
    param([hashtable]$Runner)

    if ($Runner.Args -and $Runner.Args.Count -gt 0) {
        return "$($Runner.Cmd) $($Runner.Args -join ' ')"
    }
    return $Runner.Cmd
}
