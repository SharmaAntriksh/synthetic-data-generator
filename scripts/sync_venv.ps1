# ------------------------------------------------------------
# Sync (create/update) a project virtual environment
#
# Behavior:
#   - Resolves project root (walks upward for .git / pyproject.toml / requirements.txt)
#   - Creates the venv if missing (or recreates with -Force)
#   - Installs dependencies only when inputs change (requirements/constraints content hash + python version)
#   - Uses the venv's python directly (no activation required)
#
# Exit codes:
#   0 = OK
#   1 = failure (missing python, venv create failed, pip install failed)
#
# Usage examples:
#   .\scripts\sync_venv.ps1
#   .\scripts\sync_venv.ps1 -Force
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
    [switch]$NoUpgradePip,
    [switch]$Quiet,
    [version]$MinPythonVersion = "3.10",
    [string]$ProjectRoot
)

$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest

function Resolve-ProjectRoot {
    param([string]$StartDir)

    $markers = @(".git", "pyproject.toml", "requirements.txt")
    $dir = (Resolve-Path $StartDir).Path

    while ($true) {
        foreach ($m in $markers) {
            if (Test-Path -LiteralPath (Join-Path $dir $m)) {
                return (Resolve-Path $dir).Path
            }
        }

        $parent = Split-Path $dir -Parent
        if ([string]::IsNullOrWhiteSpace($parent) -or $parent -eq $dir) { break }
        $dir = $parent
    }

    # Fallback: parent of the script directory (matches /scripts layout)
    return (Resolve-Path (Join-Path $StartDir "..")).Path
}

function Get-PythonRunner {
    param([version]$MinV)

    $maj = $MinV.Major
    $min = $MinV.Minor

    if (Get-Command py -ErrorAction SilentlyContinue) {
        # Try exact minor first (e.g. py -3.10), then generic py -3
        $candidates = @(
            @{ Cmd = "py"; Args = @("-$maj.$min") },
            @{ Cmd = "py"; Args = @("-3") }
        )

        foreach ($cand in $candidates) {
            try {
                $null = & $cand.Cmd @($cand.Args) -c "import sys" 2>$null
                if ($LASTEXITCODE -eq 0) { return $cand }
            } catch { }
        }
    }

    if (Get-Command python -ErrorAction SilentlyContinue) {
        return @{ Cmd = "python"; Args = @() }
    }

    return $null
}

function Get-PythonVersion {
    param($Runner)
    $code = 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}")'
    $out = & $Runner.Cmd @($Runner.Args) -c $code 2>$null
    if ($LASTEXITCODE -ne 0 -or -not $out) {
        throw "Python was found but could not be executed."
    }
    return [version]$out.Trim()
}

function Read-Stamp {
    param([string]$Path)
    if (-not (Test-Path -LiteralPath $Path)) { return $null }
    try {
        $raw = Get-Content -LiteralPath $Path -Raw
        if ([string]::IsNullOrWhiteSpace($raw)) { return $null }
        return ($raw | ConvertFrom-Json)
    } catch {
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

function Hash-FileOrEmpty {
    param([string]$Path)
    if ([string]::IsNullOrWhiteSpace($Path)) { return "" }
    if (-not (Test-Path -LiteralPath $Path)) { return "" }
    return (Get-FileHash -LiteralPath $Path -Algorithm SHA256).Hash
}

try {
    $RootPath = if ($ProjectRoot) { (Resolve-Path $ProjectRoot).Path } else { Resolve-ProjectRoot -StartDir $PSScriptRoot }
    $VenvPath = Join-Path $RootPath $VenvDir

    $ReqFile = Join-Path $RootPath $Requirements
    $ConFile = if ([string]::IsNullOrWhiteSpace($Constraints)) { "" } else { (Join-Path $RootPath $Constraints) }

    $VenvPython = Join-Path $VenvPath "Scripts\python.exe"
    $StampFile  = Join-Path $VenvPath ".requirements.stamp.json"

    if (-not $Quiet) {
        Write-Host ("Project root: {0}" -f $RootPath) -ForegroundColor DarkGray
        Write-Host ("Venv path:     {0}" -f $VenvPath) -ForegroundColor DarkGray
        Write-Host ("Requirements:  {0}" -f $ReqFile) -ForegroundColor DarkGray
        if ($ConFile) { Write-Host ("Constraints:   {0}" -f $ConFile) -ForegroundColor DarkGray }
    }

    # Locate runnable python for venv creation/version check
    $runner = Get-PythonRunner -MinV $MinPythonVersion
    if (-not $runner) {
        if (-not $Quiet) {
            Write-Host "Python not found." -ForegroundColor Red
            Write-Host ("Install Python {0}+ and ensure 'py' or 'python' is available." -f $MinPythonVersion) -ForegroundColor Yellow
        }
        exit 1
    }

    $sysPyVer = Get-PythonVersion -Runner $runner
    if ($sysPyVer -lt $MinPythonVersion) {
        if (-not $Quiet) {
            Write-Host ("Python version too old: {0}" -f $sysPyVer) -ForegroundColor Red
            Write-Host ("Required: {0} or newer" -f $MinPythonVersion) -ForegroundColor Yellow
        }
        exit 1
    }

    # (Re)create venv if needed
    if ($Force -and (Test-Path -LiteralPath $VenvPath)) {
        if (-not $Quiet) { Write-Host "Removing existing venv (Force)..." -ForegroundColor Yellow }
        Remove-Item -LiteralPath $VenvPath -Recurse -Force
    }

    if (-not (Test-Path -LiteralPath $VenvPython)) {
        if (-not $Quiet) { Write-Host "Creating virtual environment..." -ForegroundColor Cyan }
        & $runner.Cmd @($runner.Args) -m venv $VenvPath
        if ($LASTEXITCODE -ne 0) { throw "Failed to create venv at: $VenvPath" }
        if (-not (Test-Path -LiteralPath $VenvPython)) { throw "Venv python not found at: $VenvPython" }
    }

    # No requirements => nothing to sync (but venv ensured)
    if (-not (Test-Path -LiteralPath $ReqFile)) {
        if (-not $Quiet) { Write-Host "Requirements file not found; skipping dependency sync." -ForegroundColor Yellow }
        exit 0
    }

    if ($ConFile -and (-not (Test-Path -LiteralPath $ConFile))) {
        throw "Constraints file not found: $ConFile"
    }

    # Decide whether sync needed
    $reqHash = Hash-FileOrEmpty -Path $ReqFile
    $conHash = Hash-FileOrEmpty -Path $ConFile
    $venvPyVer = (& $VenvPython -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}")' 2>$null).Trim()

    $stamp = Read-Stamp -Path $StampFile
    $needsSync =
        (-not $stamp) -or
        ($stamp.requirements_sha256 -ne $reqHash) -or
        ($stamp.constraints_sha256  -ne $conHash) -or
        ($stamp.python_version      -ne $venvPyVer)

    if (-not $needsSync) {
        if (-not $Quiet) { Write-Host "Virtual environment already up to date." -ForegroundColor Green }
        exit 0
    }

    if (-not $Quiet) { Write-Host "Updating virtual environment..." -ForegroundColor Yellow }

    # Upgrade packaging tooling unless explicitly disabled
    if (-not $NoUpgradePip) {
        $pipArgs = @("-m","pip","install","--upgrade","pip","setuptools","wheel","--disable-pip-version-check")
        if ($Quiet) { $pipArgs += "--quiet" }
        & $VenvPython @pipArgs
        if ($LASTEXITCODE -ne 0) { throw "Failed to upgrade pip tooling." }
    }

    # Install requirements
    $installArgs = @("-m","pip","install","-r",$ReqFile,"--disable-pip-version-check")
    if ($ConFile) { $installArgs += @("-c",$ConFile) }
    if ($Quiet) { $installArgs += "--quiet" }

    & $VenvPython @installArgs
    $code = $LASTEXITCODE
    if ($code -ne 0) {
        if (-not $Quiet) { Write-Host "Dependency installation failed; stamp not updated." -ForegroundColor Red }
        exit $code
    }

    Write-Stamp -Path $StampFile -ReqHash $reqHash -ConHash $conHash -PyVer $venvPyVer
    if (-not $Quiet) { Write-Host "Virtual environment updated." -ForegroundColor Green }
    exit 0
}
catch {
    if (-not $Quiet) {
        Write-Host ("sync_venv failed: {0}" -f $_.Exception.Message) -ForegroundColor Red
    }
    exit 1
}
