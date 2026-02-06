# NOTE:
# This script is intended for CSV-based runs only.
# The run path must contain 'sql/schema/' and 'sql/load/' folders.

[CmdletBinding(DefaultParameterSetName = "Trusted")]
param (
    [Parameter(Mandatory = $true)]
    [string]$RunPath,

    [Parameter(Mandatory = $true)]
    [string]$Server,

    [Parameter(Mandatory = $true)]
    [string]$Database,

    # --- Auth (choose ONE mode) ---
    [Parameter(ParameterSetName = "Trusted")]
    [switch]$TrustedConnection,

    [Parameter(Mandatory = $true, ParameterSetName = "SqlAuth")]
    [string]$User,

    [Parameter(Mandatory = $true, ParameterSetName = "SqlAuth")]
    [string]$Password,

    # --- Optional flags ---
    [bool]$ApplyCCI = $false,

    # e.g. "ODBC Driver 18 for SQL Server"
    [string]$OdbcDriver,

    # Override python executable (defaults to active venv python if venv is activated)
    [string]$PythonExe = "python",

    # Show full absolute paths in the "Running:" line
    [switch]$ShowFullPaths
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

# Resolve paths
$ScriptRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$RepoRoot   = (Resolve-Path (Join-Path $ScriptRoot "..")).Path

# Entrypoint python script
$PyEntrypoint = Join-Path $RepoRoot "scripts\sql\run_sql_server_import.py"
if (-not (Test-Path $PyEntrypoint)) {
    Write-Error "Python entrypoint not found: $PyEntrypoint"
    exit 1
}

$ResolvedRunPath = (Resolve-Path $RunPath).Path

# CSV-only guard (early, user-friendly failure)
if (-not (Test-Path (Join-Path $ResolvedRunPath "sql\schema")) -or
    -not (Test-Path (Join-Path $ResolvedRunPath "sql\load"))) {
    Write-Error "CSV run required. Expected 'sql/schema/' and 'sql/load/' folders in run path: $ResolvedRunPath"
    exit 1
}

# Build argument list (array form handles spaces safely)
$argsList = @(
    $PyEntrypoint,
    "--server",   $Server,
    "--database", $Database,
    "--run-path", $ResolvedRunPath
)

# Auth args
if ($PSCmdlet.ParameterSetName -eq "Trusted") {
    $argsList += "--trusted"
} else {
    $argsList += @("--user", $User, "--password", $Password)
}

# Optional flags
if ($ApplyCCI) {
    $argsList += "--apply-cci"
}
if ($OdbcDriver) {
    $argsList += @("--odbc-driver", $OdbcDriver)
}

# Compact output by default
if ($ShowFullPaths) {
    Write-Host ("Running: {0} {1}" -f $PythonExe, ($argsList -join " "))
} else {
    $pyName    = Split-Path -Leaf $PythonExe
    $entryName = Split-Path -Leaf $PyEntrypoint

    # Prefer run path relative to repo root; otherwise show just the leaf folder
    $runDisplay = $ResolvedRunPath
    if ($ResolvedRunPath.StartsWith($RepoRoot, [System.StringComparison]::OrdinalIgnoreCase)) {
        $runDisplay = $ResolvedRunPath.Substring($RepoRoot.Length).TrimStart('\')
    } else {
        $runDisplay = Split-Path -Leaf $ResolvedRunPath
    }

    $authLabel = if ($PSCmdlet.ParameterSetName -eq "Trusted") { "trusted" } else { "sql-auth" }

    $flags = @()
    if ($ApplyCCI)   { $flags += "apply-cci" }
    if ($OdbcDriver) { $flags += ("odbc=" + $OdbcDriver) }

    $flagText = if ($flags.Count -gt 0) { " [" + ($flags -join ", ") + "]" } else { "" }

    Write-Host ("Running: {0} {1} --server {2} --database {3} --run {4} ({5}){6}" -f `
        $pyName, $entryName, $Server, $Database, $runDisplay, $authLabel, $flagText)
}

# Execute
& $PythonExe @argsList

if ($LASTEXITCODE -ne 0) {
    exit $LASTEXITCODE
}
