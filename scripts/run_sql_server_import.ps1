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

. (Join-Path $PSScriptRoot "_common.ps1")

try {
    $ScriptRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
    $RepoRoot   = (Resolve-Path (Join-Path $ScriptRoot "..")).Path

    $PyEntrypoint = Join-Path $RepoRoot "scripts\sql\run_sql_server_import.py"
    if (-not (Test-Path $PyEntrypoint)) {
        throw "Python entrypoint not found: $PyEntrypoint"
    }

    $ResolvedRunPath = (Resolve-Path $RunPath).Path

    # CSV-only guard
    if (-not (Test-Path (Join-Path $ResolvedRunPath "sql\schema")) -or
        -not (Test-Path (Join-Path $ResolvedRunPath "sql\load"))) {
        throw "CSV run required. Expected sql/schema/ and sql/load/ folders in: $ResolvedRunPath"
    }

    # Log the resolved python interpreter and version
    $pyResolvedPath = (Get-Command $PythonExe -ErrorAction SilentlyContinue).Source
    if ($pyResolvedPath) {
        $pyVer = & $PythonExe -c "import sys; v=sys.version_info; print(str(v.major)+'.'+str(v.minor)+'.'+str(v.micro))" 2>$null
        Write-Step "Python : $($pyVer.Trim())  [$pyResolvedPath]"
    } else {
        Write-Step "Python : $PythonExe (not resolved on PATH)" -Level warn
    }

    # Build argument list (array form handles spaces safely)
    $argsList = @(
        $PyEntrypoint,
        "--server",   $Server,
        "--database", $Database,
        "--run-path", $ResolvedRunPath
    )

    if ($PSCmdlet.ParameterSetName -eq "Trusted") {
        $argsList += "--trusted"
    } else {
        $argsList += @("--user", $User, "--password", $Password)
    }

    if ($ApplyCCI) { $argsList += "--apply-cci" }
    if ($OdbcDriver) { $argsList += @("--odbc-driver", $OdbcDriver) }

    # Log the command
    if ($ShowFullPaths) {
        Write-Step ("Running: {0} {1}" -f $PythonExe, ($argsList -join " ")) -Level cmd
    } else {
        $pyName    = Split-Path -Leaf $PythonExe
        $entryName = Split-Path -Leaf $PyEntrypoint

        $runDisplay = $ResolvedRunPath
        if ($ResolvedRunPath.StartsWith($RepoRoot, [System.StringComparison]::OrdinalIgnoreCase)) {
            $runDisplay = $ResolvedRunPath.Substring($RepoRoot.Length).TrimStart('\')
        } else {
            $runDisplay = Split-Path -Leaf $ResolvedRunPath
        }

        $authLabel = if ($PSCmdlet.ParameterSetName -eq "Trusted") { "trusted" } else { "sql-auth" }

        $flags = @()
        if ($ApplyCCI) { $flags += "apply-cci" }
        if ($OdbcDriver) { $flags += ("odbc=" + $OdbcDriver) }

        $flagText = if ($flags.Count -gt 0) { " [" + ($flags -join ", ") + "]" } else { "" }

        Write-Step ("Running: {0} {1} --server {2} --database {3} --run {4} ({5}){6}" -f $pyName, $entryName, $Server, $Database, $runDisplay, $authLabel, $flagText) -Level cmd
    }

    # Execute
    & $PythonExe @argsList
    $ec = $LASTEXITCODE
    if ($ec -ne 0) {
        Write-Step "SQL Server import exited with code $ec." -Level err
    }
    exit $ec
}
catch {
    Write-Step "run_sql_server_import failed: $($_.Exception.Message)" -Level err
    exit 1
}
