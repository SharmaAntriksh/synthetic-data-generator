[CmdletBinding(DefaultParameterSetName = "Trusted")]
param(
    [Parameter(Mandatory = $true)]
    [string]$RunPath,

    [Parameter(Mandatory = $true)]
    [string]$Server,

    [Parameter(Mandatory = $true)]
    [string]$Database,

    [Parameter(ParameterSetName = "Trusted")]
    [switch]$TrustedConnection,

    [Parameter(Mandatory = $true, ParameterSetName = "SqlAuth")]
    [string]$User,

    [Parameter(Mandatory = $true, ParameterSetName = "SqlAuth")]
    [string]$Password,

    [switch]$ApplyCCI,

    [string]$OdbcDriver,

    [string]$PythonExe = "python"
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$ScriptRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$RepoRoot   = (Resolve-Path (Join-Path $ScriptRoot "..")).Path

# ✅ Correct entrypoint
$PyEntrypoint = Join-Path $RepoRoot "scripts\sql\run_sql_server_import.py"

$ResolvedRunPath = (Resolve-Path $RunPath).Path

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

if ($ApplyCCI) {
    $argsList += "--apply-cci"
}

if ($OdbcDriver) {
    $argsList += @("--odbc-driver", $OdbcDriver)
}

Write-Host ("Running: {0} {1}" -f $PythonExe, ($argsList -join " "))

& $PythonExe @argsList
exit $LASTEXITCODE
