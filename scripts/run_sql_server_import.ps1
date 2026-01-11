param (
    [Parameter(Mandatory = $true)]
    [string]$RunPath,

    [Parameter(Mandatory = $true)]
    [string]$Server,

    [Parameter(Mandatory = $true)]
    [string]$Database,

    [string]$User,
    [string]$Password,

    [switch]$TrustedConnection
)

# Resolve run path to avoid relative path confusion
$ResolvedRunPath = Resolve-Path $RunPath

# Build authentication arguments
if ($TrustedConnection) {
    $AuthArgs = "--trusted"
}
else {
    if (-not $User -or -not $Password) {
        Write-Error "User and Password must be provided when not using -TrustedConnection"
        exit 1
    }
    $AuthArgs = "--user $User --password $Password"
}

python -m scripts.sql.run_sql_server_import `
  --server $Server `
  --database $Database `
  --run-path $ResolvedRunPath `
  $AuthArgs

if ($LASTEXITCODE -ne 0) {
    exit $LASTEXITCODE
}
