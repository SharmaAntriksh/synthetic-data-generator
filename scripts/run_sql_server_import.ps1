# NOTE:
# This script is intended for CSV-based runs only.
# The run path must contain 'sql/schema/' and 'sql/load/' folders.

param (
    [Parameter(Mandatory = $true)]
    [string]$RunPath,

    [Parameter(Mandatory = $true)]
    [string]$Server,

    [Parameter(Mandatory = $true)]
    [string]$Database,

    [string]$User,
    [string]$Password,

    [switch]$TrustedConnection,

    # Optional: only runs CCI scripts when present + flagged
    [switch]$ApplyCCI
)

# Resolve run path to avoid relative path confusion
$ResolvedRunPath = (Resolve-Path $RunPath).Path

# CSV-only guard (early, user-friendly failure)
if (-not (Test-Path (Join-Path $ResolvedRunPath "sql\schema")) -or
    -not (Test-Path (Join-Path $ResolvedRunPath "sql\load"))) {
    Write-Error "CSV run required. Expected 'sql/schema/' and 'sql/load/' folders in run path."
    exit 1
}

# Build authentication arguments (safe array form)
if ($TrustedConnection) {
    $AuthArgs = @("--trusted")
}
else {
    if (-not $User -or -not $Password) {
        Write-Error "User and Password must be provided when not using -TrustedConnection"
        exit 1
    }
    $AuthArgs = @("--user", $User, "--password", $Password)
}

# Optional flags
$OptArgs = @()
if ($ApplyCCI) {
    $OptArgs += "--apply-cci"
}

python -m scripts.sql.run_sql_server_import `
    --server $Server `
    --database $Database `
    --run-path $ResolvedRunPath `
    @AuthArgs `
    @OptArgs

if ($LASTEXITCODE -ne 0) {
    exit $LASTEXITCODE
}
