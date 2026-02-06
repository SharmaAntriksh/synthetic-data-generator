# ------------------------------------------------------------
# scripts/run_generator.ps1
# Config-first runner for the synthetic-data-generator
# ------------------------------------------------------------

[CmdletBinding()]
param(
    # Config inputs (defaults match cli.py)
    [string]$ConfigPath       = "config.yaml",
    [string]$ModelsConfigPath = "models.yaml",

    # Entrypoint (main.py imports and calls src.cli.main())
    [string]$Entrypoint = "main.py",

    # Repo / venv
    [string]$VenvDir = ".venv",
    [string]$ProjectRoot,
    [switch]$SkipSync,

    # Pipeline control
    [ValidateSet("", "dimensions", "sales")]
    [string]$Only = "",
    [bool]$Clean = $true,
    [switch]$DryRun,

    # Optional: suppress the printed config summary
    [switch]$NoConfigSummary,

    # Output/sales overrides (only applied if provided)
    [ValidateSet("csv", "parquet", "delta", "deltaparquet")]
    [string]$Format,
    [Nullable[bool]]$SkipOrderCols,
    [Nullable[int]]$SalesRows,
    [Nullable[int]]$Workers,
    [Nullable[int]]$ChunkSize,
    [Nullable[int]]$RowGroupSize,

    # Date overrides (YYYY-MM-DD)
    [string]$StartDate,
    [string]$EndDate,

    # Dimension size overrides
    [Nullable[int]]$Customers,
    [Nullable[int]]$Stores,
    [Nullable[int]]$Products,
    [Nullable[int]]$Promotions,

    # Force regeneration of specific dimensions (e.g. customers products stores all)
    [string[]]$RegenDimensions,

    # Any additional args are passed through to main.py / argparse
    [Parameter(ValueFromRemainingArguments = $true)]
    [string[]]$PassthroughArgs
)

$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest

function Resolve-RepoRoot {
    param([string]$StartDir)

    $markers = @(".git", "pyproject.toml", "requirements.txt", "config.yaml", "main.py")
    $dir = (Resolve-Path $StartDir).Path

    while ($true) {
        foreach ($m in $markers) {
            if (Test-Path -LiteralPath (Join-Path $dir $m)) { return $dir }
        }
        $parent = Split-Path $dir -Parent
        if ([string]::IsNullOrWhiteSpace($parent) -or $parent -eq $dir) { break }
        $dir = $parent
    }

    # Fallback: parent of /scripts
    return (Resolve-Path (Join-Path $StartDir "..")).Path
}

function Resolve-FromRoot {
    param([string]$Root, [string]$PathLike)

    if ([string]::IsNullOrWhiteSpace($PathLike)) { return $null }

    if ([System.IO.Path]::IsPathRooted($PathLike)) {
        return (Resolve-Path -LiteralPath $PathLike).Path
    }

    $p = Join-Path $Root $PathLike
    if (Test-Path -LiteralPath $p) { return (Resolve-Path -LiteralPath $p).Path }
    return $p
}

function Get-PythonRunner {
    param([string]$Root, [string]$VenvFolder)

    $venvPy = Join-Path (Join-Path $Root $VenvFolder) "Scripts\python.exe"
    if (Test-Path -LiteralPath $venvPy) { return @{ Cmd = $venvPy; Args = @() } }

    if (Get-Command py -ErrorAction SilentlyContinue) { return @{ Cmd = "py"; Args = @("-3") } }
    if (Get-Command python -ErrorAction SilentlyContinue) { return @{ Cmd = "python"; Args = @() } }

    return $null
}

function Invoke-SyncVenv {
    param([string]$Root, [string]$VenvFolder)

    $sync = Join-Path (Join-Path $Root "scripts") "sync_venv.ps1"
    if (-not (Test-Path -LiteralPath $sync)) { return }

    # Newer sync_venv.ps1 supports -VenvDir/-Quiet; fallback to no-args for older versions.
    try {
        & $sync -VenvDir $VenvFolder -Quiet | Out-Null
        if ($LASTEXITCODE -ne 0) { throw "sync_venv returned $LASTEXITCODE" }
    } catch {
        & $sync | Out-Null
        if ($LASTEXITCODE -ne 0) { throw "sync_venv returned $LASTEXITCODE" }
    }
}

function Assert-IsoDate {
    param([string]$Value, [string]$Name)
    if ([string]::IsNullOrWhiteSpace($Value)) { return }
    try {
        [void][DateTime]::ParseExact($Value, "yyyy-MM-dd", $null)
    } catch {
        throw "$Name must be YYYY-MM-DD. Got '$Value'."
    }
}

function Try-PrintConfigSummary {
    param($Py, [string]$CfgFile)

    # Best-effort: uses PyYAML if available. If missing, we silently skip.
    $code = @'
import json, sys
try:
    import yaml
except Exception:
    sys.exit(2)

p = sys.argv[1]
with open(p, "r", encoding="utf-8") as f:
    cfg = yaml.safe_load(f) or {}

def g(path, default=None):
    cur = cfg
    for part in path.split("."):
        if not isinstance(cur, dict) or part not in cur:
            return default
        cur = cur[part]
    return cur if cur is not None else default

defaults_start = g("defaults.dates.start")
defaults_end   = g("defaults.dates.end")
sales_end_ov   = g("sales.override.dates.end")
sales_start_ov = g("sales.override.dates.start")

eff_start = sales_start_ov or defaults_start
eff_end   = sales_end_ov   or defaults_end

fmt        = g("sales.file_format")
rows       = g("sales.total_rows")
chunk      = g("sales.chunk_size")
rg         = g("sales.row_group_size")
comp       = g("sales.compression")
out_folder = g("sales.out_folder")
pq_folder  = g("sales.parquet_folder")
delta_out  = g("sales.delta_output_folder")

customers  = g("customers.total_customers")
stores     = g("stores.num_stores")
products   = g("products.num_products")

p_seasonal  = g("promotions.num_seasonal", 0) or 0
p_clearance = g("promotions.num_clearance", 0) or 0
p_limited   = g("promotions.num_limited", 0) or 0

summary = {
  "dates": {"start": eff_start, "end": eff_end},
  "sales": {
    "file_format": fmt,
    "total_rows": rows,
    "chunk_size": chunk,
    "row_group_size": rg,
    "compression": comp,
    "out_folder": out_folder,
    "parquet_folder": pq_folder,
    "delta_output_folder": delta_out,
  },
  "dimensions": {"customers": customers, "stores": stores, "products": products},
  "promotions_total": int(p_seasonal) + int(p_clearance) + int(p_limited),
}

print(json.dumps(summary))
'@

    $raw = & $Py.Cmd @($Py.Args) -c $code $CfgFile 2>$null
    if ($LASTEXITCODE -ne 0 -or [string]::IsNullOrWhiteSpace($raw)) { return }

    try {
        $s = $raw | ConvertFrom-Json
        Write-Host "Config summary (effective):" -ForegroundColor DarkGray
        Write-Host ("  Dates:        {0} .. {1}" -f $s.dates.start, $s.dates.end) -ForegroundColor DarkGray
        Write-Host ("  Sales:        format={0} rows={1} chunk={2} row_group={3} compression={4}" -f `
            $s.sales.file_format, $s.sales.total_rows, $s.sales.chunk_size, $s.sales.row_group_size, $s.sales.compression) -ForegroundColor DarkGray
        Write-Host ("  Outputs:      out={0} parquet={1} delta={2}" -f `
            $s.sales.out_folder, $s.sales.parquet_folder, $s.sales.delta_output_folder) -ForegroundColor DarkGray
        Write-Host ("  Dimensions:   customers={0} stores={1} products={2}" -f `
            $s.dimensions.customers, $s.dimensions.stores, $s.dimensions.products) -ForegroundColor DarkGray
        Write-Host ("  Promotions:   total={0}" -f $s.promotions_total) -ForegroundColor DarkGray
    } catch {
        return
    }
}

# ------------------------------
# Main
# ------------------------------
try {
    $Root = if ($ProjectRoot) { (Resolve-Path -LiteralPath $ProjectRoot).Path } else { Resolve-RepoRoot -StartDir $PSScriptRoot }

    $MainPy  = Resolve-FromRoot -Root $Root -PathLike $Entrypoint
    $CfgFile = Resolve-FromRoot -Root $Root -PathLike $ConfigPath
    $Models  = Resolve-FromRoot -Root $Root -PathLike $ModelsConfigPath

    if (-not (Test-Path -LiteralPath $MainPy))  { throw "Entrypoint not found: $MainPy" }
    if (-not (Test-Path -LiteralPath $CfgFile)) { throw "Config not found: $CfgFile" }
    if (-not (Test-Path -LiteralPath $Models))  { throw "Models config not found: $Models" }

    if (-not $SkipSync) {
        Invoke-SyncVenv -Root $Root -VenvFolder $VenvDir
    }

    $Py = Get-PythonRunner -Root $Root -VenvFolder $VenvDir
    if (-not $Py) { throw "Python not found. Create the venv or install Python (py/python)." }

    Assert-IsoDate -Value $StartDate -Name "StartDate"
    Assert-IsoDate -Value $EndDate   -Name "EndDate"

    # Print a small config-derived summary (best-effort)
    if (-not $NoConfigSummary) {
        Try-PrintConfigSummary -Py $Py -CfgFile $CfgFile
    }

    # Build args (config-first)
    $Args = @(
        $MainPy,
        "--config", $CfgFile,
        "--models-config", $Models
    )

    if ($Only)  { $Args += @("--only", $Only) }
    if ($Clean) { $Args += "--clean" }
    if ($DryRun){ $Args += "--dry-run" }

    # Overrides (only when bound)
    if ($PSBoundParameters.ContainsKey("Format"))      { $Args += @("--format", $Format) }
    if ($PSBoundParameters.ContainsKey("SalesRows"))   { $Args += @("--sales-rows",    [string]$SalesRows) }
    if ($PSBoundParameters.ContainsKey("Workers"))     { $Args += @("--workers",      [string]$Workers) }
    if ($PSBoundParameters.ContainsKey("ChunkSize"))   { $Args += @("--chunk-size",   [string]$ChunkSize) }
    if ($PSBoundParameters.ContainsKey("RowGroupSize")){ $Args += @("--row-group-size",[string]$RowGroupSize) }
    if ($PSBoundParameters.ContainsKey("StartDate"))   { $Args += @("--start-date", $StartDate) }
    if ($PSBoundParameters.ContainsKey("EndDate"))     { $Args += @("--end-date",   $EndDate) }
    if ($PSBoundParameters.ContainsKey("Customers"))   { $Args += @("--customers",  [string]$Customers) }
    if ($PSBoundParameters.ContainsKey("Stores"))      { $Args += @("--stores",     [string]$Stores) }
    if ($PSBoundParameters.ContainsKey("Products"))    { $Args += @("--products",   [string]$Products) }
    if ($PSBoundParameters.ContainsKey("Promotions"))  { $Args += @("--promotions", [string]$Promotions) }

    if ($PSBoundParameters.ContainsKey("SkipOrderCols")) {
        # cli.py supports nargs="?" const=True:
        # - pass bare flag to mean True
        # - pass explicit "false" to disable
        if ($SkipOrderCols -eq $true) { $Args += "--skip-order-cols" }
        else { $Args += @("--skip-order-cols", "false") }
    }

    if ($PSBoundParameters.ContainsKey("RegenDimensions") -and $RegenDimensions -and $RegenDimensions.Count -gt 0) {
        $Args += "--regen-dimensions"
        $Args += $RegenDimensions
    }

    if ($PassthroughArgs) { $Args += $PassthroughArgs }

    # Print execution header
    $pyShown = if ($Py.Args -and $Py.Args.Count -gt 0) { "$($Py.Cmd) $($Py.Args -join ' ')" } else { $Py.Cmd }
    Write-Host ("Project root:  {0}" -f $Root) -ForegroundColor DarkGray
    Write-Host ("Python:        {0}" -f $pyShown) -ForegroundColor DarkGray
    Write-Host ("Command:       {0}" -f ("python " + ($Args -join " "))) -ForegroundColor DarkGray

    # Ensure relative paths inside config behave consistently
    Push-Location $Root
    try {
        & $Py.Cmd @($Py.Args) @Args
        exit $LASTEXITCODE
    }
    finally {
        Pop-Location
    }
}
catch {
    Write-Host ("run_generator failed: {0}" -f $_.Exception.Message) -ForegroundColor Red
    exit 1
}
