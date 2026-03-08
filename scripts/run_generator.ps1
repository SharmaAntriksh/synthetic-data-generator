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
    [switch]$Clean,
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

. (Join-Path $PSScriptRoot "_common.ps1")

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

function Invoke-SyncVenv {
    param([string]$Root, [string]$VenvFolder)

    $sync = Join-Path (Join-Path $Root "scripts") "sync_venv.ps1"
    if (-not (Test-Path -LiteralPath $sync)) { return }

    try {
        & $sync -VenvDir $VenvFolder -Quiet | Out-Null
        if ($LASTEXITCODE -ne 0) { throw "sync_venv returned $LASTEXITCODE" }
    } catch {
        & $sync | Out-Null
        if ($LASTEXITCODE -ne 0) { throw "sync_venv returned $LASTEXITCODE" }
    }
}

function Try-PrintConfigSummary {
    param($Py, [string]$CfgFile)

    $summaryScript = Join-Path $PSScriptRoot "helper_print_config_summary.py"

    # Use the companion Python helper if available; otherwise use inline code.
    if (Test-Path -LiteralPath $summaryScript) {
        $raw = & $Py.Cmd @($Py.Args) $summaryScript $CfgFile 2>$null
    } else {
        $code = "import json,sys`ntry:`n import yaml`nexcept Exception:`n sys.exit(2)`nwith open(sys.argv[1],'r',encoding='utf-8') as f:`n cfg=yaml.safe_load(f) or {}`ndef g(path,default=None):`n cur=cfg`n for part in path.split('.'):`n  if not isinstance(cur,dict) or part not in cur: return default`n  cur=cur[part]`n return cur if cur is not None else default`nds=g('defaults.dates.start');de=g('defaults.dates.end');ss=g('sales.override.dates.start');se=g('sales.override.dates.end')`nprint(json.dumps({'dates':{'start':ss or ds,'end':se or de},'sales':{'file_format':g('sales.file_format'),'total_rows':g('sales.total_rows'),'chunk_size':g('sales.chunk_size'),'row_group_size':g('sales.row_group_size'),'compression':g('sales.compression'),'out_folder':g('sales.out_folder'),'parquet_folder':g('sales.parquet_folder'),'delta_output_folder':g('sales.delta_output_folder')},'dimensions':{'customers':g('customers.total_customers'),'stores':g('stores.num_stores'),'products':g('products.num_products')},'promotions_total':int(g('promotions.num_seasonal',0) or 0)+int(g('promotions.num_clearance',0) or 0)+int(g('promotions.num_limited',0) or 0)}))"
        $raw = & $Py.Cmd @($Py.Args) -c $code $CfgFile 2>$null
    }

    if ($LASTEXITCODE -ne 0 -or [string]::IsNullOrWhiteSpace($raw)) { return }

    try {
        $s = $raw | ConvertFrom-Json
        Write-Step "Config summary (effective):"
        Write-Step ("  Dates       : {0} .. {1}" -f $s.dates.start, $s.dates.end)
        Write-Step ("  Sales       : format={0}  rows={1}  chunk={2}  row_group={3}  compression={4}" -f $s.sales.file_format, $s.sales.total_rows, $s.sales.chunk_size, $s.sales.row_group_size, $s.sales.compression)
        Write-Step ("  Outputs     : out={0}  parquet={1}  delta={2}" -f $s.sales.out_folder, $s.sales.parquet_folder, $s.sales.delta_output_folder)
        Write-Step ("  Dimensions  : customers={0}  stores={1}  products={2}" -f $s.dimensions.customers, $s.dimensions.stores, $s.dimensions.products)
        Write-Step ("  Promotions  : total={0}" -f $s.promotions_total)
    } catch {
        return
    }
}

# ---- Main ----

try {
    $Root = if ($ProjectRoot) {
        (Resolve-Path -LiteralPath $ProjectRoot).Path
    } else {
        Resolve-ProjectRoot -StartDir $PSScriptRoot -ExtraMarkers @("config.yaml", "main.py")
    }

    $MainPy  = Resolve-FromRoot -Root $Root -PathLike $Entrypoint
    $CfgFile = Resolve-FromRoot -Root $Root -PathLike $ConfigPath
    $Models  = Resolve-FromRoot -Root $Root -PathLike $ModelsConfigPath

    if (-not (Test-Path -LiteralPath $MainPy))  { throw "Entrypoint not found: $MainPy" }
    if (-not (Test-Path -LiteralPath $CfgFile)) { throw "Config not found: $CfgFile" }
    if (-not (Test-Path -LiteralPath $Models))  { throw "Models config not found: $Models" }

    if (-not $SkipSync) {
        Invoke-SyncVenv -Root $Root -VenvFolder $VenvDir
    }

    $VenvFullPath = Join-Path $Root $VenvDir
    $MinPyVer = [version]"3.10"
    $Py = Get-PythonRunner -MinVersion $MinPyVer -VenvPath $VenvFullPath
    if (-not $Py) { throw "Python not found. Create the venv or install Python (py/python)." }

    Assert-IsoDate -Value $StartDate -Name "StartDate"
    Assert-IsoDate -Value $EndDate   -Name "EndDate"

    # Validate numeric overrides are positive
    foreach ($numParam in @("SalesRows", "Workers", "ChunkSize", "RowGroupSize", "Customers", "Stores", "Products", "Promotions")) {
        if ($PSBoundParameters.ContainsKey($numParam)) {
            $val = $PSBoundParameters[$numParam]
            if ($null -ne $val -and $val -le 0) {
                throw "$numParam must be a positive integer. Got: $val"
            }
        }
    }

    if (-not $NoConfigSummary) {
        Try-PrintConfigSummary -Py $Py -CfgFile $CfgFile
    }

    # Build CLI args (config-first)
    $CliArgs = @(
        $MainPy,
        "--config", $CfgFile,
        "--models-config", $Models
    )

    if ($Only)   { $CliArgs += @("--only", $Only) }
    if ($Clean)  { $CliArgs += "--clean" }
    if ($DryRun) { $CliArgs += "--dry-run" }

    # Overrides (only when explicitly bound by caller)
    if ($PSBoundParameters.ContainsKey("Format"))       { $CliArgs += @("--format",         $Format) }
    if ($PSBoundParameters.ContainsKey("SalesRows"))    { $CliArgs += @("--sales-rows",     [string]$SalesRows) }
    if ($PSBoundParameters.ContainsKey("Workers"))      { $CliArgs += @("--workers",        [string]$Workers) }
    if ($PSBoundParameters.ContainsKey("ChunkSize"))    { $CliArgs += @("--chunk-size",     [string]$ChunkSize) }
    if ($PSBoundParameters.ContainsKey("RowGroupSize")) { $CliArgs += @("--row-group-size", [string]$RowGroupSize) }
    if ($PSBoundParameters.ContainsKey("StartDate"))    { $CliArgs += @("--start-date",     $StartDate) }
    if ($PSBoundParameters.ContainsKey("EndDate"))      { $CliArgs += @("--end-date",       $EndDate) }
    if ($PSBoundParameters.ContainsKey("Customers"))    { $CliArgs += @("--customers",      [string]$Customers) }
    if ($PSBoundParameters.ContainsKey("Stores"))       { $CliArgs += @("--stores",         [string]$Stores) }
    if ($PSBoundParameters.ContainsKey("Products"))     { $CliArgs += @("--products",       [string]$Products) }
    if ($PSBoundParameters.ContainsKey("Promotions"))   { $CliArgs += @("--promotions",     [string]$Promotions) }

    if ($PSBoundParameters.ContainsKey("SkipOrderCols")) {
        if ($SkipOrderCols -eq $true) { $CliArgs += "--skip-order-cols" }
        else { $CliArgs += @("--skip-order-cols", "false") }
    }

    if ($PSBoundParameters.ContainsKey("RegenDimensions") -and $RegenDimensions -and $RegenDimensions.Count -gt 0) {
        $CliArgs += "--regen-dimensions"
        $CliArgs += $RegenDimensions
    }

    if ($PassthroughArgs) { $CliArgs += $PassthroughArgs }

    # Execution header
    $pyLabel = Format-RunnerLabel -Runner $Py
    Write-Step "Project root  : $Root"
    Write-Step "Python        : $pyLabel"
    Write-Step ("Command       : {0} {1}" -f $pyLabel, ($CliArgs -join " "))

    Push-Location $Root
    try {
        & $Py.Cmd @($Py.Args) @CliArgs
        $ec = $LASTEXITCODE
        if ($ec -ne 0) {
            Write-Step "Generator exited with code $ec." -Level err
        }
        exit $ec
    }
    finally {
        Pop-Location
    }
}
catch {
    Write-Step "run_generator failed: $($_.Exception.Message)" -Level err
    exit 1
}
