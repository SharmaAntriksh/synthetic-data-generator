# ============================================
# Contoso Fake Data Generator - Run Script
# Edit values below, then run:  .\run.ps1
# ============================================

# -------------------------------
# Output / Sales configuration
# -------------------------------

$Format        = "deltaparquet"   # csv | parquet | deltaparquet
$SkipOrderCols = $true

$SalesRows     = 102500
$ChunkSize     = 2000000
$RowGroupSize  = 2000000   # Only used for parquet / deltaparquet
$Workers       = 6

# -------------------------------
# Dimension sizes
# -------------------------------

$Customers   = 10000
$Stores      = 300
$Products    = 6500
$Promotions  = 150

# -------------------------------
# Global dates (used everywhere)
# -------------------------------

$StartDate = "2024-01-01"
$EndDate   = "2024-12-31"

# -------------------------------
# Pipeline control
# -------------------------------

$Clean  = $true
$Only   = ""        # "", "dimensions", "sales"
$DryRun = $false

# ============================================
# Build CLI arguments
# ============================================

$Args = @(
    "--format", $Format
    "--sales-rows", $SalesRows
    "--chunk-size", $ChunkSize
    "--workers", $Workers
    "--customers", $Customers
    "--stores", $Stores
    "--products", $Products
    "--promotions", $Promotions
    "--start-date", $StartDate
    "--end-date", $EndDate
)

if ($SkipOrderCols) {
    $Args += "--skip-order-cols"
}

if ($RowGroupSize -and ($Format -in @("parquet", "deltaparquet"))) {
    $Args += @("--row-group-size", $RowGroupSize)
}

if ($Only) {
    $Args += @("--only", $Only)
}

if ($Clean) {
    $Args += "--clean"
}

if ($DryRun) {
    $Args += "--dry-run"
}

# ============================================
# Execute
# ============================================

python main.py @Args
