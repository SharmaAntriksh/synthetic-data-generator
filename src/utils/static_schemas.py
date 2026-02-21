# src/utils/static_schemas.py

from __future__ import annotations

from typing import Dict, List, Mapping, Sequence, Tuple


# Type aliases (lightweight, no runtime overhead)
SchemaCol = Tuple[str, str]
Schema = Tuple[SchemaCol, ...]


def _derive_schema_from_base(base: Schema, cols: Sequence[str], *, name: str) -> Schema:
    """
    Build a schema by selecting columns from a base schema, preserving the provided column order.

    This keeps datatypes consistent across derived tables (e.g., SalesOrderHeader/Detail)
    while making the column list explicit and easy to review.
    """
    base_map = {c: t for c, t in base}
    missing = [c for c in cols if c not in base_map]
    if missing:
        raise KeyError(
            f"{name}: missing columns not found in base schema: {missing}. "
            f"Update {name} column list or base schema."
        )
    return tuple((c, base_map[c]) for c in cols)


# ============================================================================
# FACT BASE SCHEMAS (single source of truth for derivations)
# ============================================================================
_SALES_SCHEMA: Schema = (
    ("SalesOrderNumber",     "BIGINT NOT NULL"),
    ("SalesOrderLineNumber", "INT NOT NULL"),

    ("CustomerKey",          "INT NOT NULL"),
    ("ProductKey",           "INT NOT NULL"),
    ("StoreKey",             "INT NOT NULL"),
    ("SalesPersonEmployeeKey","INT NOT NULL"),  
    ("PromotionKey",         "INT NOT NULL"),
    ("CurrencyKey",          "INT NOT NULL"),

    ("OrderDate",            "DATE NOT NULL"),
    ("DueDate",              "DATE NOT NULL"),
    ("DeliveryDate",         "DATE NOT NULL"),

    ("Quantity",             "INT NOT NULL"),
    ("NetPrice",             "DECIMAL(8, 2) NOT NULL"),
    ("UnitCost",             "DECIMAL(8, 2) NOT NULL"),
    ("UnitPrice",            "DECIMAL(8, 2) NOT NULL"),
    ("DiscountAmount",       "DECIMAL(8, 2) NOT NULL"),

    ("DeliveryStatus",       "VARCHAR(20) NOT NULL"),
    ("IsOrderDelayed",       "INT NOT NULL"),
)

# Derived fact tables (PascalCase table names for SQL)
_SALES_ORDER_HEADER_COLS: Tuple[str, ...] = (
    "SalesOrderNumber",
    "CustomerKey",
    "OrderDate",
    "IsOrderDelayed",
)

_SALES_ORDER_DETAIL_COLS: Tuple[str, ...] = (
    "SalesOrderNumber",
    "SalesOrderLineNumber",
    "ProductKey",
    "StoreKey",
    "SalesPersonEmployeeKey",
    "PromotionKey",
    "CurrencyKey",
    "DueDate",
    "DeliveryDate",
    "Quantity",
    "NetPrice",
    "UnitCost",
    "UnitPrice",
    "DiscountAmount",
    "DeliveryStatus",
)

_SALES_ORDER_HEADER_SCHEMA: Schema = _derive_schema_from_base(
    _SALES_SCHEMA, _SALES_ORDER_HEADER_COLS, name="SalesOrderHeader"
)
_SALES_ORDER_DETAIL_SCHEMA: Schema = _derive_schema_from_base(
    _SALES_SCHEMA, _SALES_ORDER_DETAIL_COLS, name="SalesOrderDetail"
)


# ============================================================================
# STATIC SCHEMAS (immutable intent)
# ============================================================================
STATIC_SCHEMAS: Dict[str, Schema] = {
    # -----------------------
    # DIMENSIONS
    # -----------------------
    "Customers": (
        ("CustomerKey",        "INT NOT NULL"),
        ("CustomerName",       "VARCHAR(100) NOT NULL"),
        ("DOB",                "DATE"),
        ("MaritalStatus",      "VARCHAR(10)"),
        ("Gender",             "VARCHAR(10) NOT NULL"),
        ("EmailAddress",       "VARCHAR(100)"),
        ("YearlyIncome",       "FLOAT"),
        ("TotalChildren",      "INT"),
        ("Education",          "VARCHAR(20)"),
        ("Occupation",         "VARCHAR(20)"),
        ("CustomerType",       "VARCHAR(20) NOT NULL"),
        ("CompanyName",        "VARCHAR(200)"),
        ("GeographyKey",       "INT NOT NULL"),
        ("IsActiveInSales",    "INT NOT NULL"),

        ("CustomerStartMonth", "INT NOT NULL"),     # 0..T-1 month index
        ("CustomerEndMonth",   "INT NULL"),         # nullable churn month
        ("CustomerStartDate",  "DATE NOT NULL"),    # month start date
        ("CustomerEndDate",    "DATE NULL"),        # nullable churn date

        ("CustomerWeight",      "FLOAT NOT NULL"),  # lognormal > 0
        ("CustomerTemperature", "FLOAT NOT NULL"),  # typically 0..1
        ("CustomerSegment",     "VARCHAR(30) NOT NULL"),
        ("CustomerChurnBias",   "FLOAT NOT NULL"),  # lognormal > 0
    ),
    "CustomerSegment": (
        ("SegmentKey",    "INT NOT NULL"),
        ("SegmentName",   "VARCHAR(200) NOT NULL"),
        ("SegmentType",   "VARCHAR(50) NOT NULL"),
        ("Definition",    "VARCHAR(400) NOT NULL"),
        ("IsActiveFlag",  "TINYINT NOT NULL"),
    ),

    "CustomerSegmentMembership": (
        ("CustomerKey",   "INT NOT NULL"),
        ("SegmentKey",    "INT NOT NULL"),
        ("ValidFromDate", "DATETIME2(7) NOT NULL"),
        ("ValidToDate",   "DATETIME2(7) NOT NULL"),

        # Present by default; optional in config
        ("IsPrimaryFlag", "TINYINT NOT NULL"),
        ("Score",         "REAL NOT NULL"),
    ),
    "Geography": (
        ("GeographyKey", "INT NOT NULL"),
        ("City",         "VARCHAR(100) NOT NULL"),
        ("State",        "VARCHAR(100) NOT NULL"),
        ("Country",      "VARCHAR(100) NOT NULL"),
        ("Continent",    "VARCHAR(100) NOT NULL"),
        ("ISOCode",      "VARCHAR(10) NOT NULL"),
    ),

    "Products": (
        ("ProductKey",              "INT NOT NULL"),
        ("ProductCode",             "VARCHAR(200) NOT NULL"),
        ("ProductName",             "VARCHAR(200) NOT NULL"),
        ("ProductDescription",      "VARCHAR(500) NOT NULL"),
        ("SubcategoryKey",          "INT NOT NULL"),
        ("Brand",                   "VARCHAR(30) NOT NULL"),
        ("Class",                   "VARCHAR(30) NOT NULL"),
        ("Color",                   "VARCHAR(30) NOT NULL"),
        ("StockTypeCode",           "INT NOT NULL"),
        ("StockType",               "VARCHAR(20) NOT NULL"),
        ("UnitCost",                "DECIMAL(10,2) NOT NULL"),
        ("UnitPrice",               "DECIMAL(10,2) NOT NULL"),
        ("BaseProductKey",          "INT NOT NULL"),
        ("VariantIndex",            "INT NOT NULL"),
        ("IsActiveInSales",         "INT NOT NULL"),
    ),

    "ProductCategory": (
        ("CategoryKey",             "INT NOT NULL"),
        ("Category",                "VARCHAR(100) NOT NULL"),
        ("CategoryLabel",           "VARCHAR(10) NOT NULL"),
    ),

    "ProductSubcategory": (
        ("SubcategoryKey",          "INT NOT NULL"),
        ("SubcategoryLabel",        "VARCHAR(10) NOT NULL"),
        ("Subcategory",             "VARCHAR(100) NOT NULL"),
        ("CategoryKey",             "INT"),
    ),

    "Promotions": (
        ("PromotionKey",            "INT NOT NULL"),
        ("PromotionLabel",          "VARCHAR(20) NOT NULL"),
        ("PromotionName",           "VARCHAR(50) NOT NULL"),
        ("PromotionDescription",    "VARCHAR(100) NOT NULL"),
        ("DiscountPct",             "DECIMAL(6,2)"),
        ("PromotionType",           "VARCHAR(20) NOT NULL"),
        ("PromotionCategory",       "VARCHAR(20) NOT NULL"),
        ("StartDate",               "DATE"),
        ("EndDate",                 "DATE"),
    ),

    "Stores": (
        ("StoreKey",         "INT NOT NULL"),
        ("StoreName",        "VARCHAR(100) NOT NULL"),
        ("StoreManager",     "VARCHAR(20)"),
        ("StoreType",        "VARCHAR(20) NOT NULL"),
        ("Status",           "VARCHAR(10)"),
        ("GeographyKey",     "INT NOT NULL"),
        ("OpenDate",         "DATETIME"),
        ("CloseDate",        "DATETIME"),
        ("OpenFlag",         "BIT"),
        ("SquarFootage",     "INT"),
        ("EmployeeCount",    "INT"),
        ("Phone",            "VARCHAR(20)"),
        ("StoreDescription", "VARCHAR(MAX)"),
        ("CloseReason",      "VARCHAR(MAX)"),
    ),

    "Dates": (
        ("Date",                     "DATE NOT NULL"),
        ("DateKey",                  "INT NOT NULL"),

        ("Year",                     "INT NOT NULL"),
        ("IsYearStart",              "INT NOT NULL"),
        ("IsYearEnd",                "INT NOT NULL"),

        ("Quarter",                  "INT NOT NULL"),
        ("QuarterStartDate",         "DATE NOT NULL"),
        ("QuarterEndDate",           "DATE NOT NULL"),
        ("IsQuarterStart",           "INT NOT NULL"),
        ("IsQuarterEnd",             "INT NOT NULL"),
        ("QuarterYear",              "VARCHAR(10) NOT NULL"),

        ("Month",                    "INT NOT NULL"),
        ("MonthName",                "VARCHAR(10) NOT NULL"),
        ("MonthShort",               "VARCHAR(10) NOT NULL"),
        ("MonthStartDate",           "DATE NOT NULL"),
        ("MonthEndDate",             "DATE NOT NULL"),
        ("MonthYear",                "VARCHAR(20) NOT NULL"),
        ("MonthYearNumber",          "INT NOT NULL"),
        ("CalendarMonthIndex",       "INT NOT NULL"),
        ("CalendarQuarterIndex",     "INT NOT NULL"),
        ("IsMonthStart",             "INT NOT NULL"),
        ("IsMonthEnd",               "INT NOT NULL"),

        ("WeekOfYearISO",            "INT NOT NULL"),
        ("ISOYear",                  "INT NOT NULL"),
        ("WeekOfMonth",              "INT NOT NULL"),
        ("WeekStartDate",            "DATE NOT NULL"),
        ("WeekEndDate",              "DATE NOT NULL"),

        ("Day",                      "INT NOT NULL"),
        ("DayName",                  "VARCHAR(10) NOT NULL"),
        ("DayShort",                 "VARCHAR(10) NOT NULL"),
        ("DayOfYear",                "INT NOT NULL"),
        ("DayOfWeek",                "INT NOT NULL"),
        ("IsWeekend",                "INT NOT NULL"),
        ("IsBusinessDay",            "INT NOT NULL"),
        ("NextBusinessDay",          "DATE NOT NULL"),
        ("PreviousBusinessDay",      "DATE NOT NULL"),

        ("FiscalYearStartYear",      "INT NOT NULL"),
        ("FiscalMonthNumber",        "INT NOT NULL"),
        ("FiscalQuarterNumber",      "INT NOT NULL"),
        ("FiscalQuarterName",        "VARCHAR(20) NOT NULL"),
        ("FiscalYearBin",            "VARCHAR(20) NOT NULL"),
        ("FiscalYearMonthNumber",    "INT NOT NULL"),
        ("FiscalYearQuarterNumber",  "INT NOT NULL"),
        ("FiscalMonthIndex",         "INT NOT NULL"),
        ("FiscalQuarterIndex",       "INT NOT NULL"),
        ("FiscalYearStartDate",      "DATE NOT NULL"),
        ("FiscalYearEndDate",        "DATE NOT NULL"),
        ("FiscalQuarterStartDate",   "DATE NOT NULL"),
        ("FiscalQuarterEndDate",     "DATE NOT NULL"),
        ("IsFiscalYearStart",        "BIT NOT NULL"),
        ("IsFiscalYearEnd",          "BIT NOT NULL"),
        ("IsFiscalQuarterStart",     "BIT NOT NULL"),
        ("IsFiscalQuarterEnd",       "BIT NOT NULL"),

        ("FiscalYear",               "INT NOT NULL"),
        ("FiscalYearLabel",          "VARCHAR(10) NOT NULL"),

        ("IsToday",                  "BIT NOT NULL"),
        ("IsCurrentYear",            "BIT NOT NULL"),
        ("IsCurrentMonth",           "BIT NOT NULL"),
        ("IsCurrentQuarter",         "BIT NOT NULL"),

        ("CurrentDayOffset",         "INT NOT NULL"),
    ),

    "Time": (
        ("TimeKey",        "INT NOT NULL"),        # 0..1439 minute-of-day
        ("Hour",           "INT NOT NULL"),        # 0..23
        ("Minute",         "INT NOT NULL"),        # 0..59
        ("TimeText",       "VARCHAR(5) NOT NULL"), # "HH:MM"

        # PQ/SQL-friendly time representations (include if present in parquet)
        ("TimeSeconds",    "INT NOT NULL"),        # seconds since midnight
        ("TimeOfDay",      "TIME(0) NOT NULL"),    # "HH:MM:SS"

        # Rollup bins
        ("TimeKey15",      "INT NOT NULL"),
        ("Bin15Label",     "VARCHAR(11) NOT NULL"),  # "HH:MM-HH:MM"
        ("TimeKey30",      "INT NOT NULL"),
        ("Bin30Label",     "VARCHAR(11) NOT NULL"),
        ("TimeKey60",      "INT NOT NULL"),
        ("Bin60Label",     "VARCHAR(11) NOT NULL"),
        ("TimeKey360",     "INT NOT NULL"),
        ("Bin6hLabel",     "VARCHAR(11) NOT NULL"),  # "HH:MM-HH:MM" (inclusive end style OK too)
        ("TimeKey720",     "INT NOT NULL"),
        ("Bin12hLabel",    "VARCHAR(11) NOT NULL"),

        # Coarse “4 bucket” grouping
        ("TimeBucketKey4", "INT NOT NULL"),
        ("TimeBucket4",    "VARCHAR(10) NOT NULL"),  # Night/Morning/Afternoon/Evening
    ),
    
    "Currency": (
        ("CurrencyKey",     "INT NOT NULL"),
        ("ToCurrency",      "VARCHAR(10) NOT NULL"),
        ("CurrencyName",    "VARCHAR(50) NOT NULL"),
    ),

    "Employees": (
        # Base hierarchy / placement (dict insertion order)
        ("EmployeeKey",        "BIGINT NOT NULL"),
        ("ParentEmployeeKey",  "BIGINT NULL"),
        ("EmployeeName",       "VARCHAR(200) NOT NULL"),
        ("Title",              "VARCHAR(100) NOT NULL"),
        ("OrgLevel",           "INT NOT NULL"),
        ("OrgUnitType",        "VARCHAR(20) NOT NULL"),
        ("RegionId",           "INT NULL"),
        ("DistrictId",         "INT NULL"),
        ("StoreKey",           "BIGINT NULL"),
        ("GeographyKey",       "BIGINT NULL"),

        # Dates / status (appended next)
        ("HireDate",           "DATE NOT NULL"),
        ("TerminationDate",    "DATE NULL"),
        ("IsActive",           "BIT NOT NULL"),

        # Deterministic names (appended next)
        ("FirstName",          "VARCHAR(100) NOT NULL"),
        ("LastName",           "VARCHAR(100) NOT NULL"),
        ("MiddleName",         "VARCHAR(10) NULL"),

        # HR enrichment (appended in this order)
        ("Gender",                 "CHAR(1) NOT NULL"),
        ("BirthDate",              "DATE NOT NULL"),
        ("MaritalStatus",          "CHAR(1) NOT NULL"),
        ("EmailAddress",           "VARCHAR(200) NOT NULL"),
        ("Phone",                  "VARCHAR(20) NOT NULL"),
        ("EmergencyContactName",   "VARCHAR(200) NOT NULL"),
        ("EmergencyContactPhone",  "VARCHAR(20) NOT NULL"),
        ("SalariedFlag",           "BIT NOT NULL"),
        ("PayFrequency",           "INT NOT NULL"),
        ("BaseRate",               "DECIMAL(10, 2) NOT NULL"),
        ("VacationHours",          "INT NOT NULL"),
        ("CurrentFlag",            "BIT NOT NULL"),
        ("StartDate",              "DATE NOT NULL"),
        ("EndDate",                "DATE NULL"),
        ("Status",                 "VARCHAR(20) NOT NULL"),
        ("SalesPersonFlag",        "BIT NOT NULL"),
        ("IsSalesPerson",          "BIT NOT NULL"),
        ("DepartmentName",         "VARCHAR(50) NOT NULL"),
    ),

    "EmployeeStoreAssignments": (
        ("EmployeeKey",        "BIGINT NOT NULL"),
        ("StoreKey",           "BIGINT NOT NULL"),
        ("StartDate",          "DATE NOT NULL"),
        ("EndDate",            "DATE NULL"),
        ("FTE",                "DECIMAL(4, 2) NOT NULL"),
        ("RoleAtStore",        "VARCHAR(100) NOT NULL"),
        ("IsPrimary",          "BIT NOT NULL"),
        ("AssignmentSequence", "INT NOT NULL"),
    ),

    "Superpowers": (
        ("SuperpowerKey",   "INT NOT NULL"),
        ("SuperpowerName",  "VARCHAR(200) NOT NULL"),
        ("PowerType",       "VARCHAR(50) NOT NULL"),
        ("Universe",        "VARCHAR(50) NOT NULL"),
        ("Rarity",          "VARCHAR(20) NOT NULL"),
        ("IconicExamples",  "VARCHAR(400) NOT NULL"),
        ("IsActiveFlag",    "TINYINT NOT NULL"),
    ),

    "CustomerSuperpowers": (
        ("CustomerKey",     "INT NOT NULL"),
        ("SuperpowerKey",   "INT NOT NULL"),
        ("ValidFromDate",   "DATETIME2(7) NOT NULL"),
        ("ValidToDate",     "DATETIME2(7) NOT NULL"),
        ("PowerLevel",      "TINYINT NOT NULL"),
        ("IsPrimaryFlag",   "BIT NOT NULL"),
        ("AcquiredDate",    "DATETIME2(7) NOT NULL"),
    ),
    # -----------------------
    # FACTS
    # -----------------------
    "Sales": _SALES_SCHEMA,

    # New fact tables (SQL table names stay PascalCase)
    "SalesOrderHeader": _SALES_ORDER_HEADER_SCHEMA,
    "SalesOrderDetail": _SALES_ORDER_DETAIL_SCHEMA,

    # NEW: thin SalesReturn fact (order line grain)
    "SalesReturn": (
        ("SalesOrderNumber",     "BIGINT NOT NULL"),
        ("SalesOrderLineNumber", "INT NOT NULL"),
        ("ReturnDate",           "DATE NOT NULL"),
        ("ReturnReasonKey",      "INT NOT NULL"),
        ("ReturnQuantity",       "INT NOT NULL"),
        ("ReturnNetPrice",       "DECIMAL(8, 2) NOT NULL"),
    ),
    "ExchangeRates": (
        ("Date",         "DATE NOT NULL"),
        ("FromCurrency", "VARCHAR(10) NOT NULL"),
        ("ToCurrency",   "VARCHAR(10) NOT NULL"),
        ("Rate",         "DECIMAL(10, 6) NOT NULL"),
    ),
}


# ============================================================================
# DATE COLUMN GROUPS (logical, superset)
# ============================================================================
DATE_COLUMN_GROUPS = {
    "calendar": frozenset({
        "Date", "DateKey",
        "Year", "IsYearStart", "IsYearEnd",
        "Quarter", "QuarterStartDate", "QuarterEndDate",
        "IsQuarterStart", "IsQuarterEnd",
        "QuarterYear",
        "Month", "MonthName", "MonthShort",
        "MonthStartDate", "MonthEndDate",
        "MonthYear", "MonthYearNumber",
        "CalendarMonthIndex", "CalendarQuarterIndex",
        "IsMonthStart", "IsMonthEnd",
        "WeekOfMonth",
        "Day", "DayName", "DayShort", "DayOfYear", "DayOfWeek",
        "IsWeekend", "IsBusinessDay",
        "NextBusinessDay", "PreviousBusinessDay",
        "IsToday", "IsCurrentYear", "IsCurrentMonth",
        "IsCurrentQuarter", "CurrentDayOffset",
    }),
    "iso": frozenset({
        "WeekOfYearISO",
        "ISOYear",
        "WeekStartDate",
        "WeekEndDate",
    }),
    "fiscal": frozenset({
        "FiscalYearStartYear", "FiscalMonthNumber", "FiscalQuarterNumber",
        "FiscalMonthIndex", "FiscalQuarterIndex",
        "FiscalQuarterName", "FiscalYearBin",
        "FiscalYearMonthNumber", "FiscalYearQuarterNumber",
        "FiscalYearStartDate", "FiscalYearEndDate",
        "FiscalQuarterStartDate", "FiscalQuarterEndDate",
        "IsFiscalYearStart", "IsFiscalYearEnd",
        "IsFiscalQuarterStart", "IsFiscalQuarterEnd",
        "FiscalYear", "FiscalYearLabel",
    }),
}


# ============================================================================
# Precomputed schemas / caches
# ============================================================================
_SALES_SCHEMA_NO_ORDER: Schema = tuple(
    (col, dtype)
    for col, dtype in _SALES_SCHEMA
    if col not in ("SalesOrderNumber", "SalesOrderLineNumber")
)

# Cache for get_dates_schema keyed by (calendar, iso, fiscal)
_DATES_SCHEMA_CACHE: Dict[Tuple[bool, bool, bool], List[SchemaCol]] = {}


def get_sales_schema(skip_order_cols: bool):
    """Return the Sales schema with or without order number columns."""
    return list(_SALES_SCHEMA_NO_ORDER if skip_order_cols else _SALES_SCHEMA)


def get_sales_order_header_schema() -> List[SchemaCol]:
    """Return the SalesOrderHeader schema."""
    return list(_SALES_ORDER_HEADER_SCHEMA)


def get_sales_order_detail_schema() -> List[SchemaCol]:
    """Return the SalesOrderDetail schema."""
    return list(_SALES_ORDER_DETAIL_SCHEMA)


def get_dates_schema(dates_cfg: Mapping):
    """
    Return Dates schema filtered by config include flags.
    Defaults to calendar-only (backward compatible).

    dates_cfg expected shape:
      dates:
        include:
          calendar: true
          iso: false
          fiscal: false
    """
    include_cfg = (dates_cfg.get("include", {}) or {}) if isinstance(dates_cfg, Mapping) else {}

    include_calendar = bool(include_cfg.get("calendar", True))
    include_iso = bool(include_cfg.get("iso", False))
    include_fiscal = bool(include_cfg.get("fiscal", False))

    cache_key = (include_calendar, include_iso, include_fiscal)
    cached = _DATES_SCHEMA_CACHE.get(cache_key)
    if cached is not None:
        # Return a shallow copy to prevent accidental mutation by callers
        return list(cached)

    allowed_cols = set()
    if include_calendar:
        allowed_cols.update(DATE_COLUMN_GROUPS["calendar"])
    if include_iso:
        allowed_cols.update(DATE_COLUMN_GROUPS["iso"])
    if include_fiscal:
        allowed_cols.update(DATE_COLUMN_GROUPS["fiscal"])

    # Preserve original order from STATIC_SCHEMAS["Dates"]
    out = [(col, dtype) for col, dtype in STATIC_SCHEMAS["Dates"] if col in allowed_cols]

    _DATES_SCHEMA_CACHE[cache_key] = out
    return list(out)


__all__ = [
    "STATIC_SCHEMAS",
    "DATE_COLUMN_GROUPS",
    "get_sales_schema",
    "get_sales_order_header_schema",
    "get_sales_order_detail_schema",
    "get_dates_schema",
]
