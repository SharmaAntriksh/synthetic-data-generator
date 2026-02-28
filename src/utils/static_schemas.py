# src/utils/static_schemas.py

from __future__ import annotations

from types import MappingProxyType
from typing import Dict, List, Mapping, Sequence, Tuple
import re

# Type aliases (lightweight, no runtime overhead)
SchemaCol = Tuple[str, str]
Schema = Tuple[SchemaCol, ...]


# ============================================================================
# SQL-ish type helpers (keep strings consistent + reduce repetition)
# ============================================================================
def _sql(base: str, *, not_null: bool) -> str:
    return f"{base} NOT NULL" if not_null else f"{base} NULL"


def INT(*, not_null: bool = True) -> str:
    return _sql("INT", not_null=not_null)


def BIGINT(*, not_null: bool = True) -> str:
    return _sql("BIGINT", not_null=not_null)


def SMALLINT(*, not_null: bool = True) -> str:
    return _sql("SMALLINT", not_null=not_null)


def TINYINT(*, not_null: bool = True) -> str:
    return _sql("TINYINT", not_null=not_null)


def BIT(*, not_null: bool = True) -> str:
    return _sql("BIT", not_null=not_null)


def FLOAT(*, not_null: bool = True) -> str:
    return _sql("FLOAT", not_null=not_null)


def REAL(*, not_null: bool = True) -> str:
    return _sql("REAL", not_null=not_null)


def DATE(*, not_null: bool = True) -> str:
    return _sql("DATE", not_null=not_null)


def DATETIME(*, not_null: bool = True) -> str:
    return _sql("DATETIME", not_null=not_null)


def DATETIME2(p: int = 7, *, not_null: bool = True) -> str:
    return _sql(f"DATETIME2({p})", not_null=not_null)


def TIME(p: int = 0, *, not_null: bool = True) -> str:
    return _sql(f"TIME({p})", not_null=not_null)


def VARCHAR(n: int | str, *, not_null: bool = False) -> str:
    return _sql(f"VARCHAR({n})", not_null=not_null)


def CHAR(n: int, *, not_null: bool = False) -> str:
    return _sql(f"CHAR({n})", not_null=not_null)


def DECIMAL(p: int, s: int, *, not_null: bool = True) -> str:
    return _sql(f"DECIMAL({p}, {s})", not_null=not_null)


# Common aliases (readability)
INT_NN = INT(not_null=True)
INT_NULL = INT(not_null=False)
BIGINT_NN = BIGINT(not_null=True)
BIGINT_NULL = BIGINT(not_null=False)
DATE_NN = DATE(not_null=True)
DATE_NULL = DATE(not_null=False)


# ============================================================================
# Schema derivation + validation
# ============================================================================
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


def _validate_schema(name: str, schema: Schema) -> None:
    if not isinstance(schema, tuple) or len(schema) == 0:
        raise ValueError(f"{name}: schema must be a non-empty tuple of (col, type) pairs")

    seen = set()
    for item in schema:
        if (
            not isinstance(item, tuple)
            or len(item) != 2
            or not isinstance(item[0], str)
            or not isinstance(item[1], str)
        ):
            raise TypeError(f"{name}: invalid schema entry {item!r} (expected (str, str))")

        col, typ = item
        if not col.strip():
            raise ValueError(f"{name}: empty column name")
        if col in seen:
            raise ValueError(f"{name}: duplicate column {col!r}")
        seen.add(col)

        if not typ.strip():
            raise ValueError(f"{name}: empty type for column {col!r}")


def _validate_schema_map(schema_map: Mapping[str, Schema]) -> None:
    for table, schema in schema_map.items():
        if not isinstance(table, str) or not table.strip():
            raise ValueError(f"Invalid table key: {table!r}")
        _validate_schema(table, schema)


# ============================================================================
# FACT BASE SCHEMAS (single source of truth for derivations)
# ============================================================================
_SALES_SCHEMA: Schema = (
    ("SalesOrderNumber", BIGINT_NN),
    ("SalesOrderLineNumber", INT_NN),
    ("CustomerKey", INT_NN),
    ("ProductKey", INT_NN),
    ("StoreKey", INT_NN),
    ("SalesPersonEmployeeKey", BIGINT_NN),
    ("PromotionKey", INT_NN),
    ("CurrencyKey", INT_NN),
    ("SalesChannelKey", INT_NN),
    ("TimeKey", INT_NN),
    ("OrderDate", DATE_NN),
    ("DueDate", DATE_NN),
    ("DeliveryDate", DATE_NN),
    ("Quantity", INT_NN),
    ("NetPrice", DECIMAL(8, 2, not_null=True)),
    ("UnitCost", DECIMAL(8, 2, not_null=True)),
    ("UnitPrice", DECIMAL(8, 2, not_null=True)),
    ("DiscountAmount", DECIMAL(8, 2, not_null=True)),
    ("DeliveryStatus", VARCHAR(20, not_null=True)),
    ("IsOrderDelayed", INT_NN),
)


_SALES_SCHEMA_NO_ORDER: Schema = tuple(
    (name, dtype) for (name, dtype) in _SALES_SCHEMA
    if name not in ("SalesOrderNumber", "SalesOrderLineNumber")
)

# Derived fact tables (PascalCase table names for SQL)
_SALES_ORDER_HEADER_COLS: Tuple[str, ...] = (
    "SalesOrderNumber",
    "CustomerKey",
    "StoreKey",
    "SalesPersonEmployeeKey",
    "PromotionKey",
    "CurrencyKey",
    "SalesChannelKey",
    "OrderDate",
    "TimeKey",
    "IsOrderDelayed",
)

_SALES_ORDER_DETAIL_COLS: Tuple[str, ...] = (
    "SalesOrderNumber",
    "SalesOrderLineNumber",
    "ProductKey",
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
# STATIC SCHEMAS (organized, validated, then frozen)
# ============================================================================
DIM_SCHEMAS: Dict[str, Schema] = {
    "Customers": (
        ("CustomerKey", INT_NN),
        ("CustomerName", VARCHAR(100, not_null=True)),
        ("DOB", DATE_NULL),
        ("MaritalStatus", VARCHAR(10, not_null=False)),
        ("Gender", VARCHAR(10, not_null=True)),
        ("EmailAddress", VARCHAR(100, not_null=False)),
        ("YearlyIncome", FLOAT(not_null=False)),
        ("TotalChildren", INT(not_null=False)),
        ("Education", VARCHAR(20, not_null=False)),
        ("Occupation", VARCHAR(20, not_null=False)),
        ("CustomerType", VARCHAR(20, not_null=True)),
        ("CompanyName", VARCHAR(200, not_null=False)),
        ("GeographyKey", INT_NN),
        ("LoyaltyTierKey", INT_NN),
        ("CustomerAcquisitionChannelKey", INT_NN),
        ("IsActiveInSales", INT_NN),
        ("CustomerStartMonth", INT_NN),  # 0..T-1 month index
        ("CustomerEndMonth", INT_NULL),  # nullable churn month
        ("CustomerStartDate", DATE_NN),  # month start date
        ("CustomerEndDate", DATE_NULL),  # nullable churn date
        ("CustomerWeight", FLOAT(not_null=True)),  # lognormal > 0
        ("CustomerTemperature", FLOAT(not_null=True)),  # typically 0..1
        ("CustomerSegment", VARCHAR(30, not_null=True)),
        ("CustomerChurnBias", FLOAT(not_null=True)),  # lognormal > 0
    ),
    "CustomerAcquisitionChannels": (
        ("CustomerAcquisitionChannelKey", INT_NN),
        ("AcquisitionChannel", VARCHAR(50, not_null=False)),
        ("ChannelGroup", VARCHAR(50, not_null=False)),
    ),
    "CustomerSegment": (
        ("SegmentKey", INT_NN),
        ("SegmentName", VARCHAR(200, not_null=True)),
        ("SegmentType", VARCHAR(50, not_null=True)),
        ("Definition", VARCHAR(400, not_null=True)),
        ("IsActiveFlag", TINYINT(not_null=True)),
    ),
    "CustomerSegmentMembership": (
        ("CustomerKey", INT_NN),
        ("SegmentKey", INT_NN),
        ("ValidFromDate", DATETIME2(7, not_null=True)),
        ("ValidToDate", DATETIME2(7, not_null=True)),
        ("IsPrimaryFlag", TINYINT(not_null=True)),  # present by default; optional in config
        ("Score", REAL(not_null=True)),
    ),
    "Geography": (
        ("GeographyKey", INT_NN),
        ("City", VARCHAR(100, not_null=True)),
        ("State", VARCHAR(100, not_null=True)),
        ("Country", VARCHAR(100, not_null=True)),
        ("Continent", VARCHAR(100, not_null=True)),
        ("ISOCode", VARCHAR(10, not_null=True)),
    ),
    "Products": (
        ("ProductKey", INT_NN),
        ("ProductCode", VARCHAR(200, not_null=True)),
        ("ProductName", VARCHAR(200, not_null=True)),
        ("ProductDescription", VARCHAR(500, not_null=True)),
        ("SubcategoryKey", INT_NN),
        ("Brand", VARCHAR(30, not_null=True)),
        ("Class", VARCHAR(30, not_null=True)),
        ("Color", VARCHAR(30, not_null=True)),
        ("StockTypeCode", INT_NN),
        ("StockType", VARCHAR(20, not_null=True)),
        ("UnitCost", DECIMAL(10, 2, not_null=True)),
        ("UnitPrice", DECIMAL(10, 2, not_null=True)),
        ("BaseProductKey", INT_NN),
        ("VariantIndex", INT_NN),
        ("LaunchDate", DATE_NN),
        ("LaunchDateKey", INT_NN),
        ("DiscontinuedDate", DATE_NULL),
        ("DiscontinuedDateKey", INT_NULL),
        ("IsDiscontinued", INT_NN),
        ("ColorFamily", VARCHAR(20, not_null=True)),
        ("Material", VARCHAR(30, not_null=True)),
        ("Style", VARCHAR(20, not_null=True)),
        ("AgeGroup", VARCHAR(20, not_null=True)),
        ("ProductLine", VARCHAR(30, not_null=True)),
        ("SeasonalityProfile", VARCHAR(30, not_null=True)),
        ("WeightKg", FLOAT(not_null=True)),
        ("LengthCm", FLOAT(not_null=True)),
        ("WidthCm", FLOAT(not_null=True)),
        ("HeightCm", FLOAT(not_null=True)),
        ("VolumeCm3", FLOAT(not_null=True)),
        ("ShippingClass", VARCHAR(20, not_null=True)),
        ("IsFragile", INT_NN),
        ("IsHazmat", INT_NN),
        ("LeadTimeDays", INT_NN),
        ("CasePackQty", INT_NN),
        ("FulfillmentType", VARCHAR(20, not_null=True)),
        ("BrandTier", VARCHAR(20, not_null=True)),
        ("EligibleStore", INT_NN),
        ("EligibleOnline", INT_NN),
        ("EligibleMarketplace", INT_NN),
        ("EligibleB2B", INT_NN),
        ("WarrantyMonths", INT_NN),
        ("ReturnRateBase", FLOAT(not_null=True)),
        ("DefectRateBase", FLOAT(not_null=True)),
        ("ReturnWindowDays", INT_NN),
        ("SupplierKey", INT_NN),
        ("IsActiveInSales", INT_NN),
    ),
    "ProductCategory": (
        ("CategoryKey", INT_NN),
        ("Category", VARCHAR(100, not_null=True)),
        ("CategoryLabel", VARCHAR(10, not_null=True)),
    ),
    "ProductSubcategory": (
        ("SubcategoryKey", INT_NN),
        ("SubcategoryLabel", VARCHAR(10, not_null=True)),
        ("Subcategory", VARCHAR(100, not_null=True)),
        ("CategoryKey", INT(not_null=False)),
    ),
    "Promotions": (
        ("PromotionKey", INT_NN),
        ("PromotionLabel", VARCHAR(20, not_null=True)),
        ("PromotionName", VARCHAR(100, not_null=True)),
        ("PromotionDescription", VARCHAR(100, not_null=True)),
        ("DiscountPct", DECIMAL(6, 2, not_null=False)),
        ("PromotionType", VARCHAR(20, not_null=True)),
        ("PromotionCategory", VARCHAR(20, not_null=True)),
        ("StartDate", DATE_NULL),
        ("EndDate", DATE_NULL),
    ),
    "Stores": (
        ("StoreKey", INT_NN),
        ("StoreType", VARCHAR(20, not_null=True)),
        ("Status", VARCHAR(10, not_null=False)),
        ("GeographyKey", INT_NN),
        ("StoreManager", VARCHAR(200, not_null=False)),
        ("StoreName", VARCHAR(200, not_null=True)),
        ("OpenDate", DATETIME(not_null=False)),
        ("CloseDate", DATETIME(not_null=False)),
        ("OpenFlag", BIT(not_null=False)),
        ("SquarFootage", INT(not_null=False)),
        ("EmployeeCount", INT(not_null=False)),
        ("Phone", VARCHAR(20, not_null=False)),
        ("CloseReason", VARCHAR("MAX", not_null=False)),
        ("StoreDescription", VARCHAR("MAX", not_null=False)),
    ),
    "Dates": (
        ("Date", DATE_NN),
        ("DateKey", INT_NN),
        ("SequentialDayIndex", INT_NN),

        ("Year", INT_NN),
        ("IsYearStart", INT_NN),
        ("IsYearEnd", INT_NN),

        ("Quarter", INT_NN),
        ("QuarterStartDate", DATE_NN),
        ("QuarterEndDate", DATE_NN),
        ("IsQuarterStart", INT_NN),
        ("IsQuarterEnd", INT_NN),
        ("QuarterYear", VARCHAR(10, not_null=True)),

        ("Month", INT_NN),
        ("MonthName", VARCHAR(10, not_null=True)),
        ("MonthShort", VARCHAR(10, not_null=True)),
        ("MonthNameShort", VARCHAR(10, not_null=True)),
        ("MonthStartDate", DATE_NN),
        ("MonthEndDate", DATE_NN),
        ("MonthYear", VARCHAR(20, not_null=True)),
        ("MonthYearNumber", INT_NN),
        ("YearMonthKey", INT_NN),
        ("YearMonthLabel", VARCHAR(20, not_null=True)),
        ("YearQuarterKey", INT_NN),
        ("YearQuarterLabel", VARCHAR(10, not_null=True)),
        ("CalendarMonthIndex", INT_NN),
        ("CalendarQuarterIndex", INT_NN),
        ("IsMonthStart", INT_NN),
        ("IsMonthEnd", INT_NN),

        ("WeekOfMonth", INT_NN),

        ("Day", INT_NN),
        ("DayName", VARCHAR(10, not_null=True)),
        ("DayShort", VARCHAR(10, not_null=True)),
        ("DayNameShort", VARCHAR(10, not_null=True)),
        ("DayOfYear", INT_NN),
        ("DayOfWeek", INT_NN),
        ("IsWeekend", INT_NN),
        ("IsBusinessDay", INT_NN),
        ("NextBusinessDay", DATE_NN),
        ("PreviousBusinessDay", DATE_NN),

        ("IsToday", BIT(not_null=True)),
        ("IsCurrentYear", BIT(not_null=True)),
        ("IsCurrentMonth", BIT(not_null=True)),
        ("IsCurrentQuarter", BIT(not_null=True)),
        ("CurrentDayOffset", INT_NN),
        ("YearOffset", SMALLINT(not_null=True)),
        ("CalendarMonthOffset", INT_NN),
        ("CalendarQuarterOffset", INT_NN),

        ("WeekOfYearISO", INT_NN),
        ("ISOYear", INT_NN),
        ("ISOYearWeekIndex", INT_NN),
        ("ISOWeekOffset", INT_NN),
        ("WeekStartDate", DATE_NN),
        ("WeekEndDate", DATE_NN),

        # Month-based fiscal calendar (kept as "Fiscal ..." to distinguish from Weekly Fiscal)
        ("FiscalYearStartYear", INT_NN),
        ("FiscalMonthNumber", INT_NN),
        ("FiscalQuarterNumber", INT_NN),
        ("FiscalMonthIndex", INT_NN),
        ("FiscalQuarterIndex", INT_NN),
        ("FiscalMonthOffset", INT_NN),
        ("FiscalQuarterOffset", INT_NN),
        ("FiscalQuarterName", VARCHAR(20, not_null=True)),
        ("FiscalYearBin", VARCHAR(20, not_null=True)),
        ("FiscalYearMonthNumber", INT_NN),
        ("FiscalYearQuarterNumber", INT_NN),
        ("FiscalYearStartDate", DATE_NN),
        ("FiscalYearEndDate", DATE_NN),
        ("FiscalQuarterStartDate", DATE_NN),
        ("FiscalQuarterEndDate", DATE_NN),
        ("IsFiscalYearStart", BIT(not_null=True)),
        ("IsFiscalYearEnd", BIT(not_null=True)),
        ("IsFiscalQuarterStart", BIT(not_null=True)),
        ("IsFiscalQuarterEnd", BIT(not_null=True)),
        ("FiscalYear", INT_NN),
        ("FiscalYearLabel", VARCHAR(10, not_null=True)),
        ("FiscalSystem", VARCHAR(20, not_null=True)),
        ("WeeklyFiscalSystem", VARCHAR(40, not_null=True)),

        # Weekly fiscal calendar (DAX weekly logic), disambiguated with "Weekly Fiscal ..." prefix.
        ("FWYearNumber", INT_NN),
        ("FWYearLabel", VARCHAR(10, not_null=True)),
        ("FWQuarterNumber", INT_NN),
        ("FWQuarterLabel", VARCHAR(20, not_null=True)),
        ("FWYearQuarterNumber", INT_NN),
        ("FWYearQuarterOffset", INT_NN),
        ("FWMonthNumber", INT_NN),
        ("FWMonthLabel", VARCHAR(20, not_null=True)),
        ("FWYearMonthNumber", INT_NN),
        ("FWYearMonthOffset", INT_NN),
        ("FWWeekNumber", INT_NN),
        ("FWWeekLabel", VARCHAR(20, not_null=True)),
        ("FWYearWeekNumber", INT_NN),
        ("FWYearWeekOffset", INT_NN),
        ("FWYearWeekLabel", VARCHAR(20, not_null=True)),
        ("FWPeriodNumber", INT_NN),
        ("FWPeriodLabel", VARCHAR(20, not_null=True)),
        ("FWStartOfYear", DATE_NN),
        ("FWEndOfYear", DATE_NN),
        ("FWStartOfQuarter", DATE_NN),
        ("FWEndOfQuarter", DATE_NN),
        ("FWStartOfMonth", DATE_NN),
        ("FWEndOfMonth", DATE_NN),
        ("FWStartOfWeek", DATE_NN),
        ("FWEndOfWeek", DATE_NN),
        ("WeekDayNumber", INT_NN),
        ("WeekDayNameShort", VARCHAR(10, not_null=True)),
        ("FWDayOfYearNumber", INT_NN),
        ("FWDayOfQuarterNumber", INT_NN),
        ("FWDayOfMonthNumber", INT_NN),
        ("IsWorkingDay", BIT(not_null=True)),
        ("DayType", VARCHAR(20, not_null=True)),
        ("FWWeekInQuarterNumber", INT_NN),
        ("FWYearMonthLabel", VARCHAR(20, not_null=True)),
        ("FWYearQuarterLabel", VARCHAR(20, not_null=True)),
    ),
    "Time": (
        ("TimeKey", INT_NN),  # 0..1439 minute-of-day
        ("Hour", INT_NN),  # 0..23
        ("Minute", INT_NN),  # 0..59
        ("TimeText", VARCHAR(5, not_null=True)),  # "HH:MM"
        ("TimeKey15", INT_NN),
        ("Bin15Label", VARCHAR(11, not_null=True)),
        ("TimeKey30", INT_NN),
        ("Bin30Label", VARCHAR(11, not_null=True)),
        ("TimeKey60", INT_NN),
        ("Bin60Label", VARCHAR(11, not_null=True)),
        ("TimeKey360", INT_NN),
        ("Bin6hLabel", VARCHAR(11, not_null=True)),
        ("TimeKey720", INT_NN),
        ("Bin12hLabel", VARCHAR(11, not_null=True)),
        ("TimeBucketKey4", INT_NN),
        ("TimeBucket4", VARCHAR(10, not_null=True)),
        ("TimeSeconds", INT_NN),
        ("TimeOfDay", TIME(0, not_null=True)),
    ),
    "Currency": (
        ("CurrencyKey", INT_NN),
        ("ToCurrency", VARCHAR(10, not_null=True)),
        ("CurrencyName", VARCHAR(50, not_null=True)),
    ),
    "Employees": (
        ("EmployeeKey", BIGINT_NN),
        ("ParentEmployeeKey", BIGINT_NULL),
        ("EmployeeName", VARCHAR(200, not_null=True)),
        ("Title", VARCHAR(100, not_null=True)),
        ("OrgLevel", INT_NN),
        ("OrgUnitType", VARCHAR(20, not_null=True)),
        ("RegionId", INT_NULL),
        ("DistrictId", INT_NULL),
        ("HireDate", DATE_NN),
        ("TerminationDate", DATE_NULL),
        ("IsActive", BIT(not_null=True)),
        ("Gender", VARCHAR(10, not_null=False)),
        ("FirstName", VARCHAR(100, not_null=True)),
        ("LastName", VARCHAR(100, not_null=True)),
        ("MiddleName", VARCHAR(10, not_null=False)),
        ("BirthDate", DATE_NN),
        ("MaritalStatus", CHAR(1, not_null=True)),
        ("EmailAddress", VARCHAR(200, not_null=True)),
        ("Phone", VARCHAR(20, not_null=True)),
        ("EmergencyContactName", VARCHAR(200, not_null=True)),
        ("EmergencyContactPhone", VARCHAR(20, not_null=True)),
        ("SalariedFlag", BIT(not_null=True)),
        ("PayFrequency", INT_NN),
        ("BaseRate", DECIMAL(10, 2, not_null=True)),
        ("VacationHours", INT_NN),
        ("CurrentFlag", BIT(not_null=True)),
        ("StartDate", DATE_NN),
        ("EndDate", DATE_NULL),
        ("Status", VARCHAR(20, not_null=True)),
        ("SalesPersonFlag", BIT(not_null=True)),
        ("IsSalesPerson", BIT(not_null=True)),
        ("DepartmentName", VARCHAR(50, not_null=True)),
    ),
    "EmployeeStoreAssignments": (
        ("EmployeeKey", BIGINT_NN),
        ("StoreKey", INT_NN),
        ("StartDate", DATE_NN),
        ("EndDate", DATE_NULL),
        ("FTE", DECIMAL(18, 2, not_null=True)),
        ("RoleAtStore", VARCHAR(100, not_null=True)),
        ("IsPrimary", BIT(not_null=False)),
        ("AssignmentSequence", INT_NN),
    ),
    "Suppliers": (
        ("SupplierKey", INT_NN),
        ("SupplierName", VARCHAR(200, not_null=True)),
        ("SupplierType", VARCHAR(50, not_null=True)),
        ("Country", VARCHAR(100, not_null=False)),
        ("ReliabilityScore", DECIMAL(4, 3, not_null=False)),
    ),
    "Superpowers": (
        ("SuperpowerKey", INT_NN),
        ("SuperpowerName", VARCHAR(200, not_null=True)),
        ("PowerType", VARCHAR(50, not_null=True)),
        ("Universe", VARCHAR(50, not_null=True)),
        ("Rarity", VARCHAR(20, not_null=True)),
        ("IconicExamples", VARCHAR(400, not_null=True)),
        ("IsActiveFlag", BIT(not_null=True)),
    ),
    "CustomerSuperpowers": (
        ("CustomerKey", INT_NN),
        ("SuperpowerKey", INT_NN),
        ("ValidFromDate", DATETIME2(7, not_null=True)),
        ("ValidToDate", DATETIME2(7, not_null=True)),
        ("PowerLevel", TINYINT(not_null=True)),
        ("IsPrimaryFlag", BIT(not_null=True)),
        ("AcquiredDate", DATETIME2(7, not_null=True)),
    ),
    "LoyaltyTiers": (
        ("LoyaltyTierKey", INT_NN),
        ("LoyaltyTier", VARCHAR(50, not_null=True)),
        ("TierRank", TINYINT(not_null=True)),
        ("PointsMultiplier", DECIMAL(4, 2, not_null=True)),
    ),
    "SalesChannels": (
        ("SalesChannelKey", INT_NN),
        ("SalesChannel", VARCHAR(50, not_null=False)),
        ("ChannelGroup", VARCHAR(50, not_null=False)),
        ("SalesChannelCode", VARCHAR(30, not_null=False)),
        ("SortOrder", SMALLINT(not_null=False)),
        ("IsDigital", BIT(not_null=False)),
        ("IsPhysical", BIT(not_null=False)),
        ("IsThirdParty", BIT(not_null=False)),
        ("IsB2B", BIT(not_null=False)),
        ("IsAssisted", BIT(not_null=False)),
        ("IsOwnedChannel", BIT(not_null=False)),
        ("TimeProfile", VARCHAR(20, not_null=False)),
        ("Is24x7", BIT(not_null=False)),
        ("OpenMinute", SMALLINT(not_null=False)),
        ("CloseMinute", SMALLINT(not_null=False)),
    ),
    "ReturnReason": (
        ("ReturnReasonKey", INT_NN),
        ("ReturnReason", VARCHAR(200, not_null=True)),
        ("ReturnReasonCategory", VARCHAR(100, not_null=True)),
    ),
}

FACT_SCHEMAS: Dict[str, Schema] = {
    "Sales": _SALES_SCHEMA,
    "SalesOrderHeader": _SALES_ORDER_HEADER_SCHEMA,
    "SalesOrderDetail": _SALES_ORDER_DETAIL_SCHEMA,
    "SalesReturn": (
        ("SalesOrderNumber", BIGINT_NN),
        ("SalesOrderLineNumber", INT_NN),
        ("ReturnDate", DATE_NN),
        ("ReturnReasonKey", INT_NN),
        ("ReturnQuantity", INT_NN),
        ("ReturnNetPrice", DECIMAL(8, 2, not_null=True)),
        ("ReturnEventKey", BIGINT_NN),
    ),
    "ExchangeRates": (
        ("Date", DATE_NN),
        ("FromCurrency", VARCHAR(10, not_null=True)),
        ("ToCurrency", VARCHAR(10, not_null=True)),
        ("Rate", DECIMAL(10, 6, not_null=True)),
    ),
}

# Validate before freezing (fail fast at import time)
_ALL_SCHEMAS_MUT: Dict[str, Schema] = {**DIM_SCHEMAS, **FACT_SCHEMAS}
_validate_schema_map(_ALL_SCHEMAS_MUT)

# Freeze: prevent accidental runtime mutation by callers
STATIC_SCHEMAS: Mapping[str, Schema] = MappingProxyType(_ALL_SCHEMAS_MUT)


# ============================================================================
# DATE COLUMN GROUPS (logical, superset)
# ============================================================================

# ---------------------------------------------------------------------
# Dates schema helpers
# ---------------------------------------------------------------------

_WF_INTERNAL_COLS: set[str] = {
    "FWYearNumber",
    "FWYearLabel",
    "FWQuarterNumber",
    "FWQuarterLabel",
    "FWYearQuarterNumber",
    "FWMonthNumber",
    "FWMonthLabel",
    "FWYearMonthNumber",
    "FWWeekNumber",
    "FWWeekLabel",
    "FWYearWeekNumber",
    "FWYearWeekLabel",
    "FWYearQuarterOffset",
    "FWYearMonthOffset",
    "FWYearWeekOffset",
    "FWPeriodNumber",
    "FWPeriodLabel",
    "FWStartOfYear",
    "FWEndOfYear",
    "FWStartOfQuarter",
    "FWEndOfQuarter",
    "FWStartOfMonth",
    "FWEndOfMonth",
    "FWStartOfWeek",
    "FWEndOfWeek",
    "WeekDayNumber",
    "WeekDayNameShort",
    "FWDayOfYearNumber",
    "FWDayOfQuarterNumber",
    "FWDayOfMonthNumber",
    "IsWorkingDay",
    "DayType",
    "FWWeekInQuarterNumber",
    "FWYearMonthLabel",
    "FWYearQuarterLabel",
}



# ---------------------------------------------------------------------
# Dates column naming
# ---------------------------------------------------------------------
# The generator emits SQL-friendly internal column names (no spaces).
# Any "pretty" renaming should be done downstream (e.g., Power BI display folders or SQL views).


# Internal Dates column groups (must match dates.py resolve_date_columns)
_DATES_BASE_INTERNAL = ["Date", "DateKey", "SequentialDayIndex"]

_DATES_CAL_CORE_INTERNAL = [
    "Year", "IsYearStart", "IsYearEnd",
    "Quarter", "QuarterStartDate", "QuarterEndDate",
    "IsQuarterStart", "IsQuarterEnd",
    "QuarterYear",
    "Month", "MonthName", "MonthShort", "MonthNameShort",
    "MonthStartDate", "MonthEndDate",
    "MonthYear", "MonthYearNumber", "YearMonthKey", "YearMonthLabel",
    "YearQuarterKey", "YearQuarterLabel",
    "CalendarMonthIndex", "CalendarQuarterIndex",
    "IsMonthStart", "IsMonthEnd",
    "WeekOfMonth",
    "Day", "DayName", "DayShort", "DayNameShort", "DayOfYear", "DayOfWeek",
    "IsWeekend", "IsBusinessDay",
    "NextBusinessDay", "PreviousBusinessDay",
]

_DATES_CAL_EXTRAS_INTERNAL = [
    "IsToday", "IsCurrentYear", "IsCurrentMonth", "IsCurrentQuarter",
    "CurrentDayOffset", "YearOffset", "CalendarMonthOffset", "CalendarQuarterOffset",
]

_DATES_ISO_INTERNAL = [
    "WeekOfYearISO",
    "ISOYear",
    "ISOYearWeekIndex",
    "ISOWeekOffset",
    "WeekStartDate",
    "WeekEndDate",
]

_DATES_FISCAL_INTERNAL = [
    "FiscalYearStartYear", "FiscalMonthNumber", "FiscalQuarterNumber",
    "FiscalMonthIndex", "FiscalQuarterIndex", "FiscalMonthOffset", "FiscalQuarterOffset",
    "FiscalQuarterName", "FiscalYearBin",
    "FiscalYearMonthNumber", "FiscalYearQuarterNumber",
    "FiscalYearStartDate", "FiscalYearEndDate",
    "FiscalQuarterStartDate", "FiscalQuarterEndDate",
    "IsFiscalYearStart", "IsFiscalYearEnd",
    "IsFiscalQuarterStart", "IsFiscalQuarterEnd",
    "FiscalYear", "FiscalYearLabel", "FiscalSystem", "WeeklyFiscalSystem",
]

_DATES_WEEKLY_INTERNAL = sorted(list(_WF_INTERNAL_COLS))  # order will be overridden by resolve list below


def _dates_internal_columns(dates_cfg: Mapping) -> list[str]:
    """Resolve internal output column list; mirrors dates.py resolve_date_columns semantics."""
    dates_cfg = dates_cfg or {}
    include_cfg = dates_cfg.get("include", None) if isinstance(dates_cfg, Mapping) else None
    weekly_cfg = (dates_cfg.get("weekly_calendar", {}) or {}) if isinstance(dates_cfg, Mapping) else {}

    has_include = isinstance(include_cfg, Mapping)
    include = include_cfg or {}

    if not has_include:
        include_calendar = True
        include_iso = True
        include_fiscal = True
        include_weekly = True
    else:
        include_calendar = bool(include.get("calendar", True))
        include_iso = bool(include.get("iso", False))
        include_fiscal = bool(include.get("fiscal", False))
        include_weekly = bool(include.get("weekly_fiscal", False))

    cols: list[str] = []
    cols += _DATES_BASE_INTERNAL
    cols += _DATES_CAL_CORE_INTERNAL

    if include_calendar:
        cols += _DATES_CAL_EXTRAS_INTERNAL
    if include_iso:
        cols += _DATES_ISO_INTERNAL
    if include_fiscal:
        cols += _DATES_FISCAL_INTERNAL

    if include_weekly and bool(weekly_cfg.get("enabled", True)):
        # Keep weekly columns in a stable order matching the original STATIC_SCHEMAS ordering.
        cols += [
            "FWYearNumber",
            "FWYearLabel",
            "FWQuarterNumber",
            "FWQuarterLabel",
            "FWYearQuarterNumber",
            "FWYearQuarterOffset",
            "FWMonthNumber",
            "FWMonthLabel",
            "FWYearMonthNumber",
            "FWYearMonthOffset",
            "FWWeekNumber",
            "FWWeekLabel",
            "FWYearWeekNumber",
            "FWYearWeekOffset",
            "FWYearWeekLabel",
            "FWPeriodNumber",
            "FWPeriodLabel",
            "FWStartOfYear",
            "FWEndOfYear",
            "FWStartOfQuarter",
            "FWEndOfQuarter",
            "FWStartOfMonth",
            "FWEndOfMonth",
            "FWStartOfWeek",
            "FWEndOfWeek",
            "WeekDayNumber",
            "WeekDayNameShort",
            "FWDayOfYearNumber",
            "FWDayOfQuarterNumber",
            "FWDayOfMonthNumber",
            "IsWorkingDay",
            "DayType",
            "FWWeekInQuarterNumber",
            "FWYearMonthLabel",
            "FWYearQuarterLabel",
        ]

    # Dedupe preserve order
    seen = set()
    out = []
    for c in cols:
        if c not in seen:
            seen.add(c)
            out.append(c)
    return out

DATE_COLUMN_GROUPS = {
    # Base/core columns are always present in Dates output.
    "base": frozenset(_DATES_BASE_INTERNAL),
    "calendar_core": frozenset(_DATES_CAL_CORE_INTERNAL),
    "calendar_extras": frozenset(_DATES_CAL_EXTRAS_INTERNAL),

    # Optional systems
    "iso": frozenset(_DATES_ISO_INTERNAL),
    "fiscal": frozenset(_DATES_FISCAL_INTERNAL),
    "weekly_fiscal": frozenset(_WF_INTERNAL_COLS),

    # Back-compat convenience groups (common usage)
    "calendar": frozenset(_DATES_BASE_INTERNAL + _DATES_CAL_CORE_INTERNAL + _DATES_CAL_EXTRAS_INTERNAL),
}

def get_sales_schema(skip_order_cols: bool) -> List[SchemaCol]:
    """Return the Sales schema with or without order number columns."""
    return list(_SALES_SCHEMA_NO_ORDER if skip_order_cols else _SALES_SCHEMA)


def get_sales_order_header_schema() -> List[SchemaCol]:
    """Return the SalesOrderHeader schema."""
    return list(_SALES_ORDER_HEADER_SCHEMA)


def get_sales_order_detail_schema() -> List[SchemaCol]:
    """Return the SalesOrderDetail schema."""
    return list(_SALES_ORDER_DETAIL_SCHEMA)



def get_dates_schema(dates_cfg: Mapping) -> list[SchemaCol]:
    """
    Return the Dates schema for SQL CREATE TABLE / BULK INSERT.

    The Dates generator emits SQL-friendly internal column names (no spaces).
    This function mirrors dates.py resolve_date_columns() semantics:

      - Base + calendar core columns are always included.
      - include.calendar controls only calendar extras (IsToday/offsets).
      - include.iso / include.fiscal / include.weekly_fiscal add their respective systems.
      - If dates.include is missing, assume legacy "everything on".
      - If dates.include exists, default to calendar-only unless explicitly enabled.
    """
    dates_cfg = dates_cfg or {}
    internal_cols = _dates_internal_columns(dates_cfg)

    types = dict(STATIC_SCHEMAS["Dates"])  # internal, superset
    out: list[SchemaCol] = []
    for col in internal_cols:
        dtype = types.get(col)
        if dtype is None:
            raise KeyError(f"STATIC_SCHEMAS['Dates'] missing expected column: {col}")
        out.append((col, dtype))
    return out



__all__ = [
    "STATIC_SCHEMAS",
    "DATE_COLUMN_GROUPS",
    "get_sales_schema",
    "get_sales_order_header_schema",
    "get_sales_order_detail_schema",
    "get_dates_schema",
]
