# src/utils/static_schemas.py

from __future__ import annotations

from types import MappingProxyType
from typing import Dict, List, Mapping, Sequence, Tuple

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
        ("Date Key", INT_NN),
        ("Sequential Day Index", INT_NN),

        ("Year", INT_NN),
        ("Is Year Start", INT_NN),
        ("Is Year End", INT_NN),

        ("Quarter", INT_NN),
        ("Quarter Start Date", DATE_NN),
        ("Quarter End Date", DATE_NN),
        ("Is Quarter Start", INT_NN),
        ("Is Quarter End", INT_NN),
        ("Quarter Year", VARCHAR(10, not_null=True)),

        ("Month", INT_NN),
        ("Month Name", VARCHAR(10, not_null=True)),
        ("Month Short", VARCHAR(10, not_null=True)),
        ("Month Name Short", VARCHAR(10, not_null=True)),
        ("Month Start Date", DATE_NN),
        ("Month End Date", DATE_NN),
        ("Month Year", VARCHAR(20, not_null=True)),
        ("Month Year Number", INT_NN),
        ("Year Month Key", INT_NN),
        ("Year Month Label", VARCHAR(20, not_null=True)),
        ("Year Quarter Key", INT_NN),
        ("Year Quarter Label", VARCHAR(10, not_null=True)),
        ("Calendar Month Index", INT_NN),
        ("Calendar Quarter Index", INT_NN),
        ("Is Month Start", INT_NN),
        ("Is Month End", INT_NN),

        ("Week Of Month", INT_NN),

        ("Day", INT_NN),
        ("Day Name", VARCHAR(10, not_null=True)),
        ("Day Short", VARCHAR(10, not_null=True)),
        ("Day Name Short", VARCHAR(10, not_null=True)),
        ("Day Of Year", INT_NN),
        ("Day Of Week", INT_NN),
        ("Is Weekend", INT_NN),
        ("Is Business Day", INT_NN),
        ("Next Business Day", DATE_NN),
        ("Previous Business Day", DATE_NN),

        ("Is Today", BIT(not_null=True)),
        ("Is Current Year", BIT(not_null=True)),
        ("Is Current Month", BIT(not_null=True)),
        ("Is Current Quarter", BIT(not_null=True)),
        ("Current Day Offset", INT_NN),
        ("Year Offset", SMALLINT(not_null=True)),
        ("Calendar Month Offset", INT_NN),
        ("Calendar Quarter Offset", INT_NN),

        ("Week Of Year ISO", INT_NN),
        ("ISO Year", INT_NN),
        ("ISO Year Week Index", INT_NN),
        ("ISO Week Offset", INT_NN),
        ("Week Start Date", DATE_NN),
        ("Week End Date", DATE_NN),

        # Month-based fiscal calendar (kept as "Fiscal ..." to distinguish from Weekly Fiscal)
        ("Fiscal Year Start Year", INT_NN),
        ("Fiscal Month Number", INT_NN),
        ("Fiscal Quarter Number", INT_NN),
        ("Fiscal Month Index", INT_NN),
        ("Fiscal Quarter Index", INT_NN),
        ("Fiscal Month Offset", INT_NN),
        ("Fiscal Quarter Offset", INT_NN),
        ("Fiscal Quarter Name", VARCHAR(20, not_null=True)),
        ("Fiscal Year Bin", VARCHAR(20, not_null=True)),
        ("Fiscal Year Month Number", INT_NN),
        ("Fiscal Year Quarter Number", INT_NN),
        ("Fiscal Year Start Date", DATE_NN),
        ("Fiscal Year End Date", DATE_NN),
        ("Fiscal Quarter Start Date", DATE_NN),
        ("Fiscal Quarter End Date", DATE_NN),
        ("Is Fiscal Year Start", BIT(not_null=True)),
        ("Is Fiscal Year End", BIT(not_null=True)),
        ("Is Fiscal Quarter Start", BIT(not_null=True)),
        ("Is Fiscal Quarter End", BIT(not_null=True)),
        ("Fiscal Year", INT_NN),
        ("Fiscal Year Label", VARCHAR(10, not_null=True)),
        ("Fiscal System", VARCHAR(20, not_null=True)),
        ("Weekly Fiscal System", VARCHAR(40, not_null=True)),

        # Weekly fiscal calendar (DAX weekly logic), disambiguated with "Weekly Fiscal ..." prefix.
        ("Weekly Fiscal Year Number", INT_NN),
        ("Weekly Fiscal Year Label", VARCHAR(10, not_null=True)),
        ("Weekly Fiscal Quarter Number", INT_NN),
        ("Weekly Fiscal Quarter Label", VARCHAR(20, not_null=True)),
        ("Weekly Fiscal Year Quarter Index", INT_NN),
        ("Weekly Fiscal Year Quarter Offset", INT_NN),
        ("Weekly Fiscal Month Number", INT_NN),
        ("Weekly Fiscal Month Label", VARCHAR(20, not_null=True)),
        ("Weekly Fiscal Year Month Index", INT_NN),
        ("Weekly Fiscal Year Month Offset", INT_NN),
        ("Weekly Fiscal Week Number", INT_NN),
        ("Weekly Fiscal Week Label", VARCHAR(20, not_null=True)),
        ("Weekly Fiscal Year Week Index", INT_NN),
        ("Weekly Fiscal Year Week Offset", INT_NN),
        ("Weekly Fiscal Year Week Label", VARCHAR(20, not_null=True)),
        ("Weekly Fiscal Period Number", INT_NN),
        ("Weekly Fiscal Period Label", VARCHAR(20, not_null=True)),
        ("Weekly Fiscal Start Of Year", DATE_NN),
        ("Weekly Fiscal End Of Year", DATE_NN),
        ("Weekly Fiscal Start Of Quarter", DATE_NN),
        ("Weekly Fiscal End Of Quarter", DATE_NN),
        ("Weekly Fiscal Start Of Month", DATE_NN),
        ("Weekly Fiscal End Of Month", DATE_NN),
        ("Weekly Fiscal Start Of Week", DATE_NN),
        ("Weekly Fiscal End Of Week", DATE_NN),
        ("Weekly Fiscal Week Day Number", INT_NN),
        ("Weekly Fiscal Week Day Name Short", VARCHAR(10, not_null=True)),
        ("Weekly Fiscal Day Of Year Number", INT_NN),
        ("Weekly Fiscal Day Of Quarter Number", INT_NN),
        ("Weekly Fiscal Day Of Month Number", INT_NN),
        ("Weekly Fiscal Is Working Day", BIT(not_null=True)),
        ("Weekly Fiscal Day Type", VARCHAR(20, not_null=True)),
        ("Weekly Fiscal Week In Quarter Number", INT_NN),
        ("Weekly Fiscal Year Month Label", VARCHAR(20, not_null=True)),
        ("Weekly Fiscal Year Quarter Label", VARCHAR(20, not_null=True)),
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
DATE_COLUMN_GROUPS = {
    "calendar": frozenset(
        {
            "Date",
            "Date Key",
            "Sequential Day Index",
            "Year",
            "Is Year Start",
            "Is Year End",
            "Quarter",
            "Quarter Start Date",
            "Quarter End Date",
            "Is Quarter Start",
            "Is Quarter End",
            "Quarter Year",
            "Month",
            "Month Name",
            "Month Short",
            "Month Name Short",
            "Month Start Date",
            "Month End Date",
            "Month Year",
            "Month Year Number",
            "Year Month Key",
            "Year Month Label",
            "Year Quarter Key",
            "Year Quarter Label",
            "Calendar Month Index",
            "Calendar Quarter Index",
            "Is Month Start",
            "Is Month End",
            "Week Of Month",
            "Day",
            "Day Name",
            "Day Short",
            "Day Name Short",
            "Day Of Year",
            "Day Of Week",
            "Is Weekend",
            "Is Business Day",
            "Next Business Day",
            "Previous Business Day",
            "Is Today",
            "Is Current Year",
            "Is Current Month",
            "Is Current Quarter",
            "Current Day Offset",
            "Year Offset",
            "Calendar Month Offset",
            "Calendar Quarter Offset",
        }
    ),
    "iso": frozenset(
        {
            "Week Of Year ISO",
            "ISO Year",
            "ISO Year Week Index",
            "ISO Week Offset",
            "Week Start Date",
            "Week End Date",
        }
    ),
    "fiscal": frozenset(
        {
            "Fiscal Year Start Year",
            "Fiscal Month Number",
            "Fiscal Quarter Number",
            "Fiscal Month Index",
            "Fiscal Quarter Index",
            "Fiscal Month Offset",
            "Fiscal Quarter Offset",
            "Fiscal Quarter Name",
            "Fiscal Year Bin",
            "Fiscal Year Month Number",
            "Fiscal Year Quarter Number",
            "Fiscal Year Start Date",
            "Fiscal Year End Date",
            "Fiscal Quarter Start Date",
            "Fiscal Quarter End Date",
            "Is Fiscal Year Start",
            "Is Fiscal Year End",
            "Is Fiscal Quarter Start",
            "Is Fiscal Quarter End",
            "Fiscal Year",
            "Fiscal Year Label",
            "Fiscal System",
            "Weekly Fiscal System",
        }
    ),
    "weekly_fiscal": frozenset(
        {
            "Weekly Fiscal Year Number",
            "Weekly Fiscal Year Label",
            "Weekly Fiscal Quarter Number",
            "Weekly Fiscal Quarter Label",
            "Weekly Fiscal Year Quarter Index",
            "Weekly Fiscal Year Quarter Offset",
            "Weekly Fiscal Month Number",
            "Weekly Fiscal Month Label",
            "Weekly Fiscal Year Month Index",
            "Weekly Fiscal Year Month Offset",
            "Weekly Fiscal Week Number",
            "Weekly Fiscal Week Label",
            "Weekly Fiscal Year Week Index",
            "Weekly Fiscal Year Week Offset",
            "Weekly Fiscal Year Week Label",
            "Weekly Fiscal Period Number",
            "Weekly Fiscal Period Label",
            "Weekly Fiscal Start Of Year",
            "Weekly Fiscal End Of Year",
            "Weekly Fiscal Start Of Quarter",
            "Weekly Fiscal End Of Quarter",
            "Weekly Fiscal Start Of Month",
            "Weekly Fiscal End Of Month",
            "Weekly Fiscal Start Of Week",
            "Weekly Fiscal End Of Week",
            "Weekly Fiscal Week Day Number",
            "Weekly Fiscal Week Day Name Short",
            "Weekly Fiscal Day Of Year Number",
            "Weekly Fiscal Day Of Quarter Number",
            "Weekly Fiscal Day Of Month Number",
            "Weekly Fiscal Is Working Day",
            "Weekly Fiscal Day Type",
            "Weekly Fiscal Week In Quarter Number",
            "Weekly Fiscal Year Month Label",
            "Weekly Fiscal Year Quarter Label",
        }
    ),
}


def _validate_date_groups() -> None:
    date_cols = {c for c, _ in STATIC_SCHEMAS["Dates"]}
    for grp, cols in DATE_COLUMN_GROUPS.items():
        missing = sorted(set(cols) - date_cols)
        if missing:
            raise KeyError(f"DATE_COLUMN_GROUPS[{grp!r}] references missing Dates columns: {missing}")


_validate_date_groups()


# ============================================================================
# Precomputed schemas / caches
# ============================================================================
_SALES_SCHEMA_NO_ORDER: Schema = tuple(
    (col, dtype) for col, dtype in _SALES_SCHEMA if col not in ("SalesOrderNumber", "SalesOrderLineNumber")
)

# Cache for get_dates_schema keyed by (calendar, iso, fiscal)
_DATES_SCHEMA_CACHE: Dict[Tuple[bool, bool, bool, bool], List[SchemaCol]] = {}


def get_sales_schema(skip_order_cols: bool) -> List[SchemaCol]:
    """Return the Sales schema with or without order number columns."""
    return list(_SALES_SCHEMA_NO_ORDER if skip_order_cols else _SALES_SCHEMA)


def get_sales_order_header_schema() -> List[SchemaCol]:
    """Return the SalesOrderHeader schema."""
    return list(_SALES_ORDER_HEADER_SCHEMA)


def get_sales_order_detail_schema() -> List[SchemaCol]:
    """Return the SalesOrderDetail schema."""
    return list(_SALES_ORDER_DETAIL_SCHEMA)


def get_dates_schema(dates_cfg: Mapping) -> List[SchemaCol]:
    """
    Return Dates schema filtered by config include flags.

    Supports:
      dates:
        include:
          calendar: true
          iso: false
          fiscal: false
          weekly_fiscal: false

    Defaults:
      - calendar: True
      - iso: True
      - fiscal: True
      - weekly_fiscal: True
    """
    include_cfg = (dates_cfg.get("include", {}) or {}) if isinstance(dates_cfg, Mapping) else {}

    include_calendar = bool(include_cfg.get("calendar", True))
    include_iso = bool(include_cfg.get("iso", True))
    include_fiscal = bool(include_cfg.get("fiscal", True))
    include_weekly = bool(include_cfg.get("weekly_fiscal", True))

    # Defensive fallback: if user disables everything, force calendar on
    if not (include_calendar or include_iso or include_fiscal or include_weekly):
        include_calendar = True

    cache_key = (include_calendar, include_iso, include_fiscal, include_weekly)
    cached = _DATES_SCHEMA_CACHE.get(cache_key)  # type: ignore[arg-type]
    if cached is not None:
        return list(cached)

    allowed_cols = set()
    if include_calendar:
        allowed_cols.update(DATE_COLUMN_GROUPS["calendar"])
    if include_iso:
        allowed_cols.update(DATE_COLUMN_GROUPS["iso"])
    if include_fiscal:
        allowed_cols.update(DATE_COLUMN_GROUPS["fiscal"])
    if include_weekly:
        allowed_cols.update(DATE_COLUMN_GROUPS["weekly_fiscal"])

    # Preserve original order from STATIC_SCHEMAS["Dates"]
    out = [(col, dtype) for col, dtype in STATIC_SCHEMAS["Dates"] if col in allowed_cols]

    _DATES_SCHEMA_CACHE[cache_key] = out  # type: ignore[index]
    return list(out)


__all__ = [
    "STATIC_SCHEMAS",
    "DATE_COLUMN_GROUPS",
    "get_sales_schema",
    "get_sales_order_header_schema",
    "get_sales_order_detail_schema",
    "get_dates_schema",
]