STATIC_SCHEMAS = {

    # -----------------------
    # DIMENSIONS
    # -----------------------
    "Customers": [
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
        ("IsActiveInSales",    "INT NOT NULL")
    ],

    "Geography": [
        ("GeographyKey", "INT NOT NULL"),
        ("City",         "VARCHAR(100) NOT NULL"),
        ("State",        "VARCHAR(100) NOT NULL"),
        ("Country",      "VARCHAR(100) NOT NULL"),
        ("Continent",    "VARCHAR(100) NOT NULL"),
        ("ISOCode",      "VARCHAR(10) NOT NULL")
    ],

    "Products": [
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
        ("IsActiveInSales",         "INT NOT NULL")
    ],

    "ProductCategory": [
        ("CategoryKey",             "INT NOT NULL"),
        ("Category",                "VARCHAR(100) NOT NULL"),
        ("CategoryLabel",           "VARCHAR(10) NOT NULL")
    ],

    "ProductSubcategory": [
        ("SubcategoryKey",          "INT NOT NULL"),
        ("SubcategoryLabel",        "VARCHAR(10) NOT NULL"),
        ("Subcategory",             "VARCHAR(100) NOT NULL"),
        ("CategoryKey",             "INT")
    ],

    "Promotions": [
        ("PromotionKey",            "INT NOT NULL"),
        ("PromotionLabel",          "VARCHAR(20) NOT NULL"),
        ("PromotionName",           "VARCHAR(50) NOT NULL"),
        ("PromotionDescription",    "VARCHAR(100) NOT NULL"),
        ("DiscountPct",             "DECIMAL(6,2)"),
        ("PromotionType",           "VARCHAR(20) NOT NULL"),
        ("PromotionCategory",       "VARCHAR(20) NOT NULL"),
        ("StartDate",               "DATE"),
        ("EndDate",                 "DATE")
    ],

    "Stores": [
        ("StoreKey",         "INT NOT NULL"),
        ("StoreName",        "VARCHAR(100) NOT NULL"),
        ("StoreType",        "VARCHAR(20) NOT NULL"),
        ("Status",           "VARCHAR(10)"),
        ("GeographyKey",     "INT NOT NULL"),
        ("OpenDate",         "DATETIME"),
        ("CloseDate",        "DATETIME"),
        ("OpenFlag",         "BIT"),
        ("SquarFootage",     "INT"),
        ("EmployeeCount",    "INT"),
        ("StoreManager",     "VARCHAR(20)"),
        ("Phone",            "VARCHAR(20)"),
        ("StoreDescription", "VARCHAR(MAX)"),
        ("CloseReason",      "VARCHAR(MAX)"),
    ],

    "Dates": [
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
    ],

    "Currency": [
        ("CurrencyKey",     "INT NOT NULL"),
        ("ToCurrency",      "VARCHAR(10) NOT NULL"),
        ("CurrencyName",    "VARCHAR(50) NOT NULL")
    ],

    # -----------------------
    # FACTS
    # -----------------------
    "Sales": [
        ("SalesOrderNumber",     "BIGINT NOT NULL"),
        ("SalesOrderLineNumber", "INT NOT NULL"),

        ("CustomerKey",          "INT NOT NULL"),
        ("ProductKey",           "INT NOT NULL"),
        ("StoreKey",             "INT NOT NULL"),
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
        ("IsOrderDelayed",       "INT NOT NULL")
    ],

    "ExchangeRates": [
        ("Date",         "DATE NOT NULL"),
        ("FromCurrency", "VARCHAR(10) NOT NULL"),
        ("ToCurrency",   "VARCHAR(10) NOT NULL"),
        ("Rate",         "DECIMAL(10, 6) NOT NULL")
    ]
}

# ---------------------------------------------------------
# DATE COLUMN GROUPS (logical, superset)
# ---------------------------------------------------------

DATE_COLUMN_GROUPS = {
    "calendar": {
        "Date","DateKey",
        "Year","IsYearStart","IsYearEnd",
        "Quarter","QuarterStartDate","QuarterEndDate",
        "IsQuarterStart","IsQuarterEnd",
        "QuarterYear",
        "Month","MonthName","MonthShort",
        "MonthStartDate","MonthEndDate",
        "MonthYear","MonthYearNumber",
        "CalendarMonthIndex","CalendarQuarterIndex",
        "IsMonthStart","IsMonthEnd",
        "WeekOfMonth",
        "Day","DayName","DayShort","DayOfYear","DayOfWeek",
        "IsWeekend","IsBusinessDay",
        "NextBusinessDay","PreviousBusinessDay",
        "IsToday","IsCurrentYear","IsCurrentMonth",
        "IsCurrentQuarter","CurrentDayOffset",
    },
    "iso": {
        "WeekOfYearISO",
        "ISOYear",
        "WeekStartDate",
        "WeekEndDate",
    },
    "fiscal": {
        "FiscalYearStartYear","FiscalMonthNumber","FiscalQuarterNumber",
        "FiscalMonthIndex","FiscalQuarterIndex",
        "FiscalQuarterName","FiscalYearBin",
        "FiscalYearMonthNumber","FiscalYearQuarterNumber",
        "FiscalYearStartDate","FiscalYearEndDate",
        "FiscalQuarterStartDate","FiscalQuarterEndDate",
        "IsFiscalYearStart","IsFiscalYearEnd",
        "IsFiscalQuarterStart","IsFiscalQuarterEnd",
        "FiscalYear","FiscalYearLabel",
    }
}


def get_sales_schema(skip_order_cols: bool):
    """Return the Sales schema with or without order number columns."""
    base_schema = STATIC_SCHEMAS["Sales"]

    if not skip_order_cols:
        return base_schema

    # remove order columns
    return [
        (col, dtype)
        for col, dtype in base_schema
        if col not in ("SalesOrderNumber", "SalesOrderLineNumber")
    ]


def get_dates_schema(dates_cfg: dict):
    """
    Return Dates schema filtered by config include flags.
    Defaults to calendar-only (backward compatible).
    """

    include_cfg = dates_cfg.get("include", {})

    include_calendar = include_cfg.get("calendar", True)
    include_iso = include_cfg.get("iso", False)
    include_fiscal = include_cfg.get("fiscal", False)

    allowed_cols = set()

    if include_calendar:
        allowed_cols |= DATE_COLUMN_GROUPS["calendar"]
    if include_iso:
        allowed_cols |= DATE_COLUMN_GROUPS["iso"]
    if include_fiscal:
        allowed_cols |= DATE_COLUMN_GROUPS["fiscal"]

    # Preserve original STATIC_SCHEMAS order
    return [
        (col, dtype)
        for col, dtype in STATIC_SCHEMAS["Dates"]
        if col in allowed_cols
    ]
