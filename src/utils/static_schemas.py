STATIC_SCHEMAS = {

    # -----------------------
    # DIMENSIONS
    # -----------------------
    "Customers": [
        ("CustomerKey",        "INT NOT NULL"),
        ("CustomerName",       "VARCHAR(100)"),
        ("DOB",                "DATE"),
        ("MaritalStatus",      "CHAR(1)"),
        ("Gender",             "CHAR(1)"),
        ("EmailAddress",       "VARCHAR(100)"),
        ("YearlyIncome",       "FLOAT"),
        ("TotalChildren",      "TINYINT"),
        ("Education",          "VARCHAR(20)"),
        ("Occupation",         "VARCHAR(20)"),
        ("CustomerType",       "VARCHAR(20)"),
        ("CompanyName",        "VARCHAR(200)"),
        ("GeographyKey",       "INT")
    ],

    "Geography": [
        ("GeographyKey", "SMALLINT NOT NULL"),
        ("City",         "VARCHAR(100)"),
        ("State",        "VARCHAR(100)"),
        ("Country",      "VARCHAR(100)"),
        ("Continent",    "VARCHAR(100)"),
        ("ISOCode",      "VARCHAR(10)")
    ],

    "Products": [
        ("ProductKey",              "INT NOT NULL"),
        ("ProductCode",             "VARCHAR(200)"),
        ("ProductName",             "VARCHAR(200)"),
        ("ProductDescription",      "VARCHAR(500)"),
        ("ProductSubcategoryKey",   "TINYINT"),
        ("Brand",                   "VARCHAR(30)"),
        ("Class",                   "VARCHAR(30)"),
        ("Color",                   "VARCHAR(30)"),
        ("StockTypeCode",           "TINYINT"),
        ("StockType",               "VARCHAR(20)"),
        ("UnitCost",                "DECIMAL(10,2)"),
        ("UnitPrice",               "DECIMAL(10,2)")
    ],

    "ProductCategory": [
        ("ProductCategoryKey",      "TINYINT NOT NULL"),
        ("ProductCategoryName",     "VARCHAR(100)"),
        ("CategoryLabel",           "VARCHAR(10)")
    ],

    "ProductSubcategory": [
        ("ProductSubcategoryKey",   "TINYINT NOT NULL"),
        ("SubcategoryLabel",        "VARCHAR(10)"),
        ("Subcategory",             "VARCHAR(100)"),
        ("CategoryKey",             "INT")
    ],

    "Promotions": [
        ("PromotionKey",            "SMALLINT NOT NULL"),
        ("PromotionLabel",          "VARCHAR(20)"),
        ("PromotionName",           "VARCHAR(50)"),
        ("PromotionDescription",    "VARCHAR(100)"),
        ("DiscountPct",             "DECIMAL(6,2)"),
        ("PromotionType",           "VARCHAR(20)"),
        ("PromotionCategory",       "VARCHAR(20)"),
        ("StartDate",               "DATE"),
        ("EndDate",                 "DATE")
    ],

    "Stores": [
        ("StoreKey",         "INT NOT NULL"),
        ("StoreName",        "VARCHAR(100)"),
        ("StoreType",        "VARCHAR(20)"),
        ("Status",           "VARCHAR(10)"),
        ("GeographyKey",     "SMALLINT"),
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

        ("Year",                     "INT"),
        ("IsYearStart",              "INT"),
        ("IsYearEnd",                "INT"),

        ("Quarter",                  "INT"),
        ("QuarterStartDate",         "DATE"),
        ("QuarterEndDate",           "DATE"),
        ("IsQuarterStart",           "INT"),
        ("IsQuarterEnd",             "INT"),
        ("QuarterYear",              "VARCHAR(10)"),

        ("Month",                    "INT"),
        ("MonthName",                "VARCHAR(10)"),
        ("MonthShort",               "VARCHAR(10)"),
        ("MonthStartDate",           "DATE"),
        ("MonthEndDate",             "DATE"),
        ("MonthYear",                "VARCHAR(20)"),
        ("MonthYearNumber",          "INT"),
        ("CalendarMonthIndex",       "INT"),
        ("CalendarQuarterIndex",     "INT"),
        ("IsMonthStart",             "INT"),
        ("IsMonthEnd",               "INT"),

        ("WeekOfYearISO",            "INT"),
        ("ISOYear",                  "INT"),
        ("WeekOfMonth",              "INT"),
        ("WeekStartDate",            "DATE"),
        ("WeekEndDate",              "DATE"),

        ("Day",                      "INT"),
        ("DayName",                  "VARCHAR(10)"),
        ("DayShort",                 "VARCHAR(10)"),
        ("DayOfYear",                "INT"),
        ("DayOfWeek",                "INT"),
        ("IsWeekend",                "INT"),
        ("IsBusinessDay",            "INT"),
        ("NextBusinessDay",          "DATE"),
        ("PreviousBusinessDay",      "DATE"),

        ("FiscalYearStartYear",      "INT"),
        ("FiscalMonthNumber",        "INT"),
        ("FiscalQuarterNumber",      "INT"),
        ("FiscalQuarterName",        "VARCHAR(20)"),
        ("FiscalYearBin",            "VARCHAR(20)"),
        ("FiscalYearMonthNumber",    "INT"),
        ("FiscalYearQuarterNumber",  "INT"),
        ("FiscalMonthIndex",         "INT"),
        ("FiscalQuarterIndex",       "INT"),
        ("FiscalYearStartDate",      "DATE"),
        ("FiscalYearEndDate",        "DATE"),
        ("FiscalQuarterStartDate",   "DATE"),
        ("FiscalQuarterEndDate",     "DATE"),
        ("IsFiscalYearStart",        "BIT"),
        ("IsFiscalYearEnd",          "BIT"),
        ("IsFiscalQuarterStart",     "BIT"),
        ("IsFiscalQuarterEnd",       "BIT"),

        ("FiscalYear",               "INT"),
        ("FiscalYearLabel",          "VARCHAR(10)"),

        ("IsToday",                  "BIT"),
        ("IsCurrentYear",            "BIT"),
        ("IsCurrentMonth",           "BIT"),
        ("IsCurrentQuarter",         "BIT"),

        ("CurrentDayOffset",         "INT"),
    ],


    "Currency": [
        ("CurrencyKey", "INT"),
        ("ISOCode",     "VARCHAR(10)"),
        ("CurrencyName","VARCHAR(50)")
    ],

    # -----------------------
    # FACTS
    # -----------------------
    "Sales": [
        ("SalesOrderNumber",     "BIGINT"),
        ("SalesOrderLineNumber", "TINYINT"),

        ("CustomerKey",          "INT"),
        ("ProductKey",           "INT"),
        ("StoreKey",             "INT"),
        ("PromotionKey",         "SMALLINT"),
        ("CurrencyKey",          "TINYINT"),

        ("OrderDate",            "DATE"),
        ("DueDate",              "DATE"),
        ("DeliveryDate",         "DATE"),

        ("Quantity",             "TINYINT"),
        ("NetPrice",             "DECIMAL(10, 4)"),
        ("UnitCost",             "DECIMAL(10, 4)"),
        ("UnitPrice",            "DECIMAL(10, 4)"),
        ("DiscountAmount",       "DECIMAL(10, 4)"),

        ("DeliveryStatus",       "VARCHAR(20)"),
        ("IsOrderDelayed",       "TINYINT")
    ],

    "ExchangeRates": [
        ("Date",         "DATE"),
        ("FromCurrency", "VARCHAR(10)"),
        ("ToCurrency",   "VARCHAR(10)"),
        ("Rate",         "DECIMAL(10, 6)")
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
