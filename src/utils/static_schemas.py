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
        ("GeographyKey", "INT NOT NULL"),
        ("GeographyType","VARCHAR(50)"),
        ("City",         "VARCHAR(100)"),
        ("State",        "VARCHAR(100)"),
        ("Country",      "VARCHAR(100)"),
        ("Continent",    "VARCHAR(100)"),
        ("ISOCode",      "VARCHAR(10)")
    ],

    "Products": [
        ("ProductKey",            "INT NOT NULL"),
        ("ProductName",           "VARCHAR(200)"),
        ("ProductSubcategoryKey", "INT"),
        ("UnitCost",              "DECIMAL(10,2)"),
        ("UnitPrice",             "DECIMAL(10,2)"),
        ("Status",                "VARCHAR(20)")
    ],

    "Product_Category": [
        ("ProductCategoryKey", "INT"),
        ("ProductCategoryName","VARCHAR(100)"),
        ("CategoryLabel",   "VARCHAR(10)")
    ],

    "Product_Subcategory": [
        ("ProductSubcategoryKey", "INT"),
        ("SubcategoryLabel","VARCHAR(10)"),
        ("Subcategory","VARCHAR(100)"),
        ("CategoryKey",    "INT")
    ],

    "Promotions": [
        ("PromotionKey",         "INT NOT NULL"),
        ("PromotionName",        "VARCHAR(200)"),
        ("DiscountPct",          "DECIMAL(6,2)"),
        ("StartDate",            "DATE"),
        ("EndDate",              "DATE")
    ],

    "Stores": [
        ("StoreKey",         "INT NOT NULL"),
        ("StoreName",        "VARCHAR(100)"),
        ("StoreType",        "VARCHAR(20)"),
        ("Status",           "VARCHAR(10)"),
        ("GeographyKey",     "INT"),
        ("OpenDate",         "DATE"),
        ("CloseDate",        "DATE"),
        ("OpenFlag",         "BIT"),
        ("SquarFootage",     "INT"),
        ("EmployeeCount",    "INT"),
        ("StoreManager",     "VARCHAR(20)"),
        ("Phone",            "VARCHAR(20)"),
        ("StoreDescription", "VARCHAR(MAX)"),
        ("CloseReason",      "VARCHAR(MAX)"),
    ],

    "Dates": [
        ("Date",                         "DATE NOT NULL"),
        ("Date Key",                     "INT NOT NULL"),
        ("Year",                         "INT"),
        ("Is Year Start",                "INT"),
        ("Is Year End",                  "INT"),
        ("Year Month Number",            "INT"),
        ("Year Quarter Number",          "INT"),
        ("Quarter",                      "INT"),
        ("Quarter Year",                 "VARCHAR(10)"),
        ("Quarter Start Date",           "DATE"),
        ("Quarter End Date",             "DATE"),
        ("Is Quarter Start",             "INT"),
        ("Is Quarter End",               "INT"),
        ("Month",                        "INT"),
        ("Month Name",                   "VARCHAR(10)"),
        ("Month Short",                  "VARCHAR(10)"),
        ("Month Start Date",             "DATE"),
        ("Month End Date",               "DATE"),
        ("Month Year",                   "VARCHAR(20)"),
        ("Month Year Number",            "INT"),
        ("Is Month Start",               "INT"),
        ("Is Month End",                 "INT"),
        ("Week Of Year ISO",             "INT"),
        ("ISO Year",                     "INT"),
        ("Week Of Month",                "INT"),
        ("Week Start Date",              "DATE"),
        ("Week End Date",                "DATE"),
        ("Day",                          "INT"),
        ("Day Name",                     "VARCHAR(10)"),
        ("Day Short",                    "VARCHAR(10)"),
        ("Day Of Year",                  "INT"),
        ("Day Of Week",                  "INT"),
        ("Is Weekend",                   "INT"),
        ("Is Business Day",              "INT"),
        ("Next Business Day",            "DATE"),
        ("Previous Business Day",        "DATE"),
        ("Fiscal Year Start Year",       "INT"),
        ("Fiscal Month Number",          "INT"),
        ("Fiscal Quarter Number",        "INT"),
        ("Fiscal Quarter Name",          "VARCHAR(20)"),
        ("Fiscal Year Bin",              "VARCHAR(20)"),
        ("Fiscal Year Month Number",     "INT"),
        ("Fiscal Year Quarter Number",   "INT"),
        ("Fiscal Year Start Date",       "DATE"),
        ("Fiscal Year End Date",         "DATE"),
        ("Fiscal Quarter Start Date",    "DATE"),
        ("Fiscal Quarter End Date",      "DATE"),
        ("Is Fiscal Year Start",         "INT"),
        ("Is Fiscal Year End",           "INT"),
        ("Is Fiscal Quarter Start",      "INT"),
        ("Is Fiscal Quarter End",        "INT"),
        ("Fiscal Year",                  "INT"),
        ("Fiscal Year Label",            "VARCHAR(10)"),
        ("Is Today",                     "INT"),
        ("Is Current Year",              "INT"),
        ("Is Current Month",             "INT"),
        ("Is Current Quarter",           "INT"),
        ("Current Day Offset",           "INT")
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
        ("SalesOrderNumber",     "VARCHAR(30)"),
        ("SalesOrderLineNumber", "INT"),
        ("OrderDate",            "DATE"),
        ("DueDate",              "DATE"),
        ("DeliveryDate",         "DATE"),
        ("StoreKey",             "INT"),
        ("ProductKey",           "INT"),
        ("PromotionKey",         "INT"),
        ("CurrencyKey",          "INT"),
        ("CustomerKey",          "INT"),
        ("Quantity",             "INT"),
        ("NetPrice",             "DECIMAL(10, 4)"),
        ("UnitCost",             "DECIMAL(10, 4)"),
        ("UnitPrice",            "DECIMAL(10, 4)"),
        ("DiscountAmount",       "DECIMAL(10, 4)"),
        ("DeliveryStatus",       "VARCHAR(20)"),
        ("IsOrderDelayed",       "INT")
    ],

    "Exchange_Rates": [
        ("Date",         "DATE"),
        ("FromCurrency", "VARCHAR(10)"),
        ("ToCurrency",   "VARCHAR(10)"),
        ("ExchangeRate", "FLOAT")
    ]
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
