STATIC_SCHEMAS = {
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
        ("CompanyName",        "VARCHAR(20)"),
        ("GeographyKey",       "INT")
    ],

    "Products": [
        ("ProductKey",            "INT NOT NULL"),
        ("ProductCode",           "VARCHAR(20)"),
        ("ProductName",           "VARCHAR(200)"),
        ("ProductDescription",    "VARCHAR(200)"),
        ("ProductSubcategoryKey", "INT"),
        ("Brand",                 "VARCHAR(20)"),
        ("Class",                 "VARCHAR(20)"),
        ("Color",                 "VARCHAR(20)"),
        ("StockTypeCode",         "CHAR(1)"),
        ("StockType",             "VARCHAR(10)"),
        ("UnitCost",              "DECIMAL(8, 2)"),
        ("UnitPrice",             "DECIMAL(8, 2)")
    ],

    "Geography": [
        ("GeographyKey", "INT NOT NULL"),
        ("City",         "VARCHAR(50)"),
        ("State",        "VARCHAR(50)"),
        ("Country",      "VARCHAR(50)"),
        ("GeographyType","VARCHAR(50)")
    ],

    "Stores": [
        ("StoreKey",         "INT NOT NULL"),
        ("StoreName",        "VARCHAR(20)"),
        ("StoreType",        "VARCHAR(20)"),
        ("Status",           "VARCHAR(10)"),
        ("GeographyKey",     "INT"),
        ("OpeningDate",      "DATETIME"),
        ("ClosingDate",      "DATETIME"),
        ("OpenFlag",         "BIT"),
        ("SquareFootage",    "INT"),
        ("EmployeeCount",    "INT"),
        ("StoreManager",     "VARCHAR(30)"),
        ("Phone",            "VARCHAR(20)"),
        ("StoreDescription", "VARCHAR(200)"),
        ("CloseReason",      "VARCHAR(20)")
    ],

    "Promotions": [
        ("PromotionKey",         "INT NOT NULL"),
        ("PromotionLabel",       "VARCHAR(10)"),
        ("PromotionName",        "VARCHAR(200)"),
        ("PromotionDescription", "VARCHAR(200)"),
        ("DiscountPct",          "DECIMAL(6, 2)"),
        ("PromotionType",        "VARCHAR(30)"),
        ("PromotionCategory",    "VARCHAR(20)"),
        ("StartDate",            "DATE"),
        ("EndDate",              "DATE")
    ],

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

    "ProductCategory": [
        ("CategoryKey",   "INT"),
        ("Category",      "VARCHAR(10)"),
        ("Category Label","VARCHAR(10)")
    ],

    "ProductSubcategory": [
        ("ProductSubcategoryKey", "INT"),
        ("Subcategory Label",     "VARCHAR(10)"),
        ("Subcategory",           "VARCHAR(50)"),
        ("CategoryKey",           "INT")
    ]

}
