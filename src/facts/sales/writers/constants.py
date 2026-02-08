# Columns we never dictionary-encode
DICT_EXCLUDE = {"SalesOrderNumber", "CustomerKey"}

# Columns that must always exist in Sales
REQUIRED_PRICING_COLS = {
    "UnitPrice",
    "NetPrice",
    "UnitCost",
    "DiscountAmount",
}
