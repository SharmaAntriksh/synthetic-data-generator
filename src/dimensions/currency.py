import pandas as pd

def generate_currency_dimension(currencies):
    """
    Build DimCurrency table.
    currencies = ["USD", "EUR", "INR", "GBP"]
    """
    df = pd.DataFrame({
        "CurrencyKey": range(1, len(currencies) + 1),
        "ISOCode": currencies,
        "CurrencyName": [c for c in currencies]
    })
    return df
