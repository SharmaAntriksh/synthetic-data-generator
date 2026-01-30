/* 
    NOTE:
    This script is NOT executed by the generator.
    Apply manually AFTER CSV import completes successfully.
*/

/* DIMENSION PRIMARY KEYS */
ALTER TABLE dbo.Currency
	ADD CONSTRAINT PK_Currency 
    PRIMARY KEY (CurrencyKey);

ALTER TABLE dbo.Currency
	ADD CONSTRAINT UQ_Currency_ToCurrency
	UNIQUE (ToCurrency);

ALTER TABLE dbo.Customers
	ADD CONSTRAINT PK_Customers 
    PRIMARY KEY (CustomerKey);

ALTER TABLE dbo.Dates
	ADD CONSTRAINT PK_Dates 
    PRIMARY KEY (Date);

ALTER TABLE dbo.Geography
	ADD CONSTRAINT PK_Geography 
    PRIMARY KEY (GeographyKey);

ALTER TABLE dbo.ProductCategory
	ADD CONSTRAINT PK_ProductCategory 
    PRIMARY KEY (CategoryKey);

ALTER TABLE dbo.ProductSubcategory
	ADD CONSTRAINT PK_ProductSubcategory 
    PRIMARY KEY (SubcategoryKey);

ALTER TABLE dbo.Products
	ADD CONSTRAINT PK_Products 
    PRIMARY KEY (ProductKey);

ALTER TABLE dbo.Promotions
	ADD CONSTRAINT PK_Promotions 
    PRIMARY KEY (PromotionKey);

ALTER TABLE dbo.Stores
	ADD CONSTRAINT PK_Stores 
    PRIMARY KEY (StoreKey);

/* FACT PRIMARY KEYS */
ALTER TABLE dbo.ExchangeRates
	ADD CONSTRAINT PK_ExchangeRates
	PRIMARY KEY (
		[Date],
		FromCurrency,
		ToCurrency
	);

/* FACT FOREIGN KEYS */
ALTER TABLE dbo.ExchangeRates
	ADD CONSTRAINT FK_ExchangeRates_Dates
	FOREIGN KEY ([Date])
	REFERENCES dbo.Dates ([Date]);

ALTER TABLE dbo.ExchangeRates
	ADD CONSTRAINT FK_ExchangeRates_ToCurrency
	FOREIGN KEY (ToCurrency)
	REFERENCES dbo.Currency (ToCurrency);

ALTER TABLE dbo.ProductSubcategory
	ADD CONSTRAINT FK_ProductSubcategory_ProductCategory 
	FOREIGN KEY (CategoryKey)
	REFERENCES dbo.ProductCategory (CategoryKey);

ALTER TABLE dbo.Products
	ADD CONSTRAINT FK_Products_ProductSubcategory
	FOREIGN KEY (SubcategoryKey)
	REFERENCES dbo.ProductSubcategory (SubcategoryKey);

ALTER TABLE dbo.Sales
	ADD CONSTRAINT FK_Sales_Customers
	FOREIGN KEY (CustomerKey)
	REFERENCES dbo.Customers (CustomerKey);

ALTER TABLE dbo.Sales
	ADD CONSTRAINT FK_Sales_Products
	FOREIGN KEY (ProductKey)
	REFERENCES dbo.Products (ProductKey);

ALTER TABLE dbo.Sales
	ADD CONSTRAINT FK_Sales_Stores
	FOREIGN KEY (StoreKey)
	REFERENCES dbo.Stores (StoreKey);

ALTER TABLE dbo.Sales
	ADD CONSTRAINT FK_Sales_Promotions
	FOREIGN KEY (PromotionKey)
	REFERENCES dbo.Promotions (PromotionKey);

ALTER TABLE dbo.Sales
	ADD CONSTRAINT FK_Sales_Dates
	FOREIGN KEY (OrderDate)
	REFERENCES dbo.Dates ([Date]);

ALTER TABLE dbo.Sales
	ADD CONSTRAINT FK_Sales_Currency
	FOREIGN KEY (CurrencyKey)
	REFERENCES dbo.Currency (CurrencyKey);
