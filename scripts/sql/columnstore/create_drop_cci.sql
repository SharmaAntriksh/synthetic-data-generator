/*
Optional helper script.
This file is NOT executed automatically by the generator.
Run manually before/after data load if columnstore indexes are desired.
*/

-- Execution
DECLARE @Tables dbo.TableNameList;

INSERT INTO @Tables (TableName)
VALUES
     ('Sales')
    ,('Customers')
    ,('Products')
    ,('ProductCategory')
    ,('ProductSubcategory')
    ,('Dates')
    ,('Stores')
    ,('Promotions')
    ,('Geography')
    ,('Currency')
    ,('ExchangeRates')

EXEC dbo.ManageClusteredColumnstoreIndexes
    @Action = 'CREATE',
    @Tables = @Tables;
