import pandas as pd

class StandardMetricQueries:
    NEW_CUSTOMER_GROWTH = """
    SELECT 
        ProductCategory,
        SUM(CASE WHEN RollingYearEndDate = '10/31/24' THEN Acquired END) AS Current_Acquired,
        SUM(CASE WHEN RollingYearEndDate = '10/31/23' THEN Acquired END) AS Prior_Acquired,
        ROUND((CAST(SUM(CASE WHEN RollingYearEndDate = '10/31/24' THEN Acquired END) AS FLOAT) / 
              NULLIF(SUM(CASE WHEN RollingYearEndDate = '10/31/23' THEN Acquired END), 0) - 1) * 100, 1) as YoY_Growth
    FROM marketing_data 
    WHERE CustomerSegment = 'Current Year Acquisition'
    GROUP BY ProductCategory
    """

    CUSTOMER_RETENTION = """
    SELECT 
        ProductCategory,
        ROUND(SUM(Retained_from_13_24_mo + Retained_from_25_36_mo + Retained_from_37_mo) * 100.0 / 
        NULLIF(SUM(CustomerCount), 0), 1) as Retention_Rate
    FROM marketing_data
    WHERE CustomerSegment = '0 - 12 Month' 
    AND RollingYearEndDate = '10/31/24'
    GROUP BY ProductCategory
    """

    REVENUE_PER_CUSTOMER = """
    SELECT 
        ProductCategory,
        ROUND(SUM(TotalDemand)/NULLIF(SUM(CustomerCount), 0), 2) as ARPC
    FROM marketing_data
    WHERE CustomerSegment IN ('0 - 12 Month', '13+ Month')
    AND RollingYearEndDate = '10/31/24'
    GROUP BY ProductCategory
    """

    REVENUE_GROWTH = """
    SELECT 
        ProductCategory,
        ROUND((SUM(CASE WHEN RollingYearEndDate = '10/31/24' THEN TotalDemand END)/
        NULLIF(SUM(CASE WHEN RollingYearEndDate = '10/31/23' THEN TotalDemand END), 0) - 1) * 100, 1) as Revenue_Growth
    FROM marketing_data
    GROUP BY ProductCategory
    """

    PURCHASE_FREQUENCY = """
    SELECT 
        ProductCategory,
        ROUND(SUM(OrderCount)*1.0/NULLIF(SUM(CustomerCount), 0), 2) as Orders_Per_Customer
    FROM marketing_data
    WHERE CustomerSegment IN ('0 - 12 Month', '13+ Month')
    AND RollingYearEndDate = '10/31/24'
    GROUP BY ProductCategory
    """

    HIGH_VALUE_CUSTOMER_PCT = """
    SELECT 
        ProductCategory,
        ROUND(SUM(AOV_251_PLUS)*100.0/NULLIF(SUM(CustomerCount), 0), 1) as High_Value_Customer_Pct
    FROM marketing_data
    WHERE CustomerSegment IN ('0 - 12 Month', '13+ Month')
    AND RollingYearEndDate = '10/31/24'
    GROUP BY ProductCategory
    """

    REVENUE_MIX_BY_TENURE = """
    SELECT 
    ProductCategory,
    CustomerSegment,
    ROUND(SUM(TotalDemand), 0) as Revenue,
    ROUND(SUM(TotalDemand) * 100.0 / SUM(SUM(TotalDemand)) OVER (PARTITION BY ProductCategory), 1) as Revenue_Pct
    FROM marketing_data
    WHERE RollingYearEndDate = '10/31/24'
    AND CustomerSegment IN ('0 - 12 Month', '13+ Month', '37+ Month')
    GROUP BY ProductCategory, CustomerSegment
    ORDER BY ProductCategory, Revenue DESC
    """

    BUYER_LOYALTY_PROGRESSION = """
    SELECT 
    ProductCategory,
    SUM(CASE WHEN CustomerSegment = '0 - 12 Month' THEN "2xBuyerCount" END) as Multi_Buyers_Year1,
    SUM(CASE WHEN CustomerSegment = '0 - 12 Month' THEN CustomerCount END) as Total_Buyers_Year1,
    ROUND(SUM(CASE WHEN CustomerSegment = '0 - 12 Month' THEN "2xBuyerCount" END) * 100.0 / 
        NULLIF(SUM(CASE WHEN CustomerSegment = '0 - 12 Month' THEN CustomerCount END), 0), 1) as Multi_Buyer_Rate_Pct
    FROM marketing_data
    WHERE RollingYearEndDate = '10/31/24'
    GROUP BY ProductCategory
    """

    ORDER_VALUE_DISTRIBUTION = """
    SELECT 
    ProductCategory,
    ROUND(SUM(AOV_0_50) * 100.0 / NULLIF(SUM(CustomerCount), 0), 1) as Pct_Under_50,
    ROUND(SUM(AOV_51_150) * 100.0 / NULLIF(SUM(CustomerCount), 0), 1) as Pct_51_to_150,
    ROUND(SUM(AOV_151_250) * 100.0 / NULLIF(SUM(CustomerCount), 0), 1) as Pct_151_to_250,
    ROUND(SUM(AOV_251_PLUS) * 100.0 / NULLIF(SUM(CustomerCount), 0), 1) as Pct_Over_250
    FROM marketing_data
    WHERE CustomerSegment IN ('0 - 12 Month', '13+ Month')
    AND RollingYearEndDate = '10/31/24'
    GROUP BY ProductCategory
    """

    RETENTION_RATE_TREND = """
    SELECT 
        ProductCategory,
        RollingYearEndDate,
        ROUND(SUM(Retained_from_13_24_mo + Retained_from_25_36_mo + Retained_from_37_mo) * 100.0 / 
        NULLIF(SUM(CustomerCount), 0), 1) as Retention_Rate
    FROM marketing_data
    WHERE CustomerSegment = '0 - 12 Month'
    GROUP BY ProductCategory, RollingYearEndDate
    ORDER BY ProductCategory, RollingYearEndDate
    """

    ORDER_VALUE_DISTRIBUTION_TREND = """
    SELECT 
        ProductCategory,
        RollingYearEndDate,
        ROUND(SUM(AOV_0_50) * 100.0 / NULLIF(SUM(CustomerCount), 0), 1) as Pct_Under_50,
        ROUND(SUM(AOV_51_150) * 100.0 / NULLIF(SUM(CustomerCount), 0), 1) as Pct_51_to_150,
        ROUND(SUM(AOV_151_250) * 100.0 / NULLIF(SUM(CustomerCount), 0), 1) as Pct_151_to_250,
        ROUND(SUM(AOV_251_PLUS) * 100.0 / NULLIF(SUM(CustomerCount), 0), 1) as Pct_Over_250
    FROM marketing_data
    WHERE CustomerSegment IN ('0 - 12 Month', '13+ Month')
    GROUP BY ProductCategory, RollingYearEndDate
    ORDER BY ProductCategory, RollingYearEndDate
    """

    REVENUE_PER_CUSTOMER_TREND = """
    SELECT 
    ProductCategory,
    RollingYearEndDate,
    ROUND(SUM(TotalDemand)/NULLIF(SUM(CustomerCount), 0), 2) as ARPC
    FROM marketing_data
    WHERE CustomerSegment IN ('0 - 12 Month', '13+ Month')
    GROUP BY ProductCategory, RollingYearEndDate
    ORDER BY ProductCategory, RollingYearEndDate
    """

    PURCHASE_FREQUENCY_TREND = """
    SELECT 
    ProductCategory,
    RollingYearEndDate,
    ROUND(SUM(OrderCount)*1.0/NULLIF(SUM(CustomerCount), 0), 2) as Orders_Per_Customer
    FROM marketing_data
    WHERE CustomerSegment IN ('0 - 12 Month', '13+ Month')
    GROUP BY ProductCategory, RollingYearEndDate
    ORDER BY ProductCategory, RollingYearEndDate
    """

    BUYER_LOYALTY_TREND = """
    SELECT 
    ProductCategory,
    RollingYearEndDate,
    SUM(CASE WHEN CustomerSegment = '0 - 12 Month' THEN "2xBuyerCount" END) as Multi_Buyers_Year1,
    SUM(CASE WHEN CustomerSegment = '0 - 12 Month' THEN CustomerCount END) as Total_Buyers_Year1,
    ROUND(SUM(CASE WHEN CustomerSegment = '0 - 12 Month' THEN "2xBuyerCount" END) * 100.0 / 
        NULLIF(SUM(CASE WHEN CustomerSegment = '0 - 12 Month' THEN CustomerCount END), 0), 1) as Multi_Buyer_Rate_Pct
    FROM marketing_data
    GROUP BY ProductCategory, RollingYearEndDate
    ORDER BY ProductCategory, RollingYearEndDate
    """

    REVENUE_MIX_TREND = """
    SELECT 
    ProductCategory,
    RollingYearEndDate,
    CustomerSegment,
    ROUND(SUM(TotalDemand), 0) as Revenue,
    ROUND(SUM(TotalDemand) * 100.0 / SUM(SUM(TotalDemand)) OVER (PARTITION BY ProductCategory, RollingYearEndDate), 1) as Revenue_Pct
    FROM marketing_data
    WHERE CustomerSegment IN ('0 - 12 Month', '13+ Month', '37+ Month')
    GROUP BY ProductCategory, RollingYearEndDate, CustomerSegment
    ORDER BY ProductCategory, RollingYearEndDate, Revenue DESC
    """

    HIGH_VALUE_CUSTOMER_TREND = """
    SELECT 
    ProductCategory,
    RollingYearEndDate,
    ROUND(SUM(AOV_251_PLUS)*100.0/NULLIF(SUM(CustomerCount), 0), 1) as High_Value_Customer_Pct
    FROM marketing_data
    WHERE CustomerSegment IN ('0 - 12 Month', '13+ Month')
    GROUP BY ProductCategory, RollingYearEndDate
    ORDER BY ProductCategory, RollingYearEndDate
    """

    def execute_query(self, conn, query_name):
        query = getattr(self, query_name)
        return pd.read_sql_query(query, conn)

def test_queries():
    import sqlite3
    
    # Connect to database 
    conn = sqlite3.connect('marketing_metrics.db')
    
    # Initialize queries
    queries = StandardMetricQueries()
    
    # Test each metric
    for query_name in [attr for attr in dir(StandardMetricQueries) if not attr.startswith('_') and attr != 'execute_query']:
        print(f"\n{query_name}:")
        result = queries.execute_query(conn, query_name)
        print(result)
        print("-" * 80)

if __name__ == "__main__":
    test_queries()