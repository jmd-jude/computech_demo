# src/langchain_components/qa_chain.py
from langchain_community.chat_models import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
import sqlite3
import pandas as pd

def get_table_schema():
    conn = sqlite3.connect('marketing_metrics.db')
    cursor = conn.execute("SELECT sql FROM sqlite_master WHERE type='table' AND name='marketing_data';")
    schema = cursor.fetchone()[0]
    
    # Add sample data and business context
    return f"""
    {schema}
    
    Key Business Context:
    - RollingYearEndDate format is 'MM/DD/YY' and represents the end of a 12-month period
    - CustomerSegment categories: '0 - 12 Month', '13+ Month', '37+ Month', 'Current Year Acquisition'
    - A "multi-buyer" is represented in 2xBuyerCount
    - Retention is calculated from Retained_from_13_24_mo, Retained_from_25_36_mo, and Retained_from_37_mo
    - TotalDemand represents revenue in dollars
    
    Sample valid queries:
    -- Revenue by category for current period
    SELECT ProductCategory, SUM(TotalDemand) as Revenue
    FROM marketing_data 
    WHERE RollingYearEndDate = '10/31/24'
    GROUP BY ProductCategory;
    
    -- Retention rate calculation
    SELECT ProductCategory,
           ROUND(SUM(Retained_from_13_24_mo + Retained_from_25_36_mo + Retained_from_37_mo) * 100.0 / 
           NULLIF(SUM(CustomerCount), 0), 1) as Retention_Rate
    FROM marketing_data
    WHERE CustomerSegment = '0 - 12 Month' 
    GROUP BY ProductCategory;
    """

def create_sql_generation_prompt():
    table_schema = get_table_schema()
    
    template = """You are an expert SQL query generator. 
    Given the table schema and a natural language question, generate a SQL query to answer the question.

    Table Schema:
    {schema}

    Example data context:
    - ProductCategory can be 'Equine', 'Farm', 'Pet', or 'Misc'
    - CustomerSegment shows customer recency ('0 - 12 Month', '13+ Month', etc.)
    - Dates are in format 'MM/DD/YY'

    Question: {question}

    Return only the SQL query, nothing else.
    Make sure to:
    1. Use only columns that exist in the schema
    2. Include appropriate WHERE clauses
    3. Format numbers appropriately
    4. Only add LIMIT clause if specifically requested or if returning individual records rather than aggregated results

    SQL Query:"""

    return ChatPromptTemplate.from_template(template)

def generate_dynamic_query(question: str):
    llm = ChatOpenAI(model="gpt-4")
    prompt = create_sql_generation_prompt()
    
    messages = prompt.format_messages(
        schema=get_table_schema(),
        question=question
    )
    
    response = llm.invoke(messages)
    return response.content

def execute_dynamic_query(query: str):
    conn = sqlite3.connect('marketing_metrics.db')
    try:
        result = pd.read_sql_query(query, conn)
        return result
    except Exception as e:
        return f"Error executing query: {str(e)}"