import streamlit as st
import sqlite3
import pandas as pd
from src.database.queries import StandardMetricQueries
from langchain_core.prompts import ChatPromptTemplate
import os
from dotenv import load_dotenv
from langchain_community.chat_models import ChatOpenAI

load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise ValueError("OPENAI_API_KEY environment variable not set.")

def format_currency(val):
   return f"${val:,.2f}"

def format_percent(val):
   return f"{val:.1f}%"

def load_data():
    db_path = os.path.join(os.path.dirname(__file__), 'marketing_metrics.db')
    conn = sqlite3.connect(db_path)
    queries = StandardMetricQueries()
    return {query_name: queries.execute_query(conn, query_name) 
            for query_name in dir(StandardMetricQueries) 
            if not query_name.startswith('_') and not query_name == 'execute_query'}

def generate_analysis(data):
   llm = ChatOpenAI(model="gpt-4")
   
   template = """You are a senior business analyst reviewing marketing metrics. Analyze the following data and provide a concise executive summary:

Current Metrics:
{current_metrics}

Trend Data:
{trend_data}

Focus on:
- Key performance highlights
- Notable trends
- Areas of concern
- Strategic recommendations

Provide your analysis in clear, business-focused language suitable for executive review."""

   prompt = ChatPromptTemplate.from_template(template)
   
   # Format data for prompt
   current_metrics = "\n".join([f"{k}:\n{v.to_string()}" for k,v in data.items() 
                              if not k.endswith('_TREND')])
   trend_data = "\n".join([f"{k}:\n{v.to_string()}" for k,v in data.items() 
                          if k.endswith('_TREND')])
   
   messages = prompt.format_messages(
       current_metrics=current_metrics,
       trend_data=trend_data
   )
   
   response = llm.invoke(messages)
   return response.content

def main():
   st.title("Computech Database Metrics Dashboard")
   
   if st.button("Generate Analysis"):
       with st.spinner("Loading data and generating analysis..."):
           data = load_data()
           
           # Current Metrics Section
           st.header("Current Performance")
           
           # Key metrics in columns
           col1, col2, col3 = st.columns(3)
           with col1:
               retention = data['CUSTOMER_RETENTION']
               st.metric("Avg Retention Rate", 
                        format_percent(retention['Retention_Rate'].mean()))
               
           with col2:
               revenue = data['REVENUE_GROWTH']
               st.metric("Revenue Growth", 
                        format_percent(revenue['Revenue_Growth'].mean()))
               
           with col3:
               arpc = data['REVENUE_PER_CUSTOMER']
               st.metric("Avg Revenue Per Customer",
                        format_currency(arpc['ARPC'].mean()))
           
           # Detailed metrics in expandable sections
           with st.expander("Detailed Metrics"):
               for name, df in data.items():
                   if not name.endswith('_TREND'):
                       st.subheader(name.replace('_', ' ').title())
                       st.dataframe(df)
           
           # Trends in expandable section
           with st.expander("Trend Analysis"):
            for name, df in data.items():
                if name.endswith('_TREND'):
                    st.subheader(name.replace('_', ' ').title())
                    
                    # Format numeric columns
                    for col in df.select_dtypes(include=['float64', 'float32']).columns:
                        if 'Pct' in col or 'Rate' in col or 'Growth' in col:
                            df[col] = df[col].map(lambda x: f"{x:.1f}%")
                        elif 'Revenue' in col or 'Demand' in col:
                            df[col] = df[col].map(lambda x: f"${x:,.2f}")
                        else:
                            df[col] = df[col].map(lambda x: f"{x:,.2f}")
                    
                    st.dataframe(df, use_container_width=True)
           
           # AI Analysis
           st.header("Executive Analysis")
           analysis = generate_analysis(data)
           st.markdown(analysis)

if __name__ == "__main__":
   main()