import streamlit as st
import sqlite3
import pandas as pd
import yaml
from src.database.queries import StandardMetricQueries
from langchain_core.prompts import ChatPromptTemplate
from src.langchain_components.qa_chain import generate_dynamic_query, execute_dynamic_query
import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

# Load environment variables
load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise ValueError("OPENAI_API_KEY environment variable not set.")

# Load prompts from YAML
def load_prompts():
    with open('prompts.yaml', 'r') as file:
        return yaml.safe_load(file)['prompts']

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

def generate_analysis(data, selected_prompt):
    llm = ChatOpenAI(model="gpt-4")
    
    prompt = ChatPromptTemplate.from_template(selected_prompt)
    
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
    
    tab1, tab2 = st.tabs(["Standard Metrics", "Custom Query"])
    
    with tab1:
        # Load prompts
        prompts = load_prompts()
        
        # Create prompt selection section
        st.sidebar.header("Analysis Configuration")
        
        # Radio button for choosing between predefined or custom prompt
        prompt_type = st.sidebar.radio(
            "Choose prompt type:",
            ["Predefined Prompts", "Custom Prompt"]
        )
        
        selected_prompt = None
        
        if prompt_type == "Predefined Prompts":
            prompt_options = {f"{v['name']} - {v['description']}": v['template'] 
                            for v in prompts.values()}
            selected_name = st.sidebar.selectbox(
                "Select analysis type:",
                list(prompt_options.keys())
            )
            selected_prompt = prompt_options[selected_name]
            
            with st.sidebar.expander("View Selected Prompt"):
                st.text_area("Prompt Template", selected_prompt, height=300, disabled=True)
                
        else:
            st.sidebar.markdown("""
            ### Custom Prompt Guidelines
            Your prompt must include these placeholders:
            - `{current_metrics}`
            - `{trend_data}`
            """)
            
            selected_prompt = st.sidebar.text_area(
                "Enter your custom prompt:",
                height=400,
                placeholder="Enter your analysis prompt here...\n\nCurrent Metrics:\n{current_metrics}\n\nTrend Data:\n{trend_data}\n\nFocus on:\n- ...",
                help="Make sure to include {current_metrics} and {trend_data} placeholders"
            )
        
        # Generate analysis button
        if st.button("Generate Analysis", key="generate_analysis"):
            if selected_prompt:
                with st.spinner("Loading data and generating analysis..."):
                    data = load_data()
                    
                    # Current Metrics Section
                    st.header("Current Performance")
                    
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
                    
                    with st.expander("Detailed Metrics"):
                        for name, df in data.items():
                            if not name.endswith('_TREND'):
                                st.subheader(name.replace('_', ' ').title())
                                st.dataframe(df)
                    
                    with st.expander("Trend Analysis"):
                        for name, df in data.items():
                            if name.endswith('_TREND'):
                                st.subheader(name.replace('_', ' ').title())
                                for col in df.select_dtypes(include=['float64', 'float32']).columns:
                                    if 'Pct' in col or 'Rate' in col or 'Growth' in col:
                                        df[col] = df[col].map(lambda x: f"{x:.1f}%")
                                    elif 'Revenue' in col or 'Demand' in col:
                                        df[col] = df[col].map(lambda x: f"${x:,.2f}")
                                    else:
                                        df[col] = df[col].map(lambda x: f"{x:,.2f}")
                                st.dataframe(df, use_container_width=True)
                    
                    st.header("Executive Analysis")
                    analysis = generate_analysis(data, selected_prompt)
                    st.markdown(analysis)
            else:
                st.error("Please select a prompt or enter a custom prompt before generating analysis.")
    
    with tab2:
        st.header("Ask Questions About Your Data")
        
        with st.form(key="query_form"):
            col1, col2 = st.columns([6, 1])
            with col1:
                user_question = st.text_input("Enter your question about the data:")
            with col2:
                st.write("")  # Spacer
                clear = st.form_submit_button("Clear")
            submit = st.form_submit_button("Submit")

            if clear:
                user_question = ""
            
        if submit and user_question:
            with st.spinner("Generating and executing query..."):
                try:
                    sql_query = generate_dynamic_query(user_question)
                    with st.expander("View Generated SQL", expanded=False):
                        st.code(sql_query, language="sql")
                    results = execute_dynamic_query(sql_query)
                    if isinstance(results, pd.DataFrame):
                        st.dataframe(results)
                    else:
                        st.error(results)
                except Exception as e:
                    st.error(f"Error: {str(e)}")

if __name__ == "__main__":
    main()