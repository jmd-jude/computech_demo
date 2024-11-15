import pandas as pd
import sqlite3
import os

def create_database():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(current_dir)))
    project_root = os.path.join(project_root, 'computech')
    
    data_path = os.path.join(project_root, 'data', 'marketing_data.csv')
    db_path = os.path.join(project_root, 'marketing_metrics.db')
    
    print(f"Current directory: {current_dir}")
    print(f"Project root: {project_root}")
    print(f"Data path: {data_path}")
    print(f"DB path: {db_path}")
    
    df = pd.read_csv(data_path)
    df.columns = [col.replace(' ', '_').replace('-', '_') for col in df.columns]
    conn = sqlite3.connect(db_path)
    df.to_sql('marketing_data', conn, if_exists='replace', index=False)
    return conn

if __name__ == "__main__":
    create_database()