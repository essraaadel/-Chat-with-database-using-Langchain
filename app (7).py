import streamlit as st
import google.generativeai as genai
import pandas as pd
from sqlalchemy import create_engine, text

# Configuration Constants (Updated with your new credentials)
GOOGLE_API_KEY = ""
DB_URL = ""

# Setup Google Generative AI
genai.configure(api_key=GOOGLE_API_KEY)

# Initialize the model
model = genai.GenerativeModel("models/gemini-2.5-flash")

# Streamlit UI Configuration
st.set_page_config(page_title="Postgress SQL Chatbot")
st.title("Chat with DB")

# Database Connection Function
@st.cache_resource
def get_engine():
    return create_engine(DB_URL)

# Function to Retrieve Database Schema for Context in Prompts
@st.cache_data
def get_schema():

    engine=get_engine()
    inspector_query = text("""
                        SELECT table_name, column_name
                        FROM information_schema.columns 
                        WHERE table_schema = 'public'
                        ORDER BY table_name, ordinal_position;
                     """)
    
    schema_str = ""

    try:
        with engine.connect() as conn:
            result = conn.execute(inspector_query)
            current_table = ""
            for row in result:
                table_name, column_name = row[0], row[1]
                if table_name != current_table:
                    schema_str += f"\nTable: {table_name}\nColumns: "
                    current_table = table_name
                schema_str += f"{column_name}, "
    except Exception as e:
        st.error(f"ERROR reading schema: {e}")
    
    return schema_str

def get_sql_from_gemini(question, schema):
    prompt = f"""
You are an expert PostgreSQL Data Analyst.

Here is the database schema:
{schema}

Your task:
1- Write a PostgreSQL query to answer the following question: {question}
2- IMPORTANT: the tables were created via pandas.
   -If columns or tables names are MixedCase, use double quotes around them.
3- Return ONLY the SQL query, without any explanation or comments.
"""
    
    # Generate content from the model
    response = model.generate_content(prompt)
    
    # Clean the markdown formatting (backticks) from the response
    clean_sql = response.text.replace("```sql", "").replace("```", "").strip()
    
    return clean_sql

def run_query(sql):
    engine = get_engine()
    with engine.connect() as conn:
        try:
            result = conn.execute(text(sql))
            df = pd.DataFrame(result.fetchall(), columns=result.keys())
            return df
        except Exception as e:
            return str(e)
        
def get_chat_response(question, sql, data):
    prompt = f"""
User Question: {question}
Generated SQL Query: {sql}
Data retrieved from the query:
{data}

Task: Answer the user's question in a naturel language format based on the data retrieved.
If the data is empty, say "No results found".
"""
    response = model.generate_content(prompt)

    return response.text.strip()
