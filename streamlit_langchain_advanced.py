import streamlit as st
import os
import pandas as pd
from sqlalchemy import create_engine, text
from dotenv import load_dotenv
from langchain_community.agent_toolkits import create_sql_agent, SQLDatabaseToolkit

# ===== LangChain (modern & safe) =====
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

# ===== Load env =====
load_dotenv()

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY", "")
DB_URL = os.getenv("DB_URL", "")

if not GOOGLE_API_KEY or not DB_URL:
    st.error("⚠️ Missing GOOGLE_API_KEY or DB_URL")
    st.stop()

# ===== Streamlit config =====
st.set_page_config(
    page_title="PostgreSQL AI Chatbot (LangChain)",
    page_icon="🤖",
    layout="wide"
)

st.title("🤖 PostgreSQL AI Chatbot (LangChain)")
st.markdown("Safe Streamlit + LangChain + Gemini setup")

# ===== LLM =====
@st.cache_resource
def get_llm():
    return ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        google_api_key=GOOGLE_API_KEY,
        temperature=0
    )

# ===== Database =====
@st.cache_resource
def get_engine():
    return create_engine(DB_URL)

@st.cache_data(ttl=600)
def get_schema():
    engine = get_engine()

    query = text("""
        SELECT table_name, column_name, data_type
        FROM information_schema.columns
        WHERE table_schema = 'public'
        ORDER BY table_name, ordinal_position;
    """)

    tables = {}

    with engine.connect() as conn:
        for table, col, dtype in conn.execute(query):
            tables.setdefault(table, []).append(
                f'   - "{col}" ({dtype})'
            )

    # Build available tables list (VERY important for LLM accuracy)
    available_tables = ", ".join(f'"{t}"' for t in tables.keys())

    formatted_tables = []

    for table, columns in tables.items():
        formatted_tables.append(
            f"""TABLE NAME: "{table}"
COLUMNS:
{chr(10).join(columns)}
"""
        )

    formatted_schema = "\n".join(formatted_tables)

    # Anti-hallucination header
    return f"""
YOU MUST USE ONLY THE TABLES AND COLUMNS LISTED BELOW.
DO NOT INVENT TABLES OR COLUMNS.

AVAILABLE TABLES:
{available_tables}

DATABASE SCHEMA:
{formatted_schema}
"""

from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda
import re


def clean_sql(text: str) -> str:
    """
    Removes markdown, backticks, and 'sql' labels from LLM output.
    """
    text = re.sub(r"```sql|```", "", text, flags=re.IGNORECASE)
    text = text.strip()
    return text



def create_sql_chain():
    llm = get_llm()

    prompt = PromptTemplate.from_template("""
You are an elite PostgreSQL engineer.

You MUST follow these rules:

RULES:
- Use ONLY tables listed in the schema.
- NEVER invent table names.
- NEVER invent columns.
- If the table does not exist, return EXACTLY:

TABLE_NOT_FOUND

- Output ONLY raw SQL.
- No markdown.
- No explanation.

DATABASE SCHEMA:
{schema}

QUESTION:
{question}
""")

    return prompt | llm | StrOutputParser() | RunnableLambda(clean_sql)

# ===== Natural language response chain =====
def create_nl_chain():
    llm = get_llm()

    prompt = PromptTemplate.from_template("""
User Question:
{question}

SQL Query:
{sql}

Query Results:
{data}

Answer clearly and concisely.
If no results, say so.
""")

    return prompt | llm | StrOutputParser()

# ===== Execute SQL =====
def run_query(sql):
    engine = get_engine()
    try:
        with engine.connect() as conn:
            df = pd.read_sql(text(sql), conn)
        return df, None
    except Exception as e:
        return None, str(e)

# ===== Session state =====
if "messages" not in st.session_state:
    st.session_state.messages = []

# ===== Chat history =====
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if "sql" in msg:
            with st.expander("🧾 SQL"):
                st.code(msg["sql"], language="sql")
        if "df" in msg and msg["df"] is not None:
            with st.expander("📊 Data"):
                st.dataframe(msg["df"], use_container_width=True)

# ===== Chat input =====
if question := st.chat_input("Ask about your database..."):
    st.session_state.messages.append({"role": "user", "content": question})

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            schema = get_schema()  # ✅ Streamlit-safe (main thread)

            sql_chain = create_sql_chain()
            sql = sql_chain.invoke({
                "schema": schema,
                "question": question
            }).strip()

            st.code(sql, language="sql")

            df, error = run_query(sql)

            if error:
                st.error(error)
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": f"❌ SQL Error:\n{error}",
                    "sql": sql
                })
            else:
                if not df.empty:
                    st.dataframe(df, use_container_width=True)

                nl_chain = create_nl_chain()
                answer = nl_chain.invoke({
                    "question": question,
                    "sql": sql,
                    "data": df.head(10).to_csv(index=False) if not df.empty else "No results"
                })

                st.markdown(answer)

                st.session_state.messages.append({
                    "role": "assistant",
                    "content": answer,
                    "sql": sql,
                    "df": df
                })

# ===== Clear =====
if st.button("🗑️ Clear Chat"):
    st.session_state.messages = []
    st.rerun()
