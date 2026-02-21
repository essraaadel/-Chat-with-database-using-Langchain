# LangChain Integration Guide for PostgreSQL Chatbot

## 🎯 What Changed: Original vs LangChain Version

### Original Code (Manual Approach)
```python
# You manually:
1. Created prompts
2. Called Gemini API
3. Parsed responses
4. Executed SQL
5. Formatted results
```

### LangChain Version (Framework Approach)
```python
# LangChain handles:
1. ✅ Agents (autonomous decision making)
2. ✅ Tools (SQL database toolkit)
3. ✅ Memory (conversation history)
4. ✅ Chains (structured pipelines)
5. ✅ Error handling & retries
```

## 🚀 Quick Start

### 1. Install LangChain Dependencies

```bash
pip install -r requirements_langchain.txt
```

Or install individually:
```bash
pip install langchain langchain-google-genai langchain-community
```

### 2. Run the LangChain Version

```bash
# Basic LangChain version with agents
streamlit run streamlit_langchain_chatbot.py

# Advanced version with multiple modes
streamlit run streamlit_langchain_advanced.py
```

## 📚 LangChain Components Explained

### 1. **LLM (Large Language Model)**
```python
from langchain_google_genai import ChatGoogleGenerativeAI

llm = ChatGoogleGenerativeAI(
    model="gemini-2.0-flash-exp",
    google_api_key=GOOGLE_API_KEY,
    temperature=0  # 0 = deterministic, 1 = creative
)
```

### 2. **SQL Database Connection**
```python
from langchain.sql_database import SQLDatabase

db = SQLDatabase.from_uri(DB_URL)

# Built-in methods:
db.get_usable_table_names()  # List tables
db.get_table_info(['users'])  # Get schema
db.run("SELECT * FROM users LIMIT 5")  # Execute query
```

### 3. **SQL Toolkit**
```python
from langchain.agents.agent_toolkits import SQLDatabaseToolkit

toolkit = SQLDatabaseToolkit(db=db, llm=llm)

# Toolkit provides these tools to the agent:
# - sql_db_list_tables: List all tables
# - sql_db_schema: Get table schemas
# - sql_db_query: Execute SQL queries
# - sql_db_query_checker: Validate SQL syntax
```

### 4. **SQL Agent**
```python
from langchain.agents import create_sql_agent

agent = create_sql_agent(
    llm=llm,
    toolkit=toolkit,
    agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True,  # Show reasoning steps
    handle_parsing_errors=True  # Auto-recover from errors
)

# Agent autonomously:
# 1. Analyzes the question
# 2. Decides which tools to use
# 3. Generates and executes SQL
# 4. Formats the answer
```

### 5. **SQL Chain (Alternative to Agent)**
```python
from langchain.chains import SQLDatabaseChain

chain = SQLDatabaseChain.from_llm(
    llm=llm,
    db=db,
    verbose=True,
    use_query_checker=True  # Validate SQL before running
)

# Chain is more deterministic:
# Question → SQL → Execute → Answer
```

### 6. **Memory**
```python
from langchain.memory import ConversationBufferMemory

memory = ConversationBufferMemory(
    memory_key="chat_history",
    return_messages=True
)

# Remembers conversation context
# Useful for follow-up questions
```

## 🆚 Agent vs Chain: Which to Use?

### Use **Agent** When:
- ✅ Question is complex or multi-step
- ✅ Need to explore database structure
- ✅ Want autonomous problem-solving
- ✅ Errors might occur (agent can retry)

**Example:**
```python
agent.invoke({"input": "Find the top customers and their total orders"})
# Agent will:
# 1. List tables
# 2. Check schema
# 3. Write complex JOIN query
# 4. Execute and explain results
```

### Use **Chain** When:
- ✅ Question is straightforward
- ✅ Need faster responses
- ✅ Want predictable behavior
- ✅ SQL query is simple

**Example:**
```python
chain.invoke({"query": "Show me users table"})
# Chain will:
# 1. Generate SQL
# 2. Execute query
# 3. Return results
```

## 💡 Three Versions Provided

### 1. **streamlit_langchain_chatbot.py**
**Best for:** Getting started with LangChain

Features:
- SQL Agent with toolkit
- Automatic query generation
- Error handling
- Intermediate steps display

### 2. **streamlit_langchain_advanced.py**
**Best for:** Production use with flexibility

Features:
- 🤖 Agent Mode (autonomous)
- ⛓️ Chain Mode (direct SQL)
- 💬 Hybrid Mode (best of both)
- Conversation memory
- Custom prompts
- Enhanced error handling

### 3. **Original (streamlit_postgres_chatbot_secure.py)**
**Best for:** Full control without framework

Features:
- Direct Gemini API calls
- Custom prompt engineering
- No LangChain dependency
- Simpler to understand

## 🎓 Key LangChain Benefits

### 1. **Built-in Tools**
```python
# Instead of manually coding:
query = "SELECT * FROM users"
result = conn.execute(query)

# LangChain provides:
result = db.run("SELECT * FROM users")
```

### 2. **Automatic Retry Logic**
```python
# Agent automatically retries on error:
Try SQL → Error → Fix SQL → Success
```

### 3. **Smart Prompting**
```python
# LangChain uses optimized prompts:
# - ReAct (Reasoning + Acting)
# - Few-shot examples
# - Chain-of-thought
```

### 4. **Modular Design**
```python
# Easy to swap components:
llm = ChatGoogleGenerativeAI(...)  # Use Gemini
llm = ChatOpenAI(...)              # Switch to GPT-4
llm = ChatAnthropic(...)           # Switch to Claude
```

## 🔧 Customization Examples

### Custom Agent Prompt
```python
custom_prefix = """
You are a PostgreSQL expert specializing in e-commerce databases.
Always check for data quality issues.
Suggest optimizations when appropriate.

Available tools:
"""

agent = create_sql_agent(
    llm=llm,
    toolkit=toolkit,
    prefix=custom_prefix
)
```

### Add Memory to Agent
```python
from langchain.memory import ConversationBufferMemory

memory = ConversationBufferMemory(memory_key="chat_history")

# Create agent with memory
agent = create_sql_agent(
    llm=llm,
    toolkit=toolkit,
    memory=memory
)

# Now it remembers context:
# User: "Show me users"
# Agent: [shows users]
# User: "Now filter them by age > 25"  ← Remembers "users"
```

### Custom SQL Chain
```python
from langchain.prompts import PromptTemplate

template = """
Based on the database schema, write a PostgreSQL query.

Schema: {table_info}
Question: {input}

Rules:
- Use double quotes for MixedCase names
- Limit to 100 rows
- Add helpful comments

SQL:
"""

prompt = PromptTemplate(
    input_variables=["table_info", "input"],
    template=template
)

chain = SQLDatabaseChain.from_llm(
    llm=llm,
    db=db,
    prompt=prompt
)
```

## 🐛 Troubleshooting

### Issue: Agent takes too long
**Solution:** Reduce max_iterations
```python
agent = create_sql_agent(
    max_iterations=5,  # Default is 15
    max_execution_time=30  # Timeout after 30s
)
```

### Issue: Agent makes mistakes
**Solution:** Improve the system prompt
```python
prefix = """
CRITICAL RULES:
1. Always use double quotes for table/column names
2. Test with LIMIT 1 first
3. Check for NULL values
...
"""
```

### Issue: Want to see what agent is doing
**Solution:** Enable verbose mode
```python
agent = create_sql_agent(
    verbose=True,  # Shows all steps
    return_intermediate_steps=True  # Returns steps in response
)
```

## 📊 Comparison Table

| Feature | Original | LangChain Agent | LangChain Chain |
|---------|----------|-----------------|-----------------|
| Complexity | Low | High | Medium |
| Flexibility | High | Medium | Low |
| Autonomy | None | High | Low |
| Speed | Fast | Slow | Fast |
| Error Handling | Manual | Automatic | Basic |
| Memory | Manual | Built-in | Built-in |
| Learning Curve | Easy | Hard | Medium |
| Best For | Simple apps | Complex tasks | Direct queries |

## 🎯 Recommended Approach

**For Learning:** Start with original → Try basic LangChain → Explore advanced

**For Production:** Use advanced version with hybrid mode

**For Simple Apps:** Stick with original (less overhead)

**For Complex AI:** Use LangChain agents (smarter behavior)

## 📚 Additional Resources

- [LangChain Docs](https://python.langchain.com/docs/get_started/introduction)
- [SQL Agent Guide](https://python.langchain.com/docs/use_cases/sql/)
- [Gemini + LangChain](https://python.langchain.com/docs/integrations/chat/google_generative_ai)

## 🚀 Next Steps

1. Try the basic LangChain version
2. Compare responses with original
3. Experiment with custom prompts
4. Add memory for context
5. Try different LLM providers
6. Build custom tools for your use case

---

**Happy coding with LangChain! 🦜⛓️**
