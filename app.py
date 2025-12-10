"""
LangChain Agent with Tools and Memory (LangChain v1 - Fixed)
A beginner-friendly agent using Tavily search, datetime, weather tools, and chat history
"""

# CORRECTED IMPORTS FOR LANGCHAIN V1
from langchain.agents import create_tool_calling_agent, AgentExecutor
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage, AIMessage
from langchain_openai import ChatOpenAI
from langchain_community.tools.tavily_search import TavilySearchResults
from datetime import datetime
import requests
import os
import sys
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# =============================================================================
# STEP 1: Define Custom Tools
# =============================================================================

@tool
def get_current_datetime() -> str:
    """Get the current date and time."""
    now = datetime.now()
    return now.strftime("%Y-%m-%d %H:%M:%S")

@tool
def get_weather(city: str) -> str:
    """
    Get current weather for a city.
    
    Args:
        city: Name of the city to get weather for
    """
    # Using wttr.in API (free, no key required)
    try:
        url = f"https://wttr.in/{city}?format=j1"
        response = requests.get(url, timeout=10)
        if response.status_code == 200:
            data = response.json()
            current = data['current_condition'][0]
            weather_desc = current['weatherDesc'][0]['value']
            temp_c = current['temp_C']
            feels_like = current['FeelsLikeC']
            humidity = current['humidity']
            
            return f"Weather in {city}: {weather_desc}, Temperature: {temp_c}°C (feels like {feels_like}°C), Humidity: {humidity}%"
        else:
            return f"Could not fetch weather for {city}"
    except Exception as e:
        return f"Error getting weather: {str(e)}"

# =============================================================================
# STEP 2: Initialize Chat History (Short-term Memory)
# =============================================================================

chat_history = []

# =============================================================================
# STEP 3: Create the Agent
# =============================================================================

def create_agent():
    """Initialize and return the agent executor."""
    
    # Initialize the LLM
    llm = ChatOpenAI(
        model="gpt-4o-mini",  # Use gpt-4o-mini or gpt-3.5-turbo
        temperature=0.7
    )
    
    # Initialize Tavily search tool (LangChain built-in)
    tavily_tool = TavilySearchResults(
        max_results=3,
        search_depth="basic",  # or "advanced" for more detailed results
        include_answer=True,
        include_raw_content=False
    )
    
    # Define all tools
    tools = [get_current_datetime, get_weather, tavily_tool]
    
    # Create prompt template with memory placeholder
    prompt = ChatPromptTemplate.from_messages([
        ("system", """You are a helpful AI assistant with access to tools for:
        - Getting current date and time
        - Checking weather for any city
        - Searching the web for information
        
        Use these tools when needed to provide accurate and helpful responses.
        Be conversational and remember the context from previous messages."""),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{input}"),
        MessagesPlaceholder(variable_name="agent_scratchpad"),
    ])
    
    # Create the agent (FIXED: correct function name for v1)
    agent = create_tool_calling_agent(llm, tools, prompt)
    
    # Create agent executor (FIXED: correct class name for v1)
    agent_executor = AgentExecutor(
        agent=agent,
        tools=tools,
        verbose=True,
        handle_parsing_errors=True
    )
    
    return agent_executor

# =============================================================================
# STEP 4: Run the Agent with Memory
# =============================================================================

def chat(user_input: str, agent_executor):
    """
    Process user input and maintain chat history.
    
    Args:
        user_input: The user's message
        agent_executor: The agent executor instance
    
    Returns:
        The agent's response
    """
    global chat_history
    
    # Convert chat history to proper message format for LangChain v1
    formatted_history = []
    for role, content in chat_history:
        if role == "human":
            formatted_history.append(HumanMessage(content=content))
        elif role == "assistant":
            formatted_history.append(AIMessage(content=content))
    
    # Run the agent with current chat history
    response = agent_executor.invoke({
        "input": user_input,
        "chat_history": formatted_history
    })
    
    # Update chat history
    chat_history.append(("human", user_input))
    chat_history.append(("assistant", response['output']))
    
    # Keep only last 10 exchanges (20 messages) to prevent context from growing too large
    if len(chat_history) > 20:
        chat_history = chat_history[-20:]
    
    return response['output']

# =============================================================================
# STEP 5: Main Execution
# =============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("LangChain Agent with Tools and Memory (v1)")
    print("=" * 60)
    print("\n📋 Loading API keys from .env file...")
    print("\nRequired API Keys:")
    print("- OPENAI_API_KEY (for LLM)")
    print("- TAVILY_API_KEY (for web search)")
    print("=" * 60)
    
    # Check for API keys
    if not os.getenv("OPENAI_API_KEY"):
        print("\n⚠  OPENAI_API_KEY not found in .env file!")
        print("\nPlease create a .env file with:")
        print("OPENAI_API_KEY=your-openai-key-here")
        print("TAVILY_API_KEY=your-tavily-key-here")
        sys.exit(1)
    
    if not os.getenv("TAVILY_API_KEY"):
        print("\n⚠  TAVILY_API_KEY not found in .env file!")
        print("\nPlease add to your .env file:")
        print("TAVILY_API_KEY=your-tavily-key-here")
        sys.exit(1)
    
    print("✅ API keys loaded successfully!")
    
    # Initialize agent
    print("\n🤖 Initializing agent...")
    try:
        agent_executor = create_agent()
        print("✅ Agent ready!\n")
    except Exception as e:
        print(f"\n❌ Failed to initialize agent: {str(e)}")
        sys.exit(1)
    
    # Interactive chat loop
    print("Chat with the agent (type 'quit' to exit, 'history' to see chat history):\n")
    
    while True:
        try:
            user_input = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n\nGoodbye! 👋")
            break
        
        if not user_input:
            continue
        
        if user_input.lower() in ['quit', 'exit', 'q']:
            print("\nGoodbye! 👋")
            break
        
        if user_input.lower() == 'history':
            print("\n--- Chat History ---")
            if not chat_history:
                print("No chat history yet.")
            else:
                for role, message in chat_history:
                    preview = message[:100] + "..." if len(message) > 100 else message
                    print(f"{role}: {preview}")
            print("--- End History ---\n")
            continue
        
        if user_input.lower() == 'clear':
            chat_history.clear()
            print("\n✅ Chat history cleared!\n")
            continue
        
        try:
            response = chat(user_input, agent_executor)
            print(f"\n🤖 Agent: {response}\n")
        except KeyboardInterrupt:
            print("\n\nInterrupted. Type 'quit' to exit.\n")
        except Exception as e:
            print(f"\n❌ Error: {str(e)}\n")
            """
LangChain Agent with Streamlit - Simple UI
"""

import streamlit as st
from langchain.agents import create_tool_calling_agent, AgentExecutor
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage, AIMessage
from langchain_openai import ChatOpenAI
from langchain_community.tools.tavily_search import TavilySearchResults
from datetime import datetime
import requests
import os
from dotenv import load_dotenv

load_dotenv()

# =============================================================================
# PAGE CONFIG
# =============================================================================

st.set_page_config(
    page_title="AI Agent Chat",
    page_icon="🤖",
    layout="centered"
)

# =============================================================================
# TOOLS
# =============================================================================

@tool
def get_current_datetime() -> str:
    """Get the current date and time."""
    now = datetime.now()
    return now.strftime("%Y-%m-%d %H:%M:%S")

@tool
def get_weather(city: str) -> str:
    """Get current weather for a city."""
    try:
        url = f"https://wttr.in/{city}?format=j1"
        response = requests.get(url, timeout=10)
        if response.status_code == 200:
            data = response.json()
            current = data['current_condition'][0]
            weather_desc = current['weatherDesc'][0]['value']
            temp_c = current['temp_C']
            feels_like = current['FeelsLikeC']
            humidity = current['humidity']
            return f"Weather in {city}: {weather_desc}, Temperature: {temp_c}°C (feels like {feels_like}°C), Humidity: {humidity}%"
        else:
            return f"Could not fetch weather for {city}"
    except Exception as e:
        return f"Error getting weather: {str(e)}"

# =============================================================================
# AGENT
# =============================================================================

@st.cache_resource
def create_agent():
    """Initialize the agent."""
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.7)
    tavily_tool = TavilySearchResults(max_results=3, search_depth="basic")
    tools = [get_current_datetime, get_weather, tavily_tool]
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a helpful AI assistant with access to datetime, weather, and web search tools."),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{input}"),
        MessagesPlaceholder(variable_name="agent_scratchpad"),
    ])
    
    agent = create_tool_calling_agent(llm, tools, prompt)
    return AgentExecutor(agent=agent, tools=tools, verbose=True, handle_parsing_errors=True)

# =============================================================================
# SESSION STATE
# =============================================================================

if "messages" not in st.session_state:
    st.session_state.messages = []

if "agent_executor" not in st.session_state:
    if not os.getenv("OPENAI_API_KEY") or not os.getenv("TAVILY_API_KEY"):
        st.error("⚠️ Missing API keys! Add OPENAI_API_KEY and TAVILY_API_KEY to your .env file")
        st.stop()
    st.session_state.agent_executor = create_agent()

# =============================================================================
# UI
# =============================================================================

st.title("🤖 AI Agent Chat")

# Clear chat button
if st.button("Clear Chat"):
    st.session_state.messages = []
    st.rerun()

# Display messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Chat input
if prompt := st.chat_input("Type your message..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    with st.chat_message("user"):
        st.markdown(prompt)
    
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                formatted_history = []
                for msg in st.session_state.messages[:-1]:
                    if msg["role"] == "user":
                        formatted_history.append(HumanMessage(content=msg["content"]))
                    elif msg["role"] == "assistant":
                        formatted_history.append(AIMessage(content=msg["content"]))
                
                response = st.session_state.agent_executor.invoke({
                    "input": prompt,
                    "chat_history": formatted_history
                })
                
                assistant_response = response['output']
                st.markdown(assistant_response)
                st.session_state.messages.append({"role": "assistant", "content": assistant_response})
                
            except Exception as e:
                error_msg = f"❌ Error: {str(e)}"
                st.error(error_msg)
                st.session_state.messages.append({"role": "assistant", "content": error_msg})