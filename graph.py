# import the required methods
from typing import Literal, List
import langgraph
from langgraph.prebuilt import ToolNode
from langchain_core.tools import tool
from langgraph.graph import StateGraph, MessagesState, START, END
from langchain_ollama import ChatOllama
from dotenv import load_dotenv
load_dotenv()
import os
import requests

from langchain_openai import ChatOpenAI
from langchain_community.tools.tavily_search import TavilySearchResults

#os.environ["OPENAI_API_KEY"]=os.getenv("OPENAI_API_KEY")
os.environ["TAVILY_API_KEY"]=os.getenv("TAVILY_API_KEY")
WEATHER_API_KEY=os.getenv("WEATHER_API_KEY")
os.environ["TOGETHER_API_KEY"]=os.getenv("TOGETHER_API_KEY")
# define llm
#llm = ChatOpenAI(model="gpt-4o-mini")
llm = ChatOllama(model="gemma:7b-instruct", temperature=0.2)
# define tools
@tool
def get_weather(query: str) -> list:
    """Search weatherapi to get the current weather"""
    endpoint = f"http://api.weatherapi.com/v1/current.json?key={WEATHER_API_KEY}&q={query}"
    response = requests.get(endpoint)
    data = response.json()

    if data.get("location"):
        return data
    else:
        return "Weather Data Not Found"
@tool
def search_web(query: str) -> list:
    """Search the web for a query"""
    tavily_key = os.getenv("TAVILY_API_KEY")
    tavily_search = TavilySearchResults(api_key=tavily_key, max_results=2, search_depth='advanced', max_tokens=1000)
    results = tavily_search.invoke(query)
    return results

# define a tool_node with the available tools
tools = [search_web, get_weather]
tool_node = ToolNode(tools)
llm_with_tools = llm.bind_tools(tools)

# define functions to call the LLM or the tools
def call_model(state: MessagesState):
    messages = state["messages"]
    response = llm_with_tools.invoke(messages)
    return {"messages": [response]}

def call_tools(state: MessagesState) -> Literal["tools"]:
    messages = state["messages"]
    last_message = messages[-1]
    if last_message.tool_calls:
        return "tools"
    return END

# initialize the workflow from StateGraph
workflow = StateGraph(MessagesState)

# add a node named LLM, with call_model function. This node uses an LLM to make decisions based on the input given
workflow.add_node("LLM", call_model)

# Our workflow starts with the LLM node
workflow.add_edge(START, "LLM")

# Add a tools node
workflow.add_node("tools", tool_node)

# Add a conditional edge from LLM to call_tools function. It can go tools node or end depending on the output of the LLM. 
workflow.add_conditional_edges("LLM", call_tools)

# tools node sends the information back to the LLM
workflow.add_edge("tools", "LLM")

agent = workflow.compile()
# display(Image(agent.get_graph().draw_mermaid_png()))


def query(location, question):
    print(location, question)
    location = location
    question = question
    response = []
    message = f"user: {question} in {location}?"
    for chunk in agent.stream(
        {"messages": [(message)]},
        stream_mode="values",):
        response.append(chunk["messages"][-1])
    print(response)
    return response    
