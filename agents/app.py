# app.py
from typing import Annotated
from typing_extensions import TypedDict

from langchain.chat_models import init_chat_model
from langchain_core.messages import AnyMessage
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition

from agent_keepa import keepa_search
from agent_serp import google_shopping
import os
from dotenv import load_dotenv

load_dotenv(override=False)

os.environ["OPENAI_API_KEY"]
# ---- State ----
class State(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]

# ---- LLM bound to tools ----
llm = init_chat_model("openai:gpt-4.1")
tools = [keepa_search, google_shopping]
llm_with_tools = llm.bind_tools(tools)

# ---- Agent node: let the model decide to call tools or respond ----
def call_agent(state: State):
    return {"messages": [llm_with_tools.invoke(state["messages"])]}

# ---- Tool node: executes whichever tool the model requested ----
tool_node = ToolNode(tools)

# ---- Graph ----
graph = StateGraph(State)
graph.add_node("agent", call_agent)
graph.add_node("tools", tool_node)

# Start with agent. If it asks for a tool, go to tools; otherwise end.
graph.add_edge(START, "agent")
graph.add_conditional_edges("agent", tools_condition)  # routes to "tools" or END automatically
graph.add_edge("tools", "agent")  # after tools run, go back to the agent for final answer

app = graph.compile()

if __name__ == "__main__":
    user_msg = {
        "role": "user",
        "content": "Find iPhone 15 cases under $20. Show buybox price, regular price, shipping cost, sales rank and direct link to the product."
    }

    # Easiest: single invoke
    result = app.invoke({"messages": [user_msg]})
    print("\n=== FINAL ANSWER ===")
    print(result["messages"][-1].content)

    # (Optional) Dev mode: watch the flow (tool calls, results, final)
    # for event in app.stream({"messages": [user_msg]}):
    #     print("--- EVENT ---")
    #     print(event)
