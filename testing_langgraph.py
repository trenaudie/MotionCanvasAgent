#%%
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, START, END
from typing import Annotated

from typing_extensions import TypedDict

from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langchain_core.messages import HumanMessage, SystemMessage

#%%
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

class State(TypedDict):
    messages: Annotated[list, add_messages]
# here can also do 
from langgraph.graph import START, MessagesState, StateGraph
workflow = StateGraph(state_schema=MessagesState)
graph_builder = StateGraph(State)


def chatbot(state: State):
    print(state)
    return {"messages": [llm.invoke(state["messages"])]}
graph_builder.add_node("chatbot", chatbot)
graph_builder.add_edge(START, "chatbot")
graph_builder.add_edge("chatbot", END)
graph = graph_builder.compile()

#%% 
def stream_graph_updates(user_input: str):
    for event in graph.stream({"messages": [{"role": "user", "content": user_input}]}):
        for value in event.values():
            print("Assistant:", value["messages"][-1].content)

stream_graph_updates("Hello, how are you?")
# %%
# how to get the state of the graph
from langgraph.checkpoint.memory import MemorySaver
memory = MemorySaver()
graph = graph_builder.compile(checkpointer=memory)
#%%
#  you can do streaming as above but you can also just invoke the graph
# graph.invoke(
#     {"messages": [HumanMessage(content="Translate to French: I love programming.")]},
#     config={"configurable": {"thread_id": "1"}},
# )
#%% 

# obtain the state from the memory object 
"""
memory
adelete_thread()
aget()
aget_tuple()
alist()
aput()
aput_writes()
blobs
config_specs
delete_thread()
get()
get_next_version()
get_tuple()
list()
put()
put_writes()
serde
stack
storage
writes
"""
from langchain_core.tools import tool
from langgraph.types import Command, interrupt
from langgraph.prebuilt import ToolNode
from langgraph.prebuilt import tools_condition

@tool
def human_assistance(query: str) -> str:
    """Request assistance from a human."""
    human_response = interrupt({"query": query})
    return human_response["data"]
llm_with_tools = llm.bind_tools([human_assistance])
def human_chatbot(state: State):
    message = llm_with_tools.invoke(state["messages"])
    # Because we will be interrupting during tool execution,
    # we disable parallel tool calling to avoid repeating any
    # tool invocations when we resume.
    assert len(message.tool_calls) <= 1
    return {"messages": [message]}
graph_builder.add_node("human_chatbot", human_chatbot)
tool_node = ToolNode(tools=[human_assistance])
graph_builder.add_conditional_edges(
    "human_chatbot",
    tools_condition,
)
graph_builder.edges = graph_builder.edges[:1]
graph_builder.add_edge("chatbot", "human_chatbot")
graph_builder.add_node("tools", tool_node)
graph_builder.add_edge("human_chatbot", "tools")
graph_builder.add_edge("tools", END)
graph2 = graph_builder.compile(checkpointer=memory)
graph2
# %%
