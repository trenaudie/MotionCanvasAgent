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

#%%
graph_builder = StateGraph(State)
graph_builder.edges = set()
graph_builder.edges.add((START, "chatbot"))
graph_builder.add_node("chatbot", human_chatbot)
tool_node = ToolNode(tools=[human_assistance])
graph_builder.add_conditional_edges(
    "chatbot",
    tools_condition,
    # path_map = {
    #     "tools": tool_node,
    # }
)
# graph_builder.add_edge("chatbot", tool_node)
graph_builder.add_node("tools", tool_node)
graph_builder.add_edge("tools", END)
graph2 = graph_builder.compile(checkpointer=memory)
graph2
# %%
response = graph2.invoke(
    {"messages": [HumanMessage(content="Translate to French: I love programming.")]},
    config={"configurable": {"thread_id": "1"}},)

"""
{'messages': [HumanMessage(content='Translate to French: I love programming.', additional_kwargs={}, response_metadata={}, id='a06dc884-b090-45b9-a4a1-cc3094ee5242'),
  AIMessage(content="J'aime programmer.", additional_kwargs={'refusal': None}, response_metadata={'token_usage': {'completion_tokens': 4, 'prompt_tokens': 15, 'total_tokens': 19, 'completion_tokens_details': {'accepted_prediction_tokens': 0, 'audio_tokens': 0, 'reasoning_tokens': 0, 'rejected_prediction_tokens': 0}, 'prompt_tokens_details': {'audio_tokens': 0, 'cached_tokens': 0}}, 'model_name': 'gpt-4o-mini-2024-07-18', 'system_fingerprint': None, 'id': 'chatcmpl-BsfuksM9ndUonBP6jAJaEaYYARorl', 'service_tier': 'default', 'finish_reason': 'stop', 'logprobs': None}, id='run--02038c00-5e94-4520-b972-df5da3b6e427-0', usage_metadata={'input_tokens': 15, 'output_tokens': 4, 'total_tokens': 19, 'input_token_details': {'audio': 0, 'cache_read': 0}, 'output_token_details': {'audio': 0, 'reasoning': 0}}),
  AIMessage(content='', additional_kwargs={'tool_calls': [{'id': 'call_AZWi4rvYhIaXMwUS68FRiQ8w', 'function': {'arguments': '{"query":"Translate to French: I love programming."}', 'name': 'human_assistance'}, 'type': 'function'}], 'refusal': None}, response_metadata={'token_usage': {'completion_tokens': 23, 'prompt_tokens': 60, 'total_tokens': 83, 'completion_tokens_details': {'accepted_prediction_tokens': 0, 'audio_tokens': 0, 'reasoning_tokens': 0, 'rejected_prediction_tokens': 0}, 'prompt_tokens_details': {'audio_tokens': 0, 'cached_tokens': 0}}, 'model_name': 'gpt-4o-mini-2024-07-18', 'system_fingerprint': None, 'id': 'chatcmpl-BsfukJscxDq4oFQ2NY9ne1oP4M7vf', 'service_tier': 'default', 'finish_reason': 'tool_calls', 'logprobs': None}, id='run--73ef2080-d6bc-4368-bb20-3f254020e4f2-0', tool_calls=[{'name': 'human_assistance', 'args': {'query': 'Translate to French: I love programming.'}, 'id': 'call_AZWi4rvYhIaXMwUS68FRiQ8w', 'type': 'tool_call'}], usage_metadata={'input_tokens': 60, 'output_tokens': 23, 'total_tokens': 83, 'input_token_details': {'audio': 0, 'cache_read': 0}, 'output_token_details': {'audio': 0, 'reasoning': 0}})],
 '__interrupt__': [Interrupt(value={'query': 'Translate to French: I love programming.'}, resumable=True, ns=['tools:b241ea36-3391-b44e-0fcb-484dd968a609'])]}"""

#%%
memory.get(config = {"configurable": {"thread_id": "1"}})
# %%
