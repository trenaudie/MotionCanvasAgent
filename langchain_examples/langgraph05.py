#%% 

import os
from pathlib import Path
from typing import List, Dict, Optional, Any, Callable
from langchain_core.messages import BaseMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import ChatOpenAI
from langchain_core.language_models import BaseChatModel
from langchain_core.output_parsers import StrOutputParser
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, StateGraph, START
import os
from pathlib import Path
from agents.build_prompt import build_system_prompt_from_dirs_and_yaml
from logging_config.logger import LOG
from agents.output_models.code_output import CodeOutput
from agents.output_models.graph_state import GraphState
from langgraph.checkpoint.memory import MemorySaver

from langgraph.graph import END, StateGraph, START
from typing import Callable, Optional
from agents.state_tracker import StateTracker
# --- Mocking external dependencies ---
# In a pure notebook environment without your project structure,
# we need to mock these to make the code runnable.





# --- Your original functions, adapted for a pure notebook context ---

def build_graph(generate_handler: Callable):
    workflow = StateGraph(GraphState)
    tracker = StateTracker()
    tracker.create_new_state()
    memory: Optional[MemorySaver] = tracker.memory
    thread = tracker.thread
    LOG.info(f"building graph with memory: {memory} memory id {id(memory)} and thread: {thread}")
    workflow.add_node("generate", generate_handler)
    workflow.add_edge(START, "generate")
    workflow.add_edge("generate", END)
    graph = workflow.compile(checkpointer=memory)
    return graph.with_config(thread=thread)

#%% 
if os.getenv("OPENAI_API_KEY") is None:
    raise ValueError("OPENAI_API_KEY environment variable not set. Please set it to run the LLM.")

context_dirs = [Path("frontend/src/scenes1")] # Dummy path, not used but needed for build_system_prompt
yaml_path = Path("agents/prompts/coder_general_01.yaml") # Dummy path

system_prompt = build_system_prompt_from_dirs_and_yaml(
    context_dirs, yaml_path, CodeOutput
)

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
#%% 
query = "hello world, build me a circle animation"
few_shot_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        ("human", f"{query}"),
    ],
    template_format="jinja2",
)
code_gen_chain = few_shot_prompt | llm.with_structured_output(CodeOutput)


#%% 

def generate_handler(state: GraphState) -> Dict[str, Any]:
    LOG.info("---GENERATING CODE SOLUTION---")
    messages = state["messages"]
    iterations = state["iterations"]
    error = state["error"]

    if error == "yes":
        messages += [
            (
                "user",
                "Now, try again. Invoke the code tool to structure the output with a prefix, imports, and code block:",
            )
        ]

    # Invoke the chain using the current messages
    # The chain expects a dictionary with a "messages" key if the prompt is structured this way
    try:
        code_solution = code_gen_chain.invoke({"messages": messages}, config=StateTracker().thread)
    except Exception as e:
        # Handle cases where the LLM might not return a structured output correctly
        # For this minimal example, we'll just return an error state.
        LOG.info(f"Error during code generation: {e}")
        return {
            "messages": messages + [("assistant", f"Error generating code: {e}")],
            "iterations": iterations + 1,
            "error": "yes",
            "code_generated": None
        }

    LOG.info(f"Code generation successful! Output is:\n\n {code_solution}\n\n")

    # Assuming code_solution directly matches CodeOutput if output_model is used
    messages += [
        (
            "assistant",
            f"Reasoning: {getattr(code_solution, 'reasoning', 'No reasoning provided')}\n Code: {getattr(code_solution, 'code_generated', 'No code generated')}",
        )
    ]

    iterations = iterations + 1
    return {
        "code_generated": code_solution,
        "messages": messages,
        "iterations": iterations,
        "error": "" # Clear error if successful
    }
#%% 


graph = build_graph(generate_handler)
#%%
print(f'thread id: {StateTracker().thread}')
display(StateTracker().memory)
memory = StateTracker().memory
thread = StateTracker().thread
#%%%
display(list(graph.checkpointer.list(config = StateTracker().thread)))
from langgraph.checkpoint.memory import MemorySaver
memory = StateTracker().memory
thread = StateTracker().thread
def get_messages_from_memory(memory: MemorySaver, thread: dict) -> List[BaseMessage]:
    """
    Retrieve messages from the memory for a specific thread.
    """
    # return list(graph.checkpointer.list(config = StateTracker().thread))[0].checkpoint['channel_values']['messages']
    return memory.get(thread)['channel_values']['messages']
messages = get_messages_from_memory(memory, thread)
messages
#%% 
response = graph.invoke(
    {"messages": [("user", query)], "iterations": 0, "error": ""}, config=StateTracker().thread
)
LOG.info(f"Graph response: {response}")


#%% 
 
final_response = generate_code_using_langgraph(
    system_prompt,
    llm,
    user_prompt,
    output_model=CodeOutput,
)

print("\n--- Final Graph Output ---")
print(f"Generated Code Object: {final_response.get('code_generated')}")
if final_response.get('code_generated'):
    print(f"Reasoning: {final_response['code_generated'].reasoning}")
    print(f"Code: {final_response['code_generated'].code_generated}")
else:
    print("No code was generated, check error messages.")
print(f"Messages: {final_response.get('messages')}")
print(f"Iterations: {final_response.get('iterations')}")
print(f"Error Status: {final_response.get('error')}")