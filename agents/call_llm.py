# %%
#
from typing import List, Dict, Optional, Any
from langchain_core.messages import BaseMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import ChatOpenAI
from langchain_core.language_models import BaseChatModel
from langchain_core.output_parsers import StrOutputParser
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

def build_graph(
    generate_handler: Callable,
):
    workflow = StateGraph(GraphState)
    # Define the nodes
    tracker = StateTracker()
    tracker.create_new_state()
    memory : Optional[MemorySaver] = tracker.memory
    thread = tracker.thread
    LOG.info(f"building graph with memory: {memory} memory id {id(memory)} and thread: {thread}")
    workflow.add_node("generate", generate_handler)  # generation solution
    workflow.add_edge(START, "generate")
    workflow.add_edge("generate", END)
    LOG.info(f'before compilation, current state has {len(tracker.get_current_state().get("messages", []))} messages')
    graph = workflow.compile(checkpointer=memory)
    LOG.info(f'after compilation, current state has {len(tracker.get_current_state().get("messages", []))} messages')
    return graph.with_config(thread=thread)


def generate_code_using_langgraph(
    system_prompt: str,
    llm: BaseChatModel,
    query: str,
    message_history: List[dict] = [],
    tools: Optional[List[Any]] = None,
    output_model: Optional[Any] = None,
):
    message_history_human_ai = []
    LOG.info(f'message history is {message_history}')
    for msg in message_history:
        if msg[0] == "user":
            message_history_human_ai.append(("human", msg[1]))
        elif msg[0] == "assistant":
            message_history_human_ai.append(("ai", msg[1]))
    few_shot_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            *message_history_human_ai,
            ("human", f"{query}"),
        ],
        template_format="jinja2",
    )
    code_gen_chain = few_shot_prompt | llm.with_structured_output(output_model)

    def generate_handler(state: GraphState) -> Dict[str, Any]:
        """
        Generate a code solution

        Args:
            state (dict): The current graph state

        Returns:
            state (dict): New key added to state, generation
        """

        LOG.info("---GENERATING CODE SOLUTION---")
        current_state = StateTracker().get_current_state()
        if state.get("messages", []):
            LOG.info(f'input to the generate handler has num messages: {len(state["messages"])}')
        messages =  state["messages"]
        iterations = state["iterations"]
        error = state["error"]

        # We have been routed back to generation with an error
        if error == "yes":
            messages += [
                (
                    "user",
                    "Now, try again. Invoke the code tool to structure the output with a prefix, imports, and code block:",
                )
            ]
        LOG.info(f'current state has {len(current_state.get("messages", []))} messages')
        LOG.info(f'Starting code generation with num messages : {len(messages)}')
        LOG.info(f'here are the roles in the messages: {[msg[0] for msg in messages]}')
        code_solution = code_gen_chain.invoke({"messages": messages}, config=StateTracker().get_runnable_config())
        LOG.info(f'Code generation successful')
        messages += [
            (
                "assistant",
                f"Reasoning: {code_solution.reasoning}\n Code: {code_solution.code_generated}",
            )
        ]

        # Increment
        iterations = iterations + 1
        return {
            "code_output": code_solution,
            "messages": messages,
            "iterations": iterations,
        }

    graph = build_graph(generate_handler) # it ok if it gets recompiled, just the thread needs to remain the same, and the memory object as well
    current_state = StateTracker().get_current_state()
    if current_state:
        current_state_messages = current_state.get("messages", [])
    else:
        current_state_messages = []
    current_iteration = current_state.get("iterations", 0)
    current_error = current_state.get("error", "")
    current_code_output = current_state.get("code_output", None)
    user_input= GraphState(**{"messages": current_state_messages + [("user", query)], "iterations": current_iteration, "error": current_error, "code_output": current_code_output})
    response = graph.invoke(
        user_input, config=StateTracker().get_runnable_config()
    )
    return response


def generate_code_from_query(
    prompt: str, output_file: Optional[os.PathLike] = None, dummy_code: bool = False, 
) -> str:
    """
    This function performs a code generation from a given conversation.
    If dummy_code is true, it will just copy the code from the DUMMY_CODE_FILE into the output_file.

    :param chat_messages: The conversation messages (str).
    :param agent: The agent instance that generates the code.
    :param output_file: The file where the generated code will be written.
    :param dummy_code: If true, copy from the dummy code file. Default is True.
    """

    if not dummy_code:
        context_dirs = [Path("frontend/src/scenes"), Path("frontend/src/scenes2")]
        yaml_path = Path("agents/prompts/coder_general_01.yaml")
        system_prompt = build_system_prompt_from_dirs_and_yaml(
            context_dirs, yaml_path, CodeOutput
        )
        llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

        current_state = StateTracker().get_current_state()
        LOG.info(f'state received is {current_state}')
        if current_state:
            message_history = current_state.get("messages", [])
        else:
            message_history = []

        response = generate_code_using_langgraph(
            system_prompt, llm, prompt, message_history, output_model=CodeOutput,
        )
        LOG.info(f"Response from generate_code_using_langgraph: {response}")
        code_output = response["code_output"]
        code_generated = code_output.code_generated
        _ = code_output.reasoning
        LOG.info(f"Code generation successful!")
    else:
        DUMMY_CODE_FILE = (
            "/home/bits/MotionCanvasAgent/frontend/src/scenes2/example9_circle_cat.tsx"
        )
        code_generated = Path(DUMMY_CODE_FILE).read_text()

    # Ensure the output file path exists, create it if not
    if output_file:
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Write the generated code to the output file
        with open(output_path, "w") as f:
            f.write(code_generated)

        LOG.info(f"Output written to {output_file}")

    return code_generated


# %%
if __name__ == "__main__":
    from pathlib import Path
    from agents.build_prompt import build_system_prompt_from_dirs_and_yaml
    from agents.output_models.code_output import CodeOutput

    context_dirs = [Path("frontend/src/scenes")]
    yaml_path = Path("agents/prompts/coder_general_01.yaml")
    system_prompt = build_system_prompt_from_dirs_and_yaml(
        context_dirs, yaml_path, CodeOutput
    )
    print(system_prompt)

    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

    # you can add tools here, which will then be shown as examples in the prompt
    response = generate_code_using_langgraph(
        system_prompt,
        llm,
        "can you create a simple hello world faded in animation?",
        output_model=CodeOutput,
    )
    print(response)
# %%