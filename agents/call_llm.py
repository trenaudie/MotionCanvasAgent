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
from langgraph.graph import END, StateGraph, START
from typing import Callable, Optional
from agents.state_tracker import StateTracker

def generate_code(
    system_prompt: str,
    llm: BaseChatModel,
    query: str,
    examples: List[BaseMessage] = [],
    tools: Optional[List[Any]] = None,
    output_model: Optional[Any] = None,
) -> Any:
    """
    Generate code using few-shot prompting with LangChain.

    Args:
        system_prompt: The system prompt to use
        examples: List of langchain_core.messages examples for few-shot prompting
        llm: The language model to use
        tools: Optional tools to bind to the LLM
        query: The user query/prompt for code generation

    Returns:
        Generated code as a string
    """
    # Bind tools to LLM if provided
    if tools:
        llm_final = llm.bind_tools(tools)
    if output_model:
        llm_final = llm.with_structured_output(output_model, include_raw=False)
    if not tools and not output_model:
        llm_final = llm

    # Create few-shot prompt template
    few_shot_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            *examples,
            ("human", f"{query}"),
        ],
        template_format="jinja2",
    )

    # Create the chain
    chain = {"query": RunnablePassthrough()} | few_shot_prompt | llm_final

    response = chain.invoke(query)

    return response


def build_graph(
    generate_handler: Callable,
):
    workflow = StateGraph(GraphState)
    # Define the nodes
    tracker = StateTracker()
    tracker.create_new_state()
    memory = tracker.memory
    thread = tracker.thread
    workflow.add_node("generate", generate_handler)  # generation solution
    workflow.add_edge(START, "generate")
    workflow.add_edge("generate", END)
    graph = workflow.compile(checkpointer=memory)
    return graph.with_config(thread=thread)


def generate_code_using_langgraph(
    system_prompt: str,
    llm: BaseChatModel,
    query: str,
    examples: List[BaseMessage] = [],
    tools: Optional[List[Any]] = None,
    output_model: Optional[Any] = None,
):
    # Create few-shot prompt template
    few_shot_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            *examples,
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

        # State
        messages = state["messages"]
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

        # Solution
        code_solution = code_gen_chain.invoke({"messages": messages}, config=StateTracker().thread)
        LOG.info(f"Code generation successful! Output is:\n\n {code_solution}\n\n")
        messages += [
            (
                "assistant",
                f"Reasoning: {code_solution.reasoning}\n Code: {code_solution.code_generated}",
            )
        ]

        # Increment
        iterations = iterations + 1
        return {
            "code_generated": code_solution,
            "messages": messages,
            "iterations": iterations,
        }

    graph = build_graph(generate_handler)
    response = graph.invoke(
        {"messages": [("user", query)], "iterations": 0, "error": ""}, config=StateTracker().thread
    )
    LOG.info(f"Graph response: {response}")
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
        response = generate_code_using_langgraph(
            system_prompt, llm, prompt, output_model=CodeOutput,
        )
        LOG.info(f"Response code: {response}")
        code_generated = response["code_generated"]
        code_generated = code_generated.code_generated
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
# %%
