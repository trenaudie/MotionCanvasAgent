
# %%
import getpass
import os
from bs4 import BeautifulSoup as Soup
from langchain_community.document_loaders.recursive_url_loader import RecursiveUrlLoader

from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field
from langchain_anthropic import ChatAnthropic
from langchain_core.prompts import ChatPromptTemplate
from logging_config.logger import LOG
# lets try with google 
from langchain_anthropic import ChatAnthropic
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from pprint import pprint
import json
from typing import List
from typing_extensions import TypedDict
import os 
from langchain_community.document_loaders.git import GitLoader
from agents.state_tracker import StateTracker


def _set_env(var: str):
    if not os.environ.get(var):
        os.environ[var] = getpass.getpass(f"{var}: ")


_set_env("OPENAI_API_KEY")
#%%
# CONTEXT
url = "https://python.langchain.com/docs/concepts/lcel/"
loader = RecursiveUrlLoader(
    url=url, max_depth=20, extractor=lambda x: Soup(x, "html.parser").text
)
docs = loader.load()
d_sorted = sorted(docs, key=lambda x: x.metadata["source"])
d_reversed = list(reversed(d_sorted))
concatenated_content = "\n\n\n --- \n\n\n".join(
    [doc.page_content for doc in d_reversed]
)
#%% 
repo_path = "https://github.com/langchain-ai/langgraph/tree/main/docs/docs/tutorials"
gitloader = GitLoader(repo_path="langgraph", file_filter=lambda x: "docs/docs" in x and (x.endswith(".ipynb") or x.endswith(".py")))
docs = gitloader.load()
docs
#%%
concatenated_content = ""
for doc in docs:
    print(doc.metadata["source"])
    concatenated_content += doc.page_content + "\n\n\n --- \n\n\n"



from ai_utils.count_tokens import count_openai_tokens
print(f"Context length: {count_openai_tokens(concatenated_content, 'gpt-3.5-turbo')} tokens")
#%%

# Keep docs that are not chatbot-related, **or** are chatbot+rag
docs_to_keep = [
    doc for doc in docs 
    if ("chatbot" in doc.metadata["source"] or "multi-agent-rag" in doc.metadata["source"] or "ref" in doc.metadata["source"]) 
]
doc_context_lengths = [
    count_openai_tokens(doc.page_content, "gpt-3.5-turbo") for doc in docs_to_keep
]
doc_context_lengths
# only keep the first 3 
docs_to_keep = docs_to_keep[:3]
print(f'number of docs to keep: {len(docs_to_keep)}')
concatenated_content = ""
for doc in docs_to_keep:
    concatenated_content += doc.page_content + "\n\n\n --- \n\n\n"
print(f"Context length: {count_openai_tokens(concatenated_content, 'gpt-3.5-turbo')} tokens")
#%%



#%% output model 
class code(BaseModel):
    """Schema for code solutions to questions about LCEL."""

    prefix: str = Field(description="Description of the problem and approach")
    imports: str = Field(description="Code block import statements")
    code: str = Field(description="Code block not including import statements")

### OpenAI
# Grader prompt
code_gen_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """You are a coding assistant with expertise in LCEL, LangChain expression language. \n 
    Here is a full set of LCEL documentation:  \n ------- \n  {context} \n ------- \n Answer the user 
    question based on the above provided documentation. Ensure any code you provide can be executed \n 
    with all required imports and variables defined. Structure your answer with a description of the code solution. \n
    Then list the imports. And finally list the functioning code block. Here is the user question:""",
        ),
        ("placeholder", "{messages}"),
    ]
)


expt_llm = "gpt-4o-mini"
llm = ChatOpenAI(temperature=0, model=expt_llm)
code_gen_chain = code_gen_prompt | llm.with_structured_output(code)
question = "How do I build a RAG chain in LCEL?"



# %%
# GEMINI
def build_code_gen_chain_gemini():
    """
    Build a code generation chain for Gemini.
    """
    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "auth_keys/datadog-feat-req-clustering-aeb47e2e72ee.json"
    from langchain_google_genai import ChatGoogleGenerativeAI
    from langchain_core.prompts import ChatPromptTemplate

    # Set up the prompt template
    code_gen_prompt_gemini = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """You are a coding assistant with expertise in LCEL, LangChain expression language. \n 
        Here is the LCEL documentation:  \n ------- \n  {context} \n ------- \n Answer the user 
        question based on the above provided documentation. Ensure any code you provide can be executed \n 
        with all required imports and variables defined. Structure your answer: 1) a prefix describing the code solution, 2) the imports, 3) the functioning code block. \n
        Invoke the code tool to structure the output correctly. </instructions> \n Here is the user question:""",
            ),
            ("placeholder", "{messages}"),
        ]
    )

    GOOGLE_MODEL = "gemini-2.5-flash-lite-preview-06-17"
    llm = ChatGoogleGenerativeAI(model=GOOGLE_MODEL)
    return code_gen_prompt_gemini | llm.with_structured_output(code, include_raw=True)


# %%

class GraphState(TypedDict):
    """
    Represents the state of our graph.

    Attributes:
        error : Binary flag for control flow to indicate whether test error was tripped
        messages : With user question, error messages, reasoning
        generation : Code solution
        iterations : Number of tries
    """

    error: str
    messages: List
    generation: str
    iterations: int

# %% [markdown]
# ## Graph 
# 
# Our graph lays out the logical flow shown in the figure above.

# %%
### Parameter

# Max tries
max_iterations = 3
# Reflect
# flag = 'reflect'
flag = "reflect"


#%% 
### Nodes


def generate(state: GraphState):
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
    code_solution = code_gen_chain.invoke(
        {"context": concatenated_content, "messages": messages}
    )
    messages += [
        (
            "assistant",
            f"{code_solution.prefix} \n Imports: {code_solution.imports} \n Code: {code_solution.code}",
        )
    ]

    # Increment
    iterations = iterations + 1
    return {"generation": code_solution, "messages": messages, "iterations": iterations}


def code_check(state: GraphState):
    """
    Check code

    Args:
        state (dict): The current graph state

    Returns:
        state (dict): New key added to state, error
    """

    LOG.info("---CHECKING CODE---")

    # State
    messages = state["messages"]
    code_solution = state["generation"]
    iterations = state["iterations"]

    # Get solution components
    imports = code_solution.imports
    code = code_solution.code

    # Check imports
    try:
        LOG.info("Running code import check")
        exec(imports)
    except Exception as e:
        LOG.info("---CODE IMPORT CHECK: FAILED---")
        error_message = [("user", f"Your solution failed the import test: {e}")]
        messages += error_message
        return {
            "generation": code_solution,
            "messages": messages,
            "iterations": iterations,
            "error": "yes",
        }

    # Check execution
    try:
        LOG.info("Running code block check")
        exec(imports + "\n" + code)
    except Exception as e:
        LOG.info("---CODE BLOCK CHECK: FAILED---")
        error_message = [("user", f"Your solution failed the code execution test: {e}")]
        messages += error_message
        return {
            "generation": code_solution,
            "messages": messages,
            "iterations": iterations,
            "error": "yes",
        }

    # No errors
    print("---NO CODE TEST FAILURES---")
    return {
        "generation": code_solution,
        "messages": messages,
        "iterations": iterations,
        "error": "no",
    }


def reflect(state: GraphState):
    """
    Reflect on errors

    Args:
        state (dict): The current graph state

    Returns:
        state (dict): New key added to state, generation
    """

    print("---REFLECTING CODE SOLUTION---")

    # State
    messages = state["messages"]
    iterations = state["iterations"]
    code_solution = state["generation"]

    # Prompt reflection

    # Add reflection
    reflections = code_gen_chain.invoke(
        {"context": concatenated_content, "messages": messages}
    )
    messages += [("assistant", f"Here are reflections on the error: {reflections}")]
    return {"generation": code_solution, "messages": messages, "iterations": iterations}


### Edges


def decide_to_finish(state: GraphState):
    """
    Determines whether to finish.

    Args:
        state (dict): The current graph state

    Returns:
        str: Next node to call
    """
    error = state["error"]
    iterations = state["iterations"]

    if error == "no" or iterations == max_iterations:
        print("---DECISION: FINISH---")
        return "end"
    else:
        print("---DECISION: RE-TRY SOLUTION---")
        if flag == "reflect":
            return "reflect"
        else:
            return "generate"

# %%
from langgraph.graph import END, StateGraph, START

workflow = StateGraph(GraphState)

# Define the nodes
workflow.add_node("generate", generate)  # generation solution
# workflow.add_node("check_code", code_check)  # check code
# workflow.add_node("reflect", reflect)  # reflect

# Build graph
workflow.add_edge(START, "generate")
workflow.add_edge("generate",END)
# workflow.add_edge("generate", "check_code")
# workflow.add_conditional_edges(
#     "check_code",
#     decide_to_finish,
#     {
#         "end": END,
#         "reflect": "reflect",
#         "generate": "generate",
#     },
# )
# workflow.add_edge("reflect", "generate")
app = workflow.compile()
#%% 


# %%
question = "How can I directly pass a string to a runnable and use it to construct the input needed for my prompt?"
solution = app.invoke({"messages": [("user", question)], "iterations": 0, "error": ""})
solution
# %%
print(solution["generation"].code)
