"""
This memory experiment tries out removing the MemorySaver object and seeing what happens when we do multiple graph.invoke calls
"""
import streamlit as st
import os
import getpass
from bs4 import BeautifulSoup as Soup
from langchain_community.document_loaders.recursive_url_loader import RecursiveUrlLoader
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.runnables import RunnableConfig
from pydantic import BaseModel, Field
from langchain_anthropic import ChatAnthropic # Though not used in final chain, keeping for context
from langchain_google_genai import ChatGoogleGenerativeAI

from pprint import pprint
import json
from typing import List
from typing_extensions import TypedDict
from langchain_community.document_loaders.git import GitLoader
from langgraph.graph import END, StateGraph, START
from langgraph.checkpoint.memory import MemorySaver
from dotenv import load_dotenv
import sys 
parent_dir = "/home/bits/MotionCanvasAgent"
print(f"Adding parent directory to sys.path: {parent_dir}")
sys.path.insert(0, parent_dir)
from logging_config.logger import LOG
load_dotenv()
assert os.getenv("OPENAI_API_KEY") is not None, "OPENAI_API_KEY environment variable not set. Please set it to run the LLM."
import git # For cloning the repository
from project_paths import PROJECT_ROOT
from ai_utils.count_tokens import count_openai_tokens
NUM_DOCS_TO_KEEP = 4  # Number of docs to keep for processing
@st.cache_data(show_spinner="Loading and processing documentation...")
def load_and_process_docs(num_docs_to_keep : int = 5):
    LOG.info("Loading and processing documentation from langgraph repository...")
    repo_url = "https://github.com/langchain-ai/langgraph.git"
    target_dir = PROJECT_ROOT / "trenaudie" / "langgraph"

    try:
        # Check if directory already exists and remove it for a fresh clone
        if not target_dir.exists():

            LOG.info(f"Cloning {repo_url} into {target_dir}...")
            st.info(f"Cloning {repo_url} into: {target_dir}")
            git.Repo.clone_from(repo_url, target_dir)
            st.success("Repository cloned successfully!")
        else:
            LOG.info(f"Repository already exists at {target_dir}. Skipping clone.")
            st.info(f"Repository already exists at: {target_dir}. Skipping clone.")

        LOG.info(f"Initializing GitLoader. Loading documents from {target_dir}...")
        gitloader = GitLoader(
            repo_path=str(target_dir),
            file_filter=lambda x: "docs/docs" in x and (x.endswith(".ipynb") or x.endswith(".py"))
        )
        docs = gitloader.load()
        LOG.info(f'number of docs retrieved {len(docs)}')
        for doc in docs:
            LOG.info(doc.metadata['source'])

        # Filter relevant docs
        relevant_docs = [
            doc for doc in docs
            if any(k in doc.metadata["source"] for k in ["chatbot", "multi-agent-rag", "ref", "rag", "code-assistant", "human-in-the-loop"])
        ]
        LOG.info(f"Found {len(relevant_docs)} relevant docs.")

        # Process docs to extract only code content from notebooks and add until token limit
        processed_docs = []
        total_tokens = 0
        max_tokens = 64000  # Half of 128k context window
        
        for doc in relevant_docs:
            if doc.metadata["source"].endswith(".ipynb"):
                # Extract only code cells from notebook content
                import json
                try:
                    notebook_data = json.loads(doc.page_content)
                    code_cells = []
                    if "cells" in notebook_data:
                        for cell in notebook_data["cells"]:
                            if cell.get("cell_type") == "code" and cell.get("source"):
                                # Join source lines and add to code cells
                                code_content = "".join(cell["source"])
                                if code_content.strip():  # Only add non-empty code
                                    code_cells.append(code_content)
                    
                    if code_cells:
                        processed_content = "\n\n".join(code_cells)
                        content_tokens = count_openai_tokens(processed_content, "gpt-4o-mini")
                        
                        if total_tokens + content_tokens <= max_tokens:
                            processed_docs.append(processed_content)
                            total_tokens += content_tokens
                            LOG.info(f"Added {doc.metadata['source']} ({content_tokens} tokens, total: {total_tokens})")
                        else:
                            LOG.info(f"Skipping {doc.metadata['source']} - would exceed token limit")
                            break
                            
                except json.JSONDecodeError:
                    LOG.warning(f"Failed to parse notebook: {doc.metadata['source']}")
                    continue
            else:
                # For .py files, use content as is
                content_tokens = count_openai_tokens(doc.page_content)
                
                if total_tokens + content_tokens <= max_tokens:
                    processed_docs.append(doc.page_content)
                    total_tokens += content_tokens
                    LOG.info(f"Added {doc.metadata['source']} ({content_tokens} tokens, total: {total_tokens})")
                else:
                    LOG.info(f"Skipping {doc.metadata['source']} - would exceed token limit")
                    break

        LOG.info(f"Processed {len(processed_docs)} docs with {total_tokens} total tokens")
        concatenated_content = "\n\n\n --- \n\n\n".join(processed_docs)
        return concatenated_content

    except Exception as e:
        st.error(f"An error occurred during documentation loading: {e}")
        st.stop()

class Code(BaseModel):
    reasoning: str = Field(description="Description of the problem and approach")
    code_generated: str = Field(description="Code block not including import statements")


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
    generation: Code # Changed to 'code' type for clarity
    iterations: int


def build_openai_chain(model_name: str = "gpt-4o-mini"):
    code_gen_prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """You are a coding assistant with expertise in LangGraph, LangChain library. \n
                Here is a full set of LangGraph documentation:  \n ------- \n  {context} \n ------- \n Answer the user
                question based on the above provided documentation, by giving your reasoning then the code. Ensure any code you provide can be executed\n. You must include imports, functions and a main.py script (if possible). Structure your answer with a description of the code solution. \n
                And finally provide the functioning code block. Here is the user question:""",
            ),
            ("placeholder", "{messages}"),
        ]
    )
    llm = ChatOpenAI(temperature=0, model=model_name)
    return code_gen_prompt | llm.with_structured_output(Code)

# --- LangGraph Workflow ---
concatenated_content = load_and_process_docs()
thread = {'configurable': {'thread_id': "1"}}



def generate(state: GraphState):
    """
    Generate a code solution

    Args:
        state (dict): The current graph state

    Returns:
        state (dict): New key added to state, generation
    """


    # State
    messages = state["messages"]
    st.session_state.log_messages.append(f"Current messages in state: {messages}")
    iterations = state.get('iterations',0)
    error = state.get('error', '')

    code_gen_chain = build_openai_chain() # dont give the option

    try:
        code_solution= code_gen_chain.invoke(
            {"context": concatenated_content, "messages": messages}, config =RunnableConfig(configurable = thread["configurable"])
        )
    except Exception as e:
        st.session_state.log_messages.append(f"Error during code generation: {e}")
        # In a real app, you might want to handle this more gracefully,
        # perhaps by setting an error flag in the state and retrying or informing the user.
        return {"generation": Code(reasoning= "",code_generated= ""), "messages": messages, "iterations": iterations, "error": "yes"}

    if code_solution and isinstance(code_solution, Code):
        messages.append(
            (
                "assistant",
                f"{code_solution.reasoning} \n Code: {code_solution.code_generated}",
            )
        )
    else:
        raise ValueError("Code generation did not return a valid Code object.")

    # Increment
    iterations = iterations + 1
    return {"generation": code_solution, "messages": messages, "iterations": iterations, "error": "no"}


# @st.cache_resource(show_spinner=True)
# def memory(thread_id: dict):
#     st.write("Initializing memory for the LangGraph workflow...")
#     from langgraph.checkpoint.memory import MemorySaver
#     memory = MemorySaver()
#     LOG.info(f"Memory initialized with thread ID: {thread_id}")
#     return memory



# Initialize LangGraph workflow
@st.cache_resource(show_spinner=True)
def build_graph(thread_id: str):
    st.warning(f'Compiling graph...')
    LOG.info(f'compiling graph')
    workflow = StateGraph(GraphState)
    workflow.add_node("generate", generate)  # generation solution
    workflow.add_edge(START, "generate")
    workflow.add_edge("generate", END)
    # app = workflow.compile(checkpointer=memory(thread_id=thread["configurable"]["thread_id"]))
    
    LOG.info(f'Compiling graph with MemorySaver')
    app = workflow.compile(checkpointer=MemorySaver())
    return app

# --- Streamlit UI ---

st.set_page_config(page_title="LCEL Code Generator", layout="wide")

st.title("🐍 Langgraph Code Generator")

st.markdown(
    """
    Ask a question about LangChain or Langgraph and get a code solution!
    The app uses documentation from the `langgraph` GitHub repository as context.
    """
)

# Initialize session state for messages and logs
if "log_messages" not in st.session_state:
    st.session_state.log_messages = []
if "generated_solution" not in st.session_state:
    st.session_state.generated_solution = None


st.sidebar.header("LLM Configuration")
selected_llm = st.sidebar.radio(
    "Choose your LLM:",
    ("OpenAI"),
    key="selected_llm"
)

if selected_llm == "OpenAI":
    if not os.environ.get("OPENAI_API_KEY"):
        st.sidebar.error("OpenAI API Key not found. Please add it to your Streamlit secrets.")

user_question = st.text_area(
    "Enter your question about Langgraph or Langchain:",
    height=100,
    placeholder="e.g., How can I directly pass a string to a runnable and use it to construct the input needed for my prompt?"
)
app = build_graph(str(thread['configurable']['thread_id']))

# --- Cached Function for Code Generation ---
@st.cache_data(show_spinner=True) # Spinner is handled outside
def _get_cached_solution(question: str, llm_choice: str):
    """
    Generates a code solution using the LangGraph app and caches the result.
    This function will only re-run if 'question' or 'llm_choice' changes.
    """
    # This message will only appear if the cache misses
    LOG.info(f'Starting the get cache solution')
    st.session_state.log_messages.append(f"Cache miss: Generating new solution for '{question}' with {llm_choice}.")

    # Note: We don't need to set st.session_state.selected_llm here since
    # the generate function doesn't actually use it - it uses build_openai_chain() directly
    # messages = app.get_state(thread)
    # LOG.info(f"Fetched messages using get_state: {len(messages)} total messages")
    if app.checkpointer is not None and isinstance(app.checkpointer, MemorySaver):
        checkpoint_tuple = app.checkpointer.get_tuple(RunnableConfig(configurable=thread["configurable"]))
        if checkpoint_tuple is not None:
            current_state = checkpoint_tuple.checkpoint['channel_values']
        else:
            current_state = None
    else:
        current_state = None
    LOG.info(f"Current state fetched: {current_state}")
    if current_state is None:
        current_state = {
            "messages": [],
            "generation": Code(reasoning = "", code_generated = ""),
            "iterations": 0,
            "error": "no"
        }
    current_state['messages'].append(("user", question))
    # current_state = {
    #     "messages": [("user", question)],
    # }
    LOG.info(f"Graph input: num messages {len(current_state['messages'])}")
    LOG.info(f'app id {id(app)} and thread id {thread["configurable"]["thread_id"]}')
    current_state = app.invoke(GraphState(**current_state), config=RunnableConfig(configurable=thread["configurable"]))
    LOG.info(f"Graph response acquired ! Num messages: {len(current_state['messages'])}")
    # Return a serializable dictionary instead of the BaseModel instance
    generation = current_state["generation"]
    return {
        "reasoning": generation.reasoning,
        "code_generated": generation.code_generated
    }


if st.button("Generate Code Solution"):
    if not user_question:
        st.warning("Please enter a question.")
    else:
        if (st.session_state.selected_llm == "OpenAI" and not os.environ.get("OPENAI_API_KEY")) or \
           (st.session_state.selected_llm == "Google Gemini" and not os.environ.get("GOOGLE_API_KEY")):
            st.error("Please configure your API key in Streamlit secrets before proceeding.")
        else:
            st.session_state.log_messages = [] # Clear logs for new generation
            st.session_state.log_messages.append(f"User question: {user_question}")
            with st.spinner("Generating solution... This might take a moment."):
                try:
                    # Call the cached function
                    st.session_state.generated_solution = _get_cached_solution(user_question, st.session_state.selected_llm)
                    st.session_state.log_messages.append("Code generation complete.")
                except Exception as e:
                    st.error(f"An error occurred during generation: {e}")
                    st.session_state.log_messages.append(f"Error: {e}")
                    st.session_state.generated_solution = None
                    raise e


if st.session_state.generated_solution:
    st.markdown("---")
    st.subheader("Generated Code Solution:")
    solution_code = st.session_state.generated_solution

    st.write("### Description:")
    st.info(solution_code['reasoning'])
    st.write("### Code Output:")
    st.write("### Code:")
    st.code(solution_code['code_generated'], language="python")


st.sidebar.markdown("---")
st.sidebar.subheader("Application Logs")
with st.sidebar.expander("View Logs"):
    for msg in st.session_state.log_messages:
        st.sidebar.write(msg)

st.markdown("---")
st.caption("Developed with LangChain, LangGraph, and Streamlit.")
