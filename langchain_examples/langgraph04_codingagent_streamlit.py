import streamlit as st
import os
import getpass
from bs4 import BeautifulSoup as Soup
from langchain_community.document_loaders.recursive_url_loader import RecursiveUrlLoader
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field
from langchain_anthropic import ChatAnthropic # Though not used in final chain, keeping for context
from langchain_google_genai import ChatGoogleGenerativeAI
from pprint import pprint
import json
from typing import List
from typing_extensions import TypedDict
from langchain_community.document_loaders.git import GitLoader
from langgraph.graph import END, StateGraph, START
from dotenv import load_dotenv
import sys 
parent_dir = "/home/bits/MotionCanvasAgent"
print(f"Adding parent directory to sys.path: {parent_dir}")
sys.path.append(parent_dir)
from logging_config.logger import LOG
import shutil
import tempfile
load_dotenv()
assert os.getenv("OPENAI_API_KEY") is not None, "OPENAI_API_KEY environment variable not set. Please set it to run the LLM."
import git # For cloning the repository
from project_paths import PROJECT_ROOT

@st.cache_data(show_spinner="Loading and processing documentation...")
def load_and_process_docs():
    LOG.info("Loading and processing documentation from langgraph repository...")
    repo_url = "https://github.com/langchain-ai/langgraph.git"
    target_dir = PROJECT_ROOT / "trenaudie"

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

        # Filter relevant docs
        docs_to_keep = [
            doc for doc in docs
            if any(k in doc.metadata["source"] for k in ["chatbot", "multi-agent-rag", "ref"])
        ][:3]

        concatenated_content = "\n\n\n --- \n\n\n".join(doc.page_content for doc in docs_to_keep)
        return concatenated_content

    except Exception as e:
        st.error(f"An error occurred during documentation loading: {e}")
        st.stop()

class code(BaseModel):
    """Schema for code solutions to questions about LCEL."""
    reasoning: str = Field(description="Description of the problem and approach")
    code: str = Field(description="Code block not including import statements")


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
    generation: code # Changed to 'code' type for clarity
    iterations: int


def build_openai_chain(model_name: str = "gpt-4o-mini"):
    code_gen_prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """You are a coding assistant with expertise in LangGraph, LangChain expression language. \n
                Here is a full set of LangGraph documentation:  \n ------- \n  {context} \n ------- \n Answer the user
                question based on the above provided documentation, by giving your reasoning then the code. Ensure any code you provide can be executed \n. Structure your answer with a description of the code solution. \n
                Then list the imports. And finally list the functioning code block. Here is the user question:""",
            ),
            ("placeholder", "{messages}"),
        ]
    )
    llm = ChatOpenAI(temperature=0, model=model_name)
    return code_gen_prompt | llm.with_structured_output(code)

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
    st.session_state.log_messages.append("---GENERATING CODE SOLUTION---")

    # State
    messages = state["messages"]
    iterations = state["iterations"]
    error = state["error"]

    code_gen_chain = build_openai_chain() # dont give the option

    LOG.warning('truncated content to 1000 characters')
    st.warning('truncated content to 1000 characters')
    try:
        code_solution = code_gen_chain.invoke(
            {"context": concatenated_content[:1000], "messages": messages}, config =thread
        )
    except Exception as e:
        st.session_state.log_messages.append(f"Error during code generation: {e}")
        # In a real app, you might want to handle this more gracefully,
        # perhaps by setting an error flag in the state and retrying or informing the user.
        return {"generation": code("","",""), "messages": messages, "iterations": iterations, "error": "yes"}


    messages.append(
        (
            "assistant",
            f"{code_solution.reasoning} \n Code: {code_solution.code}",
        )
    )

    # Increment
    iterations = iterations + 1
    return {"generation": code_solution, "messages": messages, "iterations": iterations, "error": "no"}


@st.cache_resource(show_spinner=True)
def memory(thread_id: dict):
    st.write("Initializing memory for the LangGraph workflow...")
    from langgraph.checkpoint.memory import MemorySaver
    memory = MemorySaver()
    LOG.info(f"Memory initialized with thread ID: {thread_id}")
    return memory



# Initialize LangGraph workflow
@st.cache_resource(show_spinner=True)
def build_graph(thread: dict):
    st.warning(f'Compiling graph...')
    workflow = StateGraph(GraphState)
    workflow.add_node("generate", generate)  # generation solution
    workflow.add_edge(START, "generate")
    workflow.add_edge("generate", END)
    app = workflow.compile(checkpointer=memory(thread_id=thread["configurable"]["thread_id"]))
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
app = build_graph(thread)

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
    current_state = app.get_state(config=thread)
    if current_state:
        LOG.info(f"Current state found: {current_state}")
        current_state_messages = current_state.values.get('messages', [])
        LOG.info(f"Current messages in state: {current_state_messages}")
    else:
        current_state_messages = []
    current_state = {
        "messages": current_state_messages + [("user", question)],
        "iterations": 0,
        "error": "",
        "generation": code(reasoning="", code="") # Initialize generation
    }
    solution = app.invoke(current_state, config=thread)
    LOG.info(f"Graph response: {solution}")
    # Return a serializable dictionary instead of the BaseModel instance
    generation = solution["generation"]
    return {
        "reasoning": generation.reasoning,
        "code": generation.code
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
    st.code(solution_code['code'], language="python")


st.sidebar.markdown("---")
st.sidebar.subheader("Application Logs")
with st.sidebar.expander("View Logs"):
    for msg in st.session_state.log_messages:
        st.sidebar.write(msg)

st.markdown("---")
st.caption("Developed with LangChain, LangGraph, and Streamlit.")
