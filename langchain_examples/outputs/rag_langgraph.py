import getpass
import os

# Set environment variables for API keys

def _set_if_undefined(var: str) -> None:
    if os.environ.get(var):
        return
    os.environ[var] = getpass.getpass(var)

_set_if_undefined("OPENAI_API_KEY")
_set_if_undefined("TAVILY_API_KEY")

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.output_parsers import StrOutputParser
from langgraph.graph import StateGraph, END, START
from langgraph.graph.message import add_messages
from typing_extensions import TypedDict
from typing import Annotated, List

# Load documents from URLs
urls = [
    "https://lilianweng.github.io/posts/2023-06-23-agent/",
    "https://lilianweng.github.io/posts/2023-03-15-prompt-engineering/",
    "https://lilianweng.github.io/posts/2023-10-25-adv-attack-llm/",
]

docs = [WebBaseLoader(url).load() for url in urls]
docs_list = [item for sublist in docs for item in sublist]

# Split documents into chunks
text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
    chunk_size=250, chunk_overlap=0
)
doc_splits = text_splitter.split_documents(docs_list)

# Create a vector store for document retrieval
vectorstore = Chroma.from_documents(
    documents=doc_splits,
    collection_name="rag-chroma",
    embedding=OpenAIEmbeddings(),
)
retriever = vectorstore.as_retriever()

# Define the state for the graph
class State(TypedDict):
    messages: Annotated[List[AIMessage], add_messages]

# Function to retrieve documents

def retrieve(state: State):
    question = state["messages"][-1].content
    documents = retriever.invoke(question)
    return {"messages": documents}

# Function to generate an answer

def generate(state: State):
    question = state["messages"][-1].content
    documents = state["messages"][:-1]  # Exclude the last message (the question)
    generation = rag_chain.invoke({"context": documents, "question": question})
    return {"messages": [AIMessage(content=generation)]}

# Create the graph
workflow = StateGraph(State)
workflow.add_node("retrieve", retrieve)
workflow.add_node("generate", generate)
workflow.add_edge(START, "retrieve")
workflow.add_edge("retrieve", "generate")
workflow.add_edge("generate", END)

# Compile the graph
app = workflow.compile()

# Run the RAG process
inputs = {"messages": [HumanMessage(content="What are the types of agent memory?")]}
for output in app.stream(inputs):
    for key, value in output.items():
        print(f"Node '{key}':")
        print(value)
    print("---")

# Final generation
print(value["messages"][-1].content)