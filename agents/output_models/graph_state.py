from typing import List, TypedDict
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
    messages: List[tuple[str, str]]  # List of tuples with role and content
    code_output: str
    iterations: int 