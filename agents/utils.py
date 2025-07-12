from pydantic import create_model, BaseModel
import re
from typing import Optional
from typing import Literal, get_args, get_origin, Any, Type, Dict, List, Tuple
from pydantic import create_model, BaseModel

def build_model_from_structure(
    name: str,
    structure: Dict[str, Any],
) -> Type[BaseModel]:
    fields: Dict[str, Tuple[Any, ...]] = {}
    for key, value in structure.items():
        # If the value is a dict, recurse to make a sub-model
        if isinstance(value, dict):
            submodel = build_model_from_structure(f"{name}_{key}", value)
            fields[key] = (submodel, ...)
        # If the value is a list, interpret it as a Literal choice
        elif isinstance(value, list) and all(isinstance(item, str) for item in value):
            literal_type = Literal[tuple(value)]  # e.g. Literal["success","update"]
            fields[key] = (literal_type, ...)
        # Otherwise fall back to str
        else:
            fields[key] = (str, ...)
    return create_model(name, **fields)


def parse_markdown_output(markdown: str) -> Optional[str]:
    """
    Extracts the content of the first triple-backtick code block from a Markdown string.

    Returns the code as a string, or None if no block is found.
    """
    match = re.search(r"```(?:\w+)?\n(.*?)```", markdown, re.DOTALL)
    if match:
        return match.group(1).strip()
    return None
