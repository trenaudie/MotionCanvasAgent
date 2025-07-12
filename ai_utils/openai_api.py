from openai import OpenAI
from pydantic import BaseModel
from enum import Enum
from typing_extensions import Type
client = OpenAI()

class OpenAIModels(Enum):
    GPT_4_1 = "gpt-4.1"
    GPT_4O_MINI = "gpt-4o-mini"
    O4_MINI = "o4-mini"
    GPT_4_1_NANO = "gpt-4.1-nano"

def chat_completion_structured(model: OpenAIModels, system_prompt: str, user_prompt: str, text_format: Type[BaseModel]):
    response = client.responses.parse(
        model=model.value,
        input=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        text_format=text_format,
    )
    return response.output_parsed
