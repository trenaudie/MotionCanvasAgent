from pydantic import BaseModel, Field

class CodeOutput(BaseModel):
    code_generated: str = Field(description="The code to be generated")
    reasoning: str = Field(description="The reasoning for the code")
