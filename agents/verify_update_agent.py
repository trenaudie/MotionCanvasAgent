#%%
from agents.agents import Agent
from pathlib import Path
import os
import yaml
from paths import AGENTS_PROMPTS_DIR
from agents.utils import build_model_from_structure, parse_markdown_output
from ai_utils.openai_api import chat_completion_structured, OpenAIModels

assert os.getenv("OPENAI_API_KEY") is not None, "OPENAI_API_KEY is not set"

class VerifyUpdateAgent(Agent):
    def __init__(self, model: OpenAIModels = OpenAIModels.GPT_4_1_NANO):
        # Initialize with the prompt YAML for the VerifyUpdateAgent
        prompt_path = os.path.join(AGENTS_PROMPTS_DIR, 'verify_update_agent.yaml')
        super().__init__('verify_update_agent', prompt_path)
        self.model = model
        # Load and parse the YAML to get the output_template and other config
        yaml_data = yaml.safe_load(open(prompt_path))
        self.system_base = yaml_data.get("system_prompt", "")
        self.examples = yaml_data.get("examples", [])
        output_template_str = yaml_data["output_template"]
        output_template_dict = yaml.safe_load(output_template_str)
        
        # Build a Pydantic model to parse structured responses
        self.ResponseModel = build_model_from_structure("VerifyUpdateResponse", output_template_dict)
        self.output_template = output_template_str

    def build_system_prompt(self) -> str:
        """
        Construct the system prompt by combining the base instructions,
        any examples defined in the YAML (if present), and the critical
        output template reminder. No extra examples are loaded from files.
        """
        parts = [self.system_base.strip()]
        if self.examples:
            formatted = "\n\n".join(
                f"INPUT: {ex['input']}\nOUTPUT:\n{ex['output']}"
                for ex in self.examples
            )
            parts.append(f"EXAMPLES:\n{formatted}")
        parts.append(
            "OUTPUT TEMPLATE (CRITICAL):\n"
            "You must follow this exact output template, including Reasoning and Output sections:\n"
            f"{self.output_template.strip()}"
        )
        return "\n\n".join(parts)

    def verify_and_update(self, file_content: str, issues_description: str = "") -> (str, str):
        """
        Verify the given code file content, apply updates to fix errors,
        and return the rewritten file plus the agent's reasoning.
        
        :param file_content: The full source code to verify/update.
        :param issues_description: Optional textual description of known issues.
        :return: Tuple of (updated_file_content, reasoning).
        """
        system_prompt = self.build_system_prompt()
        
        # Build the user prompt
        user_prompt = f"```typescript\n{file_content}\n```"
        
        # Call the model
        response = chat_completion_structured(
            self.model,
            system_prompt,
            user_prompt,
            self.ResponseModel
        )
        
        # Extract and parse
        reasoning = response.Reasoning
        raw_code = response.Output.code
        status = response.Output.status
        print(f'received status: {status}')
        print(f'received code: {raw_code[:100]}...')
        if status=="success":
            print(f'returning code: {file_content[:100]}...')
            return file_content, reasoning
        updated_file = parse_markdown_output(raw_code)
        return updated_file, status
