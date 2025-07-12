import os 
from paths import AGENTS_PROMPTS_DIR
import yaml

class Agent:
    def __init__(self, name : str, yaml_template_path : str):
        self.name = name
        self.yaml_template_path = yaml_template_path
        with open(yaml_template_path, 'r') as f:
            self.yaml_template = yaml.safe_load(f)
    def apply_prompt(self, prompt : str):
        self.yaml_template['prompt'] = prompt
    def generate_code(self, prompt : str, dummy : bool = False):
        # replace the template with the prompt
        if dummy:
            return "dummy.tsx", "success"
        else:
            return "dummy.tsx", "success"

