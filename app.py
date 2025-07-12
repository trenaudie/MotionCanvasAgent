# app.py
from flask import Flask, request, jsonify, current_app
from flask_cors import CORS  # <- Add this
import os
import yaml
from datetime import datetime
from dotenv import load_dotenv
load_dotenv()  # Load environment variables from .env
# load config from .env or similar
from paths import AGENTS_PROMPTS_DIR, SCENES_DIR, get_next_scene_filepath
from agents.coder_general_agent import CoderGeneralAgent
from agents.verify_update_agent import VerifyUpdateAgent
from ai_utils.openai_api import OpenAIModels
MODEL = OpenAIModels.O4_MINI
app = Flask(__name__)
CORS(app)  # <- Allow all origins by default

print(f"SCENES_DIR: {SCENES_DIR}")
print(f"AGENTS_PROMPTS_DIR: {AGENTS_PROMPTS_DIR}")
os.makedirs(SCENES_DIR, exist_ok=True)


def run_code_generation(prompt,model:OpenAIModels,  dummy : bool = False) -> str:
    if dummy: return "dummy.tsx", "success"
    print(f'coding....')
    coder_general_agent = CoderGeneralAgent(model)
    code_v1, reasoning_v1 = coder_general_agent.generate_code(prompt)
    print(f'verifying....')
    verify_update_agent = VerifyUpdateAgent(OpenAIModels.GPT_4O_MINI)
    try : 
        assert code_v1 is not None 
    except:
        print(f'code_v1 is None !')
    code_v2, reasoning_v2 = verify_update_agent.verify_and_update(code_v1, reasoning_v1)
    return code_v2


@app.route('/write_code', methods=['POST'])
def write_code_to_files():
    print(f'POST - WRITING CODE TO FILES')
    data = request.json or {}
    prompt = data.get('prompt')
    if not prompt:
        return jsonify({'error': 'Missing prompt'}), 400

    code_generated = run_code_generation(prompt, model=MODEL)
    filepath_return = get_next_scene_filepath()
    with open(filepath_return, 'w') as f:
        f.write(code_generated)
    print(f'Scene2d written to {filepath_return}')
    return jsonify({'filename': str(filepath_return.name)}), 200

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)