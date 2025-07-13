# app.py
from flask import Flask, request, jsonify, current_app
from flask_cors import CORS  # <- Add this
import os
import yaml
from datetime import datetime
from dotenv import load_dotenv
load_dotenv()  # Load environment variables from .env
# load config from .env or similar
from paths import OUTPUT_FILE
from agents.call_llm import generate_code_from_query
from ai_utils.openai_api import OpenAIModels
MODEL = OpenAIModels.O4_MINI
app = Flask(__name__)
CORS(app)  # <- Allow all origins by default
# have a state memory 


@app.route('/write_code', methods=['POST'])
def write_code_to_files():
    print(f'POST - WRITING CODE TO FILES')
    data = request.json or {}
    prompt = data.get('prompt')
    if not prompt:
        return jsonify({'error': 'Missing prompt'}), 400

    code_generated = generate_code_from_query(prompt, output_file=OUTPUT_FILE,dummy_code= False)
    return jsonify({'filename': str(OUTPUT_FILE.name)}),200

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)