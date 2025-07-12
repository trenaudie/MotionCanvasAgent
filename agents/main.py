import os
from agents.agent import Agent
from pathlib import Path


CHAT_FILE = "/home/bits/MotionCanvasAgent/agents/prompts/chat_message.txt"
OUTPUT_FILE = "/home/bits/MotionCanvasAgent/frontend/src/scenes/example.tsx"
DUMMY_CODE_FILE = "/home/bits/MotionCanvasAgent/frontend/src/scenes2/example9_circle_cat.tsx"


def run_code_generation(chat_messages: str, agent: Agent, output_file: os.PathLike, dummy_code: bool = False):
    """
    This function performs a code generation from a given conversation. 
    If dummy_code is true, it will just copy the code from the DUMMY_CODE_FILE into the output_file.

    :param chat_messages: The conversation messages (str).
    :param agent: The agent instance that generates the code.
    :param output_file: The file where the generated code will be written.
    :param dummy_code: If true, copy from the dummy code file. Default is True.
    """
    
    try:
        if not dummy_code:
            code_generated = agent.generate_code(chat_messages)
        else:
            code_generated = Path(DUMMY_CODE_FILE).read_text()

        # Ensure the output file path exists, create it if not
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Write the generated code to the output file
        with open(output_path, 'w') as f:
            f.write(code_generated)
        
        print(f"Code generation successful! Output written to {output_file}")
    
    except Exception as e:
        print(f"An error occurred during code generation: {e}")


# Example usage
if __name__ == "__main__":
    # Load chat messages (For now, let's just read from a file)
    # try:
    #     with open(CHAT_FILE, 'r') as f:
    #         chat_messages = f.read()
    #     print("Chat messages loaded.")
    # except FileNotFoundError:
    #     print(f"Error: The file {CHAT_FILE} was not found.")
    #     chat_messages = ""

    # # Example agent, replace with your actual agent class
    # agent = Agent()  # Assuming `Agent` is properly instantiated
    
    # # Run code generation
    # run_code_generation(chat_messages, agent, OUTPUT_FILE)

    run_code_generation("hello world", None,OUTPUT_FILE, dummy_code = True )
