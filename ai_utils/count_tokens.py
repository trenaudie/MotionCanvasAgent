import sys
try:
    import tiktoken
except ImportError:
    raise ImportError("Please install tiktoken: pip install tiktoken")

def count_openai_tokens(text, model="gpt-4o-mini"):
    """
    Count the number of tokens in a string for a given OpenAI model.
    """
    enc = tiktoken.encoding_for_model(model)
    return len(enc.encode(text))

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Count OpenAI tokens in a file or string.")
    parser.add_argument("--file", type=str, help="Path to file to count tokens.")
    parser.add_argument("--string", type=str, help="String to count tokens.")
    parser.add_argument("--model", type=str, default="gpt-3.5-turbo", help="OpenAI model name.")
    args = parser.parse_args()

    if args.file:
        with open(args.file, "r", encoding="utf-8") as f:
            text = f.read()
        print(count_openai_tokens(text, args.model))
    elif args.string:
        print(count_openai_tokens(args.string, args.model))
    else:
        print("Provide --file or --string argument.")
