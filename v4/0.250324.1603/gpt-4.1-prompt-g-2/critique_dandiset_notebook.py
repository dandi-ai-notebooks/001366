import os
import json
import time
import requests
import argparse
from pathlib import Path
from typing import Dict, Any
from typing import List, Tuple

prompt_version = 1
model_for_critique = "anthropic/claude-3.7-sonnet"
verbose = False

def create_message_content_for_cell(cell: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Create user message content for a given cell."""
    content: List[Dict[str, Any]] = []
    if cell["cell_type"] == "markdown":
        markdown_source = cell["source"]
        content.append(
            {"type": "text", "text": "INPUT-MARKDOWN: " + "".join(markdown_source)}
        )
    elif cell["cell_type"] == "code":
        code_source = cell["source"]
        content.append({"type": "text", "text": "INPUT-CODE: " + "".join(code_source)})
        for x in cell["outputs"]:
            output_type = x["output_type"]
            if output_type == "stream":
                text = "OUTPUT-TEXT: " + "\n".join(x["text"])
                if len(text) > 20_000:
                    text = text[:20_000] + " [OUTPUT-TRUNCATED]"
                content.append(
                    {"type": "text", "text": text}
                )
            elif output_type == "display_data" or output_type == "execute_result":
                if "image/png" in x["data"]:
                    png_base64 = x["data"]["image/png"]
                    image_data_url = f"data:image/png;base64,{png_base64}"
                    content.append({"type": "text", "text": "OUTPUT-IMAGE:"})
                    content.append(
                        {"type": "image_url", "image_url": {"url": image_data_url}}
                    )
                elif "text/plain" in x["data"]:
                    content.append(
                        {
                            "type": "text",
                            "text": "OUTPUT-TEXT: " + "".join(x["data"]["text/plain"]),
                        }
                    )
                elif "text/html" in x["data"]:
                    content.append(
                        {
                            "type": "text",
                            "text": "OUTPUT-TEXT: " + "".join(x["data"]["text/html"]),
                        }
                    )
                else:
                    print(
                        f"Warning: got output type {output_type} but no image/png data or text/plain or text/html"
                    )
            else:
                print(f"Warning: unsupported output type {output_type}")
    else:
        print(f'Warning: unsupported cell type {cell["cell_type"]}')
        content.append({"type": "text", "text": "Unsupported cell type"})
    return content

def run_completion(
    messages: List[Dict[str, Any]],
    *,
    model: str
) -> Tuple[str, List[Dict[str, Any]], int, int]:
    """Execute an AI completion request using the OpenRouter API

    This function manages a conversation with an AI model

    Args:
        messages: List of conversation messages, each being a dictionary with role and content.
        model: Name of the OpenRouter model to use for completion.

    Returns:
        tuple: Contains:
            - content (str): The final response content from the model
            - conversation_messages (List): Complete conversation history
            - total_prompt_tokens (int): Total tokens used in prompts
            - total_completion_tokens (int): Total tokens used in completions

    Raises:
        ValueError: If OPENROUTER_API_KEY environment variable is not set
        RuntimeError: If the OpenRouter API request fails

    Notes:

    The OPENROUTER_API_KEY environment variable must be set with a valid API key from OpenRouter.

    The messages is a list of dicts with the following structure:
    [
        {"role": "system", "content": "You are a helpful assistant... etc."},
        {"role": "user", "content": "I need help with... etc."},
        {"role": "assistant", "content": "I can help with that... etc."},
        {"role": "user", "content": "Yes, please... etc."},
        ...
    ]
    """
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        raise ValueError("OPENROUTER_API_KEY environment variable not set")

    url = "https://openrouter.ai/api/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "HTTP-Referer": "https://neurosift.app",
        "Content-Type": "application/json"
    }

    conversation_messages = [m for m in messages]

    total_prompt_tokens = 0
    total_completion_tokens = 0

    while True:
        # Make API request
        payload = {
            "model": model,
            "messages": conversation_messages
        }
        if verbose:
            print(f"Using model: {payload['model']}")
            print(f"Num. messages in conversation: {len(conversation_messages)}")

        if verbose:
            print("Submitting completion request...")
        response = requests.post(url, headers=headers, json=payload)
        if response.status_code != 200:
            raise RuntimeError(f"OpenRouter API request failed: {response.text}")

        if verbose:
            print("Processing response...")
        completion = response.json()
        prompt_tokens = completion["usage"]["prompt_tokens"]
        completion_tokens = completion["usage"]["completion_tokens"]

        total_prompt_tokens += prompt_tokens
        total_completion_tokens += completion_tokens

        message = completion["choices"][0]["message"]
        content: str = message.get("content", "")

        # Track assistant response
        current_response = {
            "role": "assistant",
            "content": content
        }
        conversation_messages.append(current_response)

        return content, conversation_messages, total_prompt_tokens, total_completion_tokens

def critique_dandiset_notebook(notebook_path: str):
    with open(notebook_path, "r") as f:
        notebook_content = f.read()
    notebook = json.loads(notebook_content)

    if not "cells" in notebook:
        raise Exception(f"Invalid notebook format. No cells found in the notebook.")

    cells = notebook["cells"]

    system_prompt = """
You are an expert in Jupyter notebooks and Python programming.

The user is going to provide the cells of a Jupyter notebook one by one.
You are then going to provide a critique of the notebook based on the below criteria.
Your critique will then be used to improve the notebook.
Be sure to point out any serious problems with the notebook such as figures that are empty or not displayed correctly.

The ideal notebook will show the user how to get started exploring the dandiset using Python. It should include the following elements:
- A title that include the name of the Dandiset
- A message that indicates the notebook is AI-generated and has not been fully verified.
- An overview of the Dandiset, including a link to the Dandiset on the DANDI archive.
- A summary of what the notebook will cover.
- A list of required packages.
- Instructions on how to load the Dandiset using the DANDI API.
- Instructions on how to load one of the NWB files in the Dandiset and show some metadata.
- A description of what data are available in the NWB file
- Instructions on how to load and visualize the different types of data in the NWB file
- Perhaps a more advanced visualization involving more than one piece of data
- A summary of the findings and possible future directions for analysis.
- Explanatory markdown cells that guide the user through the analysis process.

The notebook should include well-documented code and follow best practices for neurophysiology data analysis.

The notebook should focus on the basics of getting started with the dandiset and should not include overanalysis or overinterpretation of the data.

All of the visualizations should be clear and free from errors or misleading displays.

Keep in mind the following guiding questions for a good notebook:

How well did the notebook help you understand the purpose and content of the Dandiset?
After reviewing the notebook, do you feel confident in how to access the different types of data from this Dandiset?
Did the notebook help you understand the structure of the NWB file(s) and how to work with them?
Did the visualizations in the notebook generally help you understand key aspects of the data?
Did any of the visualizations make it harder to understand the data (e.g., due to poor formatting, unclear axes, or misleading displays)?
Do you feel more confident creating your own visualizations of the data after seeing the examples in the notebook?
How well did the visualizations show the structure or complexity of the data?
Were there any interpretations or conclusions in the notebook that felt unclear or not well supported by the data shown?
Did any of the plots or examples feel unnecessarily repetitive or redundant?
Did the notebook help you understand what kinds of questions or analyses you could do next with this Dandiset?
How clear and easy was the notebook to follow?
Did the notebook provide code you could easily reuse or adapt to explore the Dandiset yourself?
Did the notebook help you understand what kinds of questions or analyses you could do next with this Dandiset?
Overall, how helpful was this notebook for getting started with this Dandiset?

Note: Warnings in the output cells should not be considered to be problems unless they relate to other problems with the notebook.
"""
    messages: List[Dict[str, Any]] = [
        {
            "role": "system",
            "content": system_prompt,
        }
    ]
    for cell in cells:
        content = create_message_content_for_cell(cell)
        messages.append({"role": "user", "content": content})
    messages.append({
        "role": "user",
        "content": "Please provide a critique of the above notebook according to the instructions in the system message."
    })
    assistant_response, _, prompt_tokens, completion_tokens = (
        run_completion(messages=messages, model=model_for_critique)
    )
    return assistant_response, prompt_tokens, completion_tokens

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Critique a Jupyter notebook.")
    parser.add_argument(
        "notebook_path",
        type=str,
        help="Path to the Jupyter notebook file to critique.",
    )
    args = parser.parse_args()
    notebook_path = args.notebook_path
    if not os.path.exists(notebook_path):
        raise FileNotFoundError(f"Notebook file {notebook_path} does not exist.")
    if not notebook_path.endswith(".ipynb"):
        raise ValueError(f"File {notebook_path} is not a valid Jupyter notebook file.")
    if verbose:
        print(f"Critiquing notebook {notebook_path}")
    assistant_response, prompt_tokens, completion_tokens = critique_dandiset_notebook(notebook_path)
    print(f'<prompt_tokens>{prompt_tokens}</prompt_tokens>')
    print(f'<completion_tokens>{completion_tokens}</completion_tokens>')
    print("")
    print(assistant_response)
