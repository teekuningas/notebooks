import requests
import json

def generate_simple(instruction, content, model="llama3.3:70b", seed=0, temperature=0.8, n_context=8192, output_format=None):
    # model = "mistral-large"
    # model = "deepseek-r1:70b"
    messages = [
        {
            "role": "system",
            "content": instruction   
        },
        {
            "role": "user",
            "content": content
        }
    
    ]
    return generate(messages, model=model, seed=seed, temperature=temperature, n_context=n_context, output_format=output_format)


def generate(messages, model="llama3.3:70b", seed=0, temperature=0.8, n_context=8192, output_format=None):
    # model = "mistral-large"
    # model = "deepseek-r1:70b"
    api_url = "https://jyu2401-62.tail5b278e.ts.net/ollamapi/api/chat"
    headers = {
        "Content-Type": "application/json"
    }
    data = {
        "model": model,
        "messages": messages,
        "options": {
            "seed": seed,
            "temperature": temperature,
            "num_ctx": n_context
        },
        "stream": False,
        "format": output_format
    }

    response = requests.post(api_url, headers=headers, json=data) 
    
    if response.status_code == 200:
        # Return the message content directly
        return response.json()['message']['content']
    else:
        response.raise_for_status()

def embed(prompt, model="snowflake-arctic-embed2", seed=0):
    # model = "bge-m3"
    # model = "paraphrase-multilingual"
    # model = "mxbai-embed-large"
    # model = "nomic-embed-text"
    api_url = "https://jyu2401-62.tail5b278e.ts.net/ollamapi/api/embeddings"
    headers = {
        "Content-Type": "application/json"
    }
    data = {
        "model": model,
        "prompt": prompt,
        "options": {
            "seed": seed,
        }
    }
    
    response = requests.post(api_url, headers=headers, json=data) 
    
    if response.status_code == 200:
        return response.json()
    else:
        response.raise_for_status()

def generate_llamacpp(messages, model="gpt-3.5-turbo", seed=0, temperature=0.8, output_format=None):
    api_url = "https://jyu2401-62.tail5b278e.ts.net/llama-cpp/v1/chat/completions"
    headers = {
        "Content-Type": "application/json"
    }
    data = {
        "model": model,
        "messages": messages,
        "seed": seed,
        "temperature": temperature,
        "stream": False,
    }
    
    use_tools = False
    if output_format:
        if isinstance(output_format, dict):
            data['tools'] = [
                {
                    "type": "function",
                    "function": {
                        "name": "structured_output",
                        "description": "Generate a structured output based on the provided schema.",
                        "parameters": output_format
                    }
                }
            ]
            data['tool_choice'] = {"type": "function", "function": {"name": "structured_output"}}
            use_tools = True
        elif output_format == 'json':
            data['response_format'] = {"type": "json_object"}

    response = requests.post(api_url, headers=headers, json=data) 
    
    if response.status_code == 200:
        response_json = response.json()
        
        # If tools were used, extract the arguments from the tool call.
        if use_tools:
            tool_calls = response_json['choices'][0]['message'].get('tool_calls')
            if tool_calls:
                arguments = tool_calls[0]['function']['arguments']
                return arguments

        # Otherwise, return the message content directly.
        return response_json['choices'][0]['message']['content']
    else:
        response.raise_for_status()
