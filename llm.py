import requests
import json

def generate_simple(instruction, content, model=None, seed=0, temperature=0.8, n_context=None, output_format=None, provider="ollama", timeout=None):
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
    
    if provider == "llamacpp":
        if model is not None:
            print("warning: llamacpp does not respect the model parameter")
        return generate_llamacpp(messages, model=model, seed=seed, temperature=temperature, output_format=output_format, timeout=timeout)
    elif provider == "ollama":
        return generate_ollama(messages, model=model, seed=seed, temperature=temperature, n_context=n_context, output_format=output_format, timeout=timeout)
    else:
        raise ValueError(f"Unknown provider: {provider}")


def generate_ollama(messages, model="llama3.3:70b", seed=0, temperature=0.8, n_context=None, output_format=None, timeout=None):
    if n_context is None:
        n_context = 8192
        
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

    response = requests.post(api_url, headers=headers, json=data, timeout=timeout) 
    
    if response.status_code == 200:
        # Return the message content directly
        return response.json()['message']['content']
    else:
        response.raise_for_status()

def embed(prompt, model=None, seed=0, provider="ollama"):
    if provider == "llamacpp":
        if model is not None:
             print("warning: llamacpp does not respect the model parameter")
        return embed_llamacpp(prompt)
    elif provider == "ollama":
        if model is None:
            model = "snowflake-arctic-embed2"
        return embed_ollama(prompt, model=model, seed=seed)
    else:
        raise ValueError(f"Unknown provider for embed: {provider}")


def embed_ollama(prompt, model="snowflake-arctic-embed2", seed=0):
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
        return response.json()['embedding']
    else:
        response.raise_for_status()


def embed_llamacpp(prompt):
    api_url = "http://localhost:8081/v1/embeddings"
    headers = {
        "Content-Type": "application/json"
    }
    data = {
        "input": prompt,
    }
    
    response = requests.post(api_url, headers=headers, json=data) 
    
    if response.status_code == 200:
        return response.json()['data'][0]['embedding']
    else:
        response.raise_for_status()

def generate_llamacpp(messages, model="gpt-3.5-turbo", seed=0, temperature=0.8, output_format=None, timeout=None):
    api_url = "http://localhost:8080/v1/chat/completions"
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
    
    if output_format:
        if isinstance(output_format, dict):
            data['response_format'] = {
                "type": "json_schema",
                "json_schema": {
                    "name": "structured_output",
                    "strict": True,
                    "schema": output_format
                }
            }
        elif output_format == 'json':
            data['response_format'] = {"type": "json_object"}

    response = requests.post(api_url, headers=headers, json=data, timeout=timeout) 
    
    if response.status_code == 200:
        response_json = response.json()
        return response_json['choices'][0]['message']['content']
    else:
        response.raise_for_status()
