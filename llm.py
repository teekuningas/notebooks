import requests

def generate_simple(model, instruction, data, seed=0, temperature=0.8, n_context=8192, output_format=None):
    messages = [
        {
            "role": "system",
            "content": instruction   
        },
        {
            "role": "user",
            "content": data
        }
    
    ]
    return generate(model, messages, seed, temperature, n_context, output_format)

def generate(model, messages, seed=0, temperature=0.8, n_context=8192, output_format=None):
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
        return response.json()
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
