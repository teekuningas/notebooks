import requests

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
