from sentence_transformers import SentenceTransformer
import requests
import backoff
import torch
import os
emb_model = SentenceTransformer("all-MiniLM-L6-v2", model_kwargs={"torch_dtype": torch.float16})

def embedding_generator(text):
    return emb_model.encode(text)


api_key = os.environ['OPENAI_KEY']

url = "https://api.openai.com/v1/chat/completions"

headers = {
    "Authorization": f"Bearer {api_key}",
    "Content-Type": "application/json"
}

@backoff.on_exception(backoff.expo, requests.exceptions.HTTPError)
def text_generator(prompt):
    data = {
        "model": "gpt-4o-mini",
        "messages": [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt}
        ]
    }

    response = requests.post(url, headers=headers, json=data)
    response.raise_for_status()

    reply = response.json()["choices"][0]["message"]["content"]
    return reply