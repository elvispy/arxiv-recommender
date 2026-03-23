import sys
import numpy as np
from transformers import AutoTokenizer, AutoModel
import adapters
import torch

model_name = "allenai/specter2_base"
adapter_name = "allenai/specter2"

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)
adapters.init(model)

loaded_name = model.load_adapter(adapter_name, source="hf", set_active=True)
model.active_adapters = loaded_name

text = "Attention Is All You Need[SEP]The dominant sequence transduction models are based on complex recurrent or convolutional neural networks..."
inputs = tokenizer(text, padding=True, truncation=True, return_tensors="pt", max_length=512)

with torch.no_grad():
    outputs = model(**inputs)
    
vec = outputs.last_hidden_state[:, 0, :].cpu().numpy()[0]
print("Shape:", outputs.last_hidden_state.shape)
print("Norm:", np.linalg.norm(vec))

from config import SEMANTIC_SCHOLAR_API_KEY
import requests

url = "https://api.semanticscholar.org/graph/v1/paper/ARXIV:1706.03762"
headers = {"x-api-key": SEMANTIC_SCHOLAR_API_KEY}
r = requests.get(url, params={"fields": "embedding.specter_v2"}, headers=headers).json()
if "embedding" in r and "vector" in r["embedding"]:
    remote_vec = np.array(r["embedding"]["vector"])
    print("Remote Norm:", np.linalg.norm(remote_vec))
    print("Diff Norm:", np.linalg.norm(vec - remote_vec) / np.linalg.norm(remote_vec))
else:
    print("SS API returned:", r)
