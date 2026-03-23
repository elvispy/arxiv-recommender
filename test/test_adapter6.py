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
model.eval()

text = "Attention Is All You Need[SEP]The dominant sequence transduction models are based on complex recurrent or convolutional neural networks..."
inputs = tokenizer(text, padding=True, truncation=True, return_tensors="pt", max_length=512)

with torch.no_grad():
    outputs = model(**inputs)
    
vec = outputs.last_hidden_state[:, 0, :].cpu().numpy()[0]
print("Shape:", outputs.last_hidden_state.shape)
print("Norm:", np.linalg.norm(vec))

# Now try with context block for Active adapters
with adapters.AdapterSetup(loaded_name):
    outputs_context = model(**inputs)
    
vec_ctx = outputs_context.last_hidden_state[:, 0, :].cpu().numpy()[0]
print("Norm with AdapterSetup:", np.linalg.norm(vec_ctx))
