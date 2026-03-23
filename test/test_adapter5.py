from transformers import AutoTokenizer, AutoModel
import adapters
import torch

model_name = "allenai/specter2_base"
adapter_name = "allenai/specter2"

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)
adapters.init(model)
model.load_adapter(adapter_name, source="hf", set_active=True)

text = "Test paper title[SEP]Abstract content"
inputs = tokenizer(text, padding=True, truncation=True, return_tensors="pt", max_length=512)
with torch.no_grad():
    outputs = model(**inputs)
    print("Normal outputs shape:", outputs.last_hidden_state.shape)
