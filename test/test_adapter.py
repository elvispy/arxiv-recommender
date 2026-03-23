from transformers import AutoTokenizer, AutoModel
import adapters

model_name = "allenai/specter2_base"
adapter_name = "allenai/specter2"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)
adapters.init(model)

loaded_name = model.load_adapter(adapter_name, source="hf", set_active=True)
print("Loaded adapter:", loaded_name)
try:
    print("Active adapters:", model.active_adapters)
except Exception as e:
    print("Error:", e)
    pass

text = "Test paper title[SEP]Abstract content"
inputs = tokenizer(text, padding=True, truncation=True, return_tensors="pt", max_length=512)
outputs = model(**inputs)
print(outputs.last_hidden_state.shape)
