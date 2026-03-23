from transformers import AutoTokenizer
from adapters import AutoAdapterModel

model_name = "allenai/specter2_base"
adapter_name = "allenai/specter2"

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoAdapterModel.from_pretrained(model_name)
model.load_adapter(adapter_name, source="hf", load_as="specter2_proximity", set_active=True)

text = "Test paper title[SEP]Abstract content"
inputs = tokenizer(text, padding=True, truncation=True, return_tensors="pt", max_length=512)
outputs = model(**inputs)
print(outputs.last_hidden_state.shape)
