import json
import sys
import os
import torch
from transformers import AutoTokenizer, AutoModel

def run_inference(input_path, output_path, model_cache=None):
    # Load Specter2 with Proximity Adapter (allenai/specter2_base + allenai/specter2)
    model_name = "allenai/specter2_base"
    adapter_name = "allenai/specter2"
    
    print(f"Loading model {model_name} (Cache: {model_cache})...")
    tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=model_cache)
    model = AutoModel.from_pretrained(model_name, cache_dir=model_cache)
    
    # Check for adapters library
    try:
        import adapters
        adapters.init(model)
        loaded_adapter_name = model.load_adapter(adapter_name, source="hf", set_active=True, cache_dir=model_cache)
        model.set_active_adapters(loaded_adapter_name) # Explicitly set active
        print(f"Active adapters: {model.active_adapters}")
    except ImportError:
        print("Warning: 'adapters' library not found.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cpu":
        print("WARNING: CUDA not available, running on CPU. This will be slow.")
    model.to(device)
    model.eval()
    print(f"Using device: {device}")

    with open(input_path, 'r') as f:
        papers = json.load(f)

    results = []
    print(f"Processing {len(papers)} papers...")
    
    with torch.no_grad():
        for paper in papers:
            text = f"{paper['title']} [SEP] {paper['abstract']}"
            inputs = tokenizer(text, padding=True, truncation=True, return_tensors="pt", max_length=512).to(device)
            outputs = model(**inputs)
            # Specter2 uses [CLS] token (index 0)
            embedding = outputs.last_hidden_state[:, 0, :].cpu().numpy()[0]
            
            results.append({
                "id": paper["id"],
                "embedding": embedding.tolist()
            })

    with open(output_path, 'w') as f:
        json.dump(results, f)
    print(f"Results saved to {output_path}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("input", help="Input JSON path")
    parser.add_argument("output", help="Output JSON path")
    parser.add_argument("--model_cache", help="Custom model cache directory")
    args = parser.parse_args()
    
    run_inference(args.input, args.output, args.model_cache)
