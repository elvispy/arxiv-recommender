"""
Test Semantic Scholar batch API to see why papers aren't being found.
"""
import requests
from config import SEMANTIC_SCHOLAR_API_KEY

# Test with recent ArXiv IDs
test_ids = [
    "ARXIV:2601.04113",  # From Jan 7
    "ARXIV:2601.04099",
    "ARXIV:2601.04023",
]

url = "https://api.semanticscholar.org/graph/v1/paper/batch"
params = {"fields": "paperId,title,embedding.specter_v2,externalIds"}
headers = {"x-api-key": SEMANTIC_SCHOLAR_API_KEY}

print(f"Testing {len(test_ids)} recent ArXiv papers in Semantic Scholar...")
print(f"IDs: {test_ids}\n")

try:
    r = requests.post(url, json={"ids": test_ids}, params=params, headers=headers, timeout=30)
    print(f"Status: {r.status_code}")
    
    if r.status_code == 200:
        data = r.json()
        print(f"Response contains {len(data)} items\n")
        
        for idx, item in enumerate(data):
            req_id = test_ids[idx]
            print(f"[{idx+1}] {req_id}:")
            
            if item is None:
                print("   ❌ NOT FOUND in Semantic Scholar")
                print("   → This paper hasn't been indexed yet (too recent)")
            else:
                print(f"   ✓ Found: {item.get('title', 'N/A')[:60]}...")
                print(f"   Paper ID: {item.get('paperId')}")
                
                has_embedding = bool(item.get('embedding', {}).get('vector'))
                print(f"   Has embedding: {'✓ YES' if has_embedding else '✗ NO'}")
                
                doi = item.get('externalIds', {}).get('DOI')
                print(f"   DOI: {doi if doi else 'None'}")
            print()
    else:
        print(f"Error: {r.text}")
        
except Exception as e:
    print(f"Exception: {e}")

print("\n" + "="*60)
print("DIAGNOSIS:")
print("="*60)
print("If papers show 'NOT FOUND', it means Semantic Scholar hasn't")
print("indexed them yet. Jan 6-7 papers might be too recent.")
print("\nSOLUTION: We need a fallback strategy for very recent papers:")
print("1. Wait a few days for SS to index them")
print("2. Use DOI lookup (but ArXiv RSS doesn't provide DOIs)")
print("3. Accept that very recent papers won't have embeddings immediately")
