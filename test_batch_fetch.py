"""
Debug: Test the actual fetch_ss_embeddings_batch function
"""
import sys
sys.path.insert(0, '/Users/eaguerov/Documents/Github/arxiv-recommender')

from ingest import fetch_ss_embeddings_batch

# Test with papers from RSS
test_papers = [
    {'source': 'arxiv', 'external_id': '2601.04113v1'},
    {'source': 'arxiv', 'external_id': '2601.04099v1'},
    {'source': 'arxiv', 'external_id': '2601.04023v1'},
]

print("Testing fetch_ss_embeddings_batch...")
print(f"Input: {len(test_papers)} papers\n")

result = fetch_ss_embeddings_batch(test_papers)

print(f"\nResult: {len(result)} embeddings fetched")
print(f"Keys in result: {list(result.keys())}")

for pid, emb_bytes in result.items():
    print(f"  {pid}: {len(emb_bytes)} bytes")

if len(result) == 0:
    print("\n❌ ERROR: No embeddings were fetched!")
    print("This explains why papers have no emb eddings in the feed.")
else:
    print(f"\n✓ {len(result)}/{len(test_papers)} embeddings fetched successfully")
