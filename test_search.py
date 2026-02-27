# test_search.py
import json
import logging
from ingest import search_arxiv, search_semantic_scholar

# Set up logging to see what's happening
logging.basicConfig(level=logging.INFO)

def test():
    print("Testing ArXiv search...")
    # Trying the user's specific query
    arxiv_papers = search_arxiv(query="Bioreactor", author="Minki Kim", year_start="2024")
    print(f"ArXiv found {len(arxiv_papers)} papers.")
    for p in arxiv_papers:
        print(f" - [{p['source']}] {p['title']} ({p['external_id']})")

    print("\nTesting Semantic Scholar search...")
    ss_papers = search_semantic_scholar(query="Bioreactor", limit=5)
    print(f"Semantic Scholar found {len(ss_papers)} papers.")
    for p in ss_papers:
        has_emb = "Yes" if p.get('ss_embedding') else "No"
        print(f" - [{p['source']}] {p['title']} ({p['external_id']}) - Has SS Embedding: {has_emb}")

if __name__ == "__main__":
    test()
