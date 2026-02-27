"""
Update embeddings for existing papers that are missing them.
"""
import sys
sys.path.insert(0, '/Users/eaguerov/Documents/Github/arxiv-recommender')

from db import get_db_connection
from ingest import fetch_ss_embeddings_batch
import json

conn = get_db_connection()
c = conn.cursor()

# Get papers without embeddings
c.execute("""
    SELECT id, source, external_id, title
    FROM papers
    WHERE embedding IS NULL
    AND category IN ('physics.comp-ph', 'physics.flu-dyn')
    LIMIT 20
""")

papers = c.fetchall()
print(f"Found {len(papers)} papers missing embeddings in your topics\n")

if papers:
    # Convert to dict format
    papers_list = []
    for p in papers:
        papers_list.append({
            'id': p[0],
            'source': p[1],
            'external_id': p[2],
            'title': p[3]
        })
    
    # Fetch embeddings
    print("Fetching embeddings from Semantic Scholar...")
    emb_map = fetch_ss_embeddings_batch(papers_list)
    
    print(f"Got {len(emb_map)} embeddings\n")
    
    # Update database
    updated = 0
    for p in papers_list:
        pid = f"{p['source']}:{p['external_id']}"
        if pid in emb_map:
            c.execute("UPDATE papers SET embedding = ? WHERE id = ?", (emb_map[pid], p['id']))
            
            # Update VSS
            c.execute("SELECT rowid FROM papers WHERE id = ?", (p['id'],))
            row_id = c.fetchone()[0]
            try:
                c.execute('INSERT OR REPLACE INTO vss_papers(rowid, embedding) VALUES (?, ?)', 
                         (row_id, emb_map[pid]))
            except Exception as e:
                print(f"VSS error for {p['id']}: {e}")
            
            updated += 1
            print(f"✓ Updated: {p['title'][:60]}...")
    
    conn.commit()
    print(f"\nSuccessfully updated {updated} papers with embeddings")
else:
    print("No papers need embedding updates")

conn.close()
