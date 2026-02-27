
import sqlite3
import datetime
import json
import numpy as np
from db import get_db_connection

def check_batch_ranking(user_id=1):
    print("=== Checking Ranking of Recent Batch ===")
    conn = get_db_connection()
    c = conn.cursor()
    
    # 1. Get User Profile Vector
    c.execute("SELECT preference_vector FROM user_profile WHERE user_id = ?", (user_id,))
    prof_row = c.fetchone()
    
    if not prof_row or not prof_row[0]:
        print("❌ FAIL: No user profile found. Ranking is IMPOSSIBLE.")
        conn.close()
        return

    user_vector = np.frombuffer(prof_row[0], dtype=np.float32)
    print(f"✅ User Profile Found (Norm: {np.linalg.norm(user_vector):.4f})")
    
    # 2. Re-fetch the 9 papers (Same logic as feed)
    c.execute("SELECT target_subjects FROM user_settings WHERE user_id = ?", (user_id,))
    settings = c.fetchone()
    target_subjects = json.loads(settings[0])
    topic_conditions = " OR ".join([f"category LIKE '%{subj}%'" for subj in target_subjects])
    
    # Get cutoff
    c.execute(f"SELECT MAX(published_date) as latest_date FROM papers WHERE ({topic_conditions})")
    latest_date = c.fetchone()['latest_date']
    latest_dt = datetime.datetime.fromisoformat(latest_date.replace('Z', '+00:00'))
    cutoff_dt = latest_dt - datetime.timedelta(hours=24)
    cutoff_date = cutoff_dt.isoformat()
    
    query = f"SELECT id, title, published_date, embedding FROM papers WHERE published_date >= ? AND ({topic_conditions})"
    c.execute(query, (cutoff_date,))
    rows = c.fetchall()
    
    papers = []
    for r in rows:
        emb = r['embedding']
        vec = np.frombuffer(emb, dtype=np.float32) if emb else None
        papers.append({
            'id': r['id'],
            'title': r['title'],
            'date': r['published_date'],
            'vector': vec
        })
        
    print(f"\nAnalyzing {len(papers)} papers in batch:")
    
    # 3. Calculate Scores
    ranked = []
    unranked = []
    
    for p in papers:
        if p['vector'] is not None:
            # Cosine
            dot = np.dot(user_vector, p['vector'])
            norm_u = np.linalg.norm(user_vector)
            norm_p = np.linalg.norm(p['vector'])
            score = dot / (norm_u * norm_p + 1e-9)
            
            p['score'] = score
            ranked.append(p)
        else:
            p['score'] = -1.0
            unranked.append(p)
            
    # Sort by Score Descending
    ranked.sort(key=lambda x: x['score'], reverse=True)
    
    # 4. Display Results
    print(f"\n--- Ranked Order (Top 5) ---")
    for i, p in enumerate(ranked):
        print(f"{i+1}. [{p['score']:.4f}] {p['title'][:60]}... ({p['date']})")
        
    if unranked:
        print(f"\n--- Unranked (No Embedding) ---")
        for p in unranked:
            print(f"- {p['title'][:60]}...")
            
    # 5. Check if Date Order matches Rank Order (Corollary check)
    # If they are perfectly sorted by date, maybe ranking failed?
    dates_sorted = sorted(ranked, key=lambda x: x['date'], reverse=True)
    date_match = [p['id'] for p in ranked] == [p['id'] for p in dates_sorted]
    
    if date_match:
        print("\n⚠️  WARNING: The ranked order is exactly the same as Date order.")
        print("   This implies ranking might not be effective or coefficient is too weak.")
    else:
        print("\n✅ SUCCESS: Ranking order differs from Date order. Relevance is active.")

    conn.close()

if __name__ == "__main__":
    try:
        check_batch_ranking()
    except Exception as e:
        print(f"Error: {e}")
