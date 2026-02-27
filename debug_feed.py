
import sqlite3
import json
import datetime
from db import get_db_connection

def debug_feed_count(user_id=1):
    conn = get_db_connection()
    c = conn.cursor()
    
    # 1. Get Settings
    c.execute("SELECT fetch_frequency, target_subjects FROM user_settings WHERE user_id = ?", (user_id,))
    row = c.fetchone()
    if not row:
        print("No settings found.")
        return
        
    target_subjects = json.loads(row['target_subjects']) if row['target_subjects'] else []
    print(f"Target subjects: {target_subjects}")
    
    if not target_subjects:
        print("No subjects selected.")
        return

    # 2. Latest Date Logic (Conditional)
    fetch_frequency = row['fetch_frequency']
    print(f"Fetch Frequency: {fetch_frequency}")
    
    cutoff_date = None
    limit_clause = ""
    
    # Define topic_conditions here so it's available in both branches
    topic_conditions = " OR ".join([f"category LIKE '%{subj}%'" for subj in target_subjects])
    
    if fetch_frequency == 'last_100':
        print("Mode: Last 100 (ignoring date window)")
        limit_clause = "LIMIT 100"
        query = f"SELECT id, title, published_date FROM papers WHERE ({topic_conditions}) ORDER BY published_date DESC {limit_clause}"
        c.execute(query)
    else:
        # Standard logic
        c.execute(f"SELECT MAX(published_date) as latest_date FROM papers WHERE ({topic_conditions})")
        latest_row = c.fetchone()
        latest_date = latest_row['latest_date']
        print(f"Latest paper date: {latest_date}")
        
        if not latest_date:
            print("No papers found.")
            return

        try:
            latest_dt = datetime.datetime.fromisoformat(latest_date.replace('Z', '+00:00'))
            cutoff_dt = latest_dt - datetime.timedelta(hours=24)
            cutoff_date = cutoff_dt.isoformat()
            print(f"Cutoff date (Latest - 24h): {cutoff_date}")
        except Exception as e:
            print(f"Date parsing error: {e}")
            cutoff_date = latest_date
            
        # 3. Fetch Candidates
        query = f"SELECT id, title, published_date FROM papers WHERE published_date >= ? AND ({topic_conditions})"
        c.execute(query, (cutoff_date,))

    rows = c.fetchall()
    
    print(f"\nTotal papers in batch: {len(rows)}")
    for r in rows:
        print(f"- {r['title'][:50]}... ({r['published_date']})")
        
    if len(rows) <= 12:
        print("\n--> RESULT: Count is <= 12. 'Show More' button will be HIDDEN.")
    else:
        print("\n--> RESULT: Count is > 12. 'Show More' button should satisfy condition.")

if __name__ == "__main__":
    debug_feed_count()
