"""
Cleanup script to remove old papers from the database.
Only keeps:
1. Papers the user has interacted with (liked/disliked)
2. Recent papers from RSS feeds (last 30 days)
"""
import sqlite3
import datetime
from db import get_db_connection

def cleanup_old_papers(days_to_keep=30):
    """
    Remove papers that are:
    - Older than days_to_keep
    - Not interacted with by any user
    """
    conn = get_db_connection()
    c = conn.cursor()
    
    # Calculate cutoff date
    cutoff = (datetime.datetime.utcnow() - datetime.timedelta(days=days_to_keep)).isoformat()
    
    # Find papers to delete: old AND not interacted with
    c.execute("""
        SELECT p.id 
        FROM papers p
        LEFT JOIN interactions i ON p.id = i.paper_id
        WHERE p.published_date < ?
        AND i.paper_id IS NULL
    """, (cutoff,))
    
    papers_to_delete = [row[0] for row in c.fetchall()]
    
    if not papers_to_delete:
        print("No papers to clean up.")
        conn.close()
        return 0
    
    print(f"Deleting {len(papers_to_delete)} old papers...")
    
    # Delete from VSS first
    for paper_id in papers_to_delete:
        try:
            c.execute("SELECT rowid FROM papers WHERE id = ?", (paper_id,))
            row = c.fetchone()
            if row:
                c.execute("DELETE FROM vss_papers WHERE rowid = ?", (row[0],))
        except:
            pass
    
    # Delete from papers table
    placeholders = ','.join('?' * len(papers_to_delete))
    c.execute(f"DELETE FROM papers WHERE id IN ({placeholders})", papers_to_delete)
    
    conn.commit()
    conn.close()
    
    print(f"Deleted {len(papers_to_delete)} papers.")
    return len(papers_to_delete)

if __name__ == "__main__":
    cleanup_old_papers(days_to_keep=30)
