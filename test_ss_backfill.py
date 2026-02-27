import sqlite3
import logging
from ingest import backfill_ss_embeddings_for_user
from db import get_db_connection

logging.basicConfig(level=logging.INFO)

def test_backfill():
    # Ensure there is at least one interaction to backfill
    conn = get_db_connection()
    c = conn.cursor()
    
    # Check if there are any papers with interactions but no ss_embedding
    c.execute("""
        SELECT COUNT(*) FROM interactions i
        JOIN papers p ON i.paper_id = p.id
        WHERE p.ss_embedding IS NULL
    """)
    count_before = c.fetchone()[0]
    print(f"Papers needing SS embedding backfill: {count_before}")
    
    if count_before == 0:
        print("No papers to backfill. Try liking some papers in the app first.")
        return

    # Trigger backfill for DEFAULT_USER_ID = 1
    backfilled = backfill_ss_embeddings_for_user(1)
    print(f"Successfully backfilled: {backfilled}")

    c.execute("""
        SELECT COUNT(*) FROM interactions i
        JOIN papers p ON i.paper_id = p.id
        WHERE p.ss_embedding IS NOT NULL
    """)
    count_after = c.fetchone()[0]
    print(f"Papers with SS embedding: {count_after}")
    
    conn.close()

if __name__ == "__main__":
    test_backfill()
