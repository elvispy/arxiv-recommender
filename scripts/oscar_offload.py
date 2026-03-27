import os
import sys
import logging
import json
import numpy as np

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from db import get_db_connection
from core.oscar_batch import OscarBatchManager

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("oscar_offload")

def run_oscar_offload():
    """Identifies NULL-embedding papers and runs the OSCAR batch process."""
    conn = get_db_connection()
    c = conn.cursor()
    
    # 1. Select papers missing embeddings
    c.execute("SELECT id, title, abstract FROM papers WHERE embedding IS NULL")
    missing_papers = [dict(row) for row in c.fetchall()]
    
    if not missing_papers:
        logger.info("No papers missing embeddings. Skipping OSCAR offload.")
        conn.close()
        return 0
        
    logger.info(f"Found {len(missing_papers)} papers for offload.")
    
    # 2. OSCAR Batch Polling Process
    manager = OscarBatchManager()
    # Use shorter poll_interval for testing if needed
    results = manager.run_remote_batch(missing_papers, poll_interval=60) 
    
    if not results:
        logger.error("OSCAR offload returned no results.")
        conn.close()
        return 0
        
    # 3. Update DB with Retrieved Embeddings
    updated_count = 0
    for res in results:
        p_id = res["id"]
        emb_list = res["embedding"]
        emb_bytes = np.array(emb_list, dtype=np.float32).tobytes()
        
        c.execute("UPDATE papers SET embedding = ? WHERE id = ?", (emb_bytes, p_id))
        
        # Update VSS
        c.execute("SELECT rowid FROM papers WHERE id = ?", (p_id,))
        row_id_row = c.fetchone()
        if row_id_row:
            row_id = row_id_row[0]
            try:
                c.execute('INSERT OR REPLACE INTO vss_papers(rowid, embedding) VALUES (?, ?)', (row_id, emb_bytes))
            except Exception as e:
                logger.debug(f"VSS update failed for {p_id}: {e}")
        updated_count += 1
        
    conn.commit()
    conn.close()
    logger.info(f"OSCAR update complete. Updated {updated_count} papers.")
    return updated_count

if __name__ == "__main__":
    run_oscar_offload()
