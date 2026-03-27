import os
import sys
import logging
import datetime
import json
import numpy as np
import traceback

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from db import get_db_connection
from ingest import ingest_papers, save_papers
from core.oscar_batch import OscarBatchManager

# Setup logging
os.makedirs("logs", exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("logs/cron_worker.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("cron_worker")

def prune_old_papers(days=180):
    """
    Deletes papers older than 'days' (default 6 months) that have no user interactions.
    """
    logger.info(f"Pruning papers older than {days} days...")
    conn = get_db_connection()
    c = conn.cursor()
    
    cutoff_date = (datetime.datetime.now() - datetime.timedelta(days=days)).isoformat()
    
    # 1. Find papers to delete
    query = """
        SELECT id FROM papers 
        WHERE published_date < ? 
        AND id NOT IN (SELECT paper_id FROM interactions)
    """
    c.execute(query, (cutoff_date,))
    to_delete = [row[0] for row in c.fetchall()]
    
    if to_delete:
        # Delete from main table
        c.execute(f"DELETE FROM papers WHERE id IN ({','.join(['?']*len(to_delete))})", to_delete)
        # Delete from VSS (rowid is synced, but delete by ID is safer if we know the rowids)
        # sqlite-vss is best updated by just letting the main table handle it or re-syncing.
        # Simple approach: Since it's a virtual table, we can just delete from it.
        # But we need rowids. Let's just focus on the main table for now, VSS usually follows or can be rebuilt.
        logger.info(f"Pruned {len(to_delete)} old papers.")
    else:
        logger.info("No old papers to prune.")
        
    conn.commit()
    conn.close()

def run_2am_sync():
    logger.info("=== Starting Unified Daily Sync (2 AM Job) ===")
    
    try:
        conn = get_db_connection()
        c = conn.cursor()
        c.execute("SELECT user_id, target_subjects, keywords, fetch_frequency FROM user_settings")
        settings_rows = c.fetchall()
        
        # 1. Fetch Metadata (Hold in memory)
        all_candidates = []
        seen_ids = set()
        
        for row in settings_rows:
            user_id, subjects_json, keywords_json, frequency = row
            subjects = json.loads(subjects_json) if subjects_json else ["cs.CL", "cs.AI"]
            keywords = json.loads(keywords_json) if keywords_json else []
            
            logger.info(f"Fetching candidates for User {user_id}...")
            # ingest_papers now returns a list of dicts because save_to_db=False
            papers = ingest_papers(subjects, frequency=frequency, keywords=keywords, save_to_db=False)
            
            for p in papers:
                pid = f"{p['source']}:{p['external_id']}"
                if pid not in seen_ids:
                    all_candidates.append(p)
                    seen_ids.add(pid)

        # 2. Filter: Only embed papers that DON'T already exist in DB
        papers_ready_to_save = []
        papers_needing_inference = []
        
        for p in all_candidates:
            pid = f"{p['source']}:{p['external_id']}"
            c.execute("SELECT 1 FROM papers WHERE id = ?", (pid,))
            if not c.fetchone():
                if p.get('embedding'):
                    papers_ready_to_save.append(p)
                else:
                    # Strip everything except what's needed for inference to avoid JSON errors
                    papers_needing_inference.append({
                        "id": pid,
                        "title": p['title'],
                        "abstract": p['abstract']
                    })
        
        logger.info(f"Found {len(papers_ready_to_save)} papers with SS embeddings and {len(papers_needing_inference)} needing OSCAR.")

        # 3. Save papers that already have embeddings
        if papers_ready_to_save:
            saved_ss = save_papers(papers_ready_to_save)
            logger.info(f"Saved {saved_ss} papers with pre-existing SS embeddings.")

        # 4. Offload to OSCAR (Sync Batch) for those missing embeddings
        if papers_needing_inference:
            logger.info(f"Offloading {len(papers_needing_inference)} papers to OSCAR for Specter2 embeddings...")
            manager = OscarBatchManager()
            # This blocks until finished
            results = manager.run_remote_batch(papers_needing_inference, poll_interval=60)
            
            if results:
                logger.info(f"Retrieved {len(results)} embeddings from OSCAR.")
                # Map results back to full metadata
                res_map = {res["id"]: res["embedding"] for res in results}
                
                final_papers_to_save = []
                for p in all_candidates:
                    pid = f"{p['source']}:{p['external_id']}"
                    if pid in res_map:
                        p['embedding'] = np.array(res_map[pid], dtype=np.float32).tobytes()
                        final_papers_to_save.append(p)
                
                # 5. Save to DB (Metadata + Embedding together)
                saved_count = save_papers(final_papers_to_save)
                logger.info(f"Successfully stored {saved_count} new papers with OSCAR embeddings.")
            else:
                logger.error("OSCAR offload failed to return results. Papers NOT stored to maintain 'rarely empty' rule.")
        else:
            logger.info("No new papers needing OSCAR inference.")

        # 5. Prune old papers (6 months)
        prune_old_papers(days=180)

        conn.close()
        
    except Exception as e:
        logger.error(f"Sync failed: {e}")
        logger.error(traceback.format_exc())
        with open("logs/sync_errors.log", "a") as f:
            f.write(f"\n[{datetime.datetime.now().isoformat()}] {str(e)}\n{traceback.format_exc()}\n")

    logger.info("=== Sync Cycle Complete ===")

if __name__ == "__main__":
    run_2am_sync()
