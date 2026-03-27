import os
import sys
import logging
import json

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from db import get_db_connection
from ingest import ingest_papers

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("fetch_metadata")

def fetch_daily_metadata():
    """Locally identifies and saves papers published recently."""
    conn = get_db_connection()
    c = conn.cursor()
    c.execute("SELECT user_id, target_subjects, keywords, fetch_frequency FROM user_settings")
    settings_rows = c.fetchall()
    
    total_new = 0
    
    for row in settings_rows:
        user_id, subjects_json, keywords_json, frequency = row
        subjects = json.loads(subjects_json) if subjects_json else ["cs.CL", "cs.AI"]
        keywords = json.loads(keywords_json) if keywords_json else []
        
        logger.info(f"Fetching metadata for User {user_id}...")
        
        # This will save papers to 'papers' table with embedding=NULL 
        # because local models won't find the 'specter2' library on macOS easily 
        # or it will be slow. We force metadata-only by not passing use_oscar.
        new_count = ingest_papers(
            subjects=subjects, 
            frequency=frequency or 'daily', 
            keywords=keywords,
            use_oscar=False # Force local (which falls back to NULL if model fails)
        )
        total_new += new_count
        
    conn.close()
    logger.info(f"Metadata fetch complete. Total new entries: {total_new}")
    return total_new

if __name__ == "__main__":
    fetch_daily_metadata()
