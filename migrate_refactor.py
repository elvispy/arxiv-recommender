import sqlite3
import json
import requests
import time
import logging
import numpy as np
from config import DB_PATH, SEMANTIC_SCHOLAR_API_KEY as SS_API_KEY

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def migrate_and_backfill():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    c = conn.cursor()

    # 1. Migrate ss_embedding to embedding
    logger.info("Migrating ss_embedding to embedding...")
    c.execute("UPDATE papers SET embedding = ss_embedding WHERE ss_embedding IS NOT NULL")
    
    # 2. Clear embedding where ss_embedding is NULL (to force re-fetch)
    c.execute("UPDATE papers SET embedding = NULL WHERE ss_embedding IS NULL")
    conn.commit()
    logger.info("Embedding migration complete.")

    # 3. Backfill DOIs
    logger.info("Backfilling DOIs...")
    c.execute("SELECT id, source, external_id, title FROM papers WHERE doi IS NULL")
    papers = c.fetchall()
    
    if not papers:
        logger.info("No papers need DOI backfill.")
        conn.close()
        return

    logger.info(f"Found {len(papers)} papers to check for DOIs.")
    
    # Process in batches to be efficient
    updated_count = 0
    
    for paper in papers:
        paper_id = paper['id']
        source = paper['source']
        ext_id = paper['external_id']
        
        # Rate limit
        time.sleep(1.1)
        
        # Try to find via ArXiv ID first (most reliable)
        ss_id = None
        if source == 'arxiv':
            clean_id = ext_id.split('v')[0]
            ss_id = f"ARXIV:{clean_id}"
        elif source == 'semantic_scholar':
            ss_id = ext_id
            
        if not ss_id:
            logger.warning(f"Skipping {paper_id}, no mapping strategy.")
            continue
            
        try:
            url = f"https://api.semanticscholar.org/graph/v1/paper/{ss_id}"
            params = {"fields": "externalIds"}
            headers = {"x-api-key": SS_API_KEY}
            
            r = requests.get(url, params=params, headers=headers, timeout=10)
            if r.status_code == 200:
                data = r.json()
                doi = data.get('externalIds', {}).get('DOI')
                if doi:
                    c.execute("UPDATE papers SET doi = ? WHERE id = ?", (doi, paper_id))
                    updated_count += 1
                    logger.info(f"Found DOI for {paper_id}: {doi}")
                else:
                    # Mark as empty string to avoid re-checking? Or keep NULL? 
                    # Keeping NULL allows retry, empty string means "checked and none"
                    # For now keep NULL as this is a one-off
                    logger.info(f"No DOI found for {paper_id}")
            elif r.status_code == 404:
                logger.warning(f"Paper not found on SS: {ss_id}")
            else:
                logger.warning(f"SS Error {r.status_code} for {ss_id}")
                
        except Exception as e:
            logger.error(f"Error fetching DOI for {paper_id}: {e}")

        conn.commit()
        
    logger.info(f"DOI Backfill complete. Updated {updated_count} papers.")
    conn.close()

if __name__ == "__main__":
    migrate_and_backfill()
