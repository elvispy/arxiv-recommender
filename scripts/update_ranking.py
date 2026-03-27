import os
import sys
import logging
import json
import datetime

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from db import get_db_connection
from core.rocchio import update_user_profile

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("update_ranking")

def update_all_profiles():
    """Recalculates preference vectors for all users."""
    conn = get_db_connection()
    c = conn.cursor()
    c.execute("SELECT id FROM users")
    user_ids = [row[0] for row in c.fetchall()]
    
    for u_id in user_ids:
        try:
            logger.info(f"Updating preference profile for user {u_id}...")
            update_user_profile(u_id)
            
            # Record last sync time
            c.execute("UPDATE user_settings SET last_fetch_date = ? WHERE user_id = ?", 
                     (datetime.datetime.now().isoformat(), u_id))
        except Exception as e:
            logger.error(f"Failed to update profile for user {u_id}: {e}")
            
    conn.commit()
    conn.close()
    logger.info("Ranking update complete.")

if __name__ == "__main__":
    update_all_profiles()
