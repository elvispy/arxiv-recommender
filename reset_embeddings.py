import sqlite3
import sys
import os

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from config import DB_PATH

def clear_embeddings():
    print(f"Connecting to {DB_PATH}...")
    conn = sqlite3.connect(DB_PATH)
    
    # We need to load vss to delete from it? 
    # Actually standard DELETE FROM vss_papers works if extension is loaded, 
    # but we can also just drop/recreate or delete files if it was separate.
    # Safe way: Load extension.
    conn.enable_load_extension(True)
    try:
        import sqlite_vss
        sqlite_vss.load(conn)
    except ImportError:
        print("Warning: sqlite_vss not found, assuming direct SQL is fine or VSS not init yet.")

    c = conn.cursor()
    
    print("Clearing 'embedding' column in 'papers' table...")
    c.execute("UPDATE papers SET embedding = NULL")
    
    print("Clearing 'vss_papers' table...")
    try:
        c.execute("DELETE FROM vss_papers")
    except Exception as e:
        print(f"Error clearing vss_papers: {e}")
        
    print("Clearing 'user_profile' (preference vectors need recast)...")
    c.execute("DELETE FROM user_profile")
    
    conn.commit()
    conn.close()
    print("Embeddings cleared manually. System effectively reset to 'lazy' state.")

if __name__ == "__main__":
    clear_embeddings()
