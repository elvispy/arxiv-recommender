import sqlite3
import json
import sqlite_vss
from config import DB_PATH

def get_db_connection():
    """
    Returns a database connection with sqlite-vss extension loaded.
    """
    conn = sqlite3.connect(DB_PATH)
    conn.enable_load_extension(True)
    sqlite_vss.load(conn)
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    """
    Initializes the database tables and schemas.
    """
    conn = get_db_connection()
    c = conn.cursor()
    
    # Enable Write-Ahead Logging for better concurrency
    try:
        c.execute("PRAGMA journal_mode=WAL")
    except Exception as e:
        print(f"Failed to set WAL mode: {e}")

    # 1. Users
    c.execute('''
    CREATE TABLE IF NOT EXISTS users (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        username TEXT UNIQUE,
        created_at DATETIME DEFAULT CURRENT_TIMESTAMP
    );
    ''')
    
    # Create default user if not exists
    try:
        c.execute("INSERT OR IGNORE INTO users (username) VALUES ('researcher')")
    except Exception as e:
        print(f"User init warning: {e}")

    # 2. User Settings
    c.execute('''
    CREATE TABLE IF NOT EXISTS user_settings (
        user_id INTEGER PRIMARY KEY,
        fetch_frequency TEXT DEFAULT 'daily',
        target_subjects TEXT, 
        last_fetch_date DATETIME,
        keywords TEXT DEFAULT '[]',
        FOREIGN KEY(user_id) REFERENCES users(id)
    );
    ''')
    
    # 3. Papers (Metadata)
    c.execute('''
    CREATE TABLE IF NOT EXISTS papers (
        id TEXT PRIMARY KEY,
        source TEXT,
        external_id TEXT,
        title TEXT,
        abstract TEXT,
        authors TEXT, 
        published_date DATETIME,
        category TEXT,
        link TEXT,
        embedding BLOB,
        ss_embedding BLOB,
        doi TEXT
    );
    ''')

    # 4. User Profile (Rocchio Center)
    c.execute('''
    CREATE TABLE IF NOT EXISTS user_profile (
        user_id INTEGER PRIMARY KEY,
        preference_vector BLOB,
        last_updated DATETIME,
        FOREIGN KEY(user_id) REFERENCES users(id)
    );
    ''')

    # 5. Interactions
    c.execute('''
    CREATE TABLE IF NOT EXISTS interactions (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        user_id INTEGER,
        paper_id TEXT,
        action TEXT,
        timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
        FOREIGN KEY(user_id) REFERENCES users(id),
        FOREIGN KEY(paper_id) REFERENCES papers(id)
    );
    CREATE UNIQUE INDEX IF NOT EXISTS idx_interactions_user_paper ON interactions(user_id, paper_id);
    ''')

    # 6. VSS Table (Vector Search)
    c.execute('''
    CREATE VIRTUAL TABLE IF NOT EXISTS vss_papers USING vss0(
        embedding(768)
    );
    ''')

    conn.commit()
    conn.close()
    print("Database initialized successfully at", DB_PATH)

if __name__ == "__main__":
    init_db()
