import os
import sys
import time

# Ensure the project root (where `config.py` and `db.py` live)
# is on the import path, even when this script is executed
# directly from within the `tests` directory.
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from config import EMBEDDING_MODEL_NAME
from db import get_db_connection


def fetch_one_paper_text() -> str:
    """Fetch title+abstract for a single paper from the DB."""
    conn = get_db_connection()
    cur = conn.cursor()
    # Use a recent paper to approximate typical input length
    cur.execute("SELECT title, abstract FROM papers ORDER BY published_date DESC LIMIT 1")
    row = cur.fetchone()
    conn.close()

    if not row:
        raise SystemExit("No papers found in the database. Run ingestion first.")

    title, abstract = row[0], row[1]
    return f"{title} {abstract}"


def main() -> None:
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as exc:
        raise SystemExit(
            "sentence-transformers is not installed. "
            "Install dependencies (e.g. `pip install -r requirements.txt`) and retry."
        ) from exc

    text_to_embed = fetch_one_paper_text()
    print("Fetched one paper from DB for timing.")

    print(f"Loading embedding model: {EMBEDDING_MODEL_NAME}...")
    load_start = time.perf_counter()
    model = SentenceTransformer(EMBEDDING_MODEL_NAME)
    load_end = time.perf_counter()
    print(f"Model loaded in {load_end - load_start:.3f} seconds")

    # Warm-up call (helps avoid including any one-time setup in timing)
    _ = model.encode(text_to_embed)

    print("Timing a single embedding on real paper text...")
    start = time.perf_counter()
    vec = model.encode(text_to_embed)
    end = time.perf_counter()

    elapsed = end - start
    print(f"Embedding dimension: {len(vec)}")
    print(f"Time for one embedding: {elapsed:.4f} seconds")


if __name__ == "__main__":
    main()
