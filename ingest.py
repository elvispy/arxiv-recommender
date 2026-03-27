import os
import json
import logging
import requests
import arxiv
import sqlite3
import datetime
import time
import feedparser
import urllib.parse
from db import get_db_connection

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- Model Loading (Local Fallback Only) ---
_model = None
def get_model():
    global _model
    if _model is None:
        try:
            import adapters
            from transformers import AutoTokenizer, AutoModel
            model_name = "allenai/specter2_base"
            adapter_name = "allenai/specter2"
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            base_model = AutoModel.from_pretrained(model_name)
            adapters.init(base_model)
            loaded_name = base_model.load_adapter(adapter_name, source="hf", set_active=True)
            base_model.set_active_adapters(loaded_name)
            base_model.eval()
            _model = {'tokenizer': tokenizer, 'model': base_model}
        except Exception as e:
            logger.error(f"Failed to load local model: {e}")
            raise
    return _model

# --- Embedding Logic ---
def embed_papers_remote_batch(papers, batch_size=16):
    """Placeholder for the batch-offload logic. Actual offload happens in scripts/oscar_offload.py"""
    # This is kept for interface compatibility but the cron_worker will handle the actual OSCAR scp/ssh.
    return [None] * len(papers)

def fetch_ss_embeddings_batch(papers):
    from config import SEMANTIC_SCHOLAR_API_KEY
    if not papers: return {}
    payload_ids = []
    id_map = {}
    for p in papers:
        pid = f"{p['source']}:{p['external_id']}"
        ss_id = f"ARXIV:{p['external_id'].split('v')[0]}" if p['source'] == 'arxiv' else p['external_id']
        id_map[ss_id] = pid
        payload_ids.append(ss_id)
        
    url = "https://api.semanticscholar.org/graph/v1/paper/batch"
    params = {"fields": "embedding.specter_v2,externalIds"}
    headers = {"x-api-key": SEMANTIC_SCHOLAR_API_KEY}
    
    results = {}
    import numpy as np
    for i in range(0, len(payload_ids), 100):
        chunk = payload_ids[i:i+100]
        try:
            r = requests.post(url, json={"ids": chunk}, params=params, headers=headers, timeout=30)
            if r.status_code == 200:
                data = r.json()
                for idx, item in enumerate(data):
                    if item and item.get('embedding', {}).get('vector'):
                        my_id = id_map[chunk[idx]]
                        results[my_id] = np.array(item['embedding']['vector'], dtype=np.float32).tobytes()
        except Exception as e:
            logger.error(f"SS Batch fetch error: {e}")
    return results

# --- Database Storage ---
def save_papers(papers):
    conn = get_db_connection()
    c = conn.cursor()
    count = 0
    for paper in papers:
        paper_id = f"{paper['source']}:{paper['external_id']}"
        embedding_bytes = paper.get('embedding')
        
        # Check existence
        c.execute("SELECT 1 FROM papers WHERE id = ?", (paper_id,))
        if c.fetchone():
            if embedding_bytes:
                c.execute("UPDATE papers SET embedding = ? WHERE id = ?", (embedding_bytes, paper_id))
                c.execute("SELECT rowid FROM papers WHERE id = ?", (paper_id,))
                row_id = c.fetchone()[0]
                try:
                    c.execute('INSERT OR REPLACE INTO vss_papers(rowid, embedding) VALUES (?, ?)', (row_id, embedding_bytes))
                except: pass
            continue

        try:
            c.execute('''
                INSERT INTO papers (id, source, external_id, title, abstract, authors, published_date, category, link, embedding)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (paper_id, paper['source'], paper['external_id'], paper['title'], paper['abstract'], 
                  json.dumps(paper['authors']), paper['published_date'], paper['category'], paper['link'], embedding_bytes))
            
            if embedding_bytes:
                c.execute("SELECT rowid FROM papers WHERE id = ?", (paper_id,))
                row_id = c.fetchone()[0]
                c.execute('INSERT INTO vss_papers(rowid, embedding) VALUES (?, ?)', (row_id, embedding_bytes))
            count += 1
        except Exception as e:
            logger.error(f"Save error {paper_id}: {e}")
    conn.commit()
    conn.close()
    return count

# --- Ingestion Dispatches ---
def calculate_dynamic_cap(subjects, keywords, frequency, source):
    days = 7 if frequency == 'weekly' else 1
    num_subjects = len(subjects)
    num_keywords = len(keywords)
    if source == 'arxiv': cap = 30 * (num_subjects + num_keywords * 0.2) * days
    else: cap = 30 * (num_keywords * 0.2) * days
    return max(30, min(int(cap), 500))

def fetch_arxiv_api_papers(subjects, max_results=None, keywords=None, save_to_db=True):
    import arxiv
    if keywords is None: keywords = []
    if max_results is None: max_results = 100
    
    query_parts = []
    if subjects: query_parts.append("(" + " OR ".join([f"cat:{s}" for s in subjects]) + ")")
    if keywords: query_parts.append("(" + " OR ".join([f'all:"{kw}"' for kw in keywords]) + ")")
    if not query_parts: return 0 if save_to_db else []
    
    full_query = " OR ".join(query_parts)
    search = arxiv.Search(query=full_query, max_results=max_results, sort_by=arxiv.SortCriterion.SubmittedDate)
    
    papers = []
    try:
        for result in search.results():
            papers.append({
                'source': 'arxiv',
                'external_id': result.entry_id.split('/')[-1].split('v')[0],
                'title': result.title.replace('\n', ' '),
                'abstract': result.summary.replace('\n', ' '),
                'authors': [author.name for author in result.authors],
                'published_date': result.published.isoformat(),
                'category': ', '.join(result.categories),
                'link': result.entry_id
            })
    except Exception as e:
        logger.error(f"ArXiv API error: {e}")
        return 0 if save_to_db else []

    # Batch Fetch Embeddings from Semantic Scholar (Fast)
    emb_map = fetch_ss_embeddings_batch(papers)
    for p in papers:
        p['embedding'] = emb_map.get(f"arxiv:{p['external_id']}")

    if save_to_db:
        return save_papers(papers)
    return papers

def fetch_arxiv_papers(subjects, frequency='daily', save_to_db=True):
    now = datetime.datetime.now(datetime.timezone.utc)
    cutoff = now - (datetime.timedelta(days=7) if frequency == 'weekly' else datetime.timedelta(hours=24))
    
    candidates = []
    for subject in subjects:
        feed = feedparser.parse(f"http://export.arxiv.org/rss/{subject}")
        for entry in feed.entries:
            if not hasattr(entry, 'published_parsed'): continue
            pub_dt = datetime.datetime(*entry.published_parsed[:6], tzinfo=datetime.timezone.utc)
            if pub_dt < cutoff: continue
            
            candidates.append({
                'source': 'arxiv',
                'external_id': entry.link.split('/')[-1].split('v')[0],
                'title': entry.title.replace('\n', ' '),
                'abstract': entry.summary.replace('\n', ' '),
                'authors': [a.name for a in entry.authors] if 'authors' in entry else [entry.get('author', 'Unknown')],
                'published_date': pub_dt.isoformat(),
                'category': subject,
                'link': entry.link
            })
    
    unique = {p['external_id']: p for p in candidates}.values()
    papers = list(unique)
    emb_map = fetch_ss_embeddings_batch(papers)
    for p in papers:
        p['embedding'] = emb_map.get(f"arxiv:{p['external_id']}")

    if save_to_db:
        return save_papers(papers)
    return papers

def fetch_biorxiv_papers(subjects, limit=50, save_to_db=True):
    all_papers = []
    for subject in subjects:
        clean_subject = subject.replace('biorxiv.', '')
        feed = feedparser.parse(f"http://connect.biorxiv.org/biorxiv_xml.php?subject={clean_subject}")
        for entry in feed.entries[:limit]:
            all_papers.append({
                'source': 'biorxiv',
                'external_id': entry.link.split('/')[-1],
                'title': entry.title,
                'abstract': entry.summary,
                'authors': [a.name for a in entry.authors] if 'authors' in entry else [entry.get('author', 'Unknown')],
                'published_date': entry.get('updated', entry.get('date', '2025-01-01')),
                'category': subject,
                'link': entry.link
            })
    
    if all_papers:
        emb_map = fetch_ss_embeddings_batch(all_papers)
        for p in all_papers:
            p['embedding'] = emb_map.get(f"biorxiv:{p['external_id']}")

    if save_to_db:
        return save_papers(all_papers)
    return all_papers

def backfill_ss_embeddings_for_user(user_id):
    """
    Finds papers in the database missing embeddings and attempts to fetch them from Semantic Scholar.
    Used by the UI for manual 'backfill' requests.
    """
    logger.info(f"Starting SS backfill for user {user_id}...")
    conn = get_db_connection()
    c = conn.cursor()
    
    # Only fetch for papers that might be relevant to this user (simpler: fetch all missing)
    c.execute("SELECT id, source, external_id FROM papers WHERE embedding IS NULL")
    missing_rows = c.fetchall()
    
    if not missing_rows:
        logger.info("No papers missing embeddings to backfill.")
        conn.close()
        return 0
        
    papers_to_fetch = []
    for row in missing_rows:
        papers_to_fetch.append({
            'id': row[0],
            'source': row[1],
            'external_id': row[2]
        })
        
    logger.info(f"Found {len(papers_to_fetch)} papers missing embeddings. Fetching from SS...")
    emb_map = fetch_ss_embeddings_batch(papers_to_fetch)
    
    count = 0
    for p in papers_to_fetch:
        pid = p['id']
        if pid in emb_map:
            embedding_bytes = emb_map[pid]
            c.execute("UPDATE papers SET embedding = ? WHERE id = ?", (embedding_bytes, pid))
            
            # Sync to VSS
            c.execute("SELECT rowid FROM papers WHERE id = ?", (pid,))
            row_id_row = c.fetchone()
            if row_id_row:
                row_id = row_id_row[0]
                try:
                    c.execute('INSERT OR REPLACE INTO vss_papers(rowid, embedding) VALUES (?, ?)', (row_id, embedding_bytes))
                except: pass
            count += 1
            
    conn.commit()
    conn.close()
    logger.info(f"Successfully backfilled {count} SS embeddings.")
    return count

def clean_html(raw_html):
    """Utility to strip HTML tags from abstracts."""
    if not raw_html: return ""
    from bs4 import BeautifulSoup
    return BeautifulSoup(raw_html, "html.parser").get_text(separator=" ").strip()

def fetch_jfm_papers(limit=50, save_to_db=True):
    """Fetches latest papers from Journal of Fluid Mechanics via Cambridge Core RSS."""
    logger.info("Fetching JFM RSS...")
    url = "https://www.cambridge.org/core/rss/product/id/1F51BCFAA50101CAF5CB9A20F8DEA3E4"
    all_papers = []
    
    try:
        feed = feedparser.parse(url)
        for entry in feed.entries[:limit]:
            # JFM RSS provides 'prism_doi' which is perfect for external_id
            doi = entry.get('prism_doi', entry.link.split('/')[-1])
            abstract = clean_html(entry.get('summary', ''))
            
            authors = []
            if 'authors' in entry:
                authors = [a.get('name', 'Unknown') for a in entry.authors]
            elif 'author' in entry:
                authors = [entry.author]
                
            all_papers.append({
                'source': 'jfm',
                'external_id': doi,
                'title': entry.title.replace('\n', ' '),
                'abstract': abstract,
                'authors': authors,
                'published_date': entry.get('updated', entry.get('published', '2025-01-01')),
                'category': 'jfm',
                'link': entry.link
            })
    except Exception as e:
        logger.error(f"JFM fetch failed: {e}")

    if all_papers:
        emb_map = fetch_ss_embeddings_batch(all_papers)
        for p in all_papers:
            p['embedding'] = emb_map.get(f"jfm:{p['external_id']}")

    if save_to_db:
        return save_papers(all_papers)
    return all_papers

def fetch_aps_papers(subjects: list, limit=50, save_to_db=True):
    """Fetches latest papers from APS journals (e.g., prfluids, prl)."""
    all_papers = []
    for subject in subjects:
        # Expected subject format: 'aps.prfluids'
        journal = subject.split('.')[-1]
        url = f"http://feeds.aps.org/rss/recent/{journal}.xml"
        logger.info(f"Fetching APS RSS: {url}")
        
        try:
            feed = feedparser.parse(url)
            for entry in feed.entries[:limit]:
                # APS RSS link often contains DOI, but let's use the link itself as a fallback if no explicit DOI field
                doi = entry.get('id', entry.link)
                if 'doi.org/' in doi:
                    doi = doi.split('doi.org/')[-1]
                    
                abstract = clean_html(entry.get('summary', ''))
                
                authors = []
                if 'authors' in entry:
                    authors = [a.get('name', 'Unknown') for a in entry.authors]
                elif 'author' in entry:
                    authors = [entry.author]
                    
                all_papers.append({
                    'source': 'aps',
                    'external_id': doi,
                    'title': entry.title.replace('\n', ' '),
                    'abstract': abstract,
                    'authors': authors,
                    'published_date': entry.get('updated', entry.get('published', '2025-01-01')),
                    'category': subject,
                    'link': entry.link
                })
        except Exception as e:
            logger.error(f"APS fetch failed for {journal}: {e}")

    if all_papers:
        emb_map = fetch_ss_embeddings_batch(all_papers)
        for p in all_papers:
            p['embedding'] = emb_map.get(f"aps:{p['external_id']}")

    if save_to_db:
        return save_papers(all_papers)
    return all_papers

def ingest_papers(subjects, frequency='daily', keywords=None, save_to_db=True):
    if keywords is None: keywords = []
    
    arxiv_subjects = []
    biorxiv_subjects = []
    aps_subjects = []
    jfm_requested = False
    
    for s in subjects:
        if s.startswith('biorxiv.'):
            biorxiv_subjects.append(s)
        elif s.startswith('aps.'):
            aps_subjects.append(s)
        elif s == 'jfm':
            jfm_requested = True
        else:
            arxiv_subjects.append(s)
    
    total = 0
    results = []
    
    if arxiv_subjects:
        if frequency in ['weekly', 'last_100']:
            res = fetch_arxiv_api_papers(arxiv_subjects, max_results=50 if frequency == 'last_100' else None, keywords=keywords, save_to_db=save_to_db)
            if save_to_db: total += res
            else: results.extend(res)
        else:
            res = fetch_arxiv_papers(arxiv_subjects, frequency=frequency, save_to_db=save_to_db)
            if save_to_db: total += res
            else: results.extend(res)
            
    if biorxiv_subjects:
        res = fetch_biorxiv_papers(biorxiv_subjects, save_to_db=save_to_db)
        if save_to_db: total += res
        else: results.extend(res)
        
    if aps_subjects:
        res = fetch_aps_papers(aps_subjects, save_to_db=save_to_db)
        if save_to_db: total += res
        else: results.extend(res)
        
    if jfm_requested:
        res = fetch_jfm_papers(save_to_db=save_to_db)
        if save_to_db: total += res
        else: results.extend(res)
        
    return total if save_to_db else results

def search_arxiv(query: str, author: str = None, year_start: str = None, limit=10):
    """
    Search ArXiv for specific criteria.
    """
    import arxiv
    q = query
    if author: q += f' AND au:"{author}"'
    
    # Simple search
    search = arxiv.Search(
        query=q,
        max_results=limit,
        sort_by=arxiv.SortCriterion.SubmittedDate
    )
    
    papers = []
    try:
        for result in search.results():
            # Check year if needed
            if year_start:
                if result.published.year < int(year_start):
                    continue
            
            papers.append({
                'source': 'arxiv',
                'external_id': result.entry_id.split('/')[-1].split('v')[0],
                'title': result.title.replace('\n', ' '),
                'abstract': result.summary.replace('\n', ' '),
                'authors': [a.name for a in result.authors],
                'published_date': result.published.isoformat(),
                'category': ', '.join(result.categories),
                'link': result.entry_id
            })
    except Exception as e:
        logger.error(f"Search ArXiv error: {e}")
        
    return papers

def search_semantic_scholar(query: str, year: str = None, limit=10):
    """
    Search Semantic Scholar API.
    """
    from config import SEMANTIC_SCHOLAR_API_KEY
    url = "https://api.semanticscholar.org/graph/v1/paper/search"
    params = {
        "query": query,
        "limit": limit,
        "fields": "title,abstract,authors,year,url,externalIds,embedding.specter_v2"
    }
    if year:
        params["year"] = year
        
    headers = {"x-api-key": SEMANTIC_SCHOLAR_API_KEY} if SEMANTIC_SCHOLAR_API_KEY else {}
    
    papers = []
    try:
        r = requests.get(url, params=params, headers=headers, timeout=30)
        if r.status_code == 200:
            data = r.json().get('data', [])
            import numpy as np
            for item in data:
                ext_ids = item.get('externalIds', {})
                # Try to get ArXiv ID
                arxiv_id = ext_ids.get('ArXiv')
                
                paper = {
                    'source': 'semantic_scholar',
                    'external_id': arxiv_id or item.get('paperId'),
                    'title': item.get('title'),
                    'abstract': item.get('abstract', ''),
                    'authors': [a.get('name') for a in item.get('authors', [])],
                    'published_date': f"{item.get('year')}-01-01" if item.get('year') else "2025-01-01",
                    'category': 'Search Result',
                    'link': item.get('url')
                }
                
                # If it has a Specter v2 embedding, keep it
                if item.get('embedding', {}).get('vector'):
                    paper['ss_embedding'] = np.array(item['embedding']['vector'], dtype=np.float32).tobytes()
                
                papers.append(paper)
    except Exception as e:
        logger.error(f"Search SS error: {e}")
        
    return papers

if __name__ == "__main__":
    print("Ingest logic ready.")
