import os
import json
import logging
import requests
import arxiv
import sqlite3
import datetime
import sqlite_vss
from db import get_db_connection
from config import EMBEDDING_MODEL_NAME

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Check for model load
_model = None

def get_model():
    """
    Loads the Specter2 model with the proximity adapter.
    """
    global _model
    if _model is None:
        try:
            import adapters
            from transformers import AutoTokenizer, AutoModel
            
            model_name = "allenai/specter2_base"
            adapter_name = "allenai/specter2" # Proximity adapter
            
            logger.info(f"Loading embedding model base: {model_name}...")
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            
            # Load base model
            base_model = AutoModel.from_pretrained(model_name)
            adapters.init(base_model)
            
            loaded_name = base_model.load_adapter(adapter_name, source="hf", set_active=True)
            base_model.set_active_adapters(loaded_name)
            
            # KPI: Set to eval mode to disable dropout!
            base_model.eval()
            
            logger.info(f"Loaded adapter: {loaded_name} (Requested: {adapter_name})")
            
            _model = {
                'tokenizer': tokenizer,
                'model': base_model
            }
            logger.info("Model loaded successfully.")
            
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise
    return _model

def get_embedding(text):
    """
    Generates embedding for text using Specter2 Proximity Adapter.
    Recommended input format: Title + [SEP] + Abstract
    """
    model_dict = get_model()
    tokenizer = model_dict['tokenizer']
    model = model_dict['model']
    
    inputs = tokenizer(text, padding=True, truncation=True, return_tensors="pt", max_length=512)
    
    import torch
    with torch.no_grad():
        outputs = model(**inputs)
        # Specter2 uses [CLS] token embedding (first token)
        embeddings = outputs.last_hidden_state[:, 0, :]
        
        # Return numpy array of the first (and only) item
        return embeddings[0].numpy()
    
def fetch_ss_embeddings_batch(papers):
    """
    Fetches embeddings for a batch of papers from Semantic Scholar using the Batch API.
    Expects papers to be a list of dicts with 'external_id' and 'source'.
    Returns a dict mapping paper_id -> bytes embedding.
    """
    from config import SEMANTIC_SCHOLAR_API_KEY
    if not papers:
        return {}
        
    # Prepare payload
    # SS Batch API takes a list of IDs.
    # We need to map our IDs to SS IDs.
    
    # We can use the POST /graph/v1/paper/batch endpoint
    # IDs can be ArXiv IDs: "ARXIV:2106.15928"
    
    id_map = {} # ss_id -> paper_id
    payload_ids = []
    
    for p in papers:
        pid = f"{p['source']}:{p['external_id']}"
        ss_id = None
        if p['source'] == 'arxiv':
            clean_id = p['external_id'].split('v')[0] # Remove version
            ss_id = f"ARXIV:{clean_id}"
        elif p['source'] == 'semantic_scholar':
            ss_id = p['external_id']
            
        if ss_id:
            id_map[ss_id] = pid
            payload_ids.append(ss_id)
            
    if not payload_ids:
        return {}
        
    url = "https://api.semanticscholar.org/graph/v1/paper/batch"
    params = {"fields": "embedding.specter_v2,externalIds"}
    headers = {"x-api-key": SEMANTIC_SCHOLAR_API_KEY}
    
    # Chunking (max 500 per request officially, keep it safe at 100)
    chunk_size = 100
    results = {}
    
    import numpy as np
    
    for i in range(0, len(payload_ids), chunk_size):
        chunk = payload_ids[i:i+chunk_size]
        try:
            r = requests.post(url, json={"ids": chunk}, params=params, headers=headers, timeout=30)
            if r.status_code == 200:
                data = r.json()
                for item in data:
                    if not item: continue
                    # item is None if not found
                    
                    # Match back to our paper_id
                    # The response matches the order, BUT batch endpoint returns matching objects
                    # Need to check 'paperId' or handle mapping carefully.
                    # Actually, batch endpoint returns items corresponding to input IDs IF found?
                    # Documentation says: "The response is a list of paper objects, in the same order as the request."
                    # If a paper is not found, the entry is null.
                    
                    # Wait, let's rely on the input chunk index to map back if strict order is guaranteed.
                    # "same order as the request" -> Yes.
                    pass 
                
                # Correct approach:
                for idx, item in enumerate(data):
                    req_id = chunk[idx]
                    my_id = id_map[req_id]
                    
                    if item:
                        # Embedding
                        emb_vec = item.get('embedding', {}).get('vector')
                        if emb_vec:
                            results[my_id] = np.array(emb_vec, dtype=np.float32).tobytes()
                            
                        # Extra: We can snatch the DOI here if we want!
                        doi = item.get('externalIds', {}).get('DOI')
                        if doi:
                            # We can return this too or side-load it into the paper dict if passed mutable
                            # Let's side-load it into a global mapping or return tuple? 
                            # Simpler: return a dict of metadata updates
                            pass
            else:
                logger.error(f"Batch embedding fetch failed: {r.status_code} {r.text}")
                
        except Exception as e:
            logger.error(f"Error in batch embedding fetch: {e}")
            
    return results

def save_papers(papers):
    """
    Saves a list of paper dictionaries to the database.
    Expects keys: source, external_id, title, abstract, authors, published_date, category, link
    Optional keys: embedding (bytes), doi
    """
    conn = get_db_connection()
    c = conn.cursor()
    
    count = 0

    for paper in papers:
        # Generate ID
        paper_id = f"{paper['source']}:{paper['external_id']}"
        
        # Check if paper already exists
        c.execute("SELECT embedding FROM papers WHERE id = ?", (paper_id,))
        existing = c.fetchone()
        
        if existing:
            # Paper exists - update if we have new embedding
            embedding_bytes = paper.get('embedding')
            if embedding_bytes and not existing[0]:
                logger.info(f"Updating embedding for existing: {paper_id}")
                c.execute("UPDATE papers SET embedding = ? WHERE id = ?", (embedding_bytes, paper_id))
                # Update VSS
                c.execute("SELECT rowid FROM papers WHERE id = ?", (paper_id,))
                row_id = c.fetchone()[0]
                try:
                    c.execute('INSERT OR REPLACE INTO vss_papers(rowid, embedding) VALUES (?, ?)', 
                             (row_id, embedding_bytes))
                except:
                    pass
                count += 1
            continue
            
        # Check existence by DOI (Strict Duplicate Detection)
        doi = paper.get('doi')
        if doi:
            c.execute("SELECT 1 FROM papers WHERE doi = ?", (doi,))
            if c.fetchone():
                logger.info(f"Duplicate DOI found: {doi} (skipping {paper_id})")
                continue

        # Embedding: Prefetched and passed in the 'embedding' key (bytes)
        embedding_bytes = paper.get('embedding')

        # Insert metadata
        try:
            c.execute('''
                INSERT INTO papers (id, source, external_id, title, abstract, authors, published_date, category, link, embedding, doi)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                paper_id,
                paper['source'],
                paper['external_id'],
                paper['title'],
                paper['abstract'],
                json.dumps(paper['authors']),
                paper['published_date'],
                paper['category'],
                paper['link'],
                embedding_bytes,
                doi
            ))
            
            # Insert into VSS only if we have an embedding
            if embedding_bytes:
                # Get the rowid of the just inserted paper
                c.execute("SELECT rowid FROM papers WHERE id = ?", (paper_id,))
                id_row = c.fetchone()
                if id_row:
                    row_id = id_row[0]
                    c.execute('INSERT INTO vss_papers(rowid, embedding) VALUES (?, ?)', (row_id, embedding_bytes))
            
            count += 1
        except Exception as e:
            logger.error(f"Error saving paper {paper_id}: {e}")
            continue

    conn.commit()
    conn.close()
    return count

def backfill_embeddings_for_user(user_id):
    """
    Finds papers that the user has interacted with (liked/disliked) 
    which do not yet have an embedding. Calculates and saves them LOCALLY.
    """
    try:
        model_dict = get_model()
        tokenizer = model_dict['tokenizer']
    except Exception as e:
        logger.error(f"Cannot backfill embeddings: Model load failed. {e}")
        return 0

    conn = get_db_connection()
    c = conn.cursor()
    
    # metrics
    updated_count = 0
    
    try:
        # Find interacted papers with NULL embedding
        query = """
            SELECT p.id, p.title, p.abstract, p.rowid
            FROM interactions i
            JOIN papers p ON i.paper_id = p.id
            WHERE i.user_id = ? AND p.embedding IS NULL
        """
        c.execute(query, (user_id,))
        rows = c.fetchall()
        
        if not rows:
            return 0
            
        logger.info(f"Backfilling local embeddings for {len(rows)} papers...")
        
        for row in rows:
            p_id = row[0] # id
            title = row[1]
            abstract = row[2]
            row_id = row[3]
            
            # Format: Title + [SEP] + Abstract
            sep = tokenizer.sep_token
            text = f"{title}{sep}{abstract}"
            
            try:
                embedding = get_embedding(text)
                emb_bytes = embedding.tobytes()
                
                # Update main table
                c.execute("UPDATE papers SET embedding = ? WHERE id = ?", (emb_bytes, p_id))
                
                # Check VSS
                try:
                    c.execute('INSERT INTO vss_papers(rowid, embedding) VALUES (?, ?)', (row_id, emb_bytes))
                except:
                    pass
                
                updated_count += 1
            except Exception as e:
                logger.error(f"Failed to backfill embedding for {p_id}: {e}")
                
        conn.commit()
        logger.info(f"Backfilled embeddings for {updated_count} papers.")
        
    except Exception as e:
        logger.error(f"Backfill error: {e}")
        
    finally:
        conn.close()
        
    return updated_count


import feedparser
import urllib.parse
import time

# Rate limiting: Track last API call time
_last_arxiv_call = 0
_last_semantic_scholar_call = 0

def search_arxiv(query: str, author: str = None, year_start: str = None, year_end: str = None, limit=20):
    """
    Search ArXiv with advanced filters.
    Returns papers in the same format as search_semantic_scholar for consistency.
    """
    global _last_arxiv_call
    
    # ArXiv recommends 3 seconds between requests
    time_since_last = time.time() - _last_arxiv_call
    if time_since_last < 3:
        sleep_time = 3 - time_since_last
        logger.info(f"Rate limiting: waiting {sleep_time:.1f}s before ArXiv request")
        time.sleep(sleep_time)
    
    _last_arxiv_call = time.time()
    
    logger.info(f"Searching ArXiv: {query} (Author: {author}, Years: {year_start}-{year_end})")
    
    # Construct ArXiv query - be more specific to avoid timeouts
    query_parts = []
    
    # Add keyword search (prioritize title and abstract for faster results)
    if query and query.strip():
        # Use title OR abstract search (more specific than 'all:')
        query_parts.append(f"(ti:{query.strip()} OR abs:{query.strip()})")
    
    # Add author filter - ensure name is quoted for phrase search
    if author and author.strip():
        query_parts.append(f'au:"{author.strip()}"')
    
    # Combine with AND
    if not query_parts:
        # If no query at all, just get recent papers
        search_query = "all:*"
    else:
        search_query = " AND ".join(query_parts)
    
    # Reduce max_results to avoid timeout - ArXiv can be slow with large result sets
    actual_limit = min(limit, 15)  # Cap at 15 for faster response
    
    base_url = "http://export.arxiv.org/api/query?"
    params = {
        "search_query": search_query,
        "start": 0,
        "max_results": actual_limit,
        "sortBy": "submittedDate",
        "sortOrder": "descending"
    }
    
    headers = {
        "User-Agent": "DailyArXiv/1.0 (mailto:dailyarxiv@example.com)"
    }
    
    try:
        query_string = urllib.parse.urlencode(params)
        url = base_url + query_string
        
        logger.info(f"ArXiv query URL: {url[:150]}...")  # Log for debugging
        
        # Increase timeout to 30s for ArXiv
        response = requests.get(url, headers=headers, timeout=30)
        if response.status_code != 200:
            logger.error(f"ArXiv API Error: {response.status_code}")
            return []
            
        feed = feedparser.parse(response.content)
        
        if not feed.entries:
            logger.warning("ArXiv returned no entries")
            return []
        
        papers = []
        for entry in feed.entries:
            # Parse published date
            published_date = entry.get('published', '')
            
            # Filter by year if specified
            if year_start or year_end:
                try:
                    year = int(published_date[:4])
                    if year_start and year < int(year_start):
                        continue
                    if year_end and year > int(year_end):
                        continue
                except:
                    pass
            
            # Extract ID properly
            paper_id = entry.id.split('/abs/')[-1]
            
            paper = {
                'id': f'arxiv:{paper_id}',
                'source': 'arxiv',
                'external_id': paper_id,
                'title': entry.title,
                'abstract': entry.summary,
                'authors': json.dumps([a.name for a in entry.authors]),
                'published_date': published_date,
                'category': entry.arxiv_primary_category['term'] if 'arxiv_primary_category' in entry else 'arXiv',
                'link': entry.link
            }
            papers.append(paper)
            
        # Save to database (without embeddings)
        if papers:
            save_papers(papers)
            
        logger.info(f"Found {len(papers)} papers from ArXiv")
        return papers
    
    except requests.exceptions.Timeout:
        logger.error("ArXiv search timed out after 30 seconds")
        return []
    except requests.exceptions.RequestException as e:
        logger.error(f"ArXiv request failed: {e}")
        return []
    except Exception as e:
        logger.error(f"ArXiv search failed: {e}")
        import traceback
        traceback.print_exc()
        return []

def calculate_dynamic_cap(subjects: list, keywords: list, frequency: str, source: str) -> int:
    """
    Calculate dynamic fetch cap based on subjects, keywords, and frequency.
    
    Formula:
    - ArXiv: 30 * (num_subjects + num_keywords * 0.2) * days
    - BioRxiv: 30 * (num_keywords * 0.2) * days
    
    Args:
        subjects: List of subject categories (for ArXiv)
        keywords: List of keyword filters
        frequency: 'daily', 'weekly', or 'last_100'
        source: 'arxiv' or 'biorxiv'
    
    Returns:
        int: Maximum number of papers to fetch
    """
    if frequency == 'last_100':
        return 50  # Keep existing behavior for last_100 mode
    
    # Calculate days based on frequency
    if frequency == 'weekly':
        days = 7
    else:  # daily
        days = 1
    
    num_subjects = len(subjects)
    num_keywords = len(keywords)
    
    if source == 'arxiv':
        # ArXiv: subjects + keywords (keywords weighted at 0.2)
        cap = 30 * (num_subjects + num_keywords * 0.2) * days
    else:
        # BioRxiv: only keywords (weighted at 0.2)
        cap = 30 * (num_keywords * 0.2) * days
    
    # Ensure minimum cap of 30 and maximum of 500 (practical limit)
    return max(30, min(int(cap), 500))

def embed_paper_local(title, abstract):
    """
    Generate embedding locally using the loaded model.
    """
    model_dict = get_model()
    tokenizer = model_dict['tokenizer']
    model = model_dict['model']
    
    sep = tokenizer.sep_token
    text = f"{title}{sep}{abstract}"
    
    inputs = tokenizer(text, padding=True, truncation=True, return_tensors="pt", max_length=512)
    
    import torch
    device = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')
    model = model.to(device)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    with torch.no_grad():
        outputs = model(**inputs)
        embeddings = outputs.last_hidden_state[:, 0, :].cpu()
        return embeddings[0].numpy()

def embed_papers_local_batch(papers, batch_size=16):
    """
    Generate embeddings locally in batches. Returns a list of embeddings.
    """
    model_dict = get_model()
    tokenizer = model_dict['tokenizer']
    model = model_dict['model']
    sep = tokenizer.sep_token
    
    texts = [f"{p['title']}{sep}{p['abstract']}" for p in papers]
    
    import torch
    device = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')
    logger.info(f"Using device: {device} for batched inference")
    model = model.to(device)
    all_embeddings = []
    
    for i in range(0, len(texts), batch_size):
        logger.info(f"Generating local embeddings batch {i//batch_size + 1}/{(len(texts) + batch_size - 1)//batch_size}...")
        batch_texts = texts[i:i+batch_size]
        inputs = tokenizer(batch_texts, padding=True, truncation=True, return_tensors="pt", max_length=512)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = model(**inputs)
            embeddings = outputs.last_hidden_state[:, 0, :].cpu().numpy()
            for emb in embeddings:
                all_embeddings.append(emb)
                
    return all_embeddings

def fetch_arxiv_api_papers(subjects: list, max_results=None, keywords: list = None):
    """
    Fetch papers using ArXiv API (not RSS) to get 100+ papers.
    Used for 'last_100' and 'weekly' modes.
    
    Args:
        subjects: List of ArXiv categories (e.g., ['physics.flu-dyn', 'cs.AI'])
        max_results: Number of papers to fetch (if None, calculated dynamically)
        keywords: List of keywords to search across all categories (optional)
    
    Returns:
        int: Number of papers saved
    """
    import arxiv
    
    if keywords is None:
        keywords = []
    
    # Calculate dynamic cap if not provided
    if max_results is None:
        # Determine frequency from context - default to daily if unknown
        # This will be passed from ingest_papers
        max_results = 100  # fallback
    
    # Build query: (cat:X OR cat:Y) OR (all:"keyword1" OR all:"keyword2")
    query_parts = []
    
    if subjects:
        category_query = " OR ".join([f"cat:{s}" for s in subjects])
        query_parts.append(f"({category_query})")
    
    if keywords:
        keyword_query = " OR ".join([f'all:"{kw}"' for kw in keywords])
        query_parts.append(f"({keyword_query})")
    
    if not query_parts:
        logger.info("No subjects or keywords specified, skipping fetch")
        return 0
    
    full_query = " OR ".join(query_parts)
    logger.info(f"ArXiv API query: {full_query}")
    
    all_papers = []
    
    # Single combined search
    logger.info(f"Fetching up to {max_results} papers from ArXiv API")
    
    search = arxiv.Search(
        query=full_query,
        max_results=max_results,
        sort_by=arxiv.SortCriterion.SubmittedDate,
        sort_order=arxiv.SortOrder.Descending
    )
    
    try:
        for result in search.results():
            paper = {
                'source': 'arxiv',
                'external_id': result.entry_id.split('/')[-1].split('v')[0],  # Extract ID
                'title': result.title.replace('\n', ' '),
                'abstract': result.summary.replace('\n', ' '),
                'authors': [author.name for author in result.authors],
                'published_date': result.published.isoformat(),
                'category': ', '.join(result.categories),  # Use all categories
                'link': result.entry_id
            }
            all_papers.append(paper)
    except Exception as e:
        logger.error(f"Error fetching from ArXiv API: {e}")
        return 0
    
    if not all_papers:
        logger.info("No papers fetched from ArXiv API")
        return 0
    
    # Deduplicate
    unique_papers = {p['external_id']: p for p in all_papers}.values()
    candidates = list(unique_papers)
    
    # Sort by date DESC
    candidates.sort(key=lambda x: x['published_date'], reverse=True)
    
    # Limit total
    candidates = candidates[:max_results]
    
    logger.info(f"Fetched {len(candidates)} unique papers via ArXiv API. Fetching embeddings...")
    
    # Batch fetch embeddings (SS + local fallback)
    emb_map = fetch_ss_embeddings_batch(candidates)
    
    # Inject embeddings with local fallback
    papers_to_save = []
    missing_ss_papers = []
    
    for p in candidates:
        pid = f"arxiv:{p['external_id']}"
        if pid in emb_map:
            p['embedding'] = emb_map[pid]
            papers_to_save.append(p)
        else:
            missing_ss_papers.append(p)

    if missing_ss_papers:
        logger.info(f"{len(missing_ss_papers)} SS embeddings not found. Generating locally in batches...")
        try:
            local_embeddings = embed_papers_local_batch(missing_ss_papers)
            for p, emb in zip(missing_ss_papers, local_embeddings):
                p['embedding'] = emb
                papers_to_save.append(p)
        except Exception as e:
            logger.warning(f"Local batched embedding failed: {e}")
            for p in missing_ss_papers:
                p['embedding'] = None
                papers_to_save.append(p)
    
    saved_count = save_papers(papers_to_save)
    logger.info(f"Saved {saved_count} papers via ArXiv API")
    return saved_count

def fetch_arxiv_papers(subjects: list, frequency: str = 'daily'):
    """
    Fetches papers from ArXiv RSS feeds for given subjects.
    STRICT FILTERING:
    - Daily: Last 24 hours.
    - Weekly: Last 7 days.
    
    Returns count of saved papers.
    """
    import feedparser
    import datetime
    from dateutil import parser
    # from pytz import utc  <-- Removed dependency
    
    total_saved = 0
    now = datetime.datetime.now(datetime.timezone.utc)
    
    # Determine Cutoff
    cutoff = None
    max_results = 20 # Default
    
    if frequency == 'weekly':
        cutoff = now - datetime.timedelta(days=7)
    elif frequency == 'last_100':
        # No time cutoff (or very loose), but high limit
        cutoff = now - datetime.timedelta(days=365) # Just in case
        max_results = 100
    else:
        # Default daily
        cutoff = now - datetime.timedelta(hours=24)
        
    # Aggregate papers from all subjects first
    candidates = []
    
    try:
        for subject in subjects:
            # ArXiv RSS URL
            url = f"http://export.arxiv.org/rss/{subject}"
            logger.info(f"Fetching RSS: {url}")
            feed = feedparser.parse(url)
            
            for entry in feed.entries:
                # Check date strictness
                # published_parsed is time.struct_time in UTC usually
                if not hasattr(entry, 'published_parsed'):
                    continue
                    
                pub_dt = datetime.datetime(*entry.published_parsed[:6], tzinfo=datetime.timezone.utc)
                
                if pub_dt < cutoff:
                    # Too old
                    continue
                    
                # Parse ID
                # link: http://arxiv.org/abs/2301.12345
                # id might be URL or the ID string
                link = entry.link
                paper_id = link.split('/')[-1]
                if 'v' in paper_id: # 2301.123v1 -> 2301.123
                    # Actually for ArXiv ID we usually keep version for link but ID ...
                    # consistent with save_papers: source:external_id
                    pass
                
                # Authors are messy in RSS. 'authors' list of dicts or 'author' string?
                # feedparser normalizes usually.
                aut = []
                if 'authors' in entry:
                    aut = [a.name for a in entry.authors]
                elif 'author' in entry:
                    aut = [entry.author]
                    
                paper = {
                    'source': 'arxiv',
                    'external_id': paper_id,
                    'title': entry.title.replace('\n', ' '),
                    'abstract': entry.summary.replace('\n', ' '),
                    'authors': aut,
                    'published_date': pub_dt.isoformat(),
                    'category': subject,
                    'link': link
                }
                candidates.append(paper)
        
        if not candidates:
            logger.info("No candidates found in RSS time window.")
            return 0
            
        # Deduplicate candidates (same paper in multiple categories)
        unique_candidates = {p['external_id']: p for p in candidates}.values()
        candidates = list(unique_candidates)
        
        # Sort by date DESC
        candidates.sort(key=lambda x: x['published_date'], reverse=True)
        
        # Limit if needed (applies to "last_100" mode primarily)
        if frequency == 'last_100':
            candidates = candidates[:max_results]
            
        logger.info(f"Found {len(candidates)} candidates. Fetching embeddings...")
        
        # Batch Fetch Embeddings
        # This automatically handles DOIs if we implement logic inside fetch_ss_embeddings_batch to return them
        # But currently fetch_ss_embeddings_batch returns dict of ID -> bytes.
        
        emb_map = fetch_ss_embeddings_batch(candidates)
        
        # Inject embeddings (with local fallback)
        papers_to_save = []
        missing_ss_papers = []
        
        for p in candidates:
            pid = f"arxiv:{p['external_id']}"
            if pid in emb_map:
                p['embedding'] = emb_map[pid]
                papers_to_save.append(p)
            else:
                missing_ss_papers.append(p)

        if missing_ss_papers:
            logger.info(f"{len(missing_ss_papers)} SS embeddings not found. Generating locally in batches...")
            try:
                local_embeddings = embed_papers_local_batch(missing_ss_papers)
                for p, emb in zip(missing_ss_papers, local_embeddings):
                    p['embedding'] = emb
                    papers_to_save.append(p)
            except Exception as e:
                logger.warning(f"Local batched embedding failed: {e}")
                for p in missing_ss_papers:
                    p['embedding'] = None
                    papers_to_save.append(p)
                 
        saved_count = save_papers(papers_to_save)
        logger.info(f"Ingested {saved_count} new papers.")
        return saved_count
        
    except Exception as e:
        logger.error(f"Fetch failed: {e}")
        return 0

def fetch_biorxiv_papers(subjects: list, limit=50):
    """
    Fetches papers from BioRxiv via RSS feeds.
    Subject format: 'biorxiv.neuroscience' -> 'neuroscience'
    """
    total_saved = 0
    all_papers = []
    
    for subject in subjects:
        # Clean subject name (remove prefix if present)
        clean_subject = subject.replace('biorxiv.', '')
        rss_url = f"http://connect.biorxiv.org/biorxiv_xml.php?subject={clean_subject}"
        logger.info(f"Fetching BioRxiv RSS: {rss_url}")
        
        try:
            feed = feedparser.parse(rss_url)
            
            for entry in feed.entries[:limit]:
                # BioRxiv ID is usually in the link or DOI
                # Link: http://biorxiv.org/content/early/2023/10/01/2023.09.29.560123
                # We can use the DOI or the last part of URL as ID
                paper_id = entry.link.split('/')[-1]
                
                # Authors in RSS might be a string or list
                # heuristic: entry.get('authors') or entry.get('author')
                authors = [a.name for a in entry.authors] if 'authors' in entry else [entry.get('author', 'Unknown')]
                
                date_str = entry.get('updated', entry.get('date', '2025-01-01'))
                
                paper = {
                    'source': 'biorxiv',
                    'external_id': paper_id,
                    'title': entry.title,
                    'abstract': entry.summary,
                    'authors': authors,
                    'published_date': date_str,
                    'category': subject, # Use full subject (e.g. biorxiv.neuroscience) for filtering
                    'link': entry.link
                }
                all_papers.append(paper)
            
        except Exception as e:
            logger.error(f"BioRxiv fetch failed for {subject}: {e}")
    
    # Batch fetch embeddings for all biorxiv papers
    if all_papers:
        logger.info(f"Fetching embeddings for {len(all_papers)} BioRxiv papers...")
        emb_map = fetch_ss_embeddings_batch(all_papers)
        
        # Inject embeddings
        for p in all_papers:
            pid = f"biorxiv:{p['external_id']}"
            if pid in emb_map:
                p['embedding'] = emb_map[pid]
        
        total_saved = save_papers(all_papers)
            
    return total_saved

def search_semantic_scholar(query: str, year: str = None, limit=10):
    """
    Searches Semantic Scholar Graph API.
    year: specific year ('2019') or range ('2019-2023')
    """
    logger.info(f"Searching Semantic Scholar: {query} (Year: {year})")
    url = "https://api.semanticscholar.org/graph/v1/paper/search"
    params = {
        "query": query,
        "limit": limit,
        "fields": "paperId,title,abstract,authors,year,url,venue,embedding.specter_v2"
    }
    if year:
        params['year'] = year
    
    try:
        from config import SEMANTIC_SCHOLAR_API_KEY
        headers = {
            "User-Agent": "DailyArXiv/1.0 (mailto:dailyarxiv@example.com)",
            "x-api-key": SEMANTIC_SCHOLAR_API_KEY
        }
        r = requests.get(url, params=params, headers=headers)
        r.raise_for_status()
        data = r.json()
        
        papers = []
        for item in data.get('data', []):
            if not item.get('title'):
                continue
                
            paper = {
                'source': 'semantic_scholar',
                'external_id': item['paperId'],
                'title': item['title'],
                'abstract': item.get('abstract', '') or '', # Handle None
                'authors': [a['name'] for a in item.get('authors', [])],  # Keep as list for save_papers JSON conversion
                'published_date': str(item.get('year', '')), # Approx date
                'category': item.get('venue', 'Unknown'),
                'link': item.get('url', f"https://www.semanticscholar.org/paper/{item['paperId']}"),
                'ss_embedding': item.get('embedding', {}).get('vector') if item.get('embedding') else None
            }
            papers.append(paper)
            
        # Save papers to database (without embeddings for now - lazy loading)
        save_papers(papers)
        
        # Return the paper list so they can be rendered
        # Note: We need to add the database 'id' field that save_papers would create
        for paper in papers:
            paper['id'] = f"{paper['source']}:{paper['external_id']}"
            
        logger.info(f"Found {len(papers)} papers from Semantic Scholar")
        return papers
            
    except requests.exceptions.HTTPError as e:
        if e.response.status_code == 429:
            logger.error(f"Semantic Scholar rate limit exceeded. Please wait a moment and try again.")
            return []
        else:
            logger.error(f"Semantic Scholar HTTP Error: {e}")
            return []
    except Exception as e:
        logger.error(f"Semantic Scholar Search Failed: {e}")
        import traceback
        traceback.print_exc()
        return []

def backfill_ss_embeddings_for_user(user_id):
    """
    Finds papers that the user has interacted with which do not have a Semantic Scholar embedding.
    Requests them from Semantic Scholar API, respecting the 1 request per second limit.
    """
    global _last_semantic_scholar_call
    from config import SEMANTIC_SCHOLAR_API_KEY
    
    conn = get_db_connection()
    c = conn.cursor()
    
    # Find papers with interactions but no SS embedding
    query = """
        SELECT p.id, p.source, p.external_id, p.title
        FROM interactions i
        JOIN papers p ON i.paper_id = p.id
        WHERE i.user_id = ? AND p.embedding IS NULL
    """
    c.execute(query, (user_id,))
    rows = c.fetchall()
    
    if not rows:
        conn.close()
        return 0
    
    logger.info(f"Backfilling SS embeddings for {len(rows)} papers")
    
    count = 0
    import numpy as np
    
    for row in rows:
        paper_id, source, external_id, title = row
        
        # Respect rate limit (1 req/sec)
        time_since_last = time.time() - _last_semantic_scholar_call
        if time_since_last < 1.1: # Be slightly conservative
            time.sleep(1.1 - time_since_last)
        
        _last_semantic_scholar_call = time.time()
        
        # Construct lookup ID for Semantic Scholar
        ss_lookup_id = None
        if source == 'arxiv':
            # Remove version if present (e.g. 2101.12345v1 -> 2101.12345)
            clean_id = external_id.split('v')[0]
            ss_lookup_id = f"ARXIV:{clean_id}"
        elif source == 'semantic_scholar':
            ss_lookup_id = external_id
        
        if not ss_lookup_id:
            # For other sources, try searching by title
            url = "https://api.semanticscholar.org/graph/v1/paper/search/match"
            params = {"query": title}
        else:
            url = f"https://api.semanticscholar.org/graph/v1/paper/{ss_lookup_id}"
            params = {"fields": "embedding.specter_v2"}
            
        headers = {
            "User-Agent": "DailyArXiv/1.0 (mailto:dailyarxiv@example.com)",
            "x-api-key": SEMANTIC_SCHOLAR_API_KEY
        }
        
        try:
            r = requests.get(url, params=params, headers=headers, timeout=10)
            if r.status_code == 200:
                data = r.json()
                # If matched via search
                if "data" in data and len(data["data"]) > 0:
                    paper_data = data["data"][0]
                else:
                    paper_data = data
                
                if not paper_data:
                    logger.warning(f"No data returned for {paper_id}")
                    continue
                    
                emb_data = paper_data.get('embedding', {}).get('vector')
                if emb_data:
                    ss_emb_bytes = np.array(emb_data, dtype=np.float32).tobytes()
                    c.execute("UPDATE papers SET embedding = ? WHERE id = ?", (ss_emb_bytes, paper_id))
                    count += 1
            elif r.status_code == 429:
                logger.warning("SS Rate limit hit during backfill, stopping for now.")
                break
            else:
                logger.warning(f"SS Lookup failed for {paper_id}: {r.status_code}")
                
        except Exception as e:
            logger.error(f"Error backfilling SS embedding for {paper_id}: {e}")
            continue

    conn.commit()
    conn.close()
    logger.info(f"Successfully backfilled {count} SS embeddings")
    return count

def ingest_papers(subjects: list, frequency: str = 'daily', keywords: list = None):
    """
    Main entry point for ingestion. Dispatches to appropriate fetcher.
    
    Args:
        subjects: List of categories to fetch
        frequency: 'daily', 'weekly', or 'last_100'
        keywords: List of keywords to search across all categories
    """
    if keywords is None:
        keywords = []
    
    arxiv_subjects = []
    biorxiv_subjects = []
    
    for s in subjects:
        if s.startswith('biorxiv.'):
            biorxiv_subjects.append(s)
        else:
            arxiv_subjects.append(s)
            
    total = 0
    
    # Calculate dynamic caps based on formula
    arxiv_cap = calculate_dynamic_cap(arxiv_subjects, keywords, frequency, 'arxiv')
    biorxiv_cap = calculate_dynamic_cap(biorxiv_subjects, keywords, frequency, 'biorxiv')
    
    logger.info(f"Dynamic fetch caps - ArXiv: {arxiv_cap}, BioRxiv: {biorxiv_cap}")
    
    if arxiv_subjects:
        if frequency in ['weekly', 'last_100']:
            # Use ArXiv API for weekly and last_50 modes (proper query)
            # For last_100, use fixed 50; for weekly/daily, use dynamic cap
            if frequency == 'last_100':
                max_papers = 50
            else:
                max_papers = arxiv_cap
            
            total += fetch_arxiv_api_papers(arxiv_subjects, max_results=max_papers, keywords=keywords)
        else:
            # Use RSS for daily (faster, limited to ~30-50 papers)
            # Still apply dynamic cap for daily mode
            total += fetch_arxiv_papers(arxiv_subjects, frequency=frequency)
            
    if biorxiv_subjects:
        # Use dynamic cap for BioRxiv, but cap at 100 max (practical limit)
        biorxiv_limit = min(biorxiv_cap, 100)
        total += fetch_biorxiv_papers(biorxiv_subjects, limit=biorxiv_limit)
        
    return total

if __name__ == "__main__":
    # Test run
    print("Testing Ingestion...")
    # fetch_arxiv_papers(['cs.CL'], max_results=5)
    # search_semantic_scholar("Attention is all you need")
    # ingest_papers(['cs.CL', 'biorxiv.neuroscience'])
