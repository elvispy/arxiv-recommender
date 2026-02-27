from fasthtml.common import *
import sqlite3
import json
import logging
import threading
import datetime
from db import get_db_connection
from config import LAN_BIND_HOST, PORT, DEFAULT_FETCH_FREQUENCY

# Setup logger
logger = logging.getLogger(__name__)

# Define a simple default user for now (Single User Mode for MVP start)
DEFAULT_USER_ID = 1

# Background update state
update_status = {
    'in_progress': False,
    'last_update': None,
    'error': None,
    'progress': ''
} 

app, rt = fast_app(
    hdrs=(
        Meta(name="viewport", content="width=device-width, initial-scale=1"),
        Meta(http_equiv="Cache-Control", content="no-cache, no-store, must-revalidate"),
        Meta(http_equiv="Pragma", content="no-cache"),
        Meta(http_equiv="Expires", content="0"),
        Link(rel='stylesheet', href='/static/style.css?v=6'),
        # MathJax with inline configuration
        Script("""
        MathJax = {
          tex: {
            inlineMath: [['$', '$'], ['\\\\(', '\\\\)']]
          }
        };
        """),
        Script(id="MathJax-script", async_=True, src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js")
    ),
    htmlkw={"data-theme": "dark"}
)

def get_daily_feed(user_id, limit=100):
    conn = get_db_connection()
    c = conn.cursor()
    
    # 1. Get User Settings
    c.execute("SELECT fetch_frequency, target_subjects FROM user_settings WHERE user_id = ?", (user_id,))
    row = c.fetchone()
    if not row:
        conn.close()
        return [], []
    
    fetch_frequency = row['fetch_frequency']
    
    # Get selected topics
    target_subjects = json.loads(row['target_subjects']) if row['target_subjects'] else []
    
    if not target_subjects:
        # No topics selected, return empty
        conn.close()
        return [], []
    
    import datetime
    import numpy as np
    
    # 2. Find the MOST RECENT publication date in user's selected topics
    # This ensures we show the "latest batch" regardless of how old it is
    topic_conditions = " OR ".join([f"category LIKE '%{subj}%'" for subj in target_subjects])
    
    c.execute(f"""
        SELECT MAX(published_date) as latest_date
        FROM papers 
        WHERE ({topic_conditions})
    """)
    latest_row = c.fetchone()
    
    if not latest_row or not latest_row['latest_date']:
        # No papers in these topics at all
        conn.close()
        return [], []
    
    latest_date = latest_row['latest_date']
    
    # Parse the latest date to get a cutoff (same day or within 24h to catch batch publications)
    try:
        latest_dt = datetime.datetime.fromisoformat(latest_date.replace('Z', '+00:00'))
        # Show papers from the same "batch"
        if fetch_frequency == 'weekly':
            cutoff_dt = latest_dt - datetime.timedelta(days=7)
        else:
            cutoff_dt = latest_dt - datetime.timedelta(hours=24)
        cutoff_date = cutoff_dt.isoformat()
    except:
        # Fallback if date parsing fails
        cutoff_date = latest_date
        
    # 3. Get User Profile Vector
    c.execute("SELECT preference_vector FROM user_profile WHERE user_id = ?", (user_id,))
    prof_row = c.fetchone()
    user_vector = None
    if prof_row and prof_row[0]:
        user_vector = np.frombuffer(prof_row[0], dtype=np.float32)

    # 4. Fetch CANDIDATES from Database (LATEST BATCH + TOPIC FILTER)
    # CRITICAL: Exclude papers the user has already interacted with
    
    if fetch_frequency == 'last_100':
        # Last 50 papers mode (regardless of date)
        query = f"""
            SELECT * FROM papers 
            WHERE ({topic_conditions})
            AND id NOT IN (
                SELECT paper_id FROM interactions WHERE user_id = ?
            )
            ORDER BY published_date DESC
            LIMIT 50
        """
        c.execute(query, (user_id,))
    else:
        # Standard Daily/Weekly batch logic
        query = f"""
            SELECT * FROM papers 
            WHERE published_date >= ? AND ({topic_conditions})
            AND id NOT IN (
                SELECT paper_id FROM interactions WHERE user_id = ?
            )
            ORDER BY published_date DESC
        """
        c.execute(query, (cutoff_date, user_id))
    
    rows = c.fetchall()
    
    candidates = [dict(r) for r in rows]
    
    ranked_papers = []
    
    if not candidates:
        conn.close()
        return [], []

    # 5. Rank Candidates
    
    if user_vector is not None:
        logger.info(f"Ranking {len(candidates)} papers using Rocchio profile")
        # Filter candidates that have embeddings
        # Remember for refactor: we use 'embedding' column which now has SS embeddings
        embed_candidates = [p for p in candidates if p['embedding'] is not None]
        non_embed_candidates = [p for p in candidates if p['embedding'] is None]
        
        logger.info(f"  - {len(embed_candidates)} papers with embeddings")
        logger.info(f"  - {len(non_embed_candidates)} papers without embeddings")
        
        if embed_candidates:
            # Prepare matrix
            matrix = []
            valid_cands = []
            
            # Reshape user vector: (K*768,) -> (K, 768)
            K = 3
            try:
                user_centroids = user_vector.reshape(K, 768)  # Shape: (3, 768)
                logger.info(f"Using {K} interest centroids for ranking")
            except:
                # Fallback if profile is old format (single 768-dim vector)
                logger.warning("Profile in old format, treating as single centroid")
                user_centroids = np.array([user_vector])  # Shape: (1, 768)
                K = 1
            
            for p in embed_candidates:
                try:
                    vec = np.frombuffer(p['embedding'], dtype=np.float32)
                    if vec.shape[0] == 768:  # Valid embedding
                        matrix.append(vec)
                        valid_cands.append(p)
                except:
                    pass
            
            if matrix:
                # Multi-Interest Cosine Similarity
                mat = np.array(matrix)  # Shape: (N_papers, 768)
                
                # Compute similarity matrix: (N_papers, K_interests)
                # For each paper, compute cos_sim to each of K centroids
                scores_per_interest = []
                for k in range(K):
                    centroid = user_centroids[k]
                    u_norm = np.linalg.norm(centroid)
                    v_norms = np.linalg.norm(mat, axis=1)
                    
                    # Cosine similarity for this centroid
                    sims = np.dot(mat, centroid) / (u_norm * v_norms + 1e-9)
                    scores_per_interest.append(sims)
                
                # Max scoring: Take best score across all interests
                scores_matrix = np.array(scores_per_interest)  # Shape: (K, N_papers)
                scores = np.max(scores_matrix, axis=0)  # Shape: (N_papers,)
                
                # Sort
                sorted_indices = np.argsort(-scores)
                
                ranked_papers = [valid_cands[i] for i in sorted_indices]
                logger.info(f"  ✅ Ranked {len(ranked_papers)} papers by max similarity across {K} interests")
                
        # Append non-embedded at bottom
        ranked_papers.extend(non_embed_candidates)
        
        
    else:
        # Cold start: Return chronological
        logger.info(f"No user profile - showing {len(candidates)} papers chronologically")
        ranked_papers = candidates

    conn.close()
    
    return ranked_papers, []

def NavBar():
    return Div(
        Div(
            Div(
                A("Daily ArXiv", href="/", cls="nav-logo"),
                Div(
                    A("History", href="/history", cls="nav-link"),
                    A("Search", href="/search", cls="nav-link"),
                    A("Settings", href="/settings", cls="nav-link"),
                    cls="nav-links"
                ),
                style="display: flex; justify-content: space-between; align-items: center; max-width: 1400px; margin: 0 auto; padding: 0 20px;"
            ),
            cls="nav-bar"
        )
    )

def GridCard(paper, mode="feed", interaction=None):
    # Convert Row to dict if needed
    if not isinstance(paper, dict):
        paper = dict(paper)
    
    # Handle authors field - should be a JSON string in the database
    authors_field = paper.get('authors')
    
    if authors_field is None:
        authors = ["Unknown Author"]
    elif isinstance(authors_field, str):
        try:
            authors = json.loads(authors_field)
            # Handle double-encoded JSON (sometimes happens)
            if isinstance(authors, str):
                authors = json.loads(authors)
            if not isinstance(authors, list):
                logger.warning(f"Authors JSON parsed but not a list: {type(authors)}")
                authors = [str(authors)]
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse authors JSON: {authors_field[:100]}, error: {e}")
            authors = ["Unknown Author"]
    elif isinstance(authors_field, list):
        authors = authors_field
    else:
        logger.warning(f"Authors field is unexpected type {type(authors_field)}: {authors_field}")
        authors = ["Unknown Author"]
    
    # Build author string
    # Build author string: Full list, comma-separated (was previously synthesizing to 'et al.')
    if not authors or len(authors) == 0:
        author_str = "Unknown Author"
    else:
        author_str = ", ".join(authors)
    
    # Determine active states based on interaction
    liked = interaction == 'like'
    disliked = interaction == 'dismiss'
    
    # Standard Actions (Like/Dislike)
    standard_actions = Div(
        Button("👎", 
               hx_post=f"/interact/{DEFAULT_USER_ID}/{paper['id']}/dismiss", 
               hx_target="closest .grid-card", # Update the whole card to reflect state change
               hx_swap="outerHTML",
               cls=f"btn-icon like-btn like-btn-dislike action-btn {'disliked' if disliked else ''}",
               title="Dislike",
               onclick="event.stopPropagation()",
               style="margin-right: 10px;" + ("opacity: 1;" if disliked else "opacity: 0.6;")),
        Button("♥", 
               hx_post=f"/interact/{DEFAULT_USER_ID}/{paper['id']}/like", 
               hx_target="closest .grid-card",
               hx_swap="outerHTML",
               cls=f"btn-icon like-btn like-btn-heart action-btn {'liked' if liked else ''}",
               onclick="event.stopPropagation()",
               title="Like"),
        cls="action-buttons",
        style="display: flex; align-items: center;"
    )

    pdf_link = A("PDF", href=paper['link'], target="_blank", cls="btn-link action-btn", onclick="event.stopPropagation()")
    
    if mode == "history":
        # History Mode: PDF + Standard Actions + Remove
        actions = Div(
            pdf_link,
            Div(
                standard_actions,
                Button("🗑️", 
                       hx_post=f"/interact/{DEFAULT_USER_ID}/{paper['id']}/remove", 
                       hx_target="closest .grid-card", 
                       hx_swap="outerHTML",
                       cls="like-btn like-btn-remove action-btn",
                       title="Remove",
                       onclick="event.stopPropagation()",
                       style="margin-left: 10px; color: #dc3545; border-color: #dee2e6;"),
                style="display: flex; align-items: center;"
            ),
            cls="grid-actions",
            style="justify-content: space-between;"
        )
    else:
        # Feed Mode: PDF + Standard Actions
        actions = Div(
            pdf_link,
            standard_actions,
            cls="grid-actions"
        )

    return Div(
        Div(
            Span(paper['category'], cls="category-tag"),
            Span(str(paper['published_date'])[:10], cls="date-tag"),
            cls="card-top"
        ),
        H3(paper['title'], cls="grid-title"),
        P(author_str, cls="grid-author"),
        
        actions,
        
        # Button is kept for visual affordance, but clicks bubble up to the card
        Button("Read Abstract", cls="btn-expand action-btn"),
        
        cls="grid-card fade-in",
        onclick="openModal(this)",
        **{"data-title": paper['title'], "data-authors": author_str, "data-abstract": paper['abstract'], "data-pdf": paper['link']}
    )

def render_feed_content(recommended, recent):
    # Merge lists, prioritizing recommended
    seen_ids = set()
    all_papers = []
    
    for p in recommended:
        if p['id'] not in seen_ids:
            all_papers.append(p)
            seen_ids.add(p['id'])
            
    for p in recent:
        if p['id'] not in seen_ids:
            all_papers.append(p)
            seen_ids.add(p['id'])

    # Split into initial view and "show more"
    initial_count = 12
    top_papers = all_papers[:initial_count]
    more_papers = all_papers[initial_count:]

    return Div(
        # Main Grid
        Div(
            *[GridCard(p) for p in top_papers],
            cls="dashboard-grid"
        ) if top_papers else Div(
            P("Interact with papers to get personalized recommendations!", cls="text-secondary", style="text-align: center; margin: 40px 0; font-size: 1.2rem;"),
            P("Try searching for topics you like or exploring the 'Search' tab.", style="text-align: center; color: var(--text-tertiary);")
        ),
        
        # Show More
        Details(
            Summary("Show More Papers", cls="btn-primary", style="margin: 30px auto; display: block; width: fit-content; list-style: none; cursor: pointer; color: var(--bg-color);"),
            Div(
                *[GridCard(p) for p in more_papers],
                cls="dashboard-grid",
                style="margin-top: 20px;"
            ),
            style="width: 100%;"
        ) if more_papers else None
    )

@rt('/')
def get():
    recommended, recent = get_daily_feed(DEFAULT_USER_ID)
    
    # Check if we have new interactions since last update?
    # For now, we always allow update to be "proactive". 
    # To "grey it out", we would need to check if (User Last Interaction Time > User Profile Last Updated Time)
    conn = get_db_connection()
    c = conn.cursor()
    c.execute("SELECT last_updated FROM user_profile WHERE user_id = ?", (DEFAULT_USER_ID,))
    p_row = c.fetchone()
    last_profile_update = p_row[0] if (p_row and p_row[0]) else None
    
    c.execute("SELECT MAX(timestamp) FROM interactions WHERE user_id = ?", (DEFAULT_USER_ID,))
    i_row = c.fetchone()
    last_interaction = i_row[0] if (i_row and i_row[0]) else None
    needs_manual_fetch, needs_daily_fetch = check_feed_status(DEFAULT_USER_ID, conn)
    conn.close()
    
    # Auto-start daily fetch if needed and not already running
    if needs_daily_fetch and not update_status.get('in_progress'):
        logger.info("Daily fetch needed, starting background update automatically")
        update_status['in_progress'] = True
        update_status['progress'] = 'Starting automatic daily update...'
        thread = threading.Thread(target=do_update_background, daemon=True)
        thread.start()
        
    # Enable if no profile OR new interactions OR feed is manually stale
    needs_profile_update = (last_profile_update is None) or (last_interaction is not None and str(last_interaction) > str(last_profile_update))
    can_update = needs_profile_update or needs_manual_fetch
    
    btn_attrs = {
        "hx_post": "/update_feed",
        "hx_target": "#feed-container",
        "hx_indicator": "#update-progress",
        "cls": "btn-primary",
        "style": "font-size: 0.9rem; padding: 8px 16px; display: flex; align-items: center; gap: 8px;"
    }
    
    if update_status.get('in_progress'):
        btn_attrs['disabled'] = True
        btn_attrs['style'] += " background-color: #6c757d; border-color: #6c757d; cursor: not-allowed; opacity: 0.6;"
        button_text = "Update in Progress..."
        
        indicator = Div(
            P("⏳ Updating...", style="color: var(--accent); font-weight: 600;"),
            P(update_status.get('progress', 'Working...'), style="color: var(--text-secondary); font-size: 0.9rem;"),
            hx_get="/update_status",
            hx_trigger="every 1s",
            hx_swap="outerHTML",
            id="update-indicator"
        )
    else:
        if not can_update:
            btn_attrs['disabled'] = True
            btn_attrs['style'] += " background-color: #6c757d; border-color: #6c757d; cursor: not-allowed; opacity: 0.6;"
            button_text = "Recommender Up to Date"
        else:
            if needs_manual_fetch and not needs_profile_update:
                button_text = "⚡ Fetch New Papers"
            else:
                button_text = "⚡ Update Recommendations"
                
        indicator = Div(id="update-indicator", style="margin-top: 10px;")

    return Title("Daily ArXiv"), NavBar(), Main(
        
        # Dashboard Header with Update Button
        Div(
            Div(
                Div(
                    H1("Top Recommendations", style="font-size: 1.5rem; margin: 0; white-space: nowrap; color: var(--text-primary);"),
                    Div(
                        "?",
                        Div(
                            Table(
                                Tr(Td(Kbd("1"),"-",Kbd("9")), Td("Open cards 1 to 9")),
                                Tr(Td(Kbd("←"), "/", Kbd("→")), Td("Navigate previous/next card")),
                                Tr(Td(Kbd("L")), Td("Like current card")),
                                Tr(Td(Kbd("D")), Td("Dislike current card")),
                                Tr(Td(Kbd("P")), Td("Open PDF of current card in new tab")),
                                Tr(Td(Kbd("Esc")), Td("Close paper view")),
                                style="border-collapse: collapse; width: 100%;"
                            ),
                            cls="tooltip-text"
                        ),
                        cls="tooltip-container"
                    ),
                    style="display: flex; align-items: center;"
                ),
                Button(button_text, **btn_attrs),
                
                style="display: flex; justify-content: space-between; align-items: center; gap: 15px;"
            ),
            # Status indicator (will be replaced by polling)
            indicator,
            Progress(id="update-progress", cls="htmx-indicator", style="width: 100%; margin-top: 10px; height: 4px; border-radius: 2px;", max="100"),
            style="margin-bottom: 30px;"
        ),

        # Feed Container
        Div(
            render_feed_content(recommended, recent),
            id="feed-container"
        ),
        
        # Global Abstract Modal
        Div(
            Div(
                Button("×", cls="modal-close-btn", onclick="closeModal()"),
                H2("", id="modal-title", cls="modal-title"),
                P("", id="modal-authors", cls="modal-authors"),
                Div("", id="modal-abstract", cls="modal-abstract"),
                cls="modal-content",
                onclick="event.stopPropagation()" # Prevent click from bubbling to overlay
            ),
            id="abstract-modal",
            cls="modal-overlay",
            onclick="closeModal()" # Clicking background closes modal
        ),
        Script("""
            function openModal(btnEl) {
                const title = btnEl.getAttribute('data-title');
                const authors = btnEl.getAttribute('data-authors');
                const abstract = btnEl.getAttribute('data-abstract');
                window.currentPdfLink = btnEl.getAttribute('data-pdf');
                
                document.getElementById('modal-title').innerText = title;
                document.getElementById('modal-authors').innerText = authors;
                
                // Format abstract (replace newlines with br tags for nicer reading if needed)
                document.getElementById('modal-abstract').innerText = abstract;
                
                // Explicitly trigger MathJax to render LaTeX equations in the newly injected text
                if (typeof MathJax !== 'undefined' && MathJax.typesetPromise) {
                    MathJax.typesetPromise([document.getElementById('abstract-modal')]).catch(function (err) {
                        console.error('MathJax rendering failed: ' + err.message);
                    });
                }
                
                document.getElementById('abstract-modal').classList.add('active');
                document.body.classList.add('modal-open');
                
                // Track active card index for keyboard shortcuts
                const cards = Array.from(document.querySelectorAll('.grid-card'));
                window.currentActiveIndex = cards.indexOf(btnEl);
                
                // Optional: Store focus to return to it later
                window.lastFocusedElement = btnEl;
            }
            
            function closeModal() {
                document.getElementById('abstract-modal').classList.remove('active');
                document.body.classList.remove('modal-open');
                window.currentActiveIndex = -1;
                
                if (window.lastFocusedElement) {
                    window.lastFocusedElement.focus();
                }
            }
            
            // Keyboard shortcuts
            document.addEventListener('keydown', function(event) {
                // Ignore keypresses inside input fields to avoid breaking the Search bar
                if (event.target.tagName === 'INPUT' || event.target.tagName === 'TEXTAREA') return;
                
                if (event.key === 'Escape') {
                    closeModal();
                    return;
                }
                
                // 1-9 shortcuts (works both independently and when a modal is already open)
                const keyNum = parseInt(event.key);
                if (keyNum >= 1 && keyNum <= 9 && !isNaN(keyNum)) {
                    const cards = document.querySelectorAll('.grid-card');
                    if (cards.length >= keyNum) {
                        const targetCard = cards[keyNum - 1]; // 0-indexed array
                        targetCard.click();
                    }
                    return;
                }
                
                // 'l' (like), 'd' (dislike), and Arrow shortcuts only when modal is open
                if (document.body.classList.contains('modal-open') && window.currentActiveIndex !== undefined && window.currentActiveIndex >= 0) {
                    const cards = document.querySelectorAll('.grid-card');
                    const activeCard = cards[window.currentActiveIndex];
                    
                    if (event.key === 'ArrowLeft') {
                        if (window.currentActiveIndex > 0) {
                            cards[window.currentActiveIndex - 1].click();
                        }
                    } else if (event.key === 'ArrowRight') {
                        if (window.currentActiveIndex < cards.length - 1) {
                            cards[window.currentActiveIndex + 1].click();
                        }
                    } else if (activeCard) {
                        const keyLower = event.key.toLowerCase();
                        if (keyLower === 'l') {
                            const likeBtn = activeCard.querySelector('.like-btn-heart');
                            if (likeBtn) likeBtn.click();
                        } else if (keyLower === 'd') {
                            const dislikeBtn = activeCard.querySelector('.like-btn-dislike');
                            if (dislikeBtn) dislikeBtn.click();
                        } else if (keyLower === 'p') {
                            if (window.currentPdfLink) {
                                window.open(window.currentPdfLink, '_blank');
                            }
                        }
                    }
                }
            });
        """),
        
        cls="main-container"
    )

def check_feed_status(user_id, conn):
    """Returns (needs_manual_fetch, needs_daily_fetch)"""
    c = conn.cursor()
    c.execute("SELECT last_fetch_date FROM user_settings WHERE user_id = ?", (user_id,))
    row = c.fetchone()
    
    now = datetime.datetime.utcnow()
    
    if row:
        last_fetch_str = row['last_fetch_date']
        
        if not last_fetch_str:
            return True, False # Settings changed or never fetched
        else:
            try:
                last_fetch = datetime.datetime.fromisoformat(last_fetch_str)
                if (now - last_fetch).total_seconds() >= 3600 * 24:
                    return False, True # Daily fetch due
            except:
                return True, False
    else:
        return True, False
    return False, False

def do_update_background():
    """
    Background worker that fetches papers, generates embeddings, and updates profile.
    Updates global update_status to communicate with the frontend.
    """
    global update_status
    
    try:
        update_status['in_progress'] = True
        update_status['error'] = None
        update_status['progress'] = 'Checking feed status...'
        
        conn = get_db_connection()
        c = conn.cursor()
        
        needs_manual, needs_daily = check_feed_status(DEFAULT_USER_ID, conn)
        should_fetch = needs_manual or needs_daily
        
        c.execute("SELECT fetch_frequency, target_subjects, keywords FROM user_settings WHERE user_id = ?", (DEFAULT_USER_ID,))
        row = c.fetchone()
        
        subjects = []
        keywords = []
        now = datetime.datetime.utcnow()
        freq = 'daily'
        
        if row:
            freq = row['fetch_frequency']
            subjects = json.loads(row['target_subjects']) if row['target_subjects'] else []
            keywords = json.loads(row['keywords']) if row['keywords'] else []
        else:
            subjects = ['cs.AI', 'cs.LG']
            
        if should_fetch:
            update_status['progress'] = 'Fetching papers from ArXiv...'
            logger.info("Feed is stale. Fetching from ArXiv RSS...")
            from ingest import ingest_papers
            count = ingest_papers(subjects, frequency=freq if row else 'daily', keywords=keywords)
            
            update_status['progress'] = 'Cleaning up old papers...'
            try:
                from cleanup_old_papers import cleanup_old_papers
                cleanup_old_papers(days_to_keep=30)
            except Exception as e:
                logger.warning(f"Cleanup failed: {e}")
            
            c.execute("UPDATE user_settings SET last_fetch_date = ? WHERE user_id = ?", (now.isoformat(), DEFAULT_USER_ID))
            conn.commit()
        else:
            logger.info("Feed is fresh. Skipping ArXiv fetch.")
            
        conn.close()
        
        update_status['progress'] = 'Generating embeddings...'
        from ingest import backfill_ss_embeddings_for_user
        backfill_ss_embeddings_for_user(DEFAULT_USER_ID)
        
        update_status['progress'] = 'Updating profile...'
        from core.rocchio import recalculate_user_profile
        recalculate_user_profile(DEFAULT_USER_ID)
        
        update_status['in_progress'] = False
        update_status['last_update'] = datetime.datetime.now().isoformat()
        update_status['progress'] = 'Complete'
        logger.info("Background update completed successfully")
        
    except Exception as e:
        logger.error(f"Background update failed: {e}")
        update_status['in_progress'] = False
        update_status['error'] = str(e)
        update_status['progress'] = 'Error'

@rt('/update_feed')
def post():
    """
    Start background update and return loading indicator.
    """
    global update_status
    
    # Check if already running
    if update_status['in_progress']:
        logger.info("Update already in progress, ignoring request")
        return Div(
            P("⏳ Update in progress...", style="color: var(--accent); font-weight: 600;"),
            P(update_status['progress'], style="color: var(--text-secondary); font-size: 0.9rem;"),
            id="update-indicator"
        )
    
    # Start background thread
    logger.info("Starting background update thread")
    thread = threading.Thread(target=do_update_background, daemon=True)
    thread.start()
    
    # Return loading indicator immediately
    return Div(
        P("⏳ Updating...", style="color: var(--accent); font-weight: 600;"),
        P("Fetching papers...", style="color: var(--text-secondary); font-size: 0.9rem;"),
        hx_get="/update_status",
        hx_trigger="every 1s",
        hx_swap="outerHTML",
        id="update-indicator"
    )

@rt('/update_status')
def get():
    """
    Polling endpoint that returns current update status.
    Auto-refreshes feed when complete.
    """
    global update_status
    
    if update_status['in_progress']:
        # Still working
        return Div(
            P("⏳ Updating...", style="color: var(--accent); font-weight: 600;"),
            P(update_status['progress'], style="color: var(--text-secondary); font-size: 0.9rem;"),
            hx_get="/update_status",
            hx_trigger="every 1s",
            hx_swap="outerHTML",
            id="update-indicator"
        )
    elif update_status['error']:
        # Error occurred
        return Div(
            P("❌ Update failed", style="color: red; font-weight: 600;"),
            P(f"Error: {update_status['error']}", style="color: var(--text-secondary); font-size: 0.9rem;"),
            id="update-indicator"
        )
    else:
        # Complete - trigger feed refresh
        recommended, recent = get_daily_feed(DEFAULT_USER_ID)
        
        return Div(
            # Status indicator
            P("✅ Updated successfully!", style="color: green; font-weight: 600;"),
            # Feed refresh
            Div(
                render_feed_content(recommended, recent),
                id="feed-container",
                hx_swap_oob="true"  # Out-of-band swap to update feed
            ),
            id="update-indicator"
        )

@rt('/settings')
def get(saved: bool = False):
    from config import SUBJECT_CHOICES
    
    conn = get_db_connection()
    c = conn.cursor()
    c.execute("SELECT fetch_frequency, target_subjects, keywords FROM user_settings WHERE user_id = ?", (DEFAULT_USER_ID,))
    row = c.fetchone()
    conn.close()
    
    freq = row[0] if row else 'daily'
    current_topics = json.loads(row[1]) if (row and row[1]) else []
    keywords = json.loads(row[2]) if (row and row[2]) else []
    
    # Flatten subjects for JS autocomplete
    subjects_flat = []
    for category, subjects in SUBJECT_CHOICES.items():
        for code, name in subjects.items():
            subjects_flat.append({"code": code, "name": name})
    
    msg = Div(P("Settings Saved Successfully!", style="color: green; font-weight: bold; margin-bottom: 20px;")) if saved else None
    
    return Title("Settings"), NavBar(), Main(
        Div(
            H1("Settings", cls="section-title"),
            msg,
            Form(
                Div(
                    Label("Fetch Type", cls="form-label"),
                    Select(
                        Option("Last 24 Hours", value="daily", selected=(freq=='daily')),
                        Option("Last 7 Days", value="weekly", selected=(freq=='weekly')),
                        Option("Last 50 Papers", value="last_100", selected=(freq=='last_100')),
                        name="frequency",
                        cls="form-input"
                    ),
                    cls="form-group"
                ),
                Div(
                    Label("Target Topics", cls="form-label"),
                    P("Search for topics by name or abbreviation. Use ↑/↓ arrows and Enter or Tab to select.", 
                      style="font-size: 0.9rem; color: var(--text-secondary); margin-bottom: 10px;"),
                    Div(
                        Div(id="topic-tags", style="display: flex; gap: 8px; flex-wrap: wrap; margin-bottom: 10px;"),
                        Div(
                            Input(
                                type="text",
                                id="topic-input",
                                placeholder="Search target topics...",
                                cls="form-input",
                                style="margin-bottom: 0;",
                                autocomplete="off"
                            ),
                            Ul(id="topic-dropdown", cls="autocomplete-dropdown"),
                            cls="autocomplete-container"
                        ),
                        Input(type="hidden", name="topics", id="topics-hidden", value=json.dumps(current_topics)),
                        cls="keyword-input-container"
                    ),
                    cls="form-group"
                ),
                Div(
                    Label("Keywords (optional, max 5)", cls="form-label"),
                    P("Fetch papers matching keywords from any category. Press Tab to add keyword.", 
                      style="font-size: 0.9rem; color: var(--text-secondary); margin-bottom: 10px;"),
                    Div(
                        Div(id="keyword-tags", style="display: flex; gap: 8px; flex-wrap: wrap; margin-bottom: 10px;"),
                        Input(
                            type="text",
                            id="keyword-input",
                            placeholder="Type keyword and press Tab...",
                            cls="form-input",
                            style="margin-bottom: 0;"
                        ),
                        Input(type="hidden", name="keywords", id="keywords-hidden", value=json.dumps(keywords)),
                        cls="keyword-input-container"
                    ),
                    cls="form-group"
                ),
                Script(f"const ALL_TOPICS = {json.dumps(subjects_flat)}; const initialTopics = {json.dumps(current_topics)}; const initialKeywords = {json.dumps(keywords)};"),
                Script("""
                    // --- Topics Autocomplete Logic ---
                    const topicInput = document.getElementById('topic-input');
                    const topicTags = document.getElementById('topic-tags');
                    const topicsHidden = document.getElementById('topics-hidden');
                    let topics = [...initialTopics];
                    
                    function renderTopicTags() {
                        topicTags.innerHTML = topics.map((code, idx) => {
                            const topicObj = ALL_TOPICS.find(t => t.code === code);
                            const displayName = topicObj ? `${topicObj.name} (${topicObj.code})` : code;
                            return `
                                <span class="keyword-tag">
                                    ${displayName}
                                    <button type="button" class="tag-remove" onclick="removeTopic(${idx})">×</button>
                                </span>
                            `;
                        }).join('');
                        topicsHidden.value = JSON.stringify(topics);
                    }
                    
                    function removeTopic(idx) {
                        topics.splice(idx, 1);
                        renderTopicTags();
                    }
                    
                    const topicDropdown = document.getElementById('topic-dropdown');
                    let currentFocus = -1;
                    
                    topicInput.addEventListener('input', function() {
                        const val = this.value.toLowerCase().trim();
                        topicDropdown.innerHTML = '';
                        if (!val) {
                            topicDropdown.style.display = 'none';
                            return;
                        }
                        
                        let matches = ALL_TOPICS.filter(t => 
                            (t.name.toLowerCase().includes(val) || t.code.toLowerCase().includes(val)) &&
                            !topics.includes(t.code)
                        );
                        
                        currentFocus = -1;
                        if (matches.length > 0) {
                            topicDropdown.style.display = 'block';
                            matches.forEach((match, index) => {
                                const b = document.createElement('li');
                                b.className = 'autocomplete-item';
                                b.innerHTML = `<strong>${match.name}</strong> (${match.code})`;
                                b.addEventListener('click', function(e) {
                                    topics.push(match.code);
                                    topicInput.value = '';
                                    topicDropdown.style.display = 'none';
                                    renderTopicTags();
                                });
                                topicDropdown.appendChild(b);
                            });
                        } else {
                            topicDropdown.style.display = 'none';
                        }
                    });
                    
                    // Always close dropdown when clicking outside
                    document.addEventListener('click', function (e) {
                        if (e.target !== topicInput && e.target !== topicDropdown) {
                            topicDropdown.style.display = 'none';
                        }
                    });
                    
                    topicInput.addEventListener('keydown', function(e) {
                        let x = topicDropdown.getElementsByTagName('li');
                        if (e.key === 'ArrowDown') {
                            currentFocus++;
                            addActive(x);
                        } else if (e.key === 'ArrowUp') {
                            e.preventDefault();
                            currentFocus--;
                            addActive(x);
                        } else if (e.key === 'Enter' || e.key === 'Tab') {
                            if (topicDropdown.style.display === 'block') {
                                e.preventDefault();
                                if (currentFocus > -1) {
                                    if (x) x[currentFocus].click();
                                } else if (x && x.length > 0) {
                                    x[0].click(); // Auto-select first match on Tab/Enter
                                }
                            }
                        }
                    });
                    
                    function addActive(x) {
                        if (!x) return false;
                        removeActive(x);
                        if (currentFocus >= x.length) currentFocus = 0;
                        if (currentFocus < 0) currentFocus = (x.length - 1);
                        x[currentFocus].classList.add('active');
                        x[currentFocus].scrollIntoView({block: 'nearest'});
                    }
                    
                    function removeActive(x) {
                        for (let i = 0; i < x.length; i++) {
                            x[i].classList.remove('active');
                        }
                    }
                    
                    // --- Keywords Logic ---
                    const keywordInput = document.getElementById('keyword-input');
                    const keywordTags = document.getElementById('keyword-tags');
                    const keywordsHidden = document.getElementById('keywords-hidden');
                    let keywordArray = [...initialKeywords];
                    
                    function renderTags() {
                        keywordTags.innerHTML = keywordArray.map((kw, idx) => `
                            <span class="keyword-tag">
                                ${kw}
                                <button type="button" class="tag-remove" onclick="removeKeyword(${idx})">×</button>
                            </span>
                        `).join('');
                        keywordsHidden.value = JSON.stringify(keywordArray);
                    }
                    
                    function removeKeyword(idx) {
                        keywordArray.splice(idx, 1);
                        renderTags();
                    }
                    
                    keywordInput.addEventListener('keydown', (e) => {
                        if (e.key === 'Tab') {
                            e.preventDefault();
                            const value = keywordInput.value.trim();
                            if (value && keywordArray.length < 5 && !keywordArray.includes(value)) {
                                keywordArray.push(value);
                                keywordInput.value = '';
                                renderTags();
                            }
                        }
                    });
                    
                    // Initial render for both
                    renderTopicTags();
                    renderTags();
                """),
                Button("Save Settings", type="submit", cls="btn-primary"),
                action="/settings",
                method="post",
                cls="settings-form"
            ),
            cls="main-container"
        )
    )

@rt('/settings')
def post(frequency: str, topics: str = '[]', keywords: str = '[]'):
    # Parse topics JSON
    try:
        topics_list = json.loads(topics)
    except:
        topics_list = []
    
    # Parse keywords JSON and limit to 5
    try:
        keywords_list = json.loads(keywords)[:5]
    except:
        keywords_list = []
    
    conn = get_db_connection()
    c = conn.cursor()
    # Reset last_fetch_date to NULL to trigger a re-fetch (Condition a)
    c.execute("""
        UPDATE user_settings 
        SET fetch_frequency = ?, target_subjects = ?, keywords = ?, last_fetch_date = NULL
        WHERE user_id = ?
    """, (frequency, json.dumps(topics_list), json.dumps(keywords_list), DEFAULT_USER_ID))
    conn.commit()
    conn.close()
    
    return RedirectResponse(url='/settings?saved=true', status_code=303)

@rt('/interact/{user_id}/{paper_id}/remove')
def post(user_id: int, paper_id: str):
    conn = get_db_connection()
    c = conn.cursor()
    c.execute("DELETE FROM interactions WHERE user_id = ? AND paper_id = ?", (user_id, paper_id))
    conn.commit()
    conn.close()
    return "" # Swap with nothing = remove element

@rt('/history')
def get(user_id: int = DEFAULT_USER_ID, view: str = 'likes'):
    conn = get_db_connection()
    c = conn.cursor()
    
    action_filter = 'like' if view == 'likes' else 'dismiss'
    
    c.execute("""
        SELECT p.*, i.timestamp, i.action
        FROM interactions i
        JOIN papers p ON i.paper_id = p.id
        WHERE i.user_id = ? AND i.action = ?
        ORDER BY i.timestamp DESC
    """, (user_id, action_filter))
    papers = c.fetchall()
    conn.close()
    
    return Title("History"), NavBar(), Main(
        Div(
            A("Liked Papers", href="/history?view=likes", cls=f"history-tab {'active' if view=='likes' else ''}"),
            A("Disliked Papers", href="/history?view=dislikes", cls=f"history-tab {'active' if view=='dislikes' else ''}"),
            cls="history-nav"
        ),
        Div(
            *[GridCard(p, mode="history", interaction=p['action']) for p in papers],
            cls="dashboard-grid" # Reusing dashboard grid for consistency
        ) if papers else P("No papers found in this category.", cls="text-secondary"),
        cls="main-container"
    )

@rt('/interact/{user_id}/{paper_id}/{action}')
def post(user_id: int, paper_id: str, action: str, req: Request):
    # Log interaction
    conn = get_db_connection()
    c = conn.cursor()
    c.execute("""
        INSERT INTO interactions (user_id, paper_id, action) VALUES (?, ?, ?)
        ON CONFLICT(user_id, paper_id) DO UPDATE SET action=excluded.action, timestamp=CURRENT_TIMESTAMP
    """, (user_id, paper_id, action))
    conn.commit()
    
    # Mark profile as stale by clearing last_updated in user_profile
    # This makes "Update Recommendations" button available
    c.execute("""
        UPDATE user_profile 
        SET last_updated = NULL 
        WHERE user_id = ?
    """, (user_id,))
    conn.commit()
    
    # Return updated card
    c.execute("SELECT * FROM papers WHERE id = ?", (paper_id,))
    paper = c.fetchone()
    conn.close()
    
    if paper:
        paper_dict = dict(paper)
        
        # Check if we are in history mode based on referer or HX-Current-URL
        current_url = req.headers.get('hx-current-url', '')
        mode = "history" if "history" in current_url else "feed"
        
        return GridCard(dict(paper), mode=mode, interaction=action)
    
    return ""

@rt('/search')
def get():
    return Title("Search Papers"), NavBar(), Main(
        Div(
            H1("Advanced Search", cls="section-title"),
            Form(
                 # Basic Search
                 Div(
                     Label("Keywords / Title / ID", cls="form-label"),
                     Input(type="text", name="q", placeholder="e.g. 'Deep Learning' or '2101.01234'", cls="form-input", style="font-size: 1.1rem;"),
                     cls="form-group"
                 ),
                 
                 # Advanced Details
                 Details(
                     Summary("More Options (Author, Year)", style="cursor: pointer; color: var(--accent); font-weight: 600; margin-bottom: 15px;"),
                     Div(
                         Div(
                             Label("Author", cls="form-label"),
                             Input(type="text", name="author", placeholder="e.g. Yoshua Bengio", cls="form-input"),
                             cls="form-group",
                             style="flex: 1; min-width: 250px;"
                         ),
                         Div(
                             Label("Year Range", cls="form-label"),
                             Div(
                                 Input(type="number", name="year_start", placeholder="Start", cls="form-input", style="width: 48%;"),
                                 Span("-", style="font-weight: bold; color: var(--text-secondary);"),
                                 Input(type="number", name="year_end", placeholder="End", cls="form-input", style="width: 48%;"),
                                 style="display: flex; align-items: center; justify-content: space-between;"
                             ),
                             cls="form-group",
                             style="flex: 1; min-width: 200px;"
                         ),
                         style="display: flex; gap: 20px; flex-wrap: wrap;"
                     ),
                     open=True, # Default open for visibility as requested "mimic advanced search"
                     style="margin-bottom: 20px; padding: 20px; background: var(--card-bg); border: 1px solid var(--border); border-radius: 8px;"
                 ),
                 
                 Button("Search Papers", type="submit", cls="btn-primary", style="width: 100%; font-size: 1.1rem; padding: 15px;"),
                 hx_post="/search",
                 hx_target="#search-results",
                 hx_swap="innerHTML",
                 hx_indicator="#loading-indicator",
                 cls="search-form"
            ),
            
            # Loading Indicator
            Div(
                P("Searching Semantic Scholar...", style="font-weight: 600;"),
                id="loading-indicator", 
                cls="htmx-indicator", 
                style="text-align: center; margin-top: 20px; padding: 20px; background: var(--card-bg); border-radius: 8px; color: var(--text-secondary);"
            ),
            
            Div(id="search-results", style="margin-top: 40px;"),
            cls="main-container"
        )
    )

@rt('/search')
def post(q: str = "", author: str = "", year_start: str = "", year_end: str = ""):
    # Construct Query
    query_parts = []
    if q and q.strip():
        query_parts.append(q.strip())
    if author and author.strip():
        query_parts.append(author.strip())
    
    full_query = " ".join(query_parts)
    if not full_query and not year_start and not year_end:
        return Div(
            P("Please enter at least one search criteria.", style="color: #dc3545; font-weight: bold; text-align: center; margin: 40px 0;")
        )
    
    from ingest import search_arxiv, search_semantic_scholar
    
    # Try Semantic Scholar first (as requested by user)
    papers = []
    source_used = None
    error_msg = None
    
    logger.info(f"Search request: q='{q}', author='{author}', years={year_start}-{year_end}")
    
    # Construct year parameter for Semantic Scholar
    year_param = None
    if year_start:
        if year_end: year_param = f"{year_start}-{year_end}"
        else: year_param = year_start
    elif year_end:
        year_param = f"-{year_end}"
        
    try:
        ss_query = full_query
        papers = search_semantic_scholar(query=ss_query, year=year_param, limit=15)
        if papers:
            source_used = "Semantic Scholar"
    except Exception as e:
        logger.warning(f"Semantic Scholar search failed: {e}")
        error_msg = "Semantic Scholar temporarily unavailable"

    # Fallback to ArXiv if Semantic Scholar didn't return results
    if not papers:
        try:
            papers = search_arxiv(
                query=q.strip() if q else None,
                author=author.strip() if author else None, 
                year_start=year_start if year_start else None,
                year_end=year_end if year_end else None,
                limit=15
            )
            if papers:
                source_used = "ArXiv"
                error_msg = None # Clear error if fallback worked
        except Exception as e:
            logger.warning(f"ArXiv fallback also failed: {e}")
            if not error_msg: error_msg = "ArXiv temporarily unavailable"
    
    if not papers:
        return Div(
            H3("No results found" if not error_msg else error_msg, style="text-align: center; margin-top: 40px; color: var(--text-secondary);"),
            P(f"We couldn't find any papers matching your search.", style="text-align: center; color: var(--text-secondary);") if not error_msg else \
            P(f"This is likely due to API rate limiting. Please wait a few seconds and try again.", style="text-align: center; color: var(--text-secondary);"),
            P("Tip: Try more specific keywords or check if the paper ID is correct.", style="text-align: center; color: var(--accent); font-size: 0.85rem; margin-top: 20px;")
        )
    
    # Render using GridCards
    return Div(
        Div(
            H2(f"Found {len(papers)} Result{'s' if len(papers) != 1 else ''}", cls="section-title", style="margin: 0; text-align: center;"),
            P(f"Source: {source_used}", style="text-align: center; color: var(--text-secondary); font-size: 0.9rem; margin-top: 5px;"),
            style="margin-bottom: 30px;"
        ),
        Div(
            *[GridCard(p, mode="feed") for p in papers],
            cls="dashboard-grid"
        )
    )


serve(host=LAN_BIND_HOST, port=PORT)
