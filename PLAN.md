# Daily ArXiv - Project Blueprint (Phase 1)

## Goal
Build "Daily ArXiv," a local-first, high-performance research paper recommender system. The system ingests ArXiv/Bioarxiv/Semantic Scholar preprints, uses a local Large Language Model (SPECTER2) to generate semantic embeddings, and orders them based on a dynamic user profile (Rocchio Algorithm). The goal is to solve information overload via a "Zen Mode" triage interface accessible via LAN.

## Architecture & Stack
- **Stack**: Python 3.10+, FastHTML, SQLite + sqlite-vss, HuggingFace Transformers.
- **Key Decisions**:
    - **FastHTML**: Hypermedia-Driven Architecture for "Zero-Latency" UI updates using HTMX.
    - **SQLite-vss**: Single-file database for metadata and vectors (768d). Crucial for local portability.
    - **SPECTER2**: Industry standard for scientific text embeddings.
    - **Deployment**: Local host (0.0.0.0) for LAN access.

## Directory Structure
```
/
├── main.py                 # FastHTML App, Routes, UI Logic
├── db.py                   # Database connection & Schema Init
├── ingest.py               # ArXiv/BioRxiv/Semantic Scholar Fetching & Embedding
├── config.py               # Configuration constants
├── core/
│   ├── rocchio.py          # User Profile Update Logic (Rocchio Algorithm)
│   └── ranking.py          # (Planned) MMR Re-ranking Logic
├── static/
│   └── style.css           # Zen Mode Styles
└── data/
    └── arxiv.db            # SQLite Database (Metadata + VSS)
```

## Database Schema

### `papers`
Stores metadata and embeddings.
| Column | Type | Description |
|---|---|---|
| id | TEXT PK | `source:external_id` (e.g., `arxiv:2101...`) |
| source | TEXT | `arxiv`, `biorxiv`, `semantic_scholar` |
| external_id | TEXT | Original ID from source |
| title | TEXT | |
| abstract | TEXT | |
| authors | JSON | List of author names |
| published_date | DATETIME | |
| category | TEXT | |
| link | TEXT | |
| embedding | BLOB | Serialized 768d vector |

### `vss_papers` (Virtual Table)
Vector index managed by `sqlite-vss`.
- `embedding(768)`: Vector column.
- `rowid`: Synced with `papers` implicitly or explicitly.

### `users`
| Column | Type | Description |
|---|---|---|
| id | INTEGER PK | |
| username | TEXT | |

### `user_settings`
| Column | Type | Description |
|---|---|---|
| user_id | INTEGER PK | FK to `users.id` |
| fetch_frequency | TEXT | `daily`, `weekly`, `monthly` |
| target_subjects | JSON | List of subject codes (e.g., `["cs.CL"]`) |
| last_fetch_date | DATETIME | |

### `user_profile`
| Column | Type | Description |
|---|---|---|
| user_id | INTEGER PK | FK to `users.id` |
| preference_vector | BLOB | Current Rocchio center (768d) |
| last_updated | DATETIME | |

### `interactions`
| Column | Type | Description |
|---|---|---|
| id | INTEGER PK | |
| user_id | INTEGER | FK |
| paper_id | TEXT | FK |
| action | TEXT | `like`, `dismiss` |
| timestamp | DATETIME | |

## Logic Flow

### 1. Lazy Update Mechanism (Ingestion)
**Goal**: Ensure fresh papers without blocking the UI.
**Trigger**:
- **Primary**: Background Scheduled Job (if persistent process).
- **Fallback**: "Check on Load" when user visits `/`.

**Flow**:
1. User visits `/` (Landing Page).
2. **Check**: App checks `user_settings`.
    - If `frequency` == 'daily', fetch papers published since `last_fetch_date` (or yesterday).
    - If `frequency` == 'weekly', fetch on set day (e.g., Monday).
3. **Action**:
    - If update needed:
        - Trigger **Asynchronous Ingestion** (Thread/Process).
        - **UI Feedback**: Show a generic "Feed" immediately (from existing DB) but display a **Rotating Loading Symbol** indicating "Checking for updates..." (HTMX polling or SSE).
    - If no update needed, show feed immediately.
4. **Ingestion Process** (Async):
    - Fetch papers from ArXiv/BioRxiv for `target_subjects`.
    - Generate Embeddings (SPECTER2) - *Compute Intensive*.
    - Store in `papers` and `vss_papers`.
    - Update `last_fetch_date`.
5. **Completion**:
    - UI updates the feed (via polling) or notifies user "New papers available".

### 2. Ranking & Feed Generation
**Goal**: Order papers by relevance to `preference_vector`.
**Process**:
1. **Fetch Candidates**: Query `vss_papers` using `preference_vector` (KNN search).
2. **Filter**: Remove papers in `interactions` (already seen/liked/dismissed).
3. **Re-Rank (MMR)** (Phase 2):
    - Apply Maximal Marginal Relevance to ensure diversity among top candidates.
4. **Render**: Return list of `GridCard` components.

### 3. Interaction & Learning
1. User clicks "Like" or "Dismiss".
2. **State**: App records interaction in `interactions`.
3. **Profile Update**:
    - **Execution**: Done **Offline or Asynchronously** to avoid UI lag.
    - **Algorithm**: Rocchio Update.
        - `V_new = alpha * V_old + beta * V_liked - gamma * V_disliked`
    - **Effect**: Next feed generation uses updated `V_new`.
