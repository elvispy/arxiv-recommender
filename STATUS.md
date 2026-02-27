# Project Status Report

## Current State: Functional MVP (Alpha)
The project implements the core "Fetch -> Embed -> Rank" loop but lacks automation and polish.

### ✅ Implemented & Working
1.  **Ingestion**: 
    - Fetches from ArXiv (RSS) and Semantic Scholar.
    - Generates local embeddings using `Specter2` (confirmed high cosine similarity >0.96).
    - Stores metadata and vectors in SQLite.
2.  **Ranking Logic**: 
    - `main.py` correctly ranks papers by **Cosine Similarity** to the user profile.
    - **Rocchio Algorithm**: `core/rocchio.py` updates the profile based on Likes/Dislikes.
3.  **Interface**: 
    - "Zen Mode" UI is functional.
    - Interaction buttons (Like/Dismiss) work and log data.

### ⚠️ Missing / Needs Attention
1.  **Automation (Critical for "Latest Papers")**:
    - **Current**: The user must *manually* click `⚡ Update Recommendations`. This blocks the UI while fetching/embedding.
    - **Missing**: A **Background Worker**. The app should wake up periodically (e.g., every 6 hours), fetch/embed new papers silently, and have them ready instantly.
    - *Impact*: You might see stale data unless you remember to click the button.

2.  **Profile "Liveness"**:
    - **Current**: Tests show that even with interactions, the **User Profile Vector** might be stale or missing until explicitly recalculated.
    - **Impact**: The feed may revert to chronological order if the profile is not updated frequently.

3.  **Diversity (MMR) - Intentionally Disabled**:
    - The code exists in `core/ranking.py` but is **unwired** in `main.py`. This aligns with your request for *pure semantic relevance*, so we can leave this as "inactive code".

### 🧪 testing Requirements
1.  **End-to-End Feed Cycle**: 
    - Verification that "Like -> Update -> Feed Changes" works in the actual app (not just unit tests).
    - Confirmed via `test_ranking.py` that math works, but UI integration needs manual verification.
2.  **Cold Start Robustness**: 
    - Ensure the app behaves gracefully (shows chronological feed) with 0 interactions or missing profile.

## Recommendation
Implement a **Background Worker** (simple thread or cron) to fetch and embed papers automatically. This fulfills the "Latest Papers" requirement without the manual friction.
