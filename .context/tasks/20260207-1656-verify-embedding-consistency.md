# 20260207-1656-verify-embedding-consistency Verify Embedding Consistency
## Owner + Lease
- owner_session: google/antigravity/2026-02-07T16:56:00-05:00
- lease_expires: 2026-02-07T17:56:00-05:00

## Goal / Acceptance Criteria
- Verify if the local embedding model matches Semantic Scholar's API model.
- Create a test that compares S2 embedding vs Local embedding for a specific paper.
- Target relative distance < 1%.

## Constraints / Non-goals
- Do not modify application code (`ingest.py`) to enable local embeddings yet, instantiate model in test.

## Repo Touchpoints
- `config.py`
- `tests/test_embedding_consistency.py`

## Plan
- [ ] Check `config.py` for `EMBEDDING_MODEL_NAME`.
- [ ] Search Semantic Scholar documentation for their current embedding model (likely SPECTER2).
- [ ] Create `tests/test_embedding_consistency.py`.
- [ ] In test:
    - [ ] Fetch S2 embedding for a known paper (e.g., "Attention is All You Need").
    - [ ] Instantiate local model (from config).
    - [ ] Generate embedding locally.
    - [ ] Compare vectors (cosine similarity or relative error).
- [ ] Report findings to User.

## Work Log
- 2026-02-07 16:56: Task started.

## Messages

## Handoff
