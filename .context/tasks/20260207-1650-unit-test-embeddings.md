# 20260207-1650-unit-test-embeddings Unit Testing Embedding Logic (DONE)
## Owner + Lease
- owner_session: google/antigravity/2026-02-07T16:50:00-05:00
- lease_expires: 2026-02-07T17:50:00-05:00

## Goal / Acceptance Criteria
- Verify that embedding logic correctly prioritizes Semantic Scholar embeddings.
- Verify fallback to custom model embedding when Semantic Scholar data is missing.
- Create unit tests covering these scenarios without modifying existing implementation logic (unless bugs are found).

## Constraints / Non-goals
- Do not modify core embedding logic unless broken.
- Focus on `pytest` or `unittest`.

## Repo Touchpoints
- `ingest.py`
- `tests/test_embedding.py`

## Plan
- [x] Explore codebase to identify embedding generation and retrieval functions.
- [x] Create/Update test file `tests/test_embedding.py`.
- [x] Implement mocks for Semantic Scholar API.
- [x] Write test cases for:
    - [x] Successful S2 fetch.
    - [x] S2 miss (saves without embedding).
    - [x] Verify local embedding is DISABLED (as per current code).
- [x] Run tests and verify passing status.

## Work Log
- 2026-02-07 16:50: Task created.
- 2026-02-07 16:55: Analyzed `ingest.py` and found local embedding is disabled (`NotImplementedError`).
- 2026-02-07 17:00: Updated `tests/test_embedding.py` with comprehensive tests for S2 logic and disabled local logic.
- 2026-02-07 17:05: Fixed test case `test_save_papers_inserts_new_with_embedding` to handle mock side effects.
- 2026-02-07 17:06: All tests passed.

## Messages
- Note to User: The local custom model fallback is currently disabled in `ingest.py`. The tests verify this behavior. To enable it, code changes in `ingest.py` are required.

## Handoff
- Tests are in `tests/test_embedding.py`.
- Run with `python -m pytest tests/test_embedding.py`.
