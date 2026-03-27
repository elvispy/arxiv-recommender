# .context Protocol (Authoritative)
- .context is agent-only working memory. Do not quote or dump it to the user unless explicitly asked.
- Every task must have exactly one task file: .context/tasks/<task_id>.md
- Work must be traceable: all non-trivial actions update the task file (plan/log/handoff).
- Inter-agent communication happens via .context/INBOX.md and task-file “Messages” sections.
- Append-only rule: never delete other agents’ notes; prefer appending with timestamps.
- Locking rule: one active owner per task at a time (lease). If you need to take over, record the claim in TASK_INDEX + task file.

# Technical Mandates (OSCAR Integration - Phase 2)
- **Batch Sync Architecture**: 2 AM local cron job triggers a "Fetch -> Offload -> Update" cycle.
- **Remote Offload**: Synchronous batching via `sbatch` on OSCAR GPU nodes with local polling (`squeue`).
- **GPU Environment**:
    - CUDA: `12.9.0`
    - cuDNN: `9.8.0`
    - PyTorch: `2.5.1+cu121`
    - Adapters: Explicitly activated in `oscar_infer.py` via `model.set_active_adapters()`.
- **Persistence**: Remote environment (venv, model cache) is maintained in `~/scratch/arxiv-recommender`.
- **Atomic Pipeline**:
    - Metadata is fetched to memory first.
    - Offloaded to OSCAR for Specter2 embeddings.
    - Only saved to DB once embeddings are retrieved (prevents unranked papers).
- **Pruning**: Automatic 6-month cleanup of un-interacted papers.
