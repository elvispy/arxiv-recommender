# Daily ArXiv - Project Blueprint (Phase 1, 2 & 3)

## Goal
Build "Daily ArXiv," a local-first, high-performance research paper recommender system. The system ingests ArXiv/Bioarxiv/Semantic Scholar preprints, uses a remote Large Language Model (SPECTER2) on OSCAR GPUs to generate semantic embeddings, and orders them based on a dynamic user profile (Rocchio Algorithm).

**Phase 2 Success**: Fully automated 2 AM sync that offloads compute to OSCAR and prunes old data.

## Architecture (Phase 2 - Unified Sync)
```
/
├── main.py                 # FastHTML App, Routes, UI Logic
├── ingest.py               # Unified metadata-first fetching
├── core/
│   ├── oscar_batch.py      # sbatch submission & polling
│   ├── rocchio.py          # User Profile Update Logic
│   └── ranking.py          # Real-time Multi-Interest Ranking
├── scripts/
│   └── cron_worker.py      # 2 AM Orchestrator (Fetch -> Embed -> Save -> Prune)
├── remote/
│   ├── remote_setup.sh     # Remote environment (CUDA 12.9 + PT 2.5.1)
│   ├── run_inference.sh    # GPU sbatch template
│   └── oscar_infer.py      # Remote Specter2 inference engine
├── logs/                   # cron_worker.log & sync_errors.log
└── data/
    └── arxiv.db            # SQLite Database (Metadata + VSS)
```

## Logic Flow (2 AM Sync)

### 1. Fetch & Filter
`cron_worker.py` identifies new papers for all `target_subjects`. It filters out any that already exist in the local database.

### 2. Remote GPU Batch
1.  **Offloading**: New metadata is sent to OSCAR via `scp`.
2.  **Inference**: A Slurm job is submitted. It loads `Specter2` on a GPU and processes the batch.
3.  **Polling**: The local machine waits for completion.
4.  **Retrieval**: Vectors are downloaded and saved **atomically** with metadata.

### 3. Database Maintenance
`prune_old_papers` deletes entries older than 6 months that have no interactions, keeping the index under 200MB.

## Data Consistency
- **Specter2**: Mandatory for all papers.
- **Hardware Agnostic**: Similarity checks confirm identical results between local CPU and OSCAR GPU.
