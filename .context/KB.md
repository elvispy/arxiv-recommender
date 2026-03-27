# Knowledge Base (Stable Facts Only)

## OSCAR Compute Patterns
- **Discovery**: Slurm job name `openclaw-compute`.
- **Jump Hosts**: `oscar-campus`, `ssh.ccv.brown.edu`, `oscar-login`.
- **Tunnel**: Local port `11434` maps to `localhost:11434` on the GPU node.
- **Node Allocation**: `salloc -J openclaw-compute -p gpu --gres=gpu:nvidia_rtx_a5000:2 --mem=128G -t 04:00:00 --no-shell`.
- **Inference Service**: Ollama or custom `Specter2` server running on the GPU node.

## Database & Schema
- **VSS Index**: `vss_papers` table with 768-dimensional vectors.
- **Primary Source**: `papers` metadata table with `embedding` BLOB.
- **Sync Logic**: Papers are fetched metadata-first (embedding = NULL) and backfilled asynchronously.

## Style & Conventions
- **FastHTML**: Hypermedia-driven UI using HTMX.
- **Rocchio Algorithm**: Dynamic user profile update logic.
- **Metadata-First Ingestion**: Separation of "Fetching" and "Embedding" phases.
