# Daily ArXiv Recommender

Daily ArXiv is a local-first research paper recommender that uses the Rocchio algorithm and Specter2 embeddings to personalize your preprint feed.

## 🚀 Phase 2: OSCAR Acceleration
This project features an experimental asynchronous pipeline that offloads heavy embedding inference (Specter2) to **OSCAR GPU nodes**.

### Key Features
- **Zen Mode**: A clean, high-performance UI for triaging research papers.
- **Async Pipeline**: A 2 AM cron job that handles fetching, embedding, and profile updates.
- **OSCAR Bridge**: Intelligent SSH tunneling to discover and utilize remote GPU resources for Specter2 inference.
- **Privacy-First**: All interaction data and remote compute pipelines are strictly local to your machine.

## Setup

### Prerequisites
- Python 3.10+
- SQLite with `sqlite-vss` extension.
- (Optional) SSH access to a Slurm-managed GPU cluster (OSCAR) for accelerated ingestion.

### Installation
1.  Clone the repository.
2.  `pip install -r requirements.txt`
3.  `python db.py` to initialize the database.

### Running the App
- **Web App**: `python main.py`
- **Manual Sync**: `python scripts/cron_worker.py` (or wait for the 2 AM cron job).

## Branching Note
OSCAR-specific features and the async pipeline are maintained in the `feature/oscar-pipeline` branch to keep the main codebase lightweight.
