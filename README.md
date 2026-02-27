# Daily ArXiv 📑

**A Local-First, "Zen Mode" Research Paper Recommender.**

Daily ArXiv is a self-hosted tool designed to solve information overload for researchers. It ingests preprints from ArXiv, BioRxiv, and Semantic Scholar, understands them using a local Large Language Model (LLM), and curates a daily feed tailored to your evolving interests.

Built with performance, privacy, and simplicity in mind.

## 🧘 The Philosophy: Zen Mode
Daily ArXiv prioritizes **Cognitive Quietness**:
*   **Minimalist Interface**: No distractions, just papers.
*   **Swipe-to-Tune**: Like/Dismiss interactions instantly train your personal algorithm.
*   **Local & Private**: Your reading history and preference profile never leave your machine.

## 🏗️ Architecture & Stack Choices

We chose a "Hypermedia-Driven" and "Local-First" architecture to ensure the app is snappy, portable, and easy to hack on.

*   **Python 3.10+**: The lingua franca of Data Science.
*   **[FastHTML](https://fastht.ml/)**: A modern Python framework that returns HTML, not JSON. Coupled with **HTMX**, it delivers "Zero-Latency" UI updates without the complexity of React/Vue build pipelines.
*   **SQLite + [sqlite-vss](https://github.com/asg017/sqlite-vss)**: A "Single-File Database" architecture. Both metadata and 768-dimensional vector embeddings live in one `arxiv.db` file. This makes backup and migration as simple as copying a file.
*   **[SPECTER2](https://github.com/allenai/specter)**: A powerful model from AllenAI trained specifically on scientific citations. It generates high-quality embeddings so papers with similar *meanings* (not just keywords) group together.

## 🧠 How It Works

### 1. Ingestion (The "Lazy" Update)
To keep the UI instant, new papers are fetched in the background or triggered via a "Lazy Check" when you load the app.
*   Connects to ArXiv/BioRxiv APIs.
*   Filters for your target subjects (e.g., `cs.CL`, `physics.flu-dyn`).
*   Embeds titles and abstracts locally using SPECTER2.

### 2. The Brain (Rocchio Algorithm)
Your profile isn't a static list of keywords. It's a **Vector** in high-dimensional space.
*   **Initialization**: Starts at zero (neutral).
*   **Feedback Loop**:
    *   **Like (♥)**: Moves your profile vector *towards* the paper.
    *   **Dismiss (👎)**: Moves your profile vector *away* from the paper.
*   **Result**: The system learns to find papers "near" your ideal preferences.

### 3. Re-Ranking (MMR)
To avoid an "Echoboard" (seeing 10 nearly identical papers), we use **Maximal Marginal Relevance (MMR)**.
*   It fetches top 50 highly relevant papers.
*   It re-orders them to maximize diversity, ensuring your feed has a healthy mix of sub-topics.

## 🚀 Getting Started

### Prerequisites
*   Python 3.10 or higher
*   [uv](https://github.com/astral-sh/uv) (Recommended for fast package management)

### Installation
1.  **Clone the repo**
    ```bash
    git clone https://github.com/yourusername/arxiv-recommender.git
    cd arxiv-recommender
    ```

2.  **Install Dependencies**
    ```bash
    uv venv
    source .venv/bin/activate
    uv pip install -r requirements.txt
    ```
    *(Note: The first run will download the ~1GB SPECTER2 model from HuggingFace).*

3.  **Run the App**
    ```bash
    python main.py
    ```

4.  **Access**
    *   **Desktop**: Go to `http://localhost:5001`
    *   **Mobile/iPad**: Access via your LAN IP (e.g., `http://192.168.1.XX:5001`). The app binds to `0.0.0.0` by default.

## 🧩 Project Structure

*   `main.py`: The FastHTML web server and UI logic.
*   `db.py`: Database schema and connection management.
*   `ingest.py`: Logic for fetching papers and generating embeddings.
*   `core/rocchio.py`: The heart of the user profiling algorithm.
*   `core/ranking.py`: The MMR re-ranking logic.
*   `data/`: Stores your `arxiv.db`.

---
*Built with ♥ by Antigravity.*
