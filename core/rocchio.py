import sqlite3
import numpy as np
import json
import logging
from db import get_db_connection

logger = logging.getLogger(__name__)

# Rocchio Hyperparameters
ALPHA = 0.8  # Weight of old profile
BETA = 0.6   # Weight of liked papers
GAMMA = 0.2  # Weight of disliked papers

def update_user_profile(user_id):
    """
    Recalculates the user's preference vector using the Rocchio algorithm.
    Executed after interactions.
    """
    conn = get_db_connection()
    c = conn.cursor()
    
    # 1. Fetch current profile
    c.execute("SELECT preference_vector FROM user_profile WHERE user_id = ?", (user_id,))
    row = c.fetchone()
    if row and row[0]:
        current_profile = np.frombuffer(row[0], dtype=np.float32)
    else:
        current_profile = np.zeros(768, dtype=np.float32) # Init zero or random
        
    # 2. Fetch fresh interactions (BATCH or ALL?)
    # For MVP, we can recalculate from ALL recent history or just incremental.
    # Rocchio is usually iterative.
    # If we do iterative: New = Alpha * Old + Beta * Like - Gamma * Dislike
    # But this assumes we apply the update for EACH interaction ONCE.
    # If we blindly re-run over all history, we need to be careful.
    # Let's try: Fetch UNPROCESSED interactions? Or just grab last N?
    # Simpler: Recalc from scratch using last N likes/dislikes + Base Profile?
    # Or strict iterative:
    # We need to mark interactions as 'processed' to avoid double counting if we are doing incremental updates.
    # Let's add 'processed' flag to interactions? Or just fetch LAST interaction and update?
    
    # Approach for "Daily ArXiv":
    # Just fetch the interactions that happened *since last update*?
    # Or simpler: The "Zen Mode" implies one-by-one.
    # So we can just update based on the Single Latest Interaction.
    
    # Let's fetch the LATEST interaction for this user.
    c.execute("""
        SELECT i.action, p.embedding 
        FROM interactions i
        JOIN papers p ON i.paper_id = p.id
        WHERE i.user_id = ? 
        ORDER BY i.timestamp DESC 
        LIMIT 1
    """, (user_id,))
    
    interaction = c.fetchone()
    if not interaction:
        conn.close()
        return

    action = interaction[0]
    embedding_data = interaction[1]
    
    # Handle papers without embeddings (lazy loading)
    if embedding_data is None:
        logger.warning(f"Paper has no embedding yet (lazy loading), skipping Rocchio update")
        conn.close()
        return
        
    paper_vec = np.frombuffer(embedding_data, dtype=np.float32)

    # Normalize vectors? SPECTER vectors are usually normalized or close to it.
    # But usually good practice.
    
    # 3. Apply Rocchio (Incremental for Single Interaction)
    # V_new = alpha * V_old + (beta/gamma) * V_paper
    # Note: The standard formula divides by count of likes/dislikes. 
    # For single item update:
    # If Like: V_new = alpha * V_old + beta * V_paper
    # If Dislike: V_new = alpha * V_old - gamma * V_paper
    # But if we keep multiplying by Alpha (0.9), the vector shrinks?
    # Standard incremental Rocchio usually is: V_t = V_{t-1} + n * (V_doc - V_{t-1})?
    # No, Rocchio is for relevance feedback in search sessions.
    
    # Modified approach for continuous stream:
    # Moving Average:
    # V_new = (1 - lr) * V_old + lr * V_paper (for Like)
    # V_new = (1 - lr) * V_old - lr * V_paper (for Dislike)?
    
    # Let's stick to simple vector addition with decay.
    # V = V + 0.1 * Paper (Like)
    # V = V - 0.1 * Paper (Dislike)
    # Renormalize.
    
    STEP_SIZE = 0.2
    
    if action == 'like':
        new_profile = current_profile + (STEP_SIZE * paper_vec)
    elif action == 'dismiss':
        new_profile = current_profile - (STEP_SIZE * paper_vec)
    else:
        new_profile = current_profile
        
    # Normalize to unit length (optional but good for cosine sim)
    norm = np.linalg.norm(new_profile)
    if norm > 0:
        new_profile = new_profile / norm
        
    # 4. Save
    # Check if exists
    c.execute("SELECT 1 FROM user_profile WHERE user_id = ?", (user_id,))
    exists = c.fetchone()
    
    if exists:
        c.execute("UPDATE user_profile SET preference_vector = ?, last_updated = CURRENT_TIMESTAMP WHERE user_id = ?", 
                  (new_profile.tobytes(), user_id))
    else:
        c.execute("INSERT INTO user_profile (user_id, preference_vector, last_updated) VALUES (?, ?, CURRENT_TIMESTAMP)",
                  (user_id, new_profile.tobytes()))
                  
    conn.commit()
    conn.close()


def recalculate_user_profile(user_id):
    """
    Full recalculation of the user profile using Multi-Interest Retrieval.
    Uses K-means clustering (K=3) to capture diverse interests.
    """
    conn = get_db_connection()
    c = conn.cursor()
    
    # Fetch all likes
    c.execute("""
        SELECT p.embedding 
        FROM interactions i
        JOIN papers p ON i.paper_id = p.id
        WHERE i.user_id = ? AND i.action = 'like' AND p.embedding IS NOT NULL
    """, (user_id,))
    likes = [np.frombuffer(row[0], dtype=np.float32) for row in c.fetchall()]
    
    # Fetch all dislikes (we'll subtract from each centroid)
    c.execute("""
        SELECT p.embedding 
        FROM interactions i
        JOIN papers p ON i.paper_id = p.id
        WHERE i.user_id = ? AND i.action = 'dismiss' AND p.embedding IS NOT NULL
    """, (user_id,))
    dislikes = [np.frombuffer(row[0], dtype=np.float32) for row in c.fetchall()]
    
    conn.close()
    
    # Calculate Multi-Interest Profile
    K = 3  # Number of interest clusters
    
    if not likes:
        # No likes: zero vectors for all interests
        new_profile = np.zeros((K, 768), dtype=np.float32)
    elif len(likes) < K:
        # Fewer than K likes: duplicate/pad with mean
        logger.info(f"Only {len(likes)} likes, padding to {K} interests")
        centroids = []
        for emb in likes:
            centroids.append(emb)
        # Pad with mean of existing
        mean_emb = np.mean(likes, axis=0)
        while len(centroids) < K:
            centroids.append(mean_emb)
        new_profile = np.array(centroids)
    else:
        # K-Means clustering
        from sklearn.cluster import KMeans
        
        likes_matrix = np.array(likes)
        kmeans = KMeans(n_clusters=K, random_state=42, n_init=10)
        kmeans.fit(likes_matrix)
        centroids = kmeans.cluster_centers_  # Shape: (K, 768)
        
        logger.info(f"Clustered {len(likes)} likes into {K} interest groups")
        
        # Apply Rocchio: Subtract disliked papers from each centroid
        if dislikes:
            mean_dislike = np.mean(dislikes, axis=0)
            G_BATCH = 0.25  # Weight for dislikes
            for i in range(K):
                centroids[i] = centroids[i] - (G_BATCH * mean_dislike)
        
        # Normalize each centroid
        for i in range(K):
            norm = np.linalg.norm(centroids[i])
            if norm > 0:
                centroids[i] = centroids[i] / norm
        
        new_profile = centroids  # Shape: (K, 768)

    # Save to DB (concatenate all K vectors into single blob)
    # Format: [centroid_1 | centroid_2 | centroid_3] = (K*768,) = (2304,) floats
    profile_blob = new_profile.flatten().tobytes()
    
    conn = get_db_connection()
    c = conn.cursor()
    
    # Check if exists
    c.execute("SELECT 1 FROM user_profile WHERE user_id = ?", (user_id,))
    exists = c.fetchone()
    
    if exists:
        c.execute("UPDATE user_profile SET preference_vector = ?, last_updated = CURRENT_TIMESTAMP WHERE user_id = ?", 
                  (profile_blob, user_id))
    else:
        c.execute("INSERT INTO user_profile (user_id, preference_vector, last_updated) VALUES (?, ?, CURRENT_TIMESTAMP)",
                  (user_id, profile_blob))
                  
    conn.commit()
    conn.close()
    logger.info(f"Saved {K} interest centroids for user {user_id}")
    
if __name__ == "__main__":
    # Test
    # update_user_profile(1)
    pass
