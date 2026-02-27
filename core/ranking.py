import numpy as np

def cosine_similarity(v1, v2):
    """
    Compute cosine similarity between two 1D vectors.
    """
    norm1 = np.linalg.norm(v1)
    norm2 = np.linalg.norm(v2)
    if norm1 == 0 or norm2 == 0:
        return 0.0
    return np.dot(v1, v2) / (norm1 * norm2)

def mmr_rerank(query_vector, candidates, lambda_param=0.5, top_n=10):
    """
    Maximal Marginal Relevance (MMR) Re-ranking.
    
    Args:
        query_vector (np.array): User's profile vector.
        candidates (list of dict): List of papers. Must have 'embedding' (bytes or np.array) and 'id'.
        lambda_param (float): Trade-off between relevance (1.0) and diversity (0.0).
        top_n (int): Number of results to return.
        
    Returns:
        list of dict: Re-ranked papers.
    """
    if not candidates:
        return []
        
    # Ensure query vector is numpy array
    if isinstance(query_vector, bytes):
        query_vector = np.frombuffer(query_vector, dtype=np.float32)
        
    # Pre-process candidate embeddings
    candidate_embeddings = []
    processed_candidates = []
    
    for cand in candidates:
        emb = cand.get('embedding')
        if emb is None:
            continue
        if isinstance(emb, bytes):
            emb = np.frombuffer(emb, dtype=np.float32)
        candidate_embeddings.append(emb)
        processed_candidates.append(cand)
        
    if not processed_candidates:
        return []

    # Selected indices
    selected_indices = []
    candidate_indices = list(range(len(processed_candidates)))
    
    while len(selected_indices) < top_n and candidate_indices:
        best_mmr = -float('inf')
        best_idx = -1
        
        for idx in candidate_indices:
            # Relevance: Sim(cand, query)
            relevance = cosine_similarity(processed_candidates[idx]['embedding'], query_vector)
            
            # Diversity: Max Sim(cand, selected)
            if not selected_indices:
                diversity = 0
            else:
                diversity = max([cosine_similarity(processed_candidates[idx]['embedding'], processed_candidates[sel_idx]['embedding']) 
                                 for sel_idx in selected_indices])
            
            # MMR Score
            mmr_score = (lambda_param * relevance) - ((1 - lambda_param) * diversity)
            
            if mmr_score > best_mmr:
                best_mmr = mmr_score
                best_idx = idx
                
        if best_idx != -1:
            selected_indices.append(best_idx)
            candidate_indices.remove(best_idx)
        else:
            break
            
    return [processed_candidates[i] for i in selected_indices]
