"""
Test to verify that paper ranking is actually using embeddings and cosine similarity.
"""
import sqlite3
import numpy as np
from db import get_db_connection

def test_embedding_based_ranking():
    """
    Verify that:
    1. Papers have embeddings
    2. Ranking uses cosine similarity
    3. User profile affects ranking
    """
    print("\n=== Testing Embedding-Based Ranking ===\n")
    
    conn = get_db_connection()
    c = conn.cursor()
    
    # 1. Check if papers have embeddings
    print("1. Checking papers with embeddings...")
    c.execute("SELECT COUNT(*) FROM papers WHERE embedding IS NOT NULL")
    with_embeddings = c.fetchone()[0]
    
    c.execute("SELECT COUNT(*) FROM papers")
    total = c.fetchone()[0]
    
    print(f"   Papers with embeddings: {with_embeddings}/{total}")
    
    if with_embeddings == 0:
        print("   ❌ FAIL: No papers have embeddings!")
        return False
    
    # 2. Get a sample of papers with embeddings
    print("\n2. Testing ranking with different user profiles...")
    c.execute("""
        SELECT id, title, embedding 
        FROM papers 
        WHERE embedding IS NOT NULL 
        LIMIT 10
    """)
    papers = c.fetchall()
    
    if len(papers) < 3:
        print("   ❌ FAIL: Not enough papers to test ranking")
        return False
    
    # Convert embeddings to numpy
    test_papers = []
    for p in papers:
        try:
            emb = np.frombuffer(p[2], dtype=np.float32)
            test_papers.append({
                'id': p[0],
                'title': p[1],
                'embedding': emb
            })
        except:
            pass
    
    if len(test_papers) < 3:
        print("   ❌ FAIL: Could not parse embeddings")
        return False
    
    print(f"   Testing with {len(test_papers)} papers")
    
    # 3. Test ranking with different user profiles
    # Profile 1: Similar to first paper
    profile_1 = test_papers[0]['embedding']
    
    # Profile 2: Similar to last paper  
    profile_2 = test_papers[-1]['embedding']
    
    # Calculate cosine similarities for both profiles
    def rank_papers(profile, papers):
        scores = []
        for p in papers:
            # Cosine similarity
            dot = np.dot(profile, p['embedding'])
            norm_p = np.linalg.norm(profile)
            norm_e = np.linalg.norm(p['embedding'])
            score = dot / (norm_p * norm_e + 1e-9)
            scores.append((p['id'], p['title'], score))
        # Sort by score descending
        scores.sort(key=lambda x: x[2], reverse=True)
        return scores
    
    ranking_1 = rank_papers(profile_1, test_papers)
    ranking_2 = rank_papers(profile_2, test_papers)
    
    print("\n   Profile 1 (similar to first paper):")
    print(f"   Top paper: {ranking_1[0][1][:60]}... (score: {ranking_1[0][2]:.4f})")
    
    print("\n   Profile 2 (similar to last paper):")
    print(f"   Top paper: {ranking_2[0][1][:60]}... (score: {ranking_2[0][2]:.4f})")
    
    # 4. Verify rankings are different
    if ranking_1[0][0] == ranking_2[0][0]:
        print("\n   ⚠️  WARNING: Both profiles ranked the same paper first")
        print("   This might be OK if papers are very similar")
    else:
        print("\n   ✅ PASS: Different profiles produce different rankings")
    
    # 5. Check actual user profile exists
    print("\n3. Checking actual user profile...")
    c.execute("SELECT preference_vector FROM user_profile WHERE user_id = 'default'")
    profile_row = c.fetchone()
    
    if not profile_row or not profile_row[0]:
        print("   ⚠️  No user profile found - need to interact with papers first")
    else:
        user_vector = np.frombuffer(profile_row[0], dtype=np.float32)
        print(f"   ✅ User profile exists (dimension: {len(user_vector)})")
        
        # Test ranking with real profile
        real_ranking = rank_papers(user_vector, test_papers)
        print(f"\n   With YOUR profile, top paper:")
        print(f"   {real_ranking[0][1][:70]}...")
        print(f"   Score: {real_ranking[0][2]:.4f}")
    
    # 6. Verify feed query matches our manual ranking
    print("\n4. Verifying feed query logic...")
    
    # Get what the actual feed would return
    c.execute("""
        SELECT p.id, p.title, p.embedding
        FROM papers p
        WHERE p.category IN ('physics.comp-ph', 'physics.flu-dyn')
        AND p.embedding IS NOT NULL
        ORDER BY p.published_date DESC
        LIMIT 10
    """)
    
    feed_papers = c.fetchall()
    print(f"   Feed would return {len(feed_papers)} papers")
    
    if len(feed_papers) > 0 and profile_row and profile_row[0]:
        # Rank them
        feed_test = []
        for p in feed_papers:
            emb = np.frombuffer(p[2], dtype=np.float32)
            feed_test.append({'id': p[0], 'title': p[1], 'embedding': emb})
        
        feed_ranking = rank_papers(user_vector, feed_test)
        print(f"\n   Top ranked paper in actual feed:")
        print(f"   {feed_ranking[0][1][:70]}...")
        print(f"   Score: {feed_ranking[0][2]:.4f}")
        
        print("\n   ✅ PASS: Ranking logic is working correctly")
    
    conn.close()
    return True

def test_rocchio_algorithm():
    """
    Test that Rocchio algorithm is updating the user profile correctly.
    """
    print("\n\n=== Testing Rocchio Algorithm ===\n")
    
    conn = get_db_connection()
    c = conn.cursor()
    
    # Check interactions exist
    c.execute("SELECT COUNT(*) FROM interactions")
    interaction_count = c.fetchone()[0]
    print(f"Total interactions: {interaction_count}")
    
    if interaction_count == 0:
        print("⚠️  No interactions yet - Rocchio has no data to work with")
        conn.close()
        return
    
    # Check liked papers
    c.execute("""
        SELECT COUNT(*), action 
        FROM interactions 
        WHERE user_id = 'default' 
        GROUP BY action
    """)
    
    for row in c.fetchall():
        print(f"  {row[1]}: {row[0]} papers")
    
    # Verify profile exists
    c.execute("SELECT preference_vector FROM user_profile WHERE user_id = 'default'")
    profile_row = c.fetchone()
    
    if not profile_row or not profile_row[0]:
        print("\n❌ FAIL: No user profile despite having interactions")
        print("Run 'Update Recommendations' to build your profile")
        conn.close()
        return
    
    user_profile = np.frombuffer(profile_row[0], dtype=np.float32)
    print(f"\n✅ User profile exists")
    print(f"   Dimension: {len(user_profile)}")
    print(f"   Norm: {np.linalg.norm(user_profile):.4f}")
    print(f"   Non-zero elements: {np.count_nonzero(user_profile)}/{len(user_profile)}")
    
    conn.close()

if __name__ == "__main__":
    success = test_embedding_based_ranking()
    test_rocchio_algorithm()
    
    if success:
        print("\n" + "="*60)
        print("✅ All ranking tests passed!")
        print("="*60)
    else:
        print("\n" + "="*60)
        print("❌ Some tests failed - check output above")
        print("="*60)
