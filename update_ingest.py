
import os

new_function = r'''
def backfill_embeddings_for_user(user_id):
    """
    Finds papers that the user has interacted with (liked/disliked) 
    which do not yet have an embedding. Calculates and saves them LOCALLY.
    """
    try:
        model_dict = get_model()
        tokenizer = model_dict['tokenizer']
    except Exception as e:
        logger.error(f"Cannot backfill embeddings: Model load failed. {e}")
        return 0

    conn = get_db_connection()
    c = conn.cursor()
    
    # metrics
    updated_count = 0
    
    try:
        # Find interacted papers with NULL embedding
        query = """
            SELECT p.id, p.title, p.abstract, p.rowid
            FROM interactions i
            JOIN papers p ON i.paper_id = p.id
            WHERE i.user_id = ? AND p.embedding IS NULL
        """
        c.execute(query, (user_id,))
        rows = c.fetchall()
        
        if not rows:
            return 0
            
        logger.info(f"Backfilling local embeddings for {len(rows)} papers...")
        
        for row in rows:
            p_id = row[0] # id
            title = row[1]
            abstract = row[2]
            row_id = row[3]
            
            # Format: Title + [SEP] + Abstract
            sep = tokenizer.sep_token
            text = f"{title}{sep}{abstract}"
            
            try:
                embedding = get_embedding(text)
                emb_bytes = embedding.tobytes()
                
                # Update main table
                c.execute("UPDATE papers SET embedding = ? WHERE id = ?", (emb_bytes, p_id))
                
                # Check VSS
                try:
                    c.execute('INSERT INTO vss_papers(rowid, embedding) VALUES (?, ?)', (row_id, emb_bytes))
                except:
                    pass
                
                updated_count += 1
            except Exception as e:
                logger.error(f"Failed to backfill embedding for {p_id}: {e}")
                
        conn.commit()
        logger.info(f"Backfilled embeddings for {updated_count} papers.")
        
    except Exception as e:
        logger.error(f"Backfill error: {e}")
        
    finally:
        conn.close()
        
    return updated_count
'''

with open('ingest.py', 'r') as f:
    lines = f.readlines()

start_idx = -1
end_idx = -1

# Find the block to replace
for i, line in enumerate(lines):
    if line.strip().startswith("def backfill_embeddings_for_user(user_id):"):
        start_idx = i
    if start_idx != -1 and line.strip() == "return 0" and "try" not in lines[i-1]: 
        # Heuristic: return 0 is the last line of the function. 
        # Validating context: next meaningful line should be import feedparser
        if i + 2 < len(lines) and "import feedparser" in lines[i+2]:
             end_idx = i
             break
        # Or just check if it's the specific return 0 we saw in view_file (line 319)
        # Check if next line is blank and then import feedparser
        if i+1 < len(lines) and not lines[i+1].strip():
             if i+2 < len(lines) and "import feedparser" in lines[i+2]:
                 end_idx = i
                 break

if start_idx != -1 and end_idx != -1:
    print(f"Replacing lines {start_idx+1} to {end_idx+1}")
    new_lines = lines[:start_idx] + [new_function.strip() + '\n\n'] + lines[end_idx+1:]
    with open('ingest.py', 'w') as f:
        f.writelines(new_lines)
    print("Success")
else:
    print(f"Could not find block. Start: {start_idx}, End: {end_idx}")
    # Fallback debug
    if start_idx != -1:
        print("Lines around start:")
        print("".join(lines[start_idx:start_idx+5]))
    
