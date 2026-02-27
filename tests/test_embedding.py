
import pytest
import numpy as np
from unittest.mock import MagicMock, patch, ANY
import ingest
import json

# --- Mocks & Fixtures ---

@pytest.fixture
def mock_requests_post():
    with patch('ingest.requests.post') as mock_post:
        yield mock_post

@pytest.fixture
def mock_db_connection():
    with patch('ingest.get_db_connection') as mock_conn_fn:
        mock_conn = MagicMock()
        mock_cursor = MagicMock()
        mock_conn.cursor.return_value = mock_cursor
        mock_conn_fn.return_value = mock_conn
        yield mock_conn, mock_cursor

# --- Semantic Scholar Batch Fetch Tests ---

def test_fetch_ss_embeddings_batch_success(mock_requests_post):
    """Test successful fetching of embeddings from Semantic Scholar."""
    # Input papers
    papers = [
        {'source': 'arxiv', 'external_id': '2106.15928v1'},
        {'source': 'semantic_scholar', 'external_id': '1234567890abcdef'}
    ]

    # Mock response data
    # SS returns list in order of requested IDs
    mock_response_data = [
        {
            'paperId': 'SS:ARXIV:2106.15928',
            'embedding': {'vector': [0.1, 0.2, 0.3]}
        },
        {
            'paperId': '1234567890abcdef',
            'embedding': {'vector': [0.4, 0.5, 0.6]}
        }
    ]
    
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = mock_response_data
    mock_requests_post.return_value = mock_resp

    # Execute
    results = ingest.fetch_ss_embeddings_batch(papers)

    # Assertions
    # IDs mapped correctly:
    # arxiv:2106.15928v1 -> ARXIV:2106.15928 (stripped v1) -> matched back to arxiv:2106.15928v1
    assert 'arxiv:2106.15928v1' in results
    assert 'semantic_scholar:1234567890abcdef' in results
    
    # Check vector values (stored as bytes)
    vec1 = np.frombuffer(results['arxiv:2106.15928v1'], dtype=np.float32)
    assert np.allclose(vec1, [0.1, 0.2, 0.3])

def test_fetch_ss_embeddings_batch_partial_missing(mock_requests_post):
    """Test when some papers are not found in SS."""
    papers = [
        {'source': 'arxiv', 'external_id': 'found_paper'},
        {'source': 'arxiv', 'external_id': 'missing_paper'}
    ]

    # SS returns None for missing items
    mock_response_data = [
        {'paperId': 'SS:found', 'embedding': {'vector': [0.1]}},
        None 
    ]
    
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = mock_response_data
    mock_requests_post.return_value = mock_resp

    results = ingest.fetch_ss_embeddings_batch(papers)

    assert 'arxiv:found_paper' in results
    assert 'arxiv:missing_paper' not in results

def test_fetch_ss_embeddings_batch_empty():
    """Test with empty input list."""
    results = ingest.fetch_ss_embeddings_batch([])
    assert results == {}

# --- Local Embedding Logic Tests ---

def test_get_embedding_success():
    """Verify that local embedding generation works and returns correct shape."""
    # This might load the model which takes time, but it's a necessary integration test.
    # We can mock get_model if we want to test just the pooling logic, 
    # but testing the full stack is better for verification.
    
    # Use a short text
    text = "Test paper title [SEP] Abstract content"
    
    try:
        embedding = ingest.get_embedding(text)
        
        # Check shape: Specter2 base is 768 dim
        assert isinstance(embedding, np.ndarray)
        assert embedding.shape == (768,)
        assert embedding.dtype == np.float32
        
        # Check values are not all zero
        assert not np.allclose(embedding, 0)
        
    except Exception as e:
        pytest.fail(f"get_embedding failed with error: {e}")

# --- Save Papers Logic Tests ---

def test_save_papers_inserts_new_with_embedding(mock_db_connection):
    """Test saving new papers that have embeddings."""
    mock_conn, mock_cursor = mock_db_connection
    
    # Setup: First call (check exists) returns None
    # Second call (get rowid) returns (101,)
    mock_cursor.fetchone.side_effect = [None, (101,)]

    embedding_bytes = np.array([0.1, 0.2], dtype=np.float32).tobytes()
    papers = [{
        'source': 'arxiv', 
        'external_id': '1234', 
        'title': 'Test', 
        'abstract': 'Abs', 
        'authors': ['A'], 
        'published_date': '2023', 
        'category': 'cs', 
        'link': 'http',
        'embedding': embedding_bytes
    }]

    count = ingest.save_papers(papers)

    assert count == 1
    
    # Verify insert into 'papers'
    # Check that arguments to INSERT contain our embedding bytes
    insert_call = mock_cursor.execute.call_args_list[1] # 0 is SELECT check, 1 is INSERT
    assert "INSERT INTO papers" in insert_call[0][0]
    assert insert_call[0][1][9] == embedding_bytes # embedding is 10th arg (index 9)

    # Verify insert into 'vss_papers'
    vss_call = mock_cursor.execute.call_args_list[3] # 2 is SELECT rowid, 3 is INSERT vss
    assert "INSERT INTO vss_papers" in vss_call[0][0]
    assert vss_call[0][1] == (101, embedding_bytes)

def test_save_papers_skip_vss_if_no_embedding(mock_db_connection):
    """Test saving papers without embeddings skips VSS insert."""
    mock_conn, mock_cursor = mock_db_connection
    mock_cursor.fetchone.return_value = None
    
    papers = [{
        'source': 'arxiv', 'external_id': 'no_embed', 
        'title': 'T', 'abstract': 'A', 'authors': [], 
        'published_date': '', 'category': '', 'link': ''
        # No 'embedding' key
    }]

    ingest.save_papers(papers)

    # Verify regular insert happened
    assert any("INSERT INTO papers" in call[0][0] for call in mock_cursor.execute.call_args_list)
    
    # Verify VSS insert did NOT happen
    assert not any("INSERT INTO vss_papers" in call[0][0] for call in mock_cursor.execute.call_args_list)

def test_save_papers_updates_existing_embedding(mock_db_connection):
    """Test that existing papers get their embeddings updated if missing."""
    mock_conn, mock_cursor = mock_db_connection
    
    # Setup: fetchone returns a row with None embedding
    mock_cursor.fetchone.side_effect = [(None,), (101,)] # 1st call: existing paper check returns (None,) (embedding is None). 2nd call: get rowid.

    embedding_bytes = b'fake_embedding'
    papers = [{
        'source': 'arxiv', 'external_id': '1234', 
        'title': 'T', 'abstract': 'A', 'authors': [], 
        'published_date': '', 'category': '', 'link': '',
        'embedding': embedding_bytes
    }]

    ingest.save_papers(papers)

    # Verify UPDATE called
    update_calls = [call for call in mock_cursor.execute.call_args_list if "UPDATE papers" in call[0][0]]
    assert len(update_calls) == 1
    assert update_calls[0][0][1][0] == embedding_bytes

    # Verify VSS Insert/Replace called
    vss_calls = [call for call in mock_cursor.execute.call_args_list if "INSERT OR REPLACE INTO vss_papers" in call[0][0]]
    assert len(vss_calls) == 1

if __name__ == "__main__":
    pytest.main([__file__])
