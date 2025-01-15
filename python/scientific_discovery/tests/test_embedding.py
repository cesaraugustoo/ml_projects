import pytest
import numpy as np
from scientific_discovery.src.embedding_tools import EmbeddingTools

def test_embedding_generation(embedding_tools):
    test_texts = ["This is a test", "Another test text"]
    embeddings = embedding_tools.generate_batch_embeddings(test_texts)
    
    assert isinstance(embeddings, np.ndarray)
    assert embeddings.shape[0] == len(test_texts)
    assert embeddings.shape[1] > 0  # Embedding dimension

def test_embedding_comparison(embedding_tools):
    text1 = "Material science research"
    text2 = "Scientific materials study"
    similarity = embedding_tools.compare_embeddings([text1], [text2])
    
    assert isinstance(similarity, float)
    assert 0 <= similarity <= 1