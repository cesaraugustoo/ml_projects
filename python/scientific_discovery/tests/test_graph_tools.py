import pytest
import networkx as nx
from scientific_discovery.src.graph_tools import GraphTools

def test_node_embedding_generation(sample_graph, embedding_tools):
    embeddings = GraphTools.generate_node_embeddings(
        sample_graph,
        embedding_tools.tokenizer,
        embedding_tools.model
    )
    
    assert len(embeddings) == sample_graph.number_of_nodes()
    assert all(isinstance(emb, np.ndarray) for emb in embeddings.values())

def test_graph_simplification(sample_graph, embedding_tools):
    embeddings = GraphTools.generate_node_embeddings(
        sample_graph,
        embedding_tools.tokenizer,
        embedding_tools.model
    )
    
    simplified_graph = GraphTools.simplify_graph(
        sample_graph,
        embeddings,
        similarity_threshold=0.9
    )
    
    assert isinstance(simplified_graph, nx.Graph)
    assert simplified_graph.number_of_nodes() <= sample_graph.number_of_nodes()