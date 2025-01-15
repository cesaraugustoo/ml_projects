import pytest
from scientific_discovery.src.graph_gen import GraphConfig, KnowledgeGraphBuilder

def test_graph_config_validation():
    with pytest.raises(ValueError):
        GraphConfig(chunk_size=-1)
    with pytest.raises(ValueError):
        GraphConfig(similarity_threshold=2.0)

def test_graph_builder_initialization(test_data_dir):
    config = GraphConfig()
    builder = KnowledgeGraphBuilder(config, test_data_dir)
    assert builder.config.chunk_size == 2500
    assert builder.output_dir == test_data_dir

def test_graph_generation(test_data_dir, mock_llm_client):
    config = GraphConfig()
    builder = KnowledgeGraphBuilder(config, test_data_dir)
    text = "Test scientific text about materials."
    
    def mock_generate(system_prompt, user_prompt):
        return '{"edges": [{"source": "A", "target": "B", "attributes": {"type": "test"}}]}'
    
    graph, embeddings = builder.build_graph_from_text(
        text, 
        mock_generate,
        "test_graph"
    )
    
    assert len(graph.nodes) > 0
    assert len(graph.edges) > 0
    assert len(embeddings) > 0