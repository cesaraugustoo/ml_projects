import pytest
from scientific_discovery.src.graph_gen import GraphConfig, KnowledgeGraphBuilder
from scientific_discovery.src.agent_tools.science import create_science_agent_group

@pytest.fixture
def graph_config():
    return GraphConfig(
        chunk_size=2500,
        chunk_overlap=0,
        similarity_threshold=0.95,
        system_prompt="Test system prompt"
    )

def test_full_research_pipeline(
    test_data_dir,
    mock_llm_client,
    embedding_tools,
    graph_config
):
    builder = KnowledgeGraphBuilder(graph_config, test_data_dir)
    agent_group = create_science_agent_group(
        mock_llm_client,
        research_field="materials_science"
    )
    
    # Test complete pipeline
    research_text = """
    Novel materials for energy storage applications have garnered significant 
    attention due to their potential in renewable energy systems. Recent 
    developments in nanomaterials and composite structures have shown 
    promising results for improving battery performance.
    """
    
    graph, embeddings = builder.build_graph_from_text(
        research_text,
        mock_llm_client.generate_text,
        "test_research"
    )
    
    result = agent_group.process_research_task(research_text)
    
    assert graph.number_of_nodes() > 0
    assert result is not None