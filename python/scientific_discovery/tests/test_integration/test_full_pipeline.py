import pytest
from graph_reasoning.graph_gen import KnowledgeGraphBuilder
from graph_reasoning.agent_tools.science import create_science_agent_group

def test_full_research_pipeline(
    test_data_dir,
    mock_llm_client,
    embedding_tools
):
    # Configure components
    config = GraphConfig()
    builder = KnowledgeGraphBuilder(config, test_data_dir)
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
    
    # Generate knowledge graph
    graph, embeddings = builder.build_graph_from_text(
        research_text,
        mock_llm_client.generate_text,
        "test_research"
    )
    
    # Process with agent group
    result = agent_group.process_research_task(research_text)
    
    # Verify outputs
    assert len(graph.nodes) > 0
    assert len(graph.edges) > 0
    assert "plan" in result
    assert "analysis" in result
    assert agent_group.knowledge_graph.number_of_nodes() > 0