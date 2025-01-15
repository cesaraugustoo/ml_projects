import pytest
import networkx as nx
from scientific_discovery.src.graph_tools import GraphTools
from scientific_discovery.src.agent_tools.science import (
    ScienceRole,
    ScienceAgentConfig,
    ScientificAgent,
    ScienceAgentGroup,
    ResearchContext
)

@pytest.fixture
def test_scientific_agent_class():
    class TestScientificAgent(ScientificAgent):
        def process_message(self, message: str) -> str:
            return self.llm_client.generate_text(
                system_prompt=self.config.system_message,
                user_prompt=message
            )
    return TestScientificAgent

@pytest.fixture
def mock_science_agent(mock_llm_client, test_scientific_agent_class):
    config = ScienceAgentConfig(
        name="test_scientist",
        role=ScienceRole.SCIENTIST,
        system_message="Test message",
        research_field="materials_science"
    )
    return test_scientific_agent_class(config, mock_llm_client)

def test_agent_graph_integration(mock_science_agent, sample_graph):
    agent_group = ScienceAgentGroup([mock_science_agent])
    result = agent_group.process_research_task(
        "Analyze material properties in graph"
    )
    assert result is not None
    assert agent_group.knowledge_graph.number_of_nodes() > 0

def test_graph_based_research_flow(
    mock_llm_client,
    embedding_tools,
    sample_graph
):
    """Test complete research flow with graph integration."""
    # Generate embeddings for graph nodes
    embeddings = GraphTools.generate_node_embeddings(
        sample_graph,
        embedding_tools.tokenizer,
        embedding_tools.model
    )
    
    # Create research context
    context = ResearchContext(
        field="materials_science",
        concepts={
            node: f"Test concept {node}"
            for node in sample_graph.nodes
        }
    )
    
    # Create agent group with context
    agent_group = ScienceAgentGroup([
        ScientificAgent(
            config=ScienceAgentConfig(
                name="test_scientist",
                role=ScienceRole.SCIENTIST,
                system_message="Test message",
                research_field="materials_science",
                research_context=context
            ),
            llm_client=mock_llm_client
        )
    ])
    
    # Process task with graph context
    result = agent_group.process_research_task(
        "Analyze relationships in material graph"
    )
    
    # Verify integration results
    assert "analysis" in result
    assert "ontology" in result
    assert agent_group.knowledge_graph.number_of_nodes() >= sample_graph.number_of_nodes()

def test_multi_agent_graph_coordination(
    mock_llm_client,
    sample_graph
):
    """Test coordination between multiple agents using graph data."""
    # Create multiple agents
    agents = [
        ScientificAgent(
            config=ScienceAgentConfig(
                name=f"agent_{i}",
                role=ScienceRole.SCIENTIST,
                system_message="Test message",
                research_field="materials_science"
            ),
            llm_client=mock_llm_client
        )
        for i in range(3)
    ]
    
    # Create agent group
    agent_group = ScienceAgentGroup(agents)
    
    # Process tasks with graph updates
    results = []
    for i in range(3):
        result = agent_group.process_research_task(
            f"Task {i} with graph analysis"
        )
        results.append(result)
    
    # Verify coordination
    assert len(results) == 3
    assert agent_group.knowledge_graph.number_of_nodes() > 0
    assert len(agent_group.research_context.hypotheses) > 0