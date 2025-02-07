import json
import pytest
import networkx as nx
from typing import Dict, Any 
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
def mock_science_agent(mock_llm_client):
    """Create a mock scientific agent for testing."""
    class TestScientificAgent(ScientificAgent):
        def process_message(self, message: str) -> Dict[str, Any]:
            response = self.llm_client.generate_text(
                system_prompt=self.config.system_message,
                user_prompt=message
            )
            return json.loads(response)
    
    config = ScienceAgentConfig(
        name="test_scientist",
        role=ScienceRole.SCIENTIST,
        system_message="Test message",
        research_field="materials_science"
    )
    
    return TestScientificAgent(config, mock_llm_client)

# tests/test_integration/test_agent_graph.py
def test_agent_graph_integration(mock_science_agent, mock_llm_client, sample_graph):
    """Test integration using the mock agent."""
    # Create instance of test agent for each required role 
    planner = mock_science_agent.__class__(
        config=ScienceAgentConfig(
            name="test_planner",
            role=ScienceRole.PLANNER,
            system_message="Test planner message",
            research_field="materials_science"
        ),
        llm_client=mock_llm_client
    )
    
    ontologist = mock_science_agent.__class__(
        config=ScienceAgentConfig(
            name="test_ontologist", 
            role=ScienceRole.ONTOLOGIST,
            system_message="Test ontologist message",
            research_field="materials_science"
        ),
        llm_client=mock_llm_client
    )
    
    critic = mock_science_agent.__class__(
        config=ScienceAgentConfig(
            name="test_critic",
            role=ScienceRole.CRITIC, 
            system_message="Test critic message",
            research_field="materials_science"
        ),
        llm_client=mock_llm_client
    )

    # Create agent group with all required agents
    agent_group = ScienceAgentGroup([
        planner,
        ontologist, 
        mock_science_agent,  # Scientist role
        critic
    ])

    result = agent_group.process_research_task(
        "Analyze material properties in graph"
    )
    
    assert result is not None
    assert agent_group.knowledge_graph.number_of_nodes() > 0

# tests/test_integration/test_agent_graph.py
def test_graph_based_research_flow(
    mock_llm_client,
    embedding_tools,
    sample_graph,
    mock_science_agent
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
        concepts={node: f"Test concept {node}" for node in sample_graph.nodes}
    )
    
    # Create agents with all required roles
    agents = [
        mock_science_agent.__class__(
            config=ScienceAgentConfig(
                name=f"{role.name.lower()}_agent",
                role=role,
                system_message=f"Test {role.name.lower()} message",
                research_field="materials_science",
                research_context=context
            ),
            llm_client=mock_llm_client
        )
        for role in [ScienceRole.PLANNER, ScienceRole.ONTOLOGIST, ScienceRole.SCIENTIST, ScienceRole.CRITIC]
    ]
    
    agent_group = ScienceAgentGroup(agents)
    
    result = agent_group.process_research_task("Test research task")
    assert result is not None

def test_multi_agent_graph_coordination(
    mock_llm_client,
    sample_graph,
    mock_science_agent
):
    """Test coordination between multiple agents using graph data."""
    # Create agents with all required roles
    agents = [
        mock_science_agent.__class__(
            config=ScienceAgentConfig(
                name=f"{role.name.lower()}_agent",
                role=role,
                system_message=f"Test {role.name.lower()} message",
                research_field="materials_science"
            ),
            llm_client=mock_llm_client
        )
        for role in [ScienceRole.PLANNER, ScienceRole.ONTOLOGIST, ScienceRole.SCIENTIST, ScienceRole.CRITIC]
    ]
    
    agent_group = ScienceAgentGroup(agents)
    
    # Process tasks with graph updates
    results = []
    for i in range(3):
        result = agent_group.process_research_task(
            f"Task {i} with graph analysis"
        )
        results.append(result)
    
    assert len(results) == 3
    assert all(result is not None for result in results)