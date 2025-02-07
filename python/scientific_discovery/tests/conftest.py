import pytest
from pathlib import Path
from typing import Dict, Any
import json
import networkx as nx
from scientific_discovery.src.embedding_tools import EmbeddingTools
from scientific_discovery.src.llm_tools import OpenAIClient, OpenAIConfig
from scientific_discovery.src.agent_tools.science import ScientificAgent, ScienceAgentConfig, ScienceRole, ScienceAgentGroup

@pytest.fixture
def test_data_dir():
    return Path(__file__).parent / "test_data"

@pytest.fixture
def sample_graph():
    G = nx.Graph()
    G.add_nodes_from(['A', 'B', 'C'])
    G.add_edges_from([('A', 'B'), ('B', 'C')])
    return G

@pytest.fixture
def embedding_tools():
    return EmbeddingTools(model_name="bert-base-uncased")

@pytest.fixture
def mock_llm_client(mocker):
    mock_config = OpenAIConfig(
        api_key="test_key",
        organization="test_org"
    )
    client = OpenAIClient(mock_config)
    
    # Updated mock responses with correct format
    responses = [
        # Ontology concepts - with knowledge graph nodes
        '''{
            "edges": {
                "source": "material",
                "target": "property"
            }
        }''',
        # Ontology relationships - with knowledge graph edges
        '''[
            {
                "source": "material",
                "target": "property",
                "atributes": "has_property"
            }
        ]''',
        # Scientific analysis with graph updates
        '''{
            "hypothesis": "Test hypothesis",
            "mechanisms": [{"name": "mechanism1", "description": "test"}],
            "nodes": ["node1", "node2"],
            "edges": [{"source": "node1", "target": "node2", "relationship": "related_to"}]
        }''',
        # Critic analysis preserving graph structure
        '''{
            "analysis": "Test analysis",
            "feedback": "Test feedback",
            "graph_validation": true
        }'''
    ] * 5  # Multiple copies to ensure enough responses
    
    mocker.patch.object(
        client, 
        'generate_text',
        side_effect=responses
    )
    
    return client

# @pytest.fixture
# def science_agent_group(mock_llm_client):
#     return create_science_agent_group(
#         mock_llm_client,
#         research_field="materials_science",
#         analysis_depth="detailed"
#     )

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

@pytest.fixture
def science_agent_group(mock_science_agent, mock_llm_client):
    """Create a science agent group with all required roles for testing."""
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

    return ScienceAgentGroup([
        planner,
        ontologist,
        mock_science_agent,  # Scientist role
        critic
    ])