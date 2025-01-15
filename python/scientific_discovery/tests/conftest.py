import pytest
from pathlib import Path
import networkx as nx
from scientific_discovery.src.embedding_tools import EmbeddingTools
from scientific_discovery.src.llm_tools import OpenAIClient, OpenAIConfig
from scientific_discovery.src.agent_tools.science import create_science_agent_group

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
    mocker.patch.object(client, 'generate_text', return_value="Test response")
    return client

@pytest.fixture
def science_agent_group(mock_llm_client):
    return create_science_agent_group(
        mock_llm_client,
        research_field="materials_science",
        analysis_depth="detailed"
    )