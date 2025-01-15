import pytest
from scientific_discovery.src.agent_tools.science import (
    ScientistAgent,
    OntologistAgent,
    CriticAgent,
    ScienceAgentGroup
)

def test_scientist_agent_hypothesis_generation(mock_llm_client):
    agent = ScientistAgent(
        config=ScienceAgentConfig(
            name="test_scientist",
            role=ScienceRole.SCIENTIST,
            system_message="Test system message",
            research_field="materials_science"
        ),
        llm_client=mock_llm_client
    )
    
    result = agent.process_message("Test research task")
    assert "hypothesis" in result
    assert "mechanisms" in result
    assert "outcomes" in result

def test_science_agent_group_coordination(science_agent_group):
    task = "Investigate novel materials for energy storage"
    result = science_agent_group.process_research_task(task)
    
    assert "plan" in result
    assert "ontology" in result
    assert "analysis" in result
    assert "critique" in result