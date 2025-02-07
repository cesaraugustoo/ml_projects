import pytest
from scientific_discovery.src.graph_gen import GraphConfig, KnowledgeGraphBuilder
# from scientific_discovery.src.agent_tools.science import create_science_agent_group
from scientific_discovery.src.agent_tools.science import ScientificAgent, ScienceAgentConfig, ScienceRole, ScienceAgentGroup

@pytest.fixture
def graph_config():
    return GraphConfig(
        chunk_size=2500,
        chunk_overlap=0,
        similarity_threshold=0.95,
        system_prompt="Test system prompt"
    )


# def test_full_research_pipeline(
#     test_data_dir,
#     mock_llm_client,
#     embedding_tools, 
#     graph_config,
#     mock_science_agent
# ):
#     builder = KnowledgeGraphBuilder(graph_config, test_data_dir)

#     # Create proper agent group with required roles
#     planner = mock_science_agent.__class__(  # Use mock agent's class instead of ConcreteScientificAgent
#         config=ScienceAgentConfig(
#             name="test_planner",
#             role=ScienceRole.PLANNER,
#             system_message="Test planner message",
#             research_field="materials_science"
#         ),
#         llm_client=mock_llm_client
#     )
    
#     ontologist = mock_science_agent.__class__(
#         config=ScienceAgentConfig(
#             name="test_ontologist",
#             role=ScienceRole.ONTOLOGIST,
#             system_message="Test ontologist message",
#             research_field="materials_science"
#         ),
#         llm_client=mock_llm_client
#     )
    
#     agent_group = ScienceAgentGroup([
#         planner,
#         ontologist,
#         mock_science_agent
#     ])
    
#     # Test complete pipeline
#     research_text = """
#     Novel materials for energy storage applications have garnered significant 
#     attention due to their potential in renewable energy systems. Recent 
#     developments in nanomaterials and composite structures have shown 
#     promising results for improving battery performance.
#     """
    
#     graph, embeddings = builder.build_graph_from_text(
#         research_text,
#         mock_llm_client.generate_text,
#         "test_research"
#     )
    
#     result = agent_group.process_research_task(research_text)
#     assert result is not None
#     assert "plan" in result
#     assert "ontology" in result
#     assert "analysis" in result

# tests/test_integration/test_full_pipeline.py

def test_full_research_pipeline(
    test_data_dir,
    mock_llm_client,
    embedding_tools,
    graph_config,
    mock_science_agent
):
    # Override mock response with complete sequence of responses needed
    mock_llm_client.generate_text.side_effect = [
        # Graph components response
        '''{
            "edges": [
                {
                    "source": "material",
                    "target": "property",
                    "attributes": {"type": "has"}
                },
                {
                    "source": "nanomaterial",
                    "target": "battery",
                    "attributes": {"type": "improves"}
                }
            ]
        }''',
        # Planner response
        '''{
            "plan": "Analyze energy storage materials",
            "objectives": ["Understand materials", "Evaluate properties"],
            "methodology": ["Literature review", "Property analysis"]
        }''',
        # Ontologist concepts response
        '''{
            "concepts": {
                "material": "Base material class",
                "property": "Material characteristics",
                "nanomaterial": "Nano-scale material",
                "battery": "Energy storage device"
            }
        }''',
        # Ontologist relationships response
        '''[
            {
                "source": "material",
                "target": "property",
                "relationship": "has"
            },
            {
                "source": "nanomaterial",
                "target": "battery",
                "relationship": "improves"
            }
        ]''',
        # Scientist hypothesis response
        '''{
            "hypothesis": "Nanomaterials improve battery performance",
            "mechanisms": [
                {
                    "name": "surface_area",
                    "description": "Increased surface area improves reaction rates"
                }
            ],
            "outcomes": [
                {
                    "type": "performance",
                    "prediction": "Higher energy density"
                }
            ]
        }''',
        # Any additional responses needed for other agent interactions...
        '''{
            "analysis": "Valid approach",
            "recommendations": ["Further testing needed"]
        }'''
    ]

    builder = KnowledgeGraphBuilder(graph_config, test_data_dir)
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

    critic = mock_science_agent.__class__(  # Add critic agent
        config=ScienceAgentConfig(
            name="test_critic",
            role=ScienceRole.CRITIC,
            system_message="Test critic message",
            research_field="materials_science"
        ),
        llm_client=mock_llm_client
    )

    agent_group = ScienceAgentGroup([
        planner,
        ontologist,
        mock_science_agent,
        critic  # Include critic in agent group
    ])

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
    
    # Add assertions
    assert graph.number_of_nodes() > 0
    assert len(embeddings) > 0
    assert result is not None
    assert 'ontology' in result
    assert 'analysis' in result