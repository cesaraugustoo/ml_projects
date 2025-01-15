import pytest
from graph_reasoning.agent_tools.base import (
    BaseAgent,
    AgentConfig,
    AgentRole,
    Message,
    ConversationMemory,
    AgentFactory
)

def test_message_creation():
    """Test message object creation and conversion."""
    message = Message(
        role="user",
        content="Test message",
        metadata={"timestamp": "2025-01-15"}
    )
    
    message_dict = message.to_dict()
    assert message_dict["role"] == "user"
    assert message_dict["content"] == "Test message"
    assert "timestamp" in message_dict["metadata"]

def test_agent_config_validation():
    """Test agent configuration validation."""
    # Valid configuration
    valid_config = AgentConfig(
        name="test_agent",
        role=AgentRole.PLANNER,
        system_message="Test system message",
        max_consecutive_auto_reply=5
    )
    assert valid_config.name == "test_agent"
    
    # Invalid configurations
    with pytest.raises(ValueError):
        AgentConfig(
            name="",  # Empty name
            role=AgentRole.PLANNER,
            system_message="Test"
        )
    
    with pytest.raises(ValueError):
        AgentConfig(
            name="test",
            role=AgentRole.PLANNER,
            system_message="",  # Empty system message
        )

def test_conversation_memory():
    """Test conversation memory management."""
    memory = ConversationMemory(max_messages=3)
    
    # Add messages
    messages = [
        Message(role="user", content=f"Message {i}")
        for i in range(4)
    ]
    
    for msg in messages:
        memory.add_message(msg)
    
    # Check memory limit
    assert len(memory.messages) == 3
    # Check FIFO behavior
    assert memory.messages[0].content == "Message 1"
    
    # Test context retrieval
    context = memory.get_context(last_n=2)
    assert len(context) == 2
    assert context[-1].content == "Message 3"

def test_agent_factory():
    """Test agent factory creation methods."""
    config = AgentConfig(
        name="test_agent",
        role=AgentRole.PLANNER,
        system_message="Test system message"
    )
    
    # Test knowledge agent creation
    agent = AgentFactory.create_agent("knowledge", config, mock_llm_client)
    assert isinstance(agent, BaseAgent)
    
    # Test invalid agent type
    with pytest.raises(ValueError):
        AgentFactory.create_agent("invalid_type", config, mock_llm_client)

class TestBaseAgent:
    """Test base agent functionality."""
    
    @pytest.fixture
    def base_agent(self, mock_llm_client):
        config = AgentConfig(
            name="test_agent",
            role=AgentRole.PLANNER,
            system_message="Test system message"
        )
        return BaseAgent(config, mock_llm_client)
    
    def test_termination_check(self, base_agent):
        """Test message termination detection."""
        assert base_agent._should_terminate("TERMINATE")
        assert base_agent._should_terminate("Task DONE")
        assert not base_agent._should_terminate("Regular message")
    
    def test_agent_reset(self, base_agent):
        """Test agent state reset."""
        base_agent.memory.add_message(
            Message(role="user", content="Test")
        )
        base_agent.reset()
        assert len(base_agent.memory.messages) == 0