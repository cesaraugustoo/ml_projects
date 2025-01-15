"""Base agent framework providing core functionality for all agent types."""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Set
import networkx as nx
from enum import IntFlag, auto
import json
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class AgentRole(IntFlag):
    """Base enumeration of possible agent roles."""
    NONE = 0  # Always start with 0 for IntFlag
    PLANNER = auto()
    ANALYZER = auto()
    RESEARCHER = auto()
    CRITIC = auto()
    ONTOLOGIST = auto()
    COORDINATOR = auto()

@dataclass
class Message:
    """Represents a message in the agent conversation."""
    role: str
    content: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert message to dictionary format."""
        return {
            "role": self.role,
            "content": self.content,
            "metadata": self.metadata
        }

@dataclass
class AgentConfig:
    """Configuration for an agent."""
    name: str
    role: AgentRole
    system_message: str
    max_consecutive_auto_reply: int = 10
    termination_keywords: Set[str] = field(default_factory=lambda: {"TERMINATE", "DONE", "COMPLETE"})
    temperature: float = 0.2
    max_tokens: int = 2048
    
    def __post_init__(self):
        """Validate configuration after initialization."""
        if not self.name:
            raise ValueError("Agent name cannot be empty")
        if not self.system_message:
            raise ValueError("System message cannot be empty")
        if self.max_consecutive_auto_reply < 1:
            raise ValueError("max_consecutive_auto_reply must be positive")
        if not 0 <= self.temperature <= 1:
            raise ValueError("temperature must be between 0 and 1")

class ConversationMemory:
    """Manages conversation history with efficient memory usage."""
    
    def __init__(self, max_messages: int = 100):
        self.messages: List[Message] = []
        self.max_messages = max_messages
        
    def add_message(self, message: Message) -> None:
        """Add a message to the conversation history."""
        if len(self.messages) >= self.max_messages:
            self.messages.pop(0)  # Remove oldest message
        self.messages.append(message)
    
    def get_context(self, last_n: Optional[int] = None) -> List[Message]:
        """Get the conversation context, optionally limited to last n messages."""
        if last_n is None:
            return self.messages
        return self.messages[-last_n:]
    
    def clear(self) -> None:
        """Clear the conversation history."""
        self.messages.clear()

class BaseAgent(ABC):
    """Abstract base class for all agents in the system."""
    
    def __init__(self, config: AgentConfig, llm_client: Any):
        self.config = config
        self.llm_client = llm_client
        self.memory = ConversationMemory()
        self._validate_config()
    
    def _validate_config(self) -> None:
        """Validate agent configuration."""
        if not isinstance(self.config, AgentConfig):
            raise TypeError("config must be an instance of AgentConfig")
    
    @abstractmethod
    def process_message(self, message: str) -> str:
        """Process an incoming message and return a response."""
        pass
    
    def _should_terminate(self, message: str) -> bool:
        """Check if the conversation should terminate."""
        return any(keyword.lower() in message.lower() 
                  for keyword in self.config.termination_keywords)
    
    def reset(self) -> None:
        """Reset the agent's state."""
        self.memory.clear()

class KnowledgeAgent(BaseAgent):
    """Agent specialized in knowledge extraction and graph building."""
    
    def __init__(self, config: AgentConfig, llm_client: Any):
        super().__init__(config, llm_client)
        self.knowledge_graph = nx.DiGraph()
        
    def process_message(self, message: str) -> str:
        """
        Process message and extract knowledge into graph.
        
        Args:
            message: Input message to process
            
        Returns:
            str: Response with extracted knowledge
        """
        try:
            # Generate response using LLM
            response = self.llm_client.generate_text(
                system_prompt=self.config.system_message,
                user_prompt=message,
                temperature=self.config.temperature,
                max_tokens=self.config.max_tokens
            )
            
            # Extract knowledge triples
            knowledge = self._extract_knowledge(response)
            
            # Update knowledge graph
            self._update_graph(knowledge)
            
            # Store conversation
            self.memory.add_message(Message(role="user", content=message))
            self.memory.add_message(Message(role="assistant", content=response))
            
            return response
            
        except Exception as e:
            logger.error(f"Error processing message: {str(e)}")
            raise
    
    def _extract_knowledge(self, text: str) -> List[Dict[str, str]]:
        """Extract knowledge triples from text."""
        try:
            # Parse response expecting JSON format with knowledge triples
            knowledge = json.loads(text)
            if not isinstance(knowledge, list):
                raise ValueError("Expected list of knowledge triples")
            return knowledge
        except json.JSONDecodeError:
            logger.warning("Failed to parse JSON response, attempting fallback extraction")
            return self._fallback_extraction(text)
    
    def _fallback_extraction(self, text: str) -> List[Dict[str, str]]:
        """Fallback method for knowledge extraction when JSON parsing fails."""
        # Implement simple pattern matching for triples
        triples = []
        # Basic pattern: "subject ... predicate ... object"
        # This is a simplified version - in practice, use more sophisticated NLP
        sentences = text.split('.')
        for sentence in sentences:
            if ' is ' in sentence:
                parts = sentence.split(' is ')
                if len(parts) == 2:
                    triples.append({
                        "subject": parts[0].strip(),
                        "predicate": "is",
                        "object": parts[1].strip()
                    })
        return triples
    
    def _update_graph(self, knowledge: List[Dict[str, str]]) -> None:
        """Update knowledge graph with new information."""
        for triple in knowledge:
            subject = triple.get("subject", "").strip()
            predicate = triple.get("predicate", "").strip()
            obj = triple.get("object", "").strip()
            
            if subject and predicate and obj:
                self.knowledge_graph.add_edge(
                    subject, 
                    obj, 
                    relationship=predicate
                )

class AgentFactory:
    """Factory class for creating different types of agents."""
    
    @staticmethod
    def create_agent(agent_type: str, config: AgentConfig, llm_client: Any) -> BaseAgent:
        """Create an agent of the specified type."""
        agents = {
            "knowledge": KnowledgeAgent,
            # Add more agent types here
        }
        
        agent_class = agents.get(agent_type.lower())
        if agent_class is None:
            raise ValueError(f"Unknown agent type: {agent_type}")
            
        return agent_class(config, llm_client)

class AgentGroup:
    """Manages a group of collaborating agents."""
    
    def __init__(self, agents: List[BaseAgent]):
        self.agents = agents
        self.conversation_history: List[Message] = []
    
    def process_task(self, task: str) -> str:
        """Process a task using the group of agents."""
        current_message = task
        
        for agent in self.agents:
            response = agent.process_message(current_message)
            self.conversation_history.append(Message(
                role=agent.config.role.value,
                content=response
            ))
            current_message = response
            
            if agent._should_terminate(response):
                break
        
        return current_message
    
    def get_consolidated_knowledge(self) -> nx.DiGraph:
        """Get consolidated knowledge graph from all agents."""
        consolidated_graph = nx.DiGraph()
        
        for agent in self.agents:
            if isinstance(agent, KnowledgeAgent):
                consolidated_graph = nx.compose(
                    consolidated_graph, 
                    agent.knowledge_graph
                )
        
        return consolidated_graph