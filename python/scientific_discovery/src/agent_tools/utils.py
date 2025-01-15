"""Shared utilities for agent management."""

from typing import Dict, List, Any, Optional, Set, Union, Tuple
from .base import BaseAgent
from .science import ScientificAgent

def create_agent_group(agent_specs: List[Dict], llm_client: Any) -> List[BaseAgent]:
    """Create a group of agents from specifications."""
    agents = []
    for spec in agent_specs:
        agent_class = spec.pop('class')
        config = spec.pop('config')
        agents.append(agent_class(config, llm_client))
    return agents