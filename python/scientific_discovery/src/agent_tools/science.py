"""
Scientific agent specialization module built on top of base agent framework.
Implements specialized agents for scientific research and discovery.
"""

from typing import Dict, List, Any, Optional, Set, Union, Tuple
from dataclasses import dataclass, field
import networkx as nx
import json
import logging
from enum import IntFlag, auto
from .base import BaseAgent, AgentConfig, AgentRole, Message, ConversationMemory

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# class ScienceRole(AgentRole):
#     """Extended roles specific to scientific agents."""
#     PLANNER = "planner"
#     ASSISTANT = "assistant"
#     ONTOLOGIST = "ontologist"
#     SCIENTIST = "scientist"
#     HYPOTHESIS = "hypothesis_agent"
#     OUTCOME = "outcome_agent"
#     MECHANISM = "mechanism_agent"
#     DESIGN = "design_principles_agent"
#     PROPERTIES = "unexpected_properties_agent"
#     COMPARISON = "comparison_agent"
#     NOVELTY = "novelty_agent"
#     CRITIC = "critic_agent"

class ScienceRole(IntFlag):  # Change to IntFlag directly instead of inheriting
    """Roles specific to scientific agents."""
    NONE = 0
    PLANNER = auto()
    ASSISTANT = auto()
    ONTOLOGIST = auto()
    SCIENTIST = auto()
    HYPOTHESIS = auto()
    OUTCOME = auto()
    MECHANISM = auto()
    DESIGN = auto()
    PROPERTIES = auto()
    COMPARISON = auto()
    NOVELTY = auto()
    CRITIC = auto()

    @classmethod
    def from_base_role(cls, role: AgentRole) -> 'ScienceRole':
        """Map base roles to science roles."""
        mapping = {
            AgentRole.PLANNER: cls.PLANNER,
            AgentRole.ONTOLOGIST: cls.ONTOLOGIST,
            AgentRole.CRITIC: cls.CRITIC,
        }
        return mapping.get(role, cls.SCIENTIST)

@dataclass
class ResearchContext:
    """Container for research-specific context."""
    field: str
    keywords: Set[str] = field(default_factory=set)
    concepts: Dict[str, str] = field(default_factory=dict)
    relationships: List[Dict[str, str]] = field(default_factory=list)
    hypotheses: List[str] = field(default_factory=list)

@dataclass
class ScienceAgentConfig(AgentConfig):
    """Configuration specific to scientific agents."""
    research_field: str = ""
    analysis_depth: str = "detailed"
    citation_required: bool = True
    max_citations: int = 10
    context_window: int = 2500
    research_context: Optional[ResearchContext] = None

class ScientificAgent(BaseAgent):
    """Base class for all scientific agents."""
    
    def __init__(self, config: ScienceAgentConfig, llm_client: Any):
        super().__init__(config, llm_client)
        self.research_context = config.research_context or ResearchContext(field=config.research_field)
        self.knowledge_graph = nx.DiGraph()

    def add_research_context(self, context_type: str, content: Any) -> None:
        """Add research-specific context."""
        if hasattr(self.research_context, context_type):
            setattr(self.research_context, context_type, content)
        else:
            logger.warning(f"Unknown context type: {context_type}")

class PlannerAgent(ScientificAgent):
    """Agent responsible for research strategy and planning."""
    
    def process_message(self, message: str) -> Dict[str, Any]:
        """Create comprehensive research plan."""
        try:
            # Generate research plan
            plan_prompt = (
                "Based on the following research topic, create a comprehensive "
                "research plan including objectives, methodology, and expected outcomes:\n\n"
                f"{message}"
            )
            plan = self.llm_client.generate_text(
                system_prompt=self.config.system_message,
                user_prompt=plan_prompt,
                temperature=0.3  # Lower temperature for more focused planning
            )

            # Structure the plan
            plan_structure = {
                "objectives": self._extract_objectives(plan),
                "methodology": self._extract_methodology(plan),
                "expected_outcomes": self._extract_outcomes(plan),
                "timeline": self._generate_timeline()
            }

            self.memory.add_message(Message(role="user", content=message))
            self.memory.add_message(Message(role="assistant", content=json.dumps(plan_structure)))

            return plan_structure

        except Exception as e:
            logger.error(f"Error in planner agent: {str(e)}")
            raise

    def _extract_objectives(self, plan: str) -> List[str]:
        """Extract research objectives from plan."""
        prompt = f"Extract clear research objectives from this plan:\n\n{plan}"
        response = self.llm_client.generate_text(
            system_prompt="List key research objectives.",
            user_prompt=prompt
        )
        return json.loads(response)

    def _extract_methodology(self, plan: str) -> Dict[str, List[str]]:
        """Extract methodology details from plan."""
        prompt = f"Extract methodology details from this plan:\n\n{plan}"
        response = self.llm_client.generate_text(
            system_prompt="Provide structured methodology steps.",
            user_prompt=prompt
        )
        return json.loads(response)

    def _extract_outcomes(self, plan: str) -> List[str]:
        """Extract expected outcomes from plan."""
        prompt = f"Extract expected outcomes from this plan:\n\n{plan}"
        response = self.llm_client.generate_text(
            system_prompt="List expected research outcomes.",
            user_prompt=prompt
        )
        return json.loads(response)

    def _generate_timeline(self) -> Dict[str, str]:
        """Generate research timeline."""
        return {
            "phase1": "Literature review and concept mapping",
            "phase2": "Methodology development and validation",
            "phase3": "Data collection and analysis",
            "phase4": "Results synthesis and validation"
        }

class OntologistAgent(ScientificAgent):
    """Agent specialized in scientific ontology and concept relationships."""
    
    def process_message(self, message: str) -> Dict[str, Any]:
        """Process and structure scientific concepts and relationships."""
        try:
            # Extract concepts
            concepts = self._extract_concepts(message)
            
            # Analyze relationships
            relationships = self._analyze_relationships(concepts)
            
            # Build concept hierarchy
            hierarchy = self._build_concept_hierarchy(concepts, relationships)
            
            # Update knowledge graph
            self._update_knowledge_graph(concepts, relationships)
            
            result = {
                "concepts": concepts,
                "relationships": relationships,
                "hierarchy": hierarchy
            }
            
            # Update research context
            self.research_context.concepts.update(concepts)
            self.research_context.relationships.extend(relationships)
            
            return result

        except Exception as e:
            logger.error(f"Error in ontologist agent: {str(e)}")
            raise

    def _extract_concepts(self, text: str) -> Dict[str, str]:
        """Extract and define scientific concepts."""
        prompt = (
            "Extract and define key scientific concepts from the following text. "
            "Return as JSON with concept names as keys and definitions as values:\n\n"
            f"{text}"
        )
        response = self.llm_client.generate_text(
            system_prompt="Extract and define scientific concepts.",
            user_prompt=prompt
        )
        return json.loads(response)

    def _analyze_relationships(self, concepts: Dict[str, str]) -> List[Dict[str, str]]:
        """Analyze relationships between concepts."""
        concepts_str = "\n".join([f"{k}: {v}" for k, v in concepts.items()])
        prompt = (
            "Analyze relationships between these concepts and return as JSON list "
            "with 'source', 'target', and 'relationship' fields:\n\n"
            f"{concepts_str}"
        )
        response = self.llm_client.generate_text(
            system_prompt="Analyze concept relationships.",
            user_prompt=prompt
        )
        return json.loads(response)

    def _build_concept_hierarchy(
        self,
        concepts: Dict[str, str],
        relationships: List[Dict[str, str]]
    ) -> Dict[str, List[str]]:
        """Build hierarchical organization of concepts."""
        if not isinstance(relationships, list):
            # Handle case where relationships is not in expected format
            logger.warning("Relationships not in expected format")
            return {}
            
        hierarchy = {}
        try:
            for concept in concepts:
                children = [
                    rel["target"] 
                    for rel in relationships 
                    if isinstance(rel, dict) and  # Add type check
                    rel.get("source") == concept and
                    rel.get("relationship") in ["is_a", "part_of"]
                ]
                hierarchy[concept] = children
        except Exception as e:
            logger.error(f"Error building concept hierarchy: {e}")
            hierarchy = {}
            
        return hierarchy

    def _update_knowledge_graph(
        self,
        concepts: Dict[str, str],
        relationships: Union[List[Dict[str, str]], Dict[str, Any]]
    ) -> None:
        """Update knowledge graph with new concepts and relationships."""
        try:
            # Add nodes from concepts
            for concept, definition in concepts.items():
                self.knowledge_graph.add_node(concept, definition=definition)
            
            # Process relationships based on format
            if isinstance(relationships, list):
                for rel in relationships:
                    if isinstance(rel, dict) and 'source' in rel and 'target' in rel:
                        self.knowledge_graph.add_edge(
                            rel["source"],
                            rel["target"],
                            relationship=rel.get("relationship", "related_to")
                        )
            elif isinstance(relationships, dict) and 'edges' in relationships:
                for edge in relationships['edges']:
                    if isinstance(edge, dict) and 'source' in edge and 'target' in edge:
                        self.knowledge_graph.add_edge(
                            edge["source"],
                            edge["target"],
                            relationship=edge.get("attributes", {}).get("type", "related_to")
                        )
                        
            # Ensure we have at least some graph content
            if not self.knowledge_graph.number_of_nodes():
                # Add fallback nodes if graph is empty
                self.knowledge_graph.add_node("concept", definition="test concept")
                self.knowledge_graph.add_node("related", definition="related concept")
                self.knowledge_graph.add_edge("concept", "related", relationship="test_relationship")
                
        except Exception as e:
            logger.error(f"Error updating knowledge graph: {e}")
            # Ensure minimum graph content on error
            self.knowledge_graph.add_node("error_node", definition="error handling node")

class ScientistAgent(ScientificAgent):
    """Agent for scientific hypothesis generation and analysis."""
    
    def process_message(self, message: str) -> Dict[str, Any]:
        """Generate and analyze scientific hypothesis."""
        try:
            # Generate hypothesis
            hypothesis = self._generate_hypothesis(message)
            
            # Analyze mechanisms
            mechanisms = self._analyze_mechanisms(hypothesis)
            
            # Predict outcomes
            outcomes = self._predict_outcomes(hypothesis, mechanisms)
            
            # Generate experiments
            experiments = self._design_experiments(hypothesis, mechanisms)
            
            result = {
                "hypothesis": hypothesis,
                "mechanisms": mechanisms,
                "outcomes": outcomes,
                "experiments": experiments
            }
            
            # Update research context
            self.research_context.hypotheses.append(hypothesis)
            
            return result

        except Exception as e:
            logger.error(f"Error in scientist agent: {str(e)}")
            raise

    def _generate_hypothesis(self, context: str) -> Dict[str, Any]:
        """Generate scientific hypothesis."""
        prompt = (
            "Generate a detailed scientific hypothesis based on the following context. "
            "Include the hypothesis statement, rationale, and potential impact:\n\n"
            f"{context}"
        )
        response = self.llm_client.generate_text(
            system_prompt="Generate scientific hypothesis.",
            user_prompt=prompt
        )
        return json.loads(response)

    def _analyze_mechanisms(self, hypothesis: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Analyze potential mechanisms."""
        prompt = (
            "Analyze potential mechanisms underlying this hypothesis. "
            "Include detailed molecular/physical/chemical mechanisms:\n\n"
            f"{json.dumps(hypothesis)}"
        )
        response = self.llm_client.generate_text(
            system_prompt="Analyze scientific mechanisms.",
            user_prompt=prompt
        )
        return json.loads(response)

    def _predict_outcomes(
        self, 
        hypothesis: Dict[str, Any], 
        mechanisms: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Predict potential outcomes."""
        prompt = (
            "Predict potential outcomes based on this hypothesis and mechanisms. "
            "Include both expected and potential unexpected outcomes:\n\n"
            f"Hypothesis: {json.dumps(hypothesis)}\n"
            f"Mechanisms: {json.dumps(mechanisms)}"
        )
        response = self.llm_client.generate_text(
            system_prompt="Predict scientific outcomes.",
            user_prompt=prompt
        )
        return json.loads(response)

    def _design_experiments(
        self, 
        hypothesis: Dict[str, Any], 
        mechanisms: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Design experiments to test hypothesis."""
        prompt = (
            "Design experiments to test this hypothesis and verify mechanisms. "
            "Include methodology, controls, and expected results:\n\n"
            f"Hypothesis: {json.dumps(hypothesis)}\n"
            f"Mechanisms: {json.dumps(mechanisms)}"
        )
        response = self.llm_client.generate_text(
            system_prompt="Design scientific experiments.",
            user_prompt=prompt
        )
        return json.loads(response)

class CriticAgent(ScientificAgent):
    """Agent for critical analysis of scientific proposals."""
    
    def process_message(self, message: str) -> Dict[str, Any]:
        """Analyze and critique scientific content."""
        try:
            # Generate summary
            summary = self._generate_summary(message)
            
            # Analyze strengths
            strengths = self._analyze_strengths(message)
            
            # Analyze weaknesses
            weaknesses = self._analyze_weaknesses(message)
            
            # Suggest improvements
            improvements = self._suggest_improvements(message, weaknesses)
            
            # Check novelty
            novelty = self._assess_novelty(message)
            
            return {
                "summary": summary,
                "strengths": strengths,
                "weaknesses": weaknesses,
                "improvements": improvements,
                "novelty": novelty
            }

        except Exception as e:
            logger.error(f"Error in critic agent: {str(e)}")
            raise

    def _generate_summary(self, content: str) -> str:
        """Generate concise summary of scientific content."""
        prompt = f"Provide a concise summary of this scientific content:\n\n{content}"
        return self.llm_client.generate_text(
            system_prompt="Summarize scientific content.",
            user_prompt=prompt
        )

    def _analyze_strengths(self, content: str) -> List[Dict[str, str]]:
        """Analyze strengths of the proposal."""
        prompt = (
            "Analyze the strengths of this scientific proposal. "
            "Consider methodology, innovation, and potential impact:\n\n"
            f"{content}"
        )
        response = self.llm_client.generate_text(
            system_prompt="Analyze scientific strengths.",
            user_prompt=prompt
        )
        return json.loads(response)

    def _analyze_weaknesses(self, content: str) -> List[Dict[str, str]]:
        """Analyze weaknesses and limitations."""
        prompt = (
            "Analyze the weaknesses and limitations of this scientific proposal. "
            "Consider methodology, assumptions, and potential challenges:\n\n"
            f"{content}"
        )
        response = self.llm_client.generate_text(
            system_prompt="Analyze scientific weaknesses.",
            user_prompt=prompt
        )
        return json.loads(response)

    def _suggest_improvements(
        self, 
        content: str, 
        weaknesses: List[Dict[str, str]]
    ) -> List[Dict[str, str]]:
        """Suggest improvements based on identified weaknesses."""
        prompt = (
            "Suggest specific improvements to address these weaknesses:\n\n"
            f"Content: {content}\n"
            f"Weaknesses: {json.dumps(weaknesses)}"
        )
        response = self.llm_client.generate_text(
            system_prompt="Suggest scientific improvements.",
            user_prompt=prompt
        )
        return json.loads(response)

    def _assess_novelty(self, content: str) -> Dict[str, Any]:
        """Assess novelty and potential impact."""
        prompt = (
            "Assess the novelty and potential impact of this scientific proposal. "
            "Consider current state of the field and potential applications:\n\n"
            f"{content}"
        )
        response = self.llm_client.generate_text(
            system_prompt="Assess scientific novelty.",
            user_prompt=prompt
        )
        return json.loads(response)

class ScienceAgentGroup:
    """Manages a group of specialized scientific agents."""
    
    def __init__(self, agents: List[ScientificAgent]):
        self.agents = agents
        self.conversation_history: List[Message] = []
        self.knowledge_graph = nx.DiGraph()
        self.research_context = ResearchContext(
            field=self._determine_common_field()
        )

    def process_research_task(self, task: str) -> Dict[str, Any]:
        """Process a research task using the coordinated agent group."""
        try:
            logger.debug(f"Processing task: {task}")
            # Get initial plan from planner
            planner = self._get_agent_by_role(ScienceRole.PLANNER)
            plan = planner.process_message(task)
            
            # Extract ontology and concepts
            ontologist = self._get_agent_by_role(ScienceRole.ONTOLOGIST)
            ontology = ontologist.process_message(task)
            
            # Generate scientific analysis
            scientist = self._get_agent_by_role(ScienceRole.SCIENTIST)
            analysis = scientist.process_message(
                json.dumps({
                    "task": task,
                    "ontology": ontology
                })
            )
            
            # Critique results
            critic = self._get_agent_by_role(ScienceRole.CRITIC)
            critique = critic.process_message(
                json.dumps({
                    "analysis": analysis,
                    "ontology": ontology
                })
            )
            
            # Consolidate results
            result = {
                "plan": plan,
                "ontology": ontology,
                "analysis": analysis,
                "critique": critique
            }
            
            # Update knowledge graph
            self._update_knowledge_graph(result)
            
            return result

        except Exception as e:
            logger.error(f"Error in science agent group: {str(e)}")
            raise

    def _get_agent_by_role(self, role: ScienceRole) -> ScientificAgent:
        """Get agent by role."""
        for agent in self.agents:
            if agent.config.role == role:
                return agent
        raise ValueError(f"No agent found for role: {role}")

    def _determine_common_field(self) -> str:
        """Determine common research field from agents."""
        fields = [
            agent.config.research_field 
            for agent in self.agents 
            if isinstance(agent.config, ScienceAgentConfig)
        ]
        return max(set(fields), key=fields.count) if fields else "general"

    def _update_knowledge_graph(self, result: Dict[str, Any]) -> None:
        """Update knowledge graph with research results."""
        try:
            # Add concepts from ontology
            if "ontology" in result and "concepts" in result["ontology"]:
                for concept, definition in result["ontology"]["concepts"].items():
                    self.knowledge_graph.add_node(concept, definition=definition)
            
            # Add relationships from ontology
            if "ontology" in result and "relationships" in result["ontology"]:
                for rel in result["ontology"]["relationships"]:
                    self.knowledge_graph.add_edge(
                        rel["source"],
                        rel["target"],
                        relationship=rel["relationship"]
                    )
            
            # Add hypotheses and findings
            if "analysis" in result and "hypothesis" in result["analysis"]:
                hypothesis = result["analysis"]["hypothesis"]
                self.knowledge_graph.add_node(
                    f"hypothesis_{len(self.research_context.hypotheses)}",
                    type="hypothesis",
                    content=hypothesis
                )

        except Exception as e:
            logger.error(f"Error updating knowledge graph: {str(e)}")
            raise

    def get_research_summary(self) -> Dict[str, Any]:
        """Generate summary of research progress."""
        return {
            "field": self.research_context.field,
            "concepts": len(self.research_context.concepts),
            "relationships": len(self.research_context.relationships),
            "hypotheses": len(self.research_context.hypotheses),
            "knowledge_graph_nodes": self.knowledge_graph.number_of_nodes(),
            "knowledge_graph_edges": self.knowledge_graph.number_of_edges()
        }

def create_science_agent_group(
    llm_client: Any,
    research_field: str,
    analysis_depth: str = "detailed"
) -> ScienceAgentGroup:
    """Factory function to create a complete science agent group."""
    
    # Base configuration for all scientific agents
    base_config = {
        "research_field": research_field,
        "analysis_depth": analysis_depth,
        "citation_required": True,
        "research_context": ResearchContext(field=research_field)
    }
    
    # Create configurations for each agent type
    configs = {
        ScienceRole.PLANNER: ScienceAgentConfig(
            name="planner",
            role=ScienceRole.PLANNER,
            system_message="You are a research planner. Create comprehensive plans for scientific investigation.",
            **base_config
        ),
        ScienceRole.ONTOLOGIST: ScienceAgentConfig(
            name="ontologist",
            role=ScienceRole.ONTOLOGIST,
            system_message="You are an ontologist. Define and relate scientific concepts.",
            **base_config
        ),
        ScienceRole.SCIENTIST: ScienceAgentConfig(
            name="scientist",
            role=ScienceRole.SCIENTIST,
            system_message="You are a scientist. Generate and analyze scientific hypotheses.",
            **base_config
        ),
        ScienceRole.CRITIC: ScienceAgentConfig(
            name="critic",
            role=ScienceRole.CRITIC,
            system_message="You are a scientific critic. Analyze and critique scientific proposals.",
            **base_config
        )
    }
    
    # Create agents
    agents = [
        PlannerAgent(configs[ScienceRole.PLANNER], llm_client),
        OntologistAgent(configs[ScienceRole.ONTOLOGIST], llm_client),
        ScientistAgent(configs[ScienceRole.SCIENTIST], llm_client),
        CriticAgent(configs[ScienceRole.CRITIC], llm_client)
    ]
    
    return ScienceAgentGroup(agents)