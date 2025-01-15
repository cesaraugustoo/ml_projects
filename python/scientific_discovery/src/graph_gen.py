from graph_reasoning.graph_tools import *
from graph_reasoning.embedding_tools import *
import networkx as nx
import json
import logging
from pathlib import Path
from tqdm import tqdm
from dataclasses import dataclass

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class GraphConfig:
    """
    Configuration for graph generation and processing.
    
    Attributes:
        chunk_size (int): Size of text chunks for processing
        chunk_overlap (int): Overlap between chunks
        similarity_threshold (float): Threshold for node similarity
        system_prompt (str): Prompt for the generation system
        model_name (str): Name of the embedding model
    """
    chunk_size: int = 2500
    chunk_overlap: int = 0
    similarity_threshold: float = 0.95
    system_prompt: str = "Extract ontology terms and identify their relationships."
    model_name: str = "bert-base-uncased"

    def __post_init__(self):
        if self.chunk_size <= 0:
            raise ValueError("chunk_size must be positive")
        if self.chunk_overlap < 0:
            raise ValueError("chunk_overlap must be non-negative")
        if not 0 <= self.similarity_threshold <= 1:
            raise ValueError("similarity_threshold must be between 0 and 1")

class KnowledgeGraphBuilder:
    """
    Builds and processes knowledge graphs from textual data.
    """
    def __init__(self, config: GraphConfig, output_dir):
        self.config = config  # Fixed: No trailing comma
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.embedding_tools = EmbeddingTools(model_name=config.model_name)


    def build_graph_from_text(self, text: str, generate_fn: callable, graph_root: str) -> tuple[nx.Graph, dict]:
        """
        Builds a knowledge graph from input text.

        Parameters:
            text (str): The input text.
            generate_fn (callable): Function to process text and extract components.
            graph_root (str): Root name for the graph.

        Returns:
            graph (nx.Graph): Generated graph.
            embeddings (dict): Node embeddings.
        """

        if not text or not isinstance(text, str):
            raise ValueError("Text input must be a non-empty string")
        if not callable(generate_fn):
            raise TypeError("generate_fn must be callable")
        if not graph_root or not isinstance(graph_root, str):
            raise ValueError("graph_root must be a non-empty string")

        logger.info("Generating graph components from text.")
        try:
            graph_components = generate_fn(system_prompt=self.config.system_prompt, user_prompt=text)

            try:
                graph_data = json.loads(graph_components)
            except json.JSONDecodeError as e:
                logger.error(f"Invalid JSON response: {graph_components}")
                raise ValueError("Failed to decode JSON response from LLM.") from e
            
            graph = self._build_graph(graph_data)

            embeddings = GraphTools.generate_node_embeddings(graph, self.embedding_tools.tokenizer, self.embedding_tools.model)

            simplified_graph = GraphTools.simplify_graph(graph, embeddings, similarity_threshold=self.config.similarity_threshold)
            self._analyze_and_save_graph(simplified_graph, embeddings, graph_root)

            return simplified_graph, embeddings
        except Exception as e:
            logger.error(f"Error generating graph: {e}")
            raise

    def _build_graph(self, graph_data: dict) -> nx.Graph:
        """
        Constructs a graph from extracted components.

        Parameters:
            graph_data (dict): Extracted graph data containing 'edges' list with 
                            'source', 'target', and 'attributes' for each edge.

        Returns:
            nx.Graph: Constructed graph.

        Raises:
            ValueError: If graph_data is missing required fields or has invalid structure.
        """
        if not isinstance(graph_data, dict) or 'edges' not in graph_data:
            raise ValueError("Invalid graph data format: missing 'edges' key")
            
        logger.info("Building graph from components.")
        graph = nx.Graph()
        for edge in graph_data.get("edges", []):
            if not all(k in edge for k in ['source', 'target', 'attributes']):
                raise ValueError(f"Invalid edge format: {edge}")
            graph.add_edge(edge["source"], edge["target"], **edge["attributes"])
        return graph

    def _analyze_and_save_graph(self, graph, embeddings, graph_root):
        """
        Analyzes, saves, and visualizes the graph and embeddings.

        Parameters:
            graph (nx.Graph): The graph to analyze and save.
            embeddings (dict): Node embeddings.
            graph_root (str): Root name for the graph.
        """
        logger.info("Analyzing and saving graph outputs.")
        # Analyze the graph
        analysis_results = GraphTools.analyze_graph(graph)
        logger.info(f"Graph Analysis: {analysis_results}")

        # Save the graph
        graph_path = self.output_dir / f"{graph_root}_graph.graphml"
        GraphTools.save_graph(graph, graph_path)

        # Save embeddings
        embedding_path = self.output_dir / f"{graph_root}_embeddings.json"
        with open(embedding_path, "w") as f:
            json.dump({k: v.tolist() for k, v in embeddings.items()}, f, indent=4)

        # Visualize embeddings
        visualization_path = self.output_dir / f"{graph_root}_embeddings_2d.png"
        GraphTools.visualize_embeddings_2d(embeddings, visualization_path)

        # Detect and visualize communities
        community_path = self.output_dir / f"{graph_root}_community.png"
        GraphTools.detect_and_visualize_communities(graph, output_path=community_path)
