from dataclasses import dataclass
from typing import Optional, List, Dict, Any, Union, Tuple
from pathlib import Path
import logging
import json
import uuid
import random
from datetime import datetime

import networkx as nx
import pandas as pd
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModel
from scipy.spatial.distance import cosine
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import seaborn as sns
from pyvis.network import Network
from tqdm import tqdm
from IPython.display import display, Markdown
from langchain.text_splitter import RecursiveCharacterTextSplitter

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class GraphConfig:
    """Configuration for graph generation and processing."""
    chunk_size: int = 2500
    chunk_overlap: int = 0
    similarity_threshold: float = 0.95
    size_threshold: int = 10
    include_contextual_proximity: bool = False
    return_only_giant_component: bool = False
    do_louvain: bool = True
    do_simplify: bool = True
    save_common_graph: bool = False
    repeat_refine: int = 0
    palette: str = "hls"

class KnowledgeGraphBuilder:
    """Main class for building and managing knowledge graphs."""
    
    def __init__(
        self,
        config: GraphConfig,
        output_dir: Union[str, Path],
        tokenizer: Optional[AutoTokenizer] = None,
        model: Optional[AutoModel] = None
    ):
        self.config = config
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.tokenizer = tokenizer
        self.model = model
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=config.chunk_size,
            chunk_overlap=config.chunk_overlap,
            length_function=len,
            is_separator_regex=False
        )

    def build_graph_from_text(
        self,
        text: str,
        generate_fn: callable,
        graph_root: str = "graph_root",
        verbatim: bool = False
    ) -> Tuple[nx.Graph, Dict[str, np.ndarray]]:
        """
        Build a knowledge graph from text input.
        
        Args:
            text: Input text to process
            generate_fn: Function to generate graph components
            graph_root: Root name for output files
            verbatim: Whether to print verbose output
            
        Returns:
            Tuple of (NetworkX graph, node embeddings dictionary)
        """
        try:
            # Split text into chunks
            chunks = self.text_splitter.split_text(text)
            logger.info(f"Created {len(chunks)} text chunks")
            
            # Convert chunks to DataFrame
            df = self._chunks_to_dataframe(chunks)
            
            # Generate graph components
            concepts = self._generate_graph_components(
                df, 
                generate_fn,
                verbatim=verbatim
            )
            
            # Create graph DataFrame
            graph_df = self._create_graph_dataframe(concepts)
            
            # Add contextual proximity if configured
            if self.config.include_contextual_proximity:
                proximity_df = self._add_contextual_proximity(graph_df)
                graph_df = pd.concat([graph_df, proximity_df], axis=0)
            
            # Build NetworkX graph
            G = self._build_networkx_graph(graph_df)
            
            # Generate node embeddings
            embeddings = self._generate_node_embeddings(G)
            
            # Apply post-processing
            G, embeddings = self._post_process_graph(G, embeddings, verbatim)
            
            # Save outputs
            self._save_graph_outputs(G, graph_df, df, graph_root)
            
            return G, embeddings
            
        except Exception as e:
            logger.error(f"Error building graph: {str(e)}")
            raise

    def _chunks_to_dataframe(self, chunks: List[str]) -> pd.DataFrame:
        """Convert text chunks to DataFrame."""
        rows = [
            {
                "text": chunk,
                "chunk_id": uuid.uuid4().hex
            }
            for chunk in chunks
        ]
        return pd.DataFrame(rows)

    def _generate_graph_components(
        self,
        df: pd.DataFrame,
        generate_fn: callable,
        verbatim: bool = False
    ) -> List[Dict]:
        """Generate graph components from text chunks."""
        results = []
        for _, row in df.iterrows():
            try:
                result = self._process_chunk(
                    row.text,
                    generate_fn,
                    {"chunk_id": row.chunk_id},
                    verbatim
                )
                if result:
                    results.extend(result)
            except Exception as e:
                logger.warning(f"Error processing chunk: {str(e)}")
                continue
        return results

    def _process_chunk(
        self,
        text: str,
        generate_fn: callable,
        metadata: Dict,
        verbatim: bool
    ) -> Optional[List[Dict]]:
        """Process a single text chunk."""
        system_prompt = (
            "Extract ontology terms and identify their relationships from the provided context using principles of category theory."
            "- Ontology terms refer to concepts, categories, or classes in the context that need to be identified."
            "- The relationships between these terms should be articulated using concepts from category theory, such as morphisms or functors."
            "Steps:\n"
            "1. Identify Terms: Examine the context to extract relevant ontology terms that represent key concepts or classes."
            "2. Determine Relationships: Using category theory, identify how these terms are related. This can involve determining morphisms or mapping functors between categories."
            "3. Analyze Structure: Consider the structure of the terms and relationships, ensuring they align with categorical principles like objects and arrows."
            "Format your output as a list of JSON triplets. Each triplet must contain a pair of terms and the relationship between them."
            "Output Format:\n"
            "The output should be formatted as a list of JSON objects, where each object represents an ontology triplet:\n"
            "```json"
            "[\n"
            "   {\n"
            '       "node_1": "A concept from extracted ontology",\n'
            '       "node_2": "A related concept from extracted ontology",\n'
            '       "edge": "Relationship between the two concepts, node_1 and node_2, succinctly described"\n'
            "   }, {...}\n"
            "]"
            "```"
            "Try to identify and produce around 10 triplets, ensuring they effectively capture the ontology presented in the context."
            "Examples:\n"
            "Example 1\n"
            "Context:```Alice is Marc's mother.```\n"
            "Output:\n"
            "```json"
            "[\n"
            "   {\n"
            '       "node_1": "Alice",\n'
            '       "node_2": "Marc",\n'
            '       "edge": "is mother of"\n'
            "   }, "
            "{...}\n"
            "]"
            "```"
            "(For a longer text, provide more triplets reflecting the various relationships.)\n\n"
            "Example 2\n"
            "Context:```Silk is a strong natural fiber used to catch prey in a web. Beta-sheets control its strength.```\n"
            "Output:\n"
            "```json"
            "[\n"
            "   {\n"
            '       "node_1": "silk",\n'
            '       "node_2": "fiber",\n'
            '       "edge": "is"\n'
            "   }," 
            "   {\n"
            '       "node_1": "beta-sheets",\n'
            '       "node_2": "strength",\n'
            '       "edge": "control"\n'
            "   },"        
            "   {\n"
            '       "node_1": "silk",\n'
            '       "node_2": "prey",\n'
            '       "edge": "catches"\n'
            "   },"
            "{...}\n"
            "]"
            "```"
            "(Use placeholders for nodes and edges where necessary. Longer contexts should yield around 10 triplets.)"
            "Notes:\n"
            "- Pay attention to nuanced relations as portrayed in the context."
            "- Ensure that output triplets form a consistent ontology depicting the relationships clearly."
            "- Ensure each identified relationship is clearly justified with category theory concepts."
            "- Be precise in the use of terminology to reflect accurate mappings and relationships."
        )
        
        try:
            response = generate_fn(
                system_prompt=system_prompt,
                prompt=f"Context: ```{text}``` \n\nOutput: "
            )
            
            # Clean and parse response
            response = self._clean_response(response)
            result = json.loads(response)
            
            # Add metadata
            return [dict(item, **metadata) for item in result]
            
        except Exception as e:
            if verbatim:
                logger.warning(f"Error processing chunk: {str(e)}")
            return None

    def _build_networkx_graph(self, df: pd.DataFrame) -> nx.Graph:
        """Build NetworkX graph from DataFrame."""
        G = nx.Graph()
        
        # Add nodes
        nodes = pd.concat([df['node_1'], df['node_2']], axis=0).unique()
        for node in nodes:
            G.add_node(str(node))
            
        # Add edges
        for _, row in df.iterrows():
            G.add_edge(
                str(row["node_1"]),
                str(row["node_2"]),
                title=row["edge"],
                weight=row['count']/4
            )
            
        return G

    def _generate_node_embeddings(self, G: nx.Graph) -> Dict[str, np.ndarray]:
        """Generate embeddings for graph nodes."""
        if not (self.tokenizer and self.model):
            return {}
            
        embeddings = {}
        for node in G.nodes():
            try:
                inputs = self.tokenizer(
                    str(node),
                    return_tensors="pt",
                    padding=True,
                    truncation=True
                )
                with torch.no_grad():
                    outputs = self.model(**inputs)
                embeddings[node] = outputs.last_hidden_state.mean(dim=1).numpy()[0]
            except Exception as e:
                logger.warning(f"Error generating embedding for node {node}: {str(e)}")
                continue
                
        return embeddings

    def _post_process_graph(
        self,
        G: nx.Graph,
        embeddings: Dict[str, np.ndarray],
        verbatim: bool
    ) -> Tuple[nx.Graph, Dict[str, np.ndarray]]:
        """Apply post-processing steps to the graph."""
        if self.config.do_simplify:
            G, embeddings = self._simplify_graph(G, embeddings, verbatim)
            
        if self.config.size_threshold > 0:
            G = self._remove_small_components(G, self.config.size_threshold)
            embeddings = {k: v for k, v in embeddings.items() if k in G.nodes()}
            
        if self.config.return_only_giant_component:
            G = self._get_giant_component(G)
            embeddings = {k: v for k, v in embeddings.items() if k in G.nodes()}
            
        if self.config.do_louvain:
            G = self._apply_louvain(G)
            
        return G, embeddings

    def _simplify_graph(
        self,
        G: nx.Graph,
        embeddings: Dict[str, np.ndarray],
        verbatim: bool
    ) -> Tuple[nx.Graph, Dict[str, np.ndarray]]:
        """Simplify graph by merging similar nodes."""
        # Implementation details omitted for brevity
        return G, embeddings

    @staticmethod
    def _remove_small_components(G: nx.Graph, min_size: int) -> nx.Graph:
        """Remove components smaller than min_size."""
        components = list(nx.connected_components(G))
        for component in components:
            if len(component) < min_size:
                G.remove_nodes_from(component)
        return G

    @staticmethod
    def _get_giant_component(G: nx.Graph) -> nx.Graph:
        """Extract the largest connected component."""
        components = list(nx.connected_components(G))
        if not components:
            return G
        return G.subgraph(max(components, key=len)).copy()

    def _apply_louvain(self, G: nx.Graph) -> nx.Graph:
        """Apply Louvain community detection."""
        try:
            import community.community_louvain as community_louvain
            communities = community_louvain.best_partition(G)
            nx.set_node_attributes(G, communities, 'community')
            return G
        except Exception as e:
            logger.warning(f"Error applying Louvain: {str(e)}")
            return G

    def _save_graph_outputs(
        self,
        G: nx.Graph,
        graph_df: pd.DataFrame,
        chunks_df: pd.DataFrame,
        graph_root: str
    ) -> None:
        """Save graph outputs to files."""
        try:
            # Save graph in GraphML format
            nx.write_graphml(G, self.output_dir / f"{graph_root}.graphml")
            
            # Save DataFrames
            graph_df.to_csv(self.output_dir / f"{graph_root}_graph.csv", index=False)
            chunks_df.to_csv(self.output_dir / f"{graph_root}_chunks.csv", index=False)
            
            # Generate and save visualization
            net = Network(notebook=True, height="900px", width="100%")
            net.from_nx(G)
            net.save(str(self.output_dir / f"{graph_root}.html"))
            
        except Exception as e:
            logger.error(f"Error saving outputs: {str(e)}")

# Example usage:
"""
config = GraphConfig(
    chunk_size=2500,
    chunk_overlap=0,
    similarity_threshold=0.95
)

builder = KnowledgeGraphBuilder(
    config=config,
    output_dir="./output",
    tokenizer=AutoTokenizer.from_pretrained("bert-base-uncased"),
    model=AutoModel.from_pretrained("bert-base-uncased")
)

G, embeddings = builder.build_graph_from_text(
    text="Your input text here",
    generate_fn=your_generate_function
)
"""