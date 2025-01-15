import networkx as nx
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import json
from copy import deepcopy
import logging
from typing import Dict, List, Union, Optional
import torch

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class GraphTools:
    """
    A class providing utilities for graph analysis, visualization, and simplification.
    """

    @staticmethod
    def generate_node_embeddings(graph, tokenizer, model, batch_size=32):
        """Generate embeddings for nodes in a graph using batch processing."""
        embeddings = {}
        nodes = list(graph.nodes())
        
        for i in tqdm(range(0, len(nodes), batch_size), desc="Generating embeddings"):
            batch_nodes = nodes[i:i + batch_size]
            texts = [str(node) for node in batch_nodes]
            
            try:
                inputs = tokenizer(texts, padding=True, truncation=True, return_tensors="pt")
                with torch.no_grad():
                    outputs = model(**inputs)
                
                batch_embeddings = outputs.last_hidden_state.mean(dim=1).numpy()
                
                for node, embedding in zip(batch_nodes, batch_embeddings):
                    embeddings[node] = embedding
                    
            except Exception as e:
                logger.error(f"Error processing batch {i}: {e}")
                continue
                
        return embeddings

    @staticmethod
    def heuristic_path_with_embeddings(graph, embeddings, source, target):
        """
        Find a heuristic path between two nodes using embeddings.

        Parameters:
            graph (nx.Graph): The input graph.
            embeddings (dict): Node embeddings.
            source (str): Source node.
            target (str): Target node.

        Returns:
            list: The heuristic path from source to target.
        """
        if source not in embeddings or target not in embeddings:
            raise ValueError("Source or target node embeddings are missing.")

        visited = set()
        current_node = source
        path = [source]

        while current_node != target:
            visited.add(current_node)
            neighbors = [n for n in graph.neighbors(current_node) if n not in visited]

            if not neighbors:
                logger.error("No path found.")
                return []

            # Select the neighbor closest in embedding space to the target
            target_embedding = embeddings[target].flatten()
            current_embedding = embeddings[current_node].flatten()
            similarities = {
                neighbor: cosine_similarity(
                    [embeddings[neighbor].flatten()], [target_embedding]
                )[0][0]
                for neighbor in neighbors
            }

            # Choose the next node with the highest similarity to the target
            next_node = max(similarities, key=similarities.get)
            path.append(next_node)
            current_node = next_node

        return path

    @staticmethod
    def simplify_graph(
        graph: nx.Graph,
        embeddings: Dict[str, np.ndarray],
        similarity_threshold: float = 0.9,
        chunk_size: int = 1000
    ) -> nx.Graph:
        """
        Simplify a graph by merging similar nodes based on embeddings.
        
        Args:
            graph: Input graph to simplify
            embeddings: Dictionary mapping nodes to their embeddings
            similarity_threshold: Threshold for merging nodes (0.0 to 1.0)
            
        Returns:
            Simplified graph with merged nodes
            
        Raises:
            ValueError: If threshold is invalid or embeddings are missing
        """
        if not 0 <= similarity_threshold <= 1:
            raise ValueError("Similarity threshold must be between 0 and 1")
        
        if not all(node in embeddings for node in graph.nodes()):
            raise ValueError("Missing embeddings for some nodes")
        
        logger.info("Simplifying graph based on similarity threshold")
        nodes = list(embeddings.keys())
        node_mapping = {}
        for i in range(0, len(nodes), chunk_size):
            chunk_nodes = nodes[i:i + chunk_size]
            chunk_embeddings = np.array([embeddings[node].flatten() for node in chunk_nodes])
            
            for j in range(0, len(nodes), chunk_size):
                other_nodes = nodes[j:j + chunk_size]
                other_embeddings = np.array([embeddings[node].flatten() for node in other_nodes])
                
                similarities = cosine_similarity(chunk_embeddings, other_embeddings)
                similar_pairs = np.where(similarities > similarity_threshold)
                
                for idx1, idx2 in zip(*similar_pairs):
                    node1, node2 = chunk_nodes[idx1], other_nodes[idx2]
                    if node1 != node2:
                        node_mapping[node2] = node1
        
        return nx.relabel_nodes(graph, node_mapping, copy=True)

    @staticmethod
    def visualize_embeddings_2d(embeddings, output_path, title="Node Embeddings Visualization"):
        """Visualize embeddings in 2D using PCA."""
        node_ids = list(embeddings.keys())
        vectors = np.array([embeddings[node].flatten() for node in node_ids])
        pca = PCA(n_components=2)
        vectors_2d = pca.fit_transform(vectors)

        plt.figure(figsize=(10, 8))
        plt.scatter(vectors_2d[:, 0], vectors_2d[:, 1], alpha=0.5)
        for i, node_id in enumerate(node_ids):
            plt.text(vectors_2d[i, 0], vectors_2d[i, 1], str(node_id), fontsize=9)
        plt.title(title)
        plt.xlabel("PCA 1")
        plt.ylabel("PCA 2")
        plt.savefig(output_path)
        plt.show()

    @staticmethod
    def analyze_graph(graph):
        """Analyze a graph and return basic statistics."""
        num_nodes = graph.number_of_nodes()
        num_edges = graph.number_of_edges()
        density = nx.density(graph)
        components = nx.number_connected_components(graph)

        logger.info(f"Nodes: {num_nodes}, Edges: {num_edges}, Density: {density:.4f}, Components: {components}")

        return {
            "num_nodes": num_nodes,
            "num_edges": num_edges,
            "density": density,
            "connected_components": components,
        }

    @staticmethod
    def detect_and_visualize_communities(graph, output_path):
        """Detect and visualize communities in a graph."""
        from networkx.algorithms.community import greedy_modularity_communities
        logger.info("Detecting communities using greedy modularity.")
        communities = list(greedy_modularity_communities(graph))

        community_mapping = {}
        for idx, community in enumerate(communities):
            for node in community:
                community_mapping[node] = idx

        pos = nx.spring_layout(graph)
        plt.figure(figsize=(12, 10))
        nx.draw_networkx_nodes(
            graph,
            pos,
            node_color=[community_mapping[node] for node in graph.nodes()],
            cmap=plt.cm.tab20,
            node_size=50
        )
        nx.draw_networkx_edges(graph, pos, alpha=0.3)
        plt.title("Community Structure")
        plt.savefig(output_path)
        plt.show()

    @staticmethod
    def save_graph(graph, path):
        """Save a graph to a file."""
        nx.write_graphml(graph, path)
        logger.info(f"Graph saved to {path}")

    @staticmethod
    def is_scale_free(graph):
        """Determine if a graph is scale-free based on degree distribution."""
        from powerlaw import Fit

        degrees = [d for n, d in graph.degree()]
        fit = Fit(degrees, discrete=True)
        logger.info(f"Power-law alpha: {fit.power_law.alpha}, xmin: {fit.power_law.xmin}")

        return fit.distribution_compare('power_law', 'exponential')[0] > 0