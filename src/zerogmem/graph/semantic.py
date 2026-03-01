"""
Semantic Graph: Embedding-based semantic memory representation.

Handles semantic similarity and concept relationships.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional, List, Dict, Any, Tuple
import numpy as np
import networkx as nx


@dataclass
class SemanticNode:
    """A node in the semantic graph representing a memory or concept."""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    content: str = ""
    embedding: Optional[np.ndarray] = None
    concepts: List[str] = field(default_factory=list)  # Associated concepts/topics
    importance: float = 0.5  # Attention-weighted importance
    memory_id: Optional[str] = None  # Reference to source memory
    created_at: datetime = field(default_factory=datetime.now)
    access_count: int = 0
    last_accessed: Optional[datetime] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __hash__(self):
        return hash(self.id)

    def __eq__(self, other):
        if isinstance(other, SemanticNode):
            return self.id == other.id
        return False


@dataclass
class SemanticEdge:
    """An edge representing semantic relationship between nodes."""
    source_id: str = ""
    target_id: str = ""
    relation: str = ""  # is_a, part_of, related_to, similar_to, etc.
    weight: float = 1.0  # Strength of relation
    similarity: Optional[float] = None  # Cosine similarity if computed
    metadata: Dict[str, Any] = field(default_factory=dict)


class SemanticGraph:
    """
    Graph structure for semantic memory with embedding support.

    Key capabilities:
    - Store memories with embeddings
    - Compute semantic similarity
    - Find related concepts
    - Support for semantic clustering
    """

    # Semantic relation types
    RELATION_TYPES = [
        "is_a",           # Hierarchical: "dog is_a animal"
        "part_of",        # Compositional: "wheel part_of car"
        "related_to",     # General association
        "similar_to",     # Semantic similarity
        "opposite_of",    # Antonymy
        "causes",         # Causal (light version)
        "follows",        # Sequential
        "example_of",     # Instance relation
    ]

    def __init__(self, embedding_dim: int = 1536):
        self.graph = nx.Graph()  # Undirected for semantic similarity
        self.nodes: Dict[str, SemanticNode] = {}
        self.edges: List[SemanticEdge] = []
        self.embedding_dim = embedding_dim

        # Embedding index for fast similarity search
        self._embeddings: List[np.ndarray] = []
        self._embedding_ids: List[str] = []

        # Concept index
        self._concept_index: Dict[str, set] = {}

    def add_node(self, node: SemanticNode) -> str:
        """Add a semantic node to the graph."""
        self.nodes[node.id] = node
        self.graph.add_node(node.id, data=node)

        # Index embedding
        if node.embedding is not None:
            self._embeddings.append(node.embedding)
            self._embedding_ids.append(node.id)

        # Index concepts
        for concept in node.concepts:
            if concept not in self._concept_index:
                self._concept_index[concept] = set()
            self._concept_index[concept].add(node.id)

        return node.id

    def add_edge(self, edge: SemanticEdge) -> None:
        """Add a semantic edge to the graph."""
        self.edges.append(edge)
        self.graph.add_edge(
            edge.source_id,
            edge.target_id,
            relation=edge.relation,
            weight=edge.weight,
            data=edge
        )

    def compute_similarity(self, embedding1: np.ndarray, embedding2: np.ndarray) -> float:
        """Compute cosine similarity between two embeddings."""
        norm1 = np.linalg.norm(embedding1)
        norm2 = np.linalg.norm(embedding2)
        if norm1 == 0 or norm2 == 0:
            return 0.0
        return float(np.dot(embedding1, embedding2) / (norm1 * norm2))

    def find_similar(
        self,
        query_embedding: np.ndarray,
        top_k: int = 10,
        threshold: float = 0.0
    ) -> List[Tuple[SemanticNode, float]]:
        """
        Find nodes most similar to query embedding.

        Returns list of (node, similarity_score) tuples.
        """
        if not self._embeddings:
            return []

        # Compute similarities
        similarities = []
        for i, emb in enumerate(self._embeddings):
            sim = self.compute_similarity(query_embedding, emb)
            if sim >= threshold:
                node_id = self._embedding_ids[i]
                node = self.nodes.get(node_id)
                if node:
                    similarities.append((node, sim))

        # Sort by similarity descending
        similarities.sort(key=lambda x: x[1], reverse=True)
        return similarities[:top_k]

    def find_by_concept(self, concept: str) -> List[SemanticNode]:
        """Find all nodes associated with a concept."""
        if concept not in self._concept_index:
            return []
        return [self.nodes[nid] for nid in self._concept_index[concept] if nid in self.nodes]

    def find_related(
        self,
        node_id: str,
        relation_filter: Optional[List[str]] = None,
        max_depth: int = 2
    ) -> List[Tuple[SemanticNode, str, int]]:
        """
        Find nodes related to given node through graph traversal.

        Returns list of (node, relation, depth) tuples.
        """
        if node_id not in self.graph:
            return []

        results = []
        visited = {node_id}
        queue = [(node_id, 0)]

        while queue:
            current_id, depth = queue.pop(0)
            if depth >= max_depth:
                continue

            for neighbor_id in self.graph.neighbors(current_id):
                if neighbor_id in visited:
                    continue

                edge_data = self.graph.get_edge_data(current_id, neighbor_id)
                relation = edge_data.get('relation', 'related_to') if edge_data else 'related_to'

                if relation_filter and relation not in relation_filter:
                    continue

                visited.add(neighbor_id)
                neighbor_node = self.nodes.get(neighbor_id)
                if neighbor_node:
                    results.append((neighbor_node, relation, depth + 1))
                    queue.append((neighbor_id, depth + 1))

        return results

    def auto_link_similar(self, threshold: float = 0.8) -> int:
        """
        Automatically create edges between highly similar nodes.
        Returns number of edges created.
        """
        edges_created = 0

        for i, emb1 in enumerate(self._embeddings):
            for j, emb2 in enumerate(self._embeddings):
                if i >= j:
                    continue

                sim = self.compute_similarity(emb1, emb2)
                if sim >= threshold:
                    node1_id = self._embedding_ids[i]
                    node2_id = self._embedding_ids[j]

                    # Check if edge already exists
                    if not self.graph.has_edge(node1_id, node2_id):
                        edge = SemanticEdge(
                            source_id=node1_id,
                            target_id=node2_id,
                            relation="similar_to",
                            weight=sim,
                            similarity=sim
                        )
                        self.add_edge(edge)
                        edges_created += 1

        return edges_created

    def get_cluster(self, node_id: str, similarity_threshold: float = 0.7) -> List[SemanticNode]:
        """Get a cluster of semantically similar nodes."""
        node = self.nodes.get(node_id)
        if not node or node.embedding is None:
            return [node] if node else []

        similar = self.find_similar(
            node.embedding,
            top_k=20,
            threshold=similarity_threshold
        )

        return [n for n, _ in similar]

    def update_importance(self, node_id: str, delta: float = 0.1) -> None:
        """Update node importance based on access patterns."""
        node = self.nodes.get(node_id)
        if node:
            node.importance = min(1.0, max(0.0, node.importance + delta))
            node.access_count += 1
            node.last_accessed = datetime.now()

    def get_node(self, node_id: str) -> Optional[SemanticNode]:
        """Get a node by ID."""
        return self.nodes.get(node_id)

    def get_embedding_matrix(self) -> np.ndarray:
        """Get all embeddings as a matrix for batch operations."""
        if not self._embeddings:
            return np.array([])
        return np.vstack(self._embeddings)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize graph to dictionary (excluding embeddings)."""
        return {
            "nodes": [
                {
                    "id": n.id,
                    "content": n.content,
                    "concepts": n.concepts,
                    "importance": n.importance,
                    "memory_id": n.memory_id,
                    "created_at": n.created_at.isoformat(),
                    "access_count": n.access_count,
                }
                for n in self.nodes.values()
            ],
            "edges": [
                {
                    "source_id": e.source_id,
                    "target_id": e.target_id,
                    "relation": e.relation,
                    "weight": e.weight,
                }
                for e in self.edges
            ]
        }
