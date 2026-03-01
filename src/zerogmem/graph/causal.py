"""
Causal Graph: Tracks cause-effect relationships for causal reasoning.

Enables answering "why" and "what if" questions.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional, List, Dict, Any, Set
import networkx as nx


@dataclass
class CausalNode:
    """A node representing a causal event or state."""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    content: str = ""  # Description of event/state
    event_type: str = "event"  # event, state, action, condition
    preconditions: List[str] = field(default_factory=list)  # What must be true
    effects: List[str] = field(default_factory=list)  # What becomes true
    memory_id: Optional[str] = None
    timestamp: Optional[datetime] = None
    confidence: float = 1.0
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __hash__(self):
        return hash(self.id)

    def __eq__(self, other):
        if isinstance(other, CausalNode):
            return self.id == other.id
        return False


@dataclass
class CausalEdge:
    """An edge representing causal relationship."""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    cause_id: str = ""
    effect_id: str = ""
    strength: float = 1.0  # Causal strength [0, 1]
    relation_type: str = "causes"  # causes, enables, prevents, contributes_to
    evidence: List[str] = field(default_factory=list)  # Memory IDs supporting this
    temporal_lag: Optional[float] = None  # Time between cause and effect (seconds)
    confidence: float = 1.0
    metadata: Dict[str, Any] = field(default_factory=dict)


class CausalGraph:
    """
    Directed acyclic graph for causal reasoning.

    Key capabilities:
    - Track cause-effect relationships
    - Answer "why did X happen?"
    - Answer "what will happen if Y?"
    - Detect causal chains
    """

    # Causal relation types
    RELATION_TYPES = [
        "causes",          # Direct causation
        "enables",         # Makes possible (necessary but not sufficient)
        "prevents",        # Blocks effect
        "contributes_to",  # Partial cause
        "triggers",        # Immediate cause
        "leads_to",        # Eventual outcome
    ]

    def __init__(self):
        self.graph = nx.DiGraph()
        self.nodes: Dict[str, CausalNode] = {}
        self.edges: Dict[str, CausalEdge] = {}

    def add_node(self, node: CausalNode) -> str:
        """Add a causal node to the graph."""
        self.nodes[node.id] = node
        self.graph.add_node(node.id, data=node)
        return node.id

    def add_edge(self, edge: CausalEdge) -> str:
        """
        Add a causal edge to the graph.
        Validates that adding this edge doesn't create a cycle.
        """
        # Check for cycle
        if self._would_create_cycle(edge.cause_id, edge.effect_id):
            # Log warning but still add (could be a feedback loop)
            edge.metadata["has_cycle_warning"] = True

        self.edges[edge.id] = edge
        self.graph.add_edge(
            edge.cause_id,
            edge.effect_id,
            key=edge.id,
            strength=edge.strength,
            relation=edge.relation_type,
            data=edge
        )
        return edge.id

    def _would_create_cycle(self, source: str, target: str) -> bool:
        """Check if adding edge would create a cycle."""
        if source not in self.graph or target not in self.graph:
            return False
        try:
            # Check if there's already a path from target to source
            return nx.has_path(self.graph, target, source)
        except nx.NetworkXError:
            return False

    def get_causes(
        self,
        node_id: str,
        max_depth: int = 3,
        min_strength: float = 0.0
    ) -> List[List[Tuple[CausalNode, CausalEdge]]]:
        """
        Get causal chains leading to this node.
        Returns list of paths, where each path is [(node, edge), ...]
        """
        if node_id not in self.graph:
            return []

        paths = []
        self._find_cause_paths(node_id, [], paths, max_depth, min_strength, set())
        return paths

    def _find_cause_paths(
        self,
        current_id: str,
        current_path: List[Tuple[CausalNode, CausalEdge]],
        all_paths: List[List[Tuple[CausalNode, CausalEdge]]],
        max_depth: int,
        min_strength: float,
        visited: Set[str]
    ) -> None:
        """Recursive helper to find cause paths."""
        if len(current_path) >= max_depth:
            if current_path:
                all_paths.append(current_path.copy())
            return

        if current_id in visited:
            return

        visited.add(current_id)

        # Get predecessors (causes)
        predecessors = list(self.graph.predecessors(current_id))

        if not predecessors:
            # Reached a root cause
            if current_path:
                all_paths.append(current_path.copy())
            return

        for pred_id in predecessors:
            edge_data = self.graph.get_edge_data(pred_id, current_id)
            if not edge_data:
                continue

            # Get first edge (may have multiple)
            edge_key = list(edge_data.keys())[0] if isinstance(edge_data, dict) else None
            edge = self.edges.get(edge_key) if edge_key else None

            if edge and edge.strength >= min_strength:
                pred_node = self.nodes.get(pred_id)
                if pred_node:
                    new_path = current_path + [(pred_node, edge)]
                    self._find_cause_paths(
                        pred_id, new_path, all_paths, max_depth, min_strength, visited.copy()
                    )

    def get_effects(
        self,
        node_id: str,
        max_depth: int = 3,
        min_strength: float = 0.0
    ) -> List[List[Tuple[CausalNode, CausalEdge]]]:
        """
        Get causal chains originating from this node.
        Returns list of paths showing what this event causes.
        """
        if node_id not in self.graph:
            return []

        paths = []
        self._find_effect_paths(node_id, [], paths, max_depth, min_strength, set())
        return paths

    def _find_effect_paths(
        self,
        current_id: str,
        current_path: List[Tuple[CausalNode, CausalEdge]],
        all_paths: List[List[Tuple[CausalNode, CausalEdge]]],
        max_depth: int,
        min_strength: float,
        visited: Set[str]
    ) -> None:
        """Recursive helper to find effect paths."""
        if len(current_path) >= max_depth:
            if current_path:
                all_paths.append(current_path.copy())
            return

        if current_id in visited:
            return

        visited.add(current_id)

        # Get successors (effects)
        successors = list(self.graph.successors(current_id))

        if not successors:
            # Reached a terminal effect
            if current_path:
                all_paths.append(current_path.copy())
            return

        for succ_id in successors:
            edge_data = self.graph.get_edge_data(current_id, succ_id)
            if not edge_data:
                continue

            edge_key = list(edge_data.keys())[0] if isinstance(edge_data, dict) else None
            edge = self.edges.get(edge_key) if edge_key else None

            if edge and edge.strength >= min_strength:
                succ_node = self.nodes.get(succ_id)
                if succ_node:
                    new_path = current_path + [(succ_node, edge)]
                    self._find_effect_paths(
                        succ_id, new_path, all_paths, max_depth, min_strength, visited.copy()
                    )

    def get_root_causes(self, node_id: str) -> List[CausalNode]:
        """Get ultimate root causes (nodes with no predecessors in cause chain)."""
        cause_paths = self.get_causes(node_id, max_depth=10)
        root_causes = []

        for path in cause_paths:
            if path:
                root_node, _ = path[-1]  # Last in path is earliest cause
                if root_node not in root_causes:
                    root_causes.append(root_node)

        return root_causes

    def get_causal_strength(self, cause_id: str, effect_id: str) -> float:
        """
        Get aggregate causal strength between two nodes.
        Considers all paths and their combined strength.
        """
        if cause_id not in self.graph or effect_id not in self.graph:
            return 0.0

        try:
            paths = list(nx.all_simple_paths(self.graph, cause_id, effect_id, cutoff=5))
        except nx.NetworkXError:
            return 0.0

        if not paths:
            return 0.0

        # Compute path strengths
        path_strengths = []
        for path in paths:
            strength = 1.0
            for i in range(len(path) - 1):
                edge_data = self.graph.get_edge_data(path[i], path[i+1])
                if edge_data:
                    edge_key = list(edge_data.keys())[0]
                    edge = self.edges.get(edge_key)
                    if edge:
                        strength *= edge.strength
            path_strengths.append(strength)

        # Combine: 1 - product of (1 - strength) for independent paths
        combined = 1.0
        for s in path_strengths:
            combined *= (1 - s)
        return 1 - combined

    def find_common_cause(self, node_ids: List[str]) -> List[CausalNode]:
        """Find common causes for multiple effects."""
        if not node_ids:
            return []

        # Get ancestors for each node
        ancestor_sets = []
        for node_id in node_ids:
            if node_id in self.graph:
                ancestors = nx.ancestors(self.graph, node_id)
                ancestor_sets.append(ancestors)

        if not ancestor_sets:
            return []

        # Find intersection
        common_ancestors = ancestor_sets[0]
        for ancestors in ancestor_sets[1:]:
            common_ancestors = common_ancestors.intersection(ancestors)

        return [self.nodes[nid] for nid in common_ancestors if nid in self.nodes]

    def what_if(self, node_id: str, prevented: bool = False) -> Dict[str, float]:
        """
        Counterfactual reasoning: What would change if this event didn't happen?

        Returns dict of {effect_id: probability_change}
        """
        effects = {}

        for succ_id in nx.descendants(self.graph, node_id):
            # Calculate how much this node contributes to each descendant
            strength = self.get_causal_strength(node_id, succ_id)
            if prevented:
                effects[succ_id] = -strength  # Negative means less likely
            else:
                effects[succ_id] = strength

        return effects

    def get_node(self, node_id: str) -> Optional[CausalNode]:
        """Get a node by ID."""
        return self.nodes.get(node_id)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize graph to dictionary."""
        return {
            "nodes": [
                {
                    "id": n.id,
                    "content": n.content,
                    "event_type": n.event_type,
                    "preconditions": n.preconditions,
                    "effects": n.effects,
                    "memory_id": n.memory_id,
                    "timestamp": n.timestamp.isoformat() if n.timestamp else None,
                    "confidence": n.confidence,
                }
                for n in self.nodes.values()
            ],
            "edges": [
                {
                    "id": e.id,
                    "cause_id": e.cause_id,
                    "effect_id": e.effect_id,
                    "strength": e.strength,
                    "relation_type": e.relation_type,
                    "confidence": e.confidence,
                }
                for e in self.edges.values()
            ]
        }
