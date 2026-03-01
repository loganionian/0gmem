"""
Entity Graph: Tracks entities and their relationships.

Supports negative relations for adversarial robustness.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Optional, List, Dict, Set, Any, Tuple
import networkx as nx


class EntityType(Enum):
    """Types of entities in the memory system."""
    PERSON = "person"
    ORGANIZATION = "organization"
    LOCATION = "location"
    EVENT = "event"
    OBJECT = "object"
    CONCEPT = "concept"
    TIME = "time"
    UNKNOWN = "unknown"


@dataclass
class TimeRange:
    """Time range for temporal scoping of relations."""
    start: Optional[datetime] = None
    end: Optional[datetime] = None

    def is_valid_at(self, timestamp: datetime) -> bool:
        """Check if the time range is valid at a given timestamp."""
        if self.start and timestamp < self.start:
            return False
        if self.end and timestamp > self.end:
            return False
        return True


@dataclass
class EntityNode:
    """A node representing an entity in the knowledge graph."""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    name: str = ""
    entity_type: EntityType = EntityType.UNKNOWN
    aliases: List[str] = field(default_factory=list)
    attributes: Dict[str, Any] = field(default_factory=dict)
    first_seen: datetime = field(default_factory=datetime.now)
    last_seen: datetime = field(default_factory=datetime.now)
    mention_count: int = 1
    importance: float = 0.5
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __hash__(self):
        return hash(self.id)

    def __eq__(self, other):
        if isinstance(other, EntityNode):
            return self.id == other.id
        return False

    def matches_name(self, query: str) -> bool:
        """Check if query matches this entity's name or aliases."""
        query_lower = query.lower()
        if query_lower in self.name.lower():
            return True
        for alias in self.aliases:
            if query_lower in alias.lower():
                return True
        return False


@dataclass
class EntityEdge:
    """An edge representing a relationship between entities."""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    source_id: str = ""
    target_id: str = ""
    relation: str = ""  # e.g., "knows", "works_at", "lives_in", "owns"
    negated: bool = False  # Is this a negative relation? (e.g., "does NOT know")
    confidence: float = 1.0
    temporal_scope: Optional[TimeRange] = None  # When was this relation valid?
    evidence: List[str] = field(default_factory=list)  # Memory IDs supporting this
    first_seen: datetime = field(default_factory=datetime.now)
    last_confirmed: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)


class EntityGraph:
    """
    Graph structure for entity and relationship tracking.

    Key capabilities:
    - Store entities with attributes
    - Track relationships (including negative relations)
    - Entity resolution and linking
    - Relationship queries with temporal scoping
    """

    # Common relationship types
    RELATION_TYPES = {
        "person": ["knows", "friend_of", "family_of", "works_with", "married_to", "lives_with"],
        "organization": ["works_at", "member_of", "founded", "owns", "affiliated_with"],
        "location": ["lives_in", "located_at", "visited", "born_in", "from"],
        "object": ["owns", "uses", "created", "purchased"],
        "concept": ["interested_in", "believes", "likes", "dislikes", "prefers"],
    }

    def __init__(self):
        self.graph = nx.MultiDiGraph()  # Multi-graph to allow multiple relations
        self.nodes: Dict[str, EntityNode] = {}
        self.edges: Dict[str, EntityEdge] = {}
        # Indexes for efficient lookup
        self._name_index: Dict[str, Set[str]] = {}  # name -> node_ids
        self._type_index: Dict[EntityType, Set[str]] = {}  # type -> node_ids

    def add_node(self, node: EntityNode) -> str:
        """Add an entity node to the graph."""
        self.nodes[node.id] = node
        self.graph.add_node(node.id, data=node)

        # Update indexes
        self._index_node(node)

        return node.id

    def _index_node(self, node: EntityNode) -> None:
        """Index node for efficient lookup."""
        # Name index (lowercase for case-insensitive matching)
        name_key = node.name.lower()
        if name_key not in self._name_index:
            self._name_index[name_key] = set()
        self._name_index[name_key].add(node.id)

        # Also index aliases
        for alias in node.aliases:
            alias_key = alias.lower()
            if alias_key not in self._name_index:
                self._name_index[alias_key] = set()
            self._name_index[alias_key].add(node.id)

        # Type index
        if node.entity_type not in self._type_index:
            self._type_index[node.entity_type] = set()
        self._type_index[node.entity_type].add(node.id)

    def add_edge(self, edge: EntityEdge) -> str:
        """Add a relationship edge to the graph."""
        self.edges[edge.id] = edge
        self.graph.add_edge(
            edge.source_id,
            edge.target_id,
            key=edge.id,
            relation=edge.relation,
            negated=edge.negated,
            confidence=edge.confidence,
            data=edge
        )
        return edge.id

    def add_negative_relation(
        self,
        source_id: str,
        target_id: str,
        relation: str,
        evidence: List[str] = None
    ) -> str:
        """
        Add an explicit negative relation.
        Critical for adversarial robustness.
        """
        edge = EntityEdge(
            source_id=source_id,
            target_id=target_id,
            relation=relation,
            negated=True,
            evidence=evidence or []
        )
        return self.add_edge(edge)

    def find_by_name(self, name: str, fuzzy: bool = True) -> List[EntityNode]:
        """Find entities by name."""
        name_lower = name.lower()

        # Exact match first
        if name_lower in self._name_index:
            return [self.nodes[nid] for nid in self._name_index[name_lower]]

        # Fuzzy match if enabled
        if fuzzy:
            results = []
            for node in self.nodes.values():
                if node.matches_name(name):
                    results.append(node)
            return results

        return []

    def find_by_type(self, entity_type: EntityType) -> List[EntityNode]:
        """Find all entities of a specific type."""
        if entity_type not in self._type_index:
            return []
        return [self.nodes[nid] for nid in self._type_index[entity_type]]

    def get_relations(
        self,
        entity_id: str,
        relation_filter: Optional[List[str]] = None,
        include_negated: bool = True,
        at_time: Optional[datetime] = None
    ) -> List[Tuple[EntityNode, EntityEdge]]:
        """
        Get all relations for an entity.

        Args:
            entity_id: The entity to query
            relation_filter: Only return specific relation types
            include_negated: Include negative relations
            at_time: Filter by temporal validity
        """
        if entity_id not in self.graph:
            return []

        results = []

        # Outgoing edges
        for _, target_id, key in self.graph.out_edges(entity_id, keys=True):
            edge = self.edges.get(key)
            if not edge:
                continue

            # Apply filters
            if relation_filter and edge.relation not in relation_filter:
                continue
            if not include_negated and edge.negated:
                continue
            if at_time and edge.temporal_scope and not edge.temporal_scope.is_valid_at(at_time):
                continue

            target_node = self.nodes.get(target_id)
            if target_node:
                results.append((target_node, edge))

        # Incoming edges
        for source_id, _, key in self.graph.in_edges(entity_id, keys=True):
            edge = self.edges.get(key)
            if not edge:
                continue

            # Apply filters
            if relation_filter and edge.relation not in relation_filter:
                continue
            if not include_negated and edge.negated:
                continue
            if at_time and edge.temporal_scope and not edge.temporal_scope.is_valid_at(at_time):
                continue

            source_node = self.nodes.get(source_id)
            if source_node:
                results.append((source_node, edge))

        return results

    def has_relation(
        self,
        source_id: str,
        target_id: str,
        relation: str,
        check_negation: bool = True
    ) -> Tuple[bool, Optional[bool]]:
        """
        Check if a relation exists between two entities.

        Returns:
            (exists, is_negated) - exists is True if relation found,
            is_negated indicates if it's a negative relation
        """
        if not self.graph.has_edge(source_id, target_id):
            # Check reverse direction
            if not self.graph.has_edge(target_id, source_id):
                return (False, None)
            source_id, target_id = target_id, source_id

        for key in self.graph[source_id][target_id]:
            edge = self.edges.get(key)
            if edge and edge.relation == relation:
                return (True, edge.negated)

        return (False, None)

    def check_contradiction(self, entity_id: str, relation: str, target_id: str) -> Optional[EntityEdge]:
        """
        Check if asserting this relation would contradict existing knowledge.
        Returns the contradicting edge if found.
        """
        exists, is_negated = self.has_relation(entity_id, target_id, relation)

        if exists and is_negated is not None:
            # Found existing relation - check if it contradicts
            for key in self.graph[entity_id].get(target_id, {}):
                edge = self.edges.get(key)
                if edge and edge.relation == relation:
                    return edge

        return None

    def get_entity_profile(self, entity_id: str) -> Dict[str, Any]:
        """Get a comprehensive profile of an entity."""
        node = self.nodes.get(entity_id)
        if not node:
            return {}

        relations = self.get_relations(entity_id)

        profile = {
            "id": node.id,
            "name": node.name,
            "type": node.entity_type.value,
            "aliases": node.aliases,
            "attributes": node.attributes,
            "first_seen": node.first_seen.isoformat(),
            "last_seen": node.last_seen.isoformat(),
            "mention_count": node.mention_count,
            "relations": [],
            "negative_relations": [],
        }

        for related_node, edge in relations:
            rel_info = {
                "entity": related_node.name,
                "entity_id": related_node.id,
                "relation": edge.relation,
                "confidence": edge.confidence,
            }
            if edge.negated:
                profile["negative_relations"].append(rel_info)
            else:
                profile["relations"].append(rel_info)

        return profile

    def find_path(
        self,
        source_id: str,
        target_id: str,
        max_hops: int = 3
    ) -> List[List[Tuple[str, EntityEdge]]]:
        """
        Find paths between two entities.
        Used for multi-hop reasoning.
        """
        if source_id not in self.graph or target_id not in self.graph:
            return []

        try:
            # Find all simple paths up to max_hops
            paths = list(nx.all_simple_paths(
                self.graph.to_undirected(),
                source_id,
                target_id,
                cutoff=max_hops
            ))

            result_paths = []
            for path in paths:
                path_with_edges = []
                for i in range(len(path) - 1):
                    # Get edge between consecutive nodes
                    edge_keys = list(self.graph.get_edge_data(path[i], path[i+1], default={}).keys())
                    if not edge_keys:
                        edge_keys = list(self.graph.get_edge_data(path[i+1], path[i], default={}).keys())

                    if edge_keys:
                        edge = self.edges.get(edge_keys[0])
                        if edge:
                            path_with_edges.append((path[i+1], edge))

                if path_with_edges:
                    result_paths.append(path_with_edges)

            return result_paths

        except nx.NetworkXNoPath:
            return []

    def merge_entities(self, entity_ids: List[str], primary_id: Optional[str] = None) -> str:
        """
        Merge multiple entity nodes into one (entity resolution).
        Returns the ID of the merged entity.
        """
        if not entity_ids:
            return ""

        # Use first as primary if not specified
        primary_id = primary_id or entity_ids[0]
        primary = self.nodes.get(primary_id)
        if not primary:
            return ""

        # Merge attributes and relations from other entities
        for eid in entity_ids:
            if eid == primary_id:
                continue

            other = self.nodes.get(eid)
            if not other:
                continue

            # Merge aliases
            if other.name not in primary.aliases and other.name != primary.name:
                primary.aliases.append(other.name)
            primary.aliases.extend([a for a in other.aliases if a not in primary.aliases])

            # Merge attributes (prefer primary's values on conflict)
            for key, value in other.attributes.items():
                if key not in primary.attributes:
                    primary.attributes[key] = value

            # Update mention count
            primary.mention_count += other.mention_count

            # Update time bounds
            if other.first_seen < primary.first_seen:
                primary.first_seen = other.first_seen
            if other.last_seen > primary.last_seen:
                primary.last_seen = other.last_seen

            # Redirect edges
            for source_id, _, key in list(self.graph.in_edges(eid, keys=True)):
                edge = self.edges.get(key)
                if edge:
                    edge.target_id = primary_id
                    self.graph.add_edge(source_id, primary_id, key=key, data=edge)

            for _, target_id, key in list(self.graph.out_edges(eid, keys=True)):
                edge = self.edges.get(key)
                if edge:
                    edge.source_id = primary_id
                    self.graph.add_edge(primary_id, target_id, key=key, data=edge)

            # Remove old node
            self.graph.remove_node(eid)
            del self.nodes[eid]

        # Re-index primary
        self._index_node(primary)

        return primary_id

    def get_node(self, node_id: str) -> Optional[EntityNode]:
        """Get a node by ID."""
        return self.nodes.get(node_id)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize graph to dictionary."""
        return {
            "nodes": [
                {
                    "id": n.id,
                    "name": n.name,
                    "type": n.entity_type.value,
                    "aliases": n.aliases,
                    "attributes": n.attributes,
                    "first_seen": n.first_seen.isoformat(),
                    "last_seen": n.last_seen.isoformat(),
                    "mention_count": n.mention_count,
                    "importance": n.importance,
                }
                for n in self.nodes.values()
            ],
            "edges": [
                {
                    "id": e.id,
                    "source_id": e.source_id,
                    "target_id": e.target_id,
                    "relation": e.relation,
                    "negated": e.negated,
                    "confidence": e.confidence,
                }
                for e in self.edges.values()
            ]
        }
