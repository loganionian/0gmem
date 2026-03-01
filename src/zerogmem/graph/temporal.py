"""
Temporal Graph: Core component for temporal reasoning.

Implements Allen's Interval Algebra for temporal relations and
bitemporal modeling (event time + system time).
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Optional, List, Dict, Set, Tuple, Any
import networkx as nx


class TemporalRelation(Enum):
    """Allen's Interval Algebra relations."""
    BEFORE = "before"           # A ends before B starts
    AFTER = "after"             # A starts after B ends
    MEETS = "meets"             # A ends exactly when B starts
    MET_BY = "met_by"           # A starts exactly when B ends
    OVERLAPS = "overlaps"       # A starts before B, ends during B
    OVERLAPPED_BY = "overlapped_by"
    DURING = "during"           # A occurs within B's timespan
    CONTAINS = "contains"       # B occurs within A's timespan
    STARTS = "starts"           # A starts at same time as B, ends earlier
    STARTED_BY = "started_by"
    FINISHES = "finishes"       # A ends at same time as B, starts later
    FINISHED_BY = "finished_by"
    EQUALS = "equals"           # A and B have same start and end
    CONCURRENT = "concurrent"   # Simplified: A and B overlap in any way


@dataclass
class TimeInterval:
    """Represents a time interval with start and end."""
    start: datetime
    end: Optional[datetime] = None  # None means point event or ongoing

    @property
    def is_point(self) -> bool:
        return self.end is None or self.start == self.end

    @property
    def duration(self) -> Optional[timedelta]:
        if self.end is None:
            return None
        return self.end - self.start

    def contains_time(self, t: datetime) -> bool:
        """Check if time t is within this interval."""
        if self.end is None:
            return t >= self.start
        return self.start <= t <= self.end

    def overlaps_with(self, other: TimeInterval) -> bool:
        """Check if this interval overlaps with another."""
        if self.end is None and other.end is None:
            return True
        if self.end is None:
            return self.start <= other.end if other.end else True
        if other.end is None:
            return other.start <= self.end
        return self.start <= other.end and other.start <= self.end


@dataclass
class TemporalNode:
    """A node in the temporal graph representing an event or state."""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    content: str = ""
    event_time: TimeInterval = field(default_factory=lambda: TimeInterval(datetime.now()))
    ingestion_time: datetime = field(default_factory=datetime.now)  # Bitemporal: when we learned this
    memory_id: Optional[str] = None  # Reference to associated memory
    entities: List[str] = field(default_factory=list)  # Entity IDs involved
    importance: float = 0.5
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __hash__(self):
        return hash(self.id)

    def __eq__(self, other):
        if isinstance(other, TemporalNode):
            return self.id == other.id
        return False


@dataclass
class TemporalEdge:
    """An edge representing temporal relation between two nodes."""
    source_id: str
    target_id: str
    relation: TemporalRelation
    confidence: float = 1.0
    inferred: bool = False  # Was this inferred or explicitly stated?
    evidence: List[str] = field(default_factory=list)  # Memory IDs supporting this


class TemporalGraph:
    """
    Graph structure for temporal reasoning using Allen's Interval Algebra.

    Key capabilities:
    - Store events with temporal information
    - Compute temporal relations between events
    - Query events by time (point, range, relative)
    - Perform temporal chain reasoning
    """

    def __init__(self):
        self.graph = nx.DiGraph()
        self.nodes: Dict[str, TemporalNode] = {}
        self.edges: List[TemporalEdge] = []
        # Index for efficient temporal queries
        self._time_index: Dict[datetime, Set[str]] = {}

    def add_node(self, node: TemporalNode) -> str:
        """Add a temporal node to the graph."""
        self.nodes[node.id] = node
        self.graph.add_node(node.id, data=node)

        # Update time index
        self._index_node(node)

        # Compute relations with existing nodes
        self._compute_relations_for_node(node)

        return node.id

    def _index_node(self, node: TemporalNode) -> None:
        """Index node by time for efficient queries."""
        # Index by start date (day granularity for efficiency)
        start_date = node.event_time.start.replace(hour=0, minute=0, second=0, microsecond=0)
        if start_date not in self._time_index:
            self._time_index[start_date] = set()
        self._time_index[start_date].add(node.id)

        # Also index end date if different
        if node.event_time.end:
            end_date = node.event_time.end.replace(hour=0, minute=0, second=0, microsecond=0)
            if end_date != start_date:
                if end_date not in self._time_index:
                    self._time_index[end_date] = set()
                self._time_index[end_date].add(node.id)

    def _compute_relations_for_node(self, node: TemporalNode) -> None:
        """Compute temporal relations between new node and existing nodes."""
        for existing_id, existing_node in self.nodes.items():
            if existing_id == node.id:
                continue

            relation = self.compute_relation(node, existing_node)
            if relation:
                edge = TemporalEdge(
                    source_id=node.id,
                    target_id=existing_id,
                    relation=relation,
                    inferred=True
                )
                self.add_edge(edge)

    def add_edge(self, edge: TemporalEdge) -> None:
        """Add a temporal edge to the graph."""
        self.edges.append(edge)
        self.graph.add_edge(
            edge.source_id,
            edge.target_id,
            relation=edge.relation,
            confidence=edge.confidence,
            data=edge
        )

    def compute_relation(self, a: TemporalNode, b: TemporalNode) -> Optional[TemporalRelation]:
        """
        Compute the temporal relation between two events using Allen's Interval Algebra.
        """
        a_start = a.event_time.start
        a_end = a.event_time.end or a.event_time.start
        b_start = b.event_time.start
        b_end = b.event_time.end or b.event_time.start

        # Small tolerance for "meets" relation (1 second)
        tolerance = timedelta(seconds=1)

        if a_end < b_start - tolerance:
            return TemporalRelation.BEFORE
        elif a_start > b_end + tolerance:
            return TemporalRelation.AFTER
        elif abs((a_end - b_start).total_seconds()) <= tolerance.total_seconds():
            return TemporalRelation.MEETS
        elif abs((a_start - b_end).total_seconds()) <= tolerance.total_seconds():
            return TemporalRelation.MET_BY
        elif a_start < b_start and a_end > b_start and a_end < b_end:
            return TemporalRelation.OVERLAPS
        elif b_start < a_start and b_end > a_start and b_end < a_end:
            return TemporalRelation.OVERLAPPED_BY
        elif a_start > b_start and a_end < b_end:
            return TemporalRelation.DURING
        elif b_start > a_start and b_end < a_end:
            return TemporalRelation.CONTAINS
        elif a_start == b_start and a_end < b_end:
            return TemporalRelation.STARTS
        elif a_start == b_start and a_end > b_end:
            return TemporalRelation.STARTED_BY
        elif a_end == b_end and a_start > b_start:
            return TemporalRelation.FINISHES
        elif a_end == b_end and a_start < b_start:
            return TemporalRelation.FINISHED_BY
        elif a_start == b_start and a_end == b_end:
            return TemporalRelation.EQUALS
        else:
            return TemporalRelation.CONCURRENT

    def events_at(self, timestamp: datetime, tolerance: timedelta = timedelta(hours=1)) -> List[TemporalNode]:
        """Find events occurring at or near a specific timestamp."""
        results = []
        for node in self.nodes.values():
            if node.event_time.contains_time(timestamp):
                results.append(node)
            elif abs((node.event_time.start - timestamp).total_seconds()) <= tolerance.total_seconds():
                results.append(node)
        return sorted(results, key=lambda n: abs((n.event_time.start - timestamp).total_seconds()))

    def events_in_range(self, start: datetime, end: datetime) -> List[TemporalNode]:
        """Find events occurring within a time range."""
        range_interval = TimeInterval(start, end)
        results = []
        for node in self.nodes.values():
            if node.event_time.overlaps_with(range_interval):
                results.append(node)
        return sorted(results, key=lambda n: n.event_time.start)

    def events_before(self, reference: TemporalNode, limit: int = 10) -> List[TemporalNode]:
        """Find events that occurred before the reference event."""
        results = []
        for node_id, node in self.nodes.items():
            if node_id == reference.id:
                continue
            relation = self.compute_relation(node, reference)
            if relation in [TemporalRelation.BEFORE, TemporalRelation.MEETS]:
                results.append(node)
        # Sort by time, most recent first
        return sorted(results, key=lambda n: n.event_time.start, reverse=True)[:limit]

    def events_after(self, reference: TemporalNode, limit: int = 10) -> List[TemporalNode]:
        """Find events that occurred after the reference event."""
        results = []
        for node_id, node in self.nodes.items():
            if node_id == reference.id:
                continue
            relation = self.compute_relation(node, reference)
            if relation in [TemporalRelation.AFTER, TemporalRelation.MET_BY]:
                results.append(node)
        # Sort by time, earliest first
        return sorted(results, key=lambda n: n.event_time.start)[:limit]

    def events_during(self, reference: TemporalNode) -> List[TemporalNode]:
        """Find events that occurred during the reference event."""
        results = []
        for node_id, node in self.nodes.items():
            if node_id == reference.id:
                continue
            relation = self.compute_relation(node, reference)
            if relation == TemporalRelation.CONTAINS:
                results.append(node)
        return sorted(results, key=lambda n: n.event_time.start)

    def events_between(self, event_a: TemporalNode, event_b: TemporalNode) -> List[TemporalNode]:
        """
        Find events that occurred between two events.
        Critical for temporal chain reasoning.
        """
        # Ensure a is before b
        if event_a.event_time.start > event_b.event_time.start:
            event_a, event_b = event_b, event_a

        a_end = event_a.event_time.end or event_a.event_time.start
        b_start = event_b.event_time.start

        results = []
        for node_id, node in self.nodes.items():
            if node_id in [event_a.id, event_b.id]:
                continue
            node_start = node.event_time.start
            node_end = node.event_time.end or node.event_time.start

            # Check if node is between a and b
            if node_start >= a_end and node_end <= b_start:
                results.append(node)

        return sorted(results, key=lambda n: n.event_time.start)

    def find_by_entities(self, entity_ids: List[str]) -> List[TemporalNode]:
        """Find events involving specific entities."""
        results = []
        entity_set = set(entity_ids)
        for node in self.nodes.values():
            if entity_set.intersection(node.entities):
                results.append(node)
        return sorted(results, key=lambda n: n.event_time.start, reverse=True)

    def temporal_chain(self, entity_id: str, limit: int = 20) -> List[TemporalNode]:
        """
        Get the temporal chain of events for an entity.
        Returns events in chronological order.
        """
        entity_events = [n for n in self.nodes.values() if entity_id in n.entities]
        return sorted(entity_events, key=lambda n: n.event_time.start)[:limit]

    def get_node(self, node_id: str) -> Optional[TemporalNode]:
        """Get a node by ID."""
        return self.nodes.get(node_id)

    def get_neighbors(self, node_id: str, relation_filter: Optional[List[TemporalRelation]] = None) -> List[Tuple[TemporalNode, TemporalRelation]]:
        """Get neighboring nodes with their relations."""
        if node_id not in self.graph:
            return []

        neighbors = []
        for neighbor_id in self.graph.successors(node_id):
            edge_data = self.graph.get_edge_data(node_id, neighbor_id)
            if edge_data:
                relation = edge_data.get('relation')
                if relation_filter is None or relation in relation_filter:
                    neighbor_node = self.nodes.get(neighbor_id)
                    if neighbor_node:
                        neighbors.append((neighbor_node, relation))

        return neighbors

    def to_dict(self) -> Dict[str, Any]:
        """Serialize graph to dictionary."""
        return {
            "nodes": [
                {
                    "id": n.id,
                    "content": n.content,
                    "event_time_start": n.event_time.start.isoformat(),
                    "event_time_end": n.event_time.end.isoformat() if n.event_time.end else None,
                    "ingestion_time": n.ingestion_time.isoformat(),
                    "memory_id": n.memory_id,
                    "entities": n.entities,
                    "importance": n.importance,
                    "metadata": n.metadata,
                }
                for n in self.nodes.values()
            ],
            "edges": [
                {
                    "source_id": e.source_id,
                    "target_id": e.target_id,
                    "relation": e.relation.value,
                    "confidence": e.confidence,
                    "inferred": e.inferred,
                }
                for e in self.edges
            ]
        }
