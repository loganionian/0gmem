"""
Unified Memory Graph (UMG): Combines all four graph views.

Provides a unified interface for memory operations across
temporal, semantic, causal, and entity graphs.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional, List, Dict, Any, Tuple, Set
import numpy as np

from zerogmem.graph.temporal import TemporalGraph, TemporalNode, TimeInterval, TemporalRelation
from zerogmem.graph.semantic import SemanticGraph, SemanticNode
from zerogmem.graph.causal import CausalGraph, CausalNode, CausalEdge
from zerogmem.graph.entity import EntityGraph, EntityNode, EntityEdge, EntityType


@dataclass
class UnifiedMemoryItem:
    """A unified memory item that can be represented in all four graphs."""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    content: str = ""
    summary: str = ""  # Compressed version
    embedding: Optional[np.ndarray] = None

    # Temporal info
    event_time: Optional[TimeInterval] = None
    ingestion_time: datetime = field(default_factory=datetime.now)

    # Entity info
    entities: List[str] = field(default_factory=list)  # Entity IDs
    entity_names: List[str] = field(default_factory=list)  # For display

    # Causal info
    causes: List[str] = field(default_factory=list)  # Memory IDs that caused this
    effects: List[str] = field(default_factory=list)  # Memory IDs caused by this

    # Semantic info
    concepts: List[str] = field(default_factory=list)
    importance: float = 0.5

    # Metadata
    source: str = ""  # conversation, document, etc.
    session_id: Optional[str] = None
    turn_number: Optional[int] = None
    speaker: Optional[str] = None
    negated_facts: List[str] = field(default_factory=list)  # Explicit negations

    # Graph node references
    temporal_node_id: Optional[str] = None
    semantic_node_id: Optional[str] = None
    causal_node_id: Optional[str] = None

    metadata: Dict[str, Any] = field(default_factory=dict)


class UnifiedMemoryGraph:
    """
    The central memory structure combining four orthogonal graph views.

    Provides:
    - Unified CRUD operations
    - Cross-graph queries
    - Multi-hop reasoning across graph types
    - Automatic graph synchronization
    """

    def __init__(self, embedding_dim: int = 1536):
        self.temporal_graph = TemporalGraph()
        self.semantic_graph = SemanticGraph(embedding_dim=embedding_dim)
        self.causal_graph = CausalGraph()
        self.entity_graph = EntityGraph()

        # Unified memory store
        self.memories: Dict[str, UnifiedMemoryItem] = {}

        # Cross-references
        self._entity_to_memories: Dict[str, Set[str]] = {}  # entity_id -> memory_ids
        self._concept_to_memories: Dict[str, Set[str]] = {}  # concept -> memory_ids

    def add_memory(self, memory: UnifiedMemoryItem) -> str:
        """
        Add a unified memory item, creating nodes in all relevant graphs.
        """
        self.memories[memory.id] = memory

        # Add to temporal graph
        if memory.event_time:
            temporal_node = TemporalNode(
                id=f"temporal_{memory.id}",
                content=memory.summary or memory.content[:200],
                event_time=memory.event_time,
                ingestion_time=memory.ingestion_time,
                memory_id=memory.id,
                entities=memory.entities,
                importance=memory.importance,
            )
            self.temporal_graph.add_node(temporal_node)
            memory.temporal_node_id = temporal_node.id

        # Add to semantic graph
        if memory.embedding is not None:
            semantic_node = SemanticNode(
                id=f"semantic_{memory.id}",
                content=memory.content,
                embedding=memory.embedding,
                concepts=memory.concepts,
                importance=memory.importance,
                memory_id=memory.id,
            )
            self.semantic_graph.add_node(semantic_node)
            memory.semantic_node_id = semantic_node.id

        # Add to causal graph if has causal info
        if memory.causes or memory.effects:
            causal_node = CausalNode(
                id=f"causal_{memory.id}",
                content=memory.content,
                memory_id=memory.id,
                timestamp=memory.event_time.start if memory.event_time else None,
            )
            self.causal_graph.add_node(causal_node)
            memory.causal_node_id = causal_node.id

            # Add causal edges
            for cause_id in memory.causes:
                cause_memory = self.memories.get(cause_id)
                if cause_memory and cause_memory.causal_node_id:
                    edge = CausalEdge(
                        cause_id=cause_memory.causal_node_id,
                        effect_id=causal_node.id,
                        evidence=[memory.id],
                    )
                    self.causal_graph.add_edge(edge)

        # Update cross-references
        for entity_id in memory.entities:
            if entity_id not in self._entity_to_memories:
                self._entity_to_memories[entity_id] = set()
            self._entity_to_memories[entity_id].add(memory.id)

        for concept in memory.concepts:
            if concept not in self._concept_to_memories:
                self._concept_to_memories[concept] = set()
            self._concept_to_memories[concept].add(memory.id)

        return memory.id

    def add_entity(self, entity: EntityNode) -> str:
        """Add an entity to the entity graph."""
        return self.entity_graph.add_node(entity)

    def add_entity_relation(
        self,
        source_id: str,
        target_id: str,
        relation: str,
        negated: bool = False,
        evidence: List[str] = None
    ) -> str:
        """Add a relation between entities."""
        edge = EntityEdge(
            source_id=source_id,
            target_id=target_id,
            relation=relation,
            negated=negated,
            evidence=evidence or [],
        )
        return self.entity_graph.add_edge(edge)

    def get_memory(self, memory_id: str) -> Optional[UnifiedMemoryItem]:
        """Get a memory by ID."""
        return self.memories.get(memory_id)

    def get_entity(self, entity_id: str) -> Optional[EntityNode]:
        """Get an entity by ID."""
        return self.entity_graph.get_node(entity_id)

    # ==================== Query Methods ====================

    def query_by_time(
        self,
        start: datetime,
        end: Optional[datetime] = None,
        entities: Optional[List[str]] = None
    ) -> List[UnifiedMemoryItem]:
        """Query memories by time range."""
        if end is None:
            # Point query
            temporal_nodes = self.temporal_graph.events_at(start)
        else:
            temporal_nodes = self.temporal_graph.events_in_range(start, end)

        # Filter by entities if specified
        if entities:
            entity_set = set(entities)
            temporal_nodes = [n for n in temporal_nodes if entity_set.intersection(n.entities)]

        # Map back to memories
        memories = []
        for node in temporal_nodes:
            if node.memory_id and node.memory_id in self.memories:
                memories.append(self.memories[node.memory_id])

        return memories

    def query_by_similarity(
        self,
        query_embedding: np.ndarray,
        top_k: int = 10,
        threshold: float = 0.0
    ) -> List[Tuple[UnifiedMemoryItem, float]]:
        """Query memories by semantic similarity."""
        similar_nodes = self.semantic_graph.find_similar(
            query_embedding, top_k=top_k, threshold=threshold
        )

        results = []
        for node, score in similar_nodes:
            if node.memory_id and node.memory_id in self.memories:
                results.append((self.memories[node.memory_id], score))

        return results

    def query_by_entity(
        self,
        entity_id: str,
        relation_filter: Optional[List[str]] = None
    ) -> List[UnifiedMemoryItem]:
        """Query memories involving an entity."""
        memory_ids = self._entity_to_memories.get(entity_id, set())
        return [self.memories[mid] for mid in memory_ids if mid in self.memories]

    def query_by_concept(self, concept: str) -> List[UnifiedMemoryItem]:
        """Query memories related to a concept."""
        memory_ids = self._concept_to_memories.get(concept, set())
        return [self.memories[mid] for mid in memory_ids if mid in self.memories]

    def query_temporal_relative(
        self,
        reference_memory_id: str,
        relation: TemporalRelation,
        limit: int = 10
    ) -> List[UnifiedMemoryItem]:
        """Query memories with temporal relation to reference."""
        ref_memory = self.memories.get(reference_memory_id)
        if not ref_memory or not ref_memory.temporal_node_id:
            return []

        ref_node = self.temporal_graph.get_node(ref_memory.temporal_node_id)
        if not ref_node:
            return []

        if relation == TemporalRelation.BEFORE:
            nodes = self.temporal_graph.events_before(ref_node, limit)
        elif relation == TemporalRelation.AFTER:
            nodes = self.temporal_graph.events_after(ref_node, limit)
        elif relation == TemporalRelation.DURING:
            nodes = self.temporal_graph.events_during(ref_node)
        else:
            return []

        return [
            self.memories[n.memory_id]
            for n in nodes
            if n.memory_id and n.memory_id in self.memories
        ]

    def query_causal_chain(
        self,
        memory_id: str,
        direction: str = "causes",  # "causes" or "effects"
        max_depth: int = 3
    ) -> List[List[UnifiedMemoryItem]]:
        """Query causal chains related to a memory."""
        memory = self.memories.get(memory_id)
        if not memory or not memory.causal_node_id:
            return []

        if direction == "causes":
            paths = self.causal_graph.get_causes(memory.causal_node_id, max_depth)
        else:
            paths = self.causal_graph.get_effects(memory.causal_node_id, max_depth)

        # Convert to memory paths
        memory_paths = []
        for path in paths:
            memory_path = []
            for node, edge in path:
                if node.memory_id and node.memory_id in self.memories:
                    memory_path.append(self.memories[node.memory_id])
            if memory_path:
                memory_paths.append(memory_path)

        return memory_paths

    # ==================== Multi-hop Reasoning ====================

    def multi_hop_query(
        self,
        start_entities: List[str],
        query_embedding: Optional[np.ndarray] = None,
        max_hops: int = 3,
        top_k: int = 10
    ) -> List[Tuple[UnifiedMemoryItem, float, List[str]]]:
        """
        Perform multi-hop reasoning across graphs.

        Returns: List of (memory, score, reasoning_path)
        """
        results = []
        visited_memories = set()

        # Start from entity-related memories
        frontier = []
        for entity_id in start_entities:
            memory_ids = self._entity_to_memories.get(entity_id, set())
            for mid in memory_ids:
                if mid not in visited_memories:
                    frontier.append((mid, 1.0, [f"entity:{entity_id}"]))
                    visited_memories.add(mid)

        # BFS with scoring
        for hop in range(max_hops):
            next_frontier = []

            for memory_id, score, path in frontier:
                memory = self.memories.get(memory_id)
                if not memory:
                    continue

                # Score based on similarity if embedding provided
                if query_embedding is not None and memory.embedding is not None:
                    sim = self.semantic_graph.compute_similarity(query_embedding, memory.embedding)
                    score *= (0.5 + 0.5 * sim)  # Blend path score with similarity

                results.append((memory, score, path))

                # Expand through different graph views

                # 1. Entity graph expansion
                for entity_id in memory.entities:
                    relations = self.entity_graph.get_relations(entity_id)
                    for related_entity, edge in relations[:5]:  # Limit expansion
                        rel_memory_ids = self._entity_to_memories.get(related_entity.id, set())
                        for mid in rel_memory_ids:
                            if mid not in visited_memories:
                                new_path = path + [f"{edge.relation}:{related_entity.name}"]
                                next_frontier.append((mid, score * 0.8, new_path))
                                visited_memories.add(mid)

                # 2. Temporal graph expansion
                if memory.temporal_node_id:
                    temp_node = self.temporal_graph.get_node(memory.temporal_node_id)
                    if temp_node:
                        neighbors = self.temporal_graph.get_neighbors(
                            temp_node.id,
                            relation_filter=[TemporalRelation.BEFORE, TemporalRelation.AFTER]
                        )
                        for neighbor, relation in neighbors[:3]:
                            if neighbor.memory_id and neighbor.memory_id not in visited_memories:
                                new_path = path + [f"temporal:{relation.value}"]
                                next_frontier.append((neighbor.memory_id, score * 0.7, new_path))
                                visited_memories.add(neighbor.memory_id)

                # 3. Semantic graph expansion
                if memory.semantic_node_id:
                    sem_node = self.semantic_graph.get_node(memory.semantic_node_id)
                    if sem_node:
                        related = self.semantic_graph.find_related(sem_node.id, max_depth=1)
                        for rel_node, relation, _ in related[:3]:
                            if rel_node.memory_id and rel_node.memory_id not in visited_memories:
                                new_path = path + [f"semantic:{relation}"]
                                next_frontier.append((rel_node.memory_id, score * 0.7, new_path))
                                visited_memories.add(rel_node.memory_id)

            frontier = next_frontier

        # Sort by score and return top_k
        results.sort(key=lambda x: x[1], reverse=True)
        return results[:top_k]

    def find_temporal_chain(
        self,
        entity_id: str,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None
    ) -> List[UnifiedMemoryItem]:
        """
        Get chronological chain of events for an entity.
        Critical for temporal reasoning in LoCoMo.
        """
        memory_ids = self._entity_to_memories.get(entity_id, set())
        memories = [self.memories[mid] for mid in memory_ids if mid in self.memories]

        # Filter by time if specified
        if start_time:
            memories = [m for m in memories if m.event_time and m.event_time.start >= start_time]
        if end_time:
            memories = [m for m in memories if m.event_time and m.event_time.start <= end_time]

        # Sort chronologically
        memories.sort(key=lambda m: m.event_time.start if m.event_time else datetime.min)

        return memories

    def find_events_between(
        self,
        event_a_id: str,
        event_b_id: str
    ) -> List[UnifiedMemoryItem]:
        """
        Find events that occurred between two events.
        Critical for temporal chain reasoning.
        """
        mem_a = self.memories.get(event_a_id)
        mem_b = self.memories.get(event_b_id)

        if not mem_a or not mem_b:
            return []
        if not mem_a.temporal_node_id or not mem_b.temporal_node_id:
            return []

        node_a = self.temporal_graph.get_node(mem_a.temporal_node_id)
        node_b = self.temporal_graph.get_node(mem_b.temporal_node_id)

        if not node_a or not node_b:
            return []

        between_nodes = self.temporal_graph.events_between(node_a, node_b)

        return [
            self.memories[n.memory_id]
            for n in between_nodes
            if n.memory_id and n.memory_id in self.memories
        ]

    # ==================== Negative Fact Handling ====================

    def add_negative_fact(
        self,
        subject_entity_id: str,
        relation: str,
        object_entity_id: str,
        evidence_memory_id: str
    ) -> None:
        """
        Store an explicit negative fact.
        Critical for adversarial robustness.
        """
        self.entity_graph.add_negative_relation(
            subject_entity_id,
            object_entity_id,
            relation,
            evidence=[evidence_memory_id]
        )

    def check_negation(
        self,
        subject_entity_id: str,
        relation: str,
        object_entity_id: str
    ) -> Tuple[bool, Optional[str]]:
        """
        Check if a fact is explicitly negated.

        Returns: (is_negated, evidence_memory_id)
        """
        exists, is_negated = self.entity_graph.has_relation(
            subject_entity_id, object_entity_id, relation
        )

        if exists and is_negated:
            # Find evidence
            for key in self.entity_graph.graph[subject_entity_id].get(object_entity_id, {}):
                edge = self.entity_graph.edges.get(key)
                if edge and edge.relation == relation and edge.negated:
                    evidence = edge.evidence[0] if edge.evidence else None
                    return (True, evidence)

        return (False, None)

    # ==================== Statistics ====================

    def get_stats(self) -> Dict[str, Any]:
        """Get statistics about the unified graph."""
        return {
            "total_memories": len(self.memories),
            "temporal_nodes": len(self.temporal_graph.nodes),
            "semantic_nodes": len(self.semantic_graph.nodes),
            "causal_nodes": len(self.causal_graph.nodes),
            "entity_nodes": len(self.entity_graph.nodes),
            "entity_edges": len(self.entity_graph.edges),
            "unique_concepts": len(self._concept_to_memories),
        }

    def to_dict(self) -> Dict[str, Any]:
        """Serialize the unified graph.

        Note: Embeddings are NOT included. They must be saved separately
        and passed to from_dict().
        """
        return {
            "memories": [
                {
                    "id": m.id,
                    "content": m.content,
                    "summary": m.summary,
                    "event_time_start": m.event_time.start.isoformat() if m.event_time else None,
                    "event_time_end": (
                        m.event_time.end.isoformat()
                        if m.event_time and m.event_time.end
                        else None
                    ),
                    "ingestion_time": m.ingestion_time.isoformat(),
                    "entities": m.entities,
                    "entity_names": m.entity_names,
                    "causes": m.causes,
                    "effects": m.effects,
                    "concepts": m.concepts,
                    "importance": m.importance,
                    "source": m.source,
                    "session_id": m.session_id,
                    "turn_number": m.turn_number,
                    "speaker": m.speaker,
                    "negated_facts": m.negated_facts,
                    "temporal_node_id": m.temporal_node_id,
                    "semantic_node_id": m.semantic_node_id,
                    "causal_node_id": m.causal_node_id,
                    "metadata": m.metadata,
                }
                for m in self.memories.values()
            ],
            "temporal_graph": self.temporal_graph.to_dict(),
            "entity_graph": self.entity_graph.to_dict(),
            "causal_graph": self.causal_graph.to_dict(),
            "semantic_graph": self.semantic_graph.to_dict(),
        }

    @classmethod
    def from_dict(
        cls,
        data: Dict[str, Any],
        embeddings_map: Optional[Dict[str, np.ndarray]] = None,
    ) -> "UnifiedMemoryGraph":
        """Deserialize the unified graph from dictionary.

        Args:
            data: Output of to_dict().
            embeddings_map: Map of memory_id -> embedding (and also
                semantic_node_id -> embedding for the semantic sub-graph).
        """
        embeddings_map = embeddings_map or {}
        embedding_dim = data.get("semantic_graph", {}).get("embedding_dim", 1536)
        graph = cls(embedding_dim=embedding_dim)

        # Restore sub-graphs via their own from_dict
        if "temporal_graph" in data:
            graph.temporal_graph = TemporalGraph.from_dict(data["temporal_graph"])
        if "entity_graph" in data:
            graph.entity_graph = EntityGraph.from_dict(data["entity_graph"])
        if "causal_graph" in data:
            graph.causal_graph = CausalGraph.from_dict(data["causal_graph"])
        if "semantic_graph" in data:
            # Build semantic embeddings map from "semantic_<memory_id>" keys
            sem_embeddings = {}
            for key, emb in embeddings_map.items():
                if key.startswith("semantic_"):
                    sem_embeddings[key] = emb
                else:
                    sem_embeddings[f"semantic_{key}"] = emb
            graph.semantic_graph = SemanticGraph.from_dict(
                data["semantic_graph"], sem_embeddings
            )

        # Restore memories directly (NOT via add_memory which creates sub-nodes)
        for md in data.get("memories", []):
            event_time = None
            if md.get("event_time_start"):
                event_time = TimeInterval(
                    start=datetime.fromisoformat(md["event_time_start"]),
                    end=(
                        datetime.fromisoformat(md["event_time_end"])
                        if md.get("event_time_end")
                        else None
                    ),
                )
            memory = UnifiedMemoryItem(
                id=md["id"],
                content=md.get("content", ""),
                summary=md.get("summary", ""),
                embedding=embeddings_map.get(md["id"]),
                event_time=event_time,
                ingestion_time=(
                    datetime.fromisoformat(md["ingestion_time"])
                    if md.get("ingestion_time")
                    else datetime.now()
                ),
                entities=md.get("entities", []),
                entity_names=md.get("entity_names", []),
                causes=md.get("causes", []),
                effects=md.get("effects", []),
                concepts=md.get("concepts", []),
                importance=md.get("importance", 0.5),
                source=md.get("source", ""),
                session_id=md.get("session_id"),
                turn_number=md.get("turn_number"),
                speaker=md.get("speaker"),
                negated_facts=md.get("negated_facts", []),
                temporal_node_id=md.get("temporal_node_id"),
                semantic_node_id=md.get("semantic_node_id"),
                causal_node_id=md.get("causal_node_id"),
                metadata=md.get("metadata", {}),
            )
            graph.memories[memory.id] = memory

            # Rebuild cross-reference indexes
            for entity_id in memory.entities:
                if entity_id not in graph._entity_to_memories:
                    graph._entity_to_memories[entity_id] = set()
                graph._entity_to_memories[entity_id].add(memory.id)

            for concept in memory.concepts:
                if concept not in graph._concept_to_memories:
                    graph._concept_to_memories[concept] = set()
                graph._concept_to_memories[concept].add(memory.id)

        return graph
