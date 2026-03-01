"""
Memory Manager: Central orchestrator for all memory operations.

Coordinates the Unified Memory Graph, memory hierarchy,
and provides the main interface for the 0GMem system.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional, List, Dict, Any, Tuple
import numpy as np

from zerogmem.graph.unified import UnifiedMemoryGraph, UnifiedMemoryItem
from zerogmem.graph.entity import EntityNode, EntityType
from zerogmem.graph.temporal import TimeInterval
from zerogmem.memory.working import WorkingMemory, WorkingMemoryItem
from zerogmem.memory.episodic import EpisodicMemory, Episode, EpisodeMessage
from zerogmem.memory.semantic import SemanticMemoryStore, Fact


@dataclass
class Conversation:
    """Represents a conversation to be processed."""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    session_id: str = ""
    messages: List[Dict[str, Any]] = field(default_factory=list)
    participants: List[str] = field(default_factory=list)
    start_time: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class MemoryConfig:
    """Configuration for memory manager."""
    working_memory_capacity: int = 20
    working_memory_decay_rate: float = 0.05
    embedding_dim: int = 1536
    auto_consolidate: bool = True
    consolidation_threshold_days: int = 7
    consolidation_min_retrievals: int = 3
    max_episodes: int = 500
    max_facts: int = 5000
    eviction_batch_size: int = 10


class MemoryManager:
    """
    Central orchestrator for the 0GMem system.

    Manages:
    - Unified Memory Graph (temporal, semantic, causal, entity)
    - Memory hierarchy (working, episodic, semantic)
    - Memory encoding and consolidation
    - Query routing and retrieval
    """

    def __init__(self, config: Optional[MemoryConfig] = None):
        self.config = config or MemoryConfig()

        # Core components
        self.graph = UnifiedMemoryGraph(embedding_dim=self.config.embedding_dim)
        self.working_memory = WorkingMemory(
            capacity=self.config.working_memory_capacity,
            decay_rate=self.config.working_memory_decay_rate,
        )
        self.episodic_memory = EpisodicMemory()
        self.semantic_memory = SemanticMemoryStore()

        # Session tracking
        self.current_session_id: Optional[str] = None
        self.current_episode: Optional[Episode] = None

        # Embedding function (to be set by encoder)
        self._embed_fn: Optional[callable] = None

    def set_embedding_function(self, embed_fn: callable) -> None:
        """Set the embedding function for memory encoding."""
        self._embed_fn = embed_fn

    def start_session(self, session_id: Optional[str] = None) -> str:
        """Start a new conversation session."""
        self.current_session_id = session_id or str(uuid.uuid4())

        # Create new episode for this session
        self.current_episode = Episode(
            session_id=self.current_session_id,
            start_time=datetime.now(),
        )

        return self.current_session_id

    def end_session(self) -> Optional[str]:
        """End the current session and finalize the episode."""
        if self.current_episode:
            self.current_episode.end_time = datetime.now()

            # Generate summary if we have an embedding function
            if self._embed_fn and self.current_episode.messages:
                full_text = self.current_episode.get_full_text()
                self.current_episode.full_embedding = self._embed_fn(full_text)

            # Add to episodic memory
            self.episodic_memory.add_episode(self.current_episode)

            # Enforce capacity limits
            removed_episodes = self.episodic_memory.enforce_capacity(
                self.config.max_episodes
            )
            for _, session_id in removed_episodes:
                if session_id:
                    self._cascade_remove_by_session(session_id)

            self.semantic_memory.enforce_capacity(self.config.max_facts)

            episode_id = self.current_episode.id
            self.current_episode = None
            self.current_session_id = None
            return episode_id

        return None

    def _cascade_remove_by_session(self, session_id: str) -> None:
        """Remove unified memories linked to an evicted session.

        Finds memories whose session_id matches and removes them
        from the unified graph.
        """
        to_remove = [
            mid for mid, mem in self.graph.memories.items()
            if mem.session_id == session_id
        ]
        for mid in to_remove:
            self.graph.remove_memory(mid)

    def add_message(
        self,
        speaker: str,
        content: str,
        timestamp: Optional[datetime] = None,
        entities: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Add a message to the current session.

        Returns the memory ID for the message.
        """
        timestamp = timestamp or datetime.now()
        entities = entities or []

        # Create episode message
        message = EpisodeMessage(
            speaker=speaker,
            content=content,
            timestamp=timestamp,
            entities_mentioned=entities,
            metadata=metadata or {},
        )

        # Add to current episode
        if self.current_episode:
            self.current_episode.add_message(message)

        # Create unified memory item
        embedding = self._embed_fn(content) if self._embed_fn else None

        memory_item = UnifiedMemoryItem(
            content=content,
            embedding=embedding,
            event_time=TimeInterval(start=timestamp),
            entities=entities,
            source="conversation",
            session_id=self.current_session_id,
            speaker=speaker,
            metadata=metadata or {},
        )

        # Add to unified graph
        self.graph.add_memory(memory_item)

        # Add to working memory
        working_item = WorkingMemoryItem(
            id=memory_item.id,
            content=content,
            embedding=embedding,
            source_memory_id=memory_item.id,
        )
        self.working_memory.add(working_item)

        return memory_item.id

    def add_entity(
        self,
        name: str,
        entity_type: EntityType = EntityType.UNKNOWN,
        attributes: Optional[Dict[str, Any]] = None,
        aliases: Optional[List[str]] = None
    ) -> str:
        """Add or update an entity in the entity graph."""
        # Check if entity exists
        existing = self.graph.entity_graph.find_by_name(name)
        if existing:
            # Update existing entity
            entity = existing[0]
            entity.last_seen = datetime.now()
            entity.mention_count += 1
            if attributes:
                entity.attributes.update(attributes)
            if aliases:
                entity.aliases.extend([a for a in aliases if a not in entity.aliases])
            return entity.id

        # Create new entity
        entity = EntityNode(
            name=name,
            entity_type=entity_type,
            attributes=attributes or {},
            aliases=aliases or [],
        )

        return self.graph.add_entity(entity)

    def add_relation(
        self,
        source_entity: str,
        relation: str,
        target_entity: str,
        negated: bool = False,
        evidence_memory_id: Optional[str] = None
    ) -> str:
        """Add a relation between entities."""
        # Ensure entities exist
        source_id = self.add_entity(source_entity)
        target_id = self.add_entity(target_entity)

        # Add relation
        return self.graph.add_entity_relation(
            source_id=source_id,
            target_id=target_id,
            relation=relation,
            negated=negated,
            evidence=[evidence_memory_id] if evidence_memory_id else [],
        )

    def add_fact(
        self,
        subject: str,
        predicate: str,
        obj: str,
        category: str = "",
        source_episode_id: Optional[str] = None,
        negated: bool = False
    ) -> str:
        """Add a fact to semantic memory."""
        embedding = self._embed_fn(f"{subject} {predicate} {obj}") if self._embed_fn else None

        fact = Fact(
            content=f"{subject} {predicate} {obj}",
            subject=subject,
            predicate=predicate,
            object=obj,
            category=category,
            sources=[source_episode_id] if source_episode_id else [],
            embedding=embedding,
            negated=negated,
        )

        fact_id, _ = self.semantic_memory.add_fact(fact)
        return fact_id

    def add_negative_fact(
        self,
        subject: str,
        predicate: str,
        obj: str,
        source_episode_id: Optional[str] = None
    ) -> str:
        """Add an explicit negation (what is NOT true)."""
        return self.semantic_memory.add_negation(
            subject=subject,
            predicate=predicate,
            obj=obj,
            source_id=source_episode_id or "unknown",
        )

    # ==================== Query Methods ====================

    def query(
        self,
        query_text: str,
        query_type: str = "auto",
        top_k: int = 10,
        include_working_memory: bool = True
    ) -> List[Dict[str, Any]]:
        """
        Main query interface for retrieving relevant memories.

        Args:
            query_text: The query string
            query_type: "semantic", "temporal", "entity", "fact", or "auto"
            top_k: Number of results to return
            include_working_memory: Whether to include working memory in results

        Returns:
            List of memory items with scores and metadata
        """
        results = []

        # Get query embedding
        query_embedding = self._embed_fn(query_text) if self._embed_fn else None

        # Include working memory context
        if include_working_memory and query_embedding is not None:
            wm_items = self.working_memory.get_context(query_embedding, top_k=5)
            for item in wm_items:
                results.append({
                    "id": item.id,
                    "content": item.content,
                    "source": "working_memory",
                    "score": item.attention_weight,
                    "type": "working",
                })

        # Semantic search in unified graph
        if query_embedding is not None:
            similar = self.graph.query_by_similarity(
                query_embedding, top_k=top_k
            )
            for memory, score in similar:
                results.append({
                    "id": memory.id,
                    "content": memory.content,
                    "source": memory.source,
                    "score": score,
                    "type": "semantic",
                    "entities": memory.entity_names,
                    "timestamp": memory.event_time.start.isoformat() if memory.event_time else None,
                })

        # Search episodic memory
        if query_embedding is not None:
            episodes = self.episodic_memory.search_similar(
                query_embedding, top_k=top_k // 2
            )
            for episode, score in episodes:
                results.append({
                    "id": episode.id,
                    "content": episode.summary or episode.get_full_text()[:500],
                    "source": "episodic",
                    "score": score,
                    "type": "episode",
                    "participants": episode.participant_names,
                    "timestamp": episode.start_time.isoformat(),
                })

        # Search semantic facts
        if query_embedding is not None:
            facts = self.semantic_memory.search_similar(
                query_embedding, top_k=top_k // 2
            )
            for fact, score in facts:
                results.append({
                    "id": fact.id,
                    "content": fact.content,
                    "source": "semantic_fact",
                    "score": score * fact.confidence,  # Weight by confidence
                    "type": "fact",
                    "subject": fact.subject,
                    "predicate": fact.predicate,
                    "object": fact.object,
                    "negated": fact.negated,
                    "confidence": fact.confidence,
                })

        # Sort by score and deduplicate
        results.sort(key=lambda x: x["score"], reverse=True)
        seen_ids = set()
        unique_results = []
        for r in results:
            if r["id"] not in seen_ids:
                seen_ids.add(r["id"])
                unique_results.append(r)

        return unique_results[:top_k]

    def query_temporal(
        self,
        start_time: datetime,
        end_time: Optional[datetime] = None,
        entities: Optional[List[str]] = None
    ) -> List[Dict[str, Any]]:
        """Query memories by time range."""
        memories = self.graph.query_by_time(start_time, end_time, entities)

        return [
            {
                "id": m.id,
                "content": m.content,
                "source": m.source,
                "timestamp": m.event_time.start.isoformat() if m.event_time else None,
                "entities": m.entity_names,
            }
            for m in memories
        ]

    def query_entity(self, entity_name: str) -> Dict[str, Any]:
        """Get comprehensive information about an entity."""
        # Find entity
        entities = self.graph.entity_graph.find_by_name(entity_name)
        if not entities:
            return {"error": f"Entity '{entity_name}' not found"}

        entity = entities[0]

        # Get profile from entity graph
        profile = self.graph.entity_graph.get_entity_profile(entity.id)

        # Get facts from semantic memory
        facts = self.semantic_memory.get_facts_about(entity.id)
        facts.extend(self.semantic_memory.get_facts_about(entity_name))

        # Get related memories
        memories = self.graph.query_by_entity(entity.id)

        # Get temporal chain
        temporal_chain = self.graph.find_temporal_chain(entity.id)

        return {
            "entity": profile,
            "facts": [
                {
                    "content": f.content,
                    "confidence": f.confidence,
                    "negated": f.negated,
                }
                for f in facts[:20]
            ],
            "memory_count": len(memories),
            "recent_events": [
                {
                    "content": m.content[:200],
                    "timestamp": m.event_time.start.isoformat() if m.event_time else None,
                }
                for m in temporal_chain[:10]
            ],
        }

    def check_fact(
        self,
        subject: str,
        predicate: str,
        obj: str
    ) -> Dict[str, Any]:
        """Check if a fact is true, false, or unknown."""
        # Check for explicit negation
        is_negated, negation_fact = self.semantic_memory.check_negation(
            subject, predicate, obj
        )

        if is_negated:
            return {
                "status": "negated",
                "message": f"Explicitly stored as NOT true",
                "confidence": 1.0,
                "evidence": negation_fact.negation_source if negation_fact else None,
            }

        # Check for positive fact
        facts = self.semantic_memory.get_facts_about(subject, predicate)
        matching = [f for f in facts if f.object == obj and not f.negated]

        if matching:
            best = max(matching, key=lambda f: f.confidence)
            return {
                "status": "confirmed",
                "message": "Fact found in memory",
                "confidence": best.confidence,
                "confirmations": best.confirmation_count,
                "evidence": best.sources,
            }

        # Check entity relations
        source_entities = self.graph.entity_graph.find_by_name(subject)
        target_entities = self.graph.entity_graph.find_by_name(obj)

        if source_entities and target_entities:
            exists, is_neg = self.graph.entity_graph.has_relation(
                source_entities[0].id,
                target_entities[0].id,
                predicate
            )
            if exists:
                return {
                    "status": "negated" if is_neg else "confirmed",
                    "message": "Relation found in entity graph",
                    "confidence": 0.8,
                }

        return {
            "status": "unknown",
            "message": "No information found about this fact",
            "confidence": 0.0,
        }

    # ==================== Consolidation ====================

    def consolidate(self) -> Dict[str, int]:
        """
        Perform memory consolidation.

        - Extract facts from frequently accessed episodes
        - Update entity profiles
        - Archive old episodes
        """
        stats = {
            "facts_extracted": 0,
            "episodes_archived": 0,
            "entities_updated": 0,
        }

        # Find episodes ready for consolidation
        candidates = self.episodic_memory.get_candidates_for_consolidation(
            min_retrievals=self.config.consolidation_min_retrievals,
            min_age_days=self.config.consolidation_threshold_days,
        )

        for episode in candidates:
            # Extract facts (simplified - in production, use LLM)
            for fact_text in episode.extracted_facts:
                # This would be LLM-extracted in production
                stats["facts_extracted"] += 1

        return stats

    # ==================== Statistics ====================

    def get_stats(self) -> Dict[str, Any]:
        """Get comprehensive statistics about the memory system."""
        ep_count = len(self.episodic_memory.episodes)
        fact_count = len(self.semantic_memory.facts)
        return {
            "graph": self.graph.get_stats(),
            "working_memory": self.working_memory.get_stats(),
            "episodic_memory": self.episodic_memory.get_stats(),
            "semantic_memory": self.semantic_memory.get_stats(),
            "current_session": self.current_session_id,
            "capacity": {
                "max_episodes": self.config.max_episodes,
                "max_facts": self.config.max_facts,
                "episode_utilization": ep_count / self.config.max_episodes if self.config.max_episodes else 0,
                "fact_utilization": fact_count / self.config.max_facts if self.config.max_facts else 0,
            },
        }

    def get_context_for_response(
        self,
        query: str,
        max_tokens: int = 4000
    ) -> str:
        """
        Get formatted context string for LLM response generation.

        Uses position-aware composition to combat lost-in-the-middle.
        """
        results = self.query(query, top_k=20)

        if not results:
            return ""

        # Position-aware composition
        parts = []

        # High relevance at start
        parts.append("## Most Relevant Context")
        for r in results[:3]:
            parts.append(f"- {r['content']}")

        # Medium relevance in middle with structure
        if len(results) > 6:
            parts.append("\n## Additional Context")
            for i, r in enumerate(results[3:-3], 1):
                parts.append(f"[{i}] {r['content'][:200]}")

        # Reinforce key points at end
        parts.append("\n## Key Points")
        for r in results[:2]:
            key_point = r['content'][:100]
            parts.append(f"- {key_point}")

        context = "\n".join(parts)

        # Truncate if needed (rough estimate)
        if len(context) > max_tokens * 4:
            context = context[:max_tokens * 4]

        return context

    def to_dict(self) -> Dict[str, Any]:
        """Serialize the entire memory system.

        Note: Working memory is ephemeral and NOT serialized.
        Embeddings are NOT included — save separately via persistence module.
        """
        return {
            "config": {
                "working_memory_capacity": self.config.working_memory_capacity,
                "working_memory_decay_rate": self.config.working_memory_decay_rate,
                "embedding_dim": self.config.embedding_dim,
                "auto_consolidate": self.config.auto_consolidate,
                "consolidation_threshold_days": self.config.consolidation_threshold_days,
                "consolidation_min_retrievals": self.config.consolidation_min_retrievals,
                "max_episodes": self.config.max_episodes,
                "max_facts": self.config.max_facts,
                "eviction_batch_size": self.config.eviction_batch_size,
            },
            "current_session_id": self.current_session_id,
            "graph": self.graph.to_dict(),
            "episodic": self.episodic_memory.to_dict(),
            "semantic": self.semantic_memory.to_dict(),
        }

    @classmethod
    def from_dict(
        cls,
        data: Dict[str, Any],
        embedding_fn: Optional[callable] = None,
        embeddings_map: Optional[Dict[str, np.ndarray]] = None,
    ) -> "MemoryManager":
        """Deserialize memory manager from dictionary.

        Args:
            data: Output of to_dict().
            embedding_fn: Embedding function to set on the restored manager.
            embeddings_map: Map of id -> embedding for all components.
        """
        embeddings_map = embeddings_map or {}
        config_data = data.get("config", {})
        config = MemoryConfig(
            working_memory_capacity=config_data.get("working_memory_capacity", 20),
            working_memory_decay_rate=config_data.get("working_memory_decay_rate", 0.05),
            embedding_dim=config_data.get("embedding_dim", 1536),
            auto_consolidate=config_data.get("auto_consolidate", True),
            consolidation_threshold_days=config_data.get("consolidation_threshold_days", 7),
            consolidation_min_retrievals=config_data.get("consolidation_min_retrievals", 3),
            max_episodes=config_data.get("max_episodes", 500),
            max_facts=config_data.get("max_facts", 5000),
            eviction_batch_size=config_data.get("eviction_batch_size", 10),
        )

        manager = cls(config=config)

        if embedding_fn:
            manager.set_embedding_function(embedding_fn)

        # Restore sub-components
        if "graph" in data:
            manager.graph = UnifiedMemoryGraph.from_dict(
                data["graph"], embeddings_map
            )
        if "episodic" in data:
            manager.episodic_memory = EpisodicMemory.from_dict(
                data["episodic"], embeddings_map
            )
        if "semantic" in data:
            manager.semantic_memory = SemanticMemoryStore.from_dict(
                data["semantic"], embeddings_map
            )

        manager.current_session_id = data.get("current_session_id")

        return manager
