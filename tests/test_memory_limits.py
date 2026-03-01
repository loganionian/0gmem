"""Tests for memory limits, capacity enforcement, and garbage collection."""

from datetime import datetime, timedelta
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from zerogmem.graph.unified import UnifiedMemoryGraph, UnifiedMemoryItem
from zerogmem.graph.temporal import TimeInterval
from zerogmem.memory.episodic import EpisodicMemory, Episode, EpisodeMessage
from zerogmem.memory.semantic import SemanticMemoryStore, Fact
from zerogmem.memory.manager import MemoryManager, MemoryConfig


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _mock_embedding(seed: int = 0) -> np.ndarray:
    np.random.seed(seed)
    return np.random.randn(1536).astype(np.float32)


def _mock_embedding_fn(text: str) -> np.ndarray:
    np.random.seed(hash(text) % (2**31))
    return np.random.randn(1536).astype(np.float32)


def _make_memory(graph: UnifiedMemoryGraph, content: str = "test",
                 session_id: str = None, entities: list = None,
                 concepts: list = None) -> UnifiedMemoryItem:
    """Add a memory with embedding to the graph and return it."""
    item = UnifiedMemoryItem(
        content=content,
        embedding=_mock_embedding_fn(content),
        event_time=TimeInterval(start=datetime.now()),
        entities=entities or [],
        concepts=concepts or [],
        session_id=session_id,
    )
    graph.add_memory(item)
    return item


# ---------------------------------------------------------------------------
# UnifiedMemoryGraph.remove_memory()
# ---------------------------------------------------------------------------

class TestRemoveMemory:
    """Tests for UnifiedMemoryGraph.remove_memory()."""

    def test_remove_memory_cleans_all_subgraphs(self):
        graph = UnifiedMemoryGraph()
        mem = _make_memory(graph, "Hello world")

        assert mem.id in graph.memories
        assert mem.temporal_node_id in graph.temporal_graph.nodes
        assert mem.semantic_node_id in graph.semantic_graph.nodes

        result = graph.remove_memory(mem.id)

        assert result is True
        assert mem.id not in graph.memories
        assert mem.temporal_node_id not in graph.temporal_graph.nodes
        assert mem.semantic_node_id not in graph.semantic_graph.nodes
        # Embedding should be gone
        assert mem.semantic_node_id not in graph.semantic_graph._embedding_ids

    def test_remove_memory_cleans_crossrefs(self):
        graph = UnifiedMemoryGraph()
        mem = _make_memory(graph, "test", entities=["e1"], concepts=["c1"])

        assert mem.id in graph._entity_to_memories.get("e1", set())
        assert mem.id in graph._concept_to_memories.get("c1", set())

        graph.remove_memory(mem.id)

        assert mem.id not in graph._entity_to_memories.get("e1", set())
        assert mem.id not in graph._concept_to_memories.get("c1", set())

    def test_remove_nonexistent_returns_false(self):
        graph = UnifiedMemoryGraph()
        assert graph.remove_memory("nonexistent") is False

    def test_remove_memory_with_causal_node(self):
        graph = UnifiedMemoryGraph()
        # First add a memory to serve as cause
        cause = _make_memory(graph, "cause event")
        # Add a memory with causal info
        item = UnifiedMemoryItem(
            content="effect event",
            embedding=_mock_embedding_fn("effect"),
            event_time=TimeInterval(start=datetime.now()),
            causes=[cause.id],
        )
        graph.add_memory(item)

        assert item.causal_node_id is not None
        assert item.causal_node_id in graph.causal_graph.nodes

        graph.remove_memory(item.id)
        assert item.causal_node_id not in graph.causal_graph.nodes


# ---------------------------------------------------------------------------
# EpisodicMemory
# ---------------------------------------------------------------------------

class TestEpisodicRemoval:
    """Tests for EpisodicMemory.remove_episode() and enforce_capacity()."""

    def _make_episode(self, store: EpisodicMemory, age_days: int = 0,
                      retrieval_count: int = 0, importance: float = 0.5,
                      session_id: str = None,
                      participant_names: list = None,
                      topics: list = None) -> Episode:
        ep = Episode(
            session_id=session_id or "sess",
            start_time=datetime.now() - timedelta(days=age_days),
            created_at=datetime.now() - timedelta(days=age_days),
            retrieval_count=retrieval_count,
            importance=importance,
            participant_names=participant_names or [],
            topics=topics or [],
            summary_embedding=_mock_embedding(hash(str(age_days)) % 1000),
        )
        store.add_episode(ep)
        return ep

    def test_remove_episode_cleans_indexes(self):
        store = EpisodicMemory()
        ep = self._make_episode(
            store, participant_names=["Alice"], topics=["work"],
            session_id="s1",
        )

        assert ep.id in store.episodes
        assert ep.id in store._embedding_ids

        store.remove_episode(ep.id)

        assert ep.id not in store.episodes
        assert ep.id not in store._embedding_ids
        assert ep.id not in store._participant_index.get("Alice", set())
        assert ep.id not in store._topic_index.get("work", set())

    def test_enforce_capacity_no_op_under_limit(self):
        store = EpisodicMemory()
        self._make_episode(store)
        self._make_episode(store)

        removed = store.enforce_capacity(max_episodes=5)
        assert removed == []
        assert len(store.episodes) == 2

    def test_enforce_capacity_evicts_lowest_scored(self):
        store = EpisodicMemory()
        # Add 5 episodes
        episodes = []
        for i in range(5):
            episodes.append(self._make_episode(store, age_days=i))

        removed = store.enforce_capacity(max_episodes=3)

        assert len(removed) == 2
        assert len(store.episodes) == 3

    def test_eviction_prefers_old_unaccessed(self):
        store = EpisodicMemory()
        # Old, unaccessed episode (should be evicted first)
        old_ep = self._make_episode(store, age_days=365, retrieval_count=0,
                                     importance=0.1)
        # Recent, accessed episode (should be kept)
        new_ep = self._make_episode(store, age_days=1, retrieval_count=10,
                                     importance=0.9)
        # Mid episode
        mid_ep = self._make_episode(store, age_days=30, retrieval_count=2,
                                     importance=0.5)

        removed = store.enforce_capacity(max_episodes=2)

        assert len(removed) == 1
        removed_ids = [eid for eid, _ in removed]
        assert old_ep.id in removed_ids
        assert new_ep.id in store.episodes
        assert mid_ep.id in store.episodes


# ---------------------------------------------------------------------------
# SemanticMemoryStore
# ---------------------------------------------------------------------------

class TestSemanticRemoval:
    """Tests for SemanticMemoryStore.remove_fact() and enforce_capacity()."""

    def _make_fact(self, store: SemanticMemoryStore, subject: str = "Alice",
                   predicate: str = "likes", obj: str = "cats",
                   category: str = "preference",
                   confidence: float = 1.0,
                   confirmation_count: int = 1,
                   negated: bool = False,
                   age_days: int = 0) -> Fact:
        fact = Fact(
            content=f"{subject} {predicate} {obj}",
            subject=subject,
            predicate=predicate,
            object=obj,
            category=category,
            confidence=confidence,
            confirmation_count=confirmation_count,
            negated=negated,
            first_learned=datetime.now() - timedelta(days=age_days),
            last_confirmed=datetime.now() - timedelta(days=age_days),
            embedding=_mock_embedding_fn(f"{subject} {predicate} {obj}"),
        )
        # Directly populate to avoid merge logic
        store.facts[fact.id] = fact
        store._index_fact(fact)
        if fact.negated:
            store._negated_facts.add(fact.id)
        return fact

    def test_remove_fact_cleans_indexes(self):
        store = SemanticMemoryStore()
        fact = self._make_fact(store, subject="Bob", predicate="lives_in",
                               obj="NYC", category="biographical")

        assert fact.id in store.facts
        assert fact.id in store._subject_index["Bob"]
        assert fact.id in store._predicate_index["lives_in"]
        assert fact.id in store._object_index["NYC"]
        assert fact.id in store._category_index["biographical"]
        assert fact.id in store._embedding_ids

        store.remove_fact(fact.id)

        assert fact.id not in store.facts
        assert fact.id not in store._subject_index.get("Bob", set())
        assert fact.id not in store._predicate_index.get("lives_in", set())
        assert fact.id not in store._object_index.get("NYC", set())
        assert fact.id not in store._category_index.get("biographical", set())
        assert fact.id not in store._embedding_ids

    def test_remove_negated_fact_cleans_negated_set(self):
        store = SemanticMemoryStore()
        fact = self._make_fact(store, negated=True)

        assert fact.id in store._negated_facts

        store.remove_fact(fact.id)
        assert fact.id not in store._negated_facts

    def test_enforce_capacity_no_op_under_limit(self):
        store = SemanticMemoryStore()
        self._make_fact(store)

        removed = store.enforce_capacity(max_facts=10)
        assert removed == []
        assert len(store.facts) == 1

    def test_enforce_capacity_evicts_lowest_scored(self):
        store = SemanticMemoryStore()
        for i in range(5):
            self._make_fact(store, subject=f"S{i}", obj=f"O{i}")

        removed = store.enforce_capacity(max_facts=3)

        assert len(removed) == 2
        assert len(store.facts) == 3

    def test_eviction_prefers_low_confidence_negated(self):
        store = SemanticMemoryStore()
        # Low confidence, negated (should be evicted first)
        bad_fact = self._make_fact(store, subject="X", obj="Y",
                                    confidence=0.1, negated=True,
                                    age_days=100)
        # High confidence, confirmed (should be kept)
        good_fact = self._make_fact(store, subject="A", obj="B",
                                     confidence=1.0,
                                     confirmation_count=5,
                                     age_days=1)

        removed = store.enforce_capacity(max_facts=1)

        assert bad_fact.id in removed
        assert good_fact.id in store.facts


# ---------------------------------------------------------------------------
# MemoryManager integration
# ---------------------------------------------------------------------------

class TestManagerEviction:
    """Tests for MemoryManager capacity enforcement wiring."""

    def test_end_session_enforces_episode_capacity(self):
        config = MemoryConfig(max_episodes=2)
        mm = MemoryManager(config=config)
        mm.set_embedding_function(_mock_embedding_fn)

        # Create 3 sessions (each produces 1 episode)
        for i in range(3):
            mm.start_session()
            mm.add_message("user", f"Message {i}")
            mm.end_session()

        # Should have evicted 1 episode to stay at max_episodes=2
        assert len(mm.episodic_memory.episodes) == 2

    def test_end_session_cascades_memory_removal(self):
        config = MemoryConfig(max_episodes=1)
        mm = MemoryManager(config=config)
        mm.set_embedding_function(_mock_embedding_fn)

        # Session 1
        mm.start_session()
        mm.add_message("user", "First session message")
        mm.end_session()
        initial_memory_count = len(mm.graph.memories)
        assert initial_memory_count == 1

        # Session 2 — should evict session 1's episode and its memories
        mm.start_session()
        mm.add_message("user", "Second session message")
        mm.end_session()

        assert len(mm.episodic_memory.episodes) == 1
        # The old memory should be gone, new one present
        assert len(mm.graph.memories) == 1

    def test_config_serialization_roundtrip(self):
        config = MemoryConfig(max_episodes=100, max_facts=200,
                              eviction_batch_size=5)
        mm = MemoryManager(config=config)
        mm.set_embedding_function(_mock_embedding_fn)

        data = mm.to_dict()
        restored = MemoryManager.from_dict(data, _mock_embedding_fn)

        assert restored.config.max_episodes == 100
        assert restored.config.max_facts == 200
        assert restored.config.eviction_batch_size == 5

    def test_get_stats_includes_capacity(self):
        mm = MemoryManager()
        mm.set_embedding_function(_mock_embedding_fn)

        stats = mm.get_stats()
        assert "capacity" in stats
        assert stats["capacity"]["max_episodes"] == 500
        assert stats["capacity"]["max_facts"] == 5000
        assert stats["capacity"]["episode_utilization"] == 0
        assert stats["capacity"]["fact_utilization"] == 0


# ---------------------------------------------------------------------------
# MCP summary display
# ---------------------------------------------------------------------------

class TestMCPCapacityDisplay:
    """Test that get_memory_summary shows capacity info."""

    @pytest.mark.asyncio
    async def test_memory_summary_shows_capacity(self):
        from zerogmem import mcp_server

        mm = MagicMock()
        mm.get_stats.return_value = {
            "episodic_memory": {"total_episodes": 10, "total_messages": 50,
                                "unique_participants": 2},
            "semantic_memory": {"total_facts": 25},
            "graph": {"entity_nodes": 5, "semantic_nodes": 10,
                       "temporal_nodes": 10},
            "current_session": None,
            "capacity": {
                "max_episodes": 500,
                "max_facts": 5000,
                "episode_utilization": 0.02,
                "fact_utilization": 0.005,
            },
        }

        original = mcp_server._memory_manager
        old_init = mcp_server._initialized
        mcp_server._memory_manager = mm
        mcp_server._initialized = True

        try:
            result = await mcp_server.get_memory_summary()
            assert "Capacity" in result
            assert "10/500" in result
            assert "25/5000" in result
        finally:
            mcp_server._memory_manager = original
            mcp_server._initialized = old_init
