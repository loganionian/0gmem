"""Tests for persistence: save/load memory state to disk."""

import shutil
from datetime import datetime
from pathlib import Path

import numpy as np
import pytest

from zerogmem.memory.manager import MemoryManager, MemoryConfig
from zerogmem.memory.episodic import Episode, EpisodeMessage
from zerogmem.memory.semantic import Fact
from zerogmem.graph.entity import EntityNode, EntityType
from zerogmem.graph.temporal import TimeInterval
from zerogmem.graph.unified import UnifiedMemoryItem
from zerogmem.persistence import (
    save_memory_state,
    load_memory_state,
    EmbeddingRegistry,
    STATE_FILENAME,
    EMBEDDINGS_FILENAME,
)


@pytest.fixture
def tmp_persist_dir(tmp_path):
    """Temp directory for persistence tests."""
    d = tmp_path / "persist"
    d.mkdir()
    return d


@pytest.fixture
def mock_embedding_fn():
    """Deterministic hash-based embedding function."""
    def _embed(text: str) -> np.ndarray:
        np.random.seed(hash(text) % (2**31))
        return np.random.randn(1536).astype(np.float32)
    return _embed


@pytest.fixture
def populated_manager(mock_embedding_fn):
    """Create a MemoryManager with data in all components."""
    config = MemoryConfig(working_memory_capacity=10, embedding_dim=1536)
    mm = MemoryManager(config=config)
    mm.set_embedding_function(mock_embedding_fn)

    # Start a session and add messages
    mm.start_session(session_id="session-1")
    mm.add_message("Alice", "I love hiking in the mountains.", timestamp=datetime(2024, 6, 1, 10, 0))
    mm.add_message("Bob", "Have you tried the Alps?", timestamp=datetime(2024, 6, 1, 10, 5))
    mm.end_session()

    mm.start_session(session_id="session-2")
    mm.add_message("Alice", "I went swimming yesterday.", timestamp=datetime(2024, 7, 1, 9, 0))
    mm.end_session()

    # Add entities
    alice_id = mm.add_entity("Alice", EntityType.PERSON, attributes={"age": 30})
    bob_id = mm.add_entity("Bob", EntityType.PERSON)

    # Add relation
    mm.add_relation("Alice", "knows", "Bob")

    # Add facts
    mm.add_fact("Alice", "likes", "hiking", category="preference")
    mm.add_fact("Alice", "lives_in", "Colorado")
    mm.add_negative_fact("Alice", "likes", "spiders")

    return mm


class TestEmbeddingRegistry:
    """Tests for the EmbeddingRegistry."""

    def test_register_and_count(self):
        reg = EmbeddingRegistry()
        emb = np.random.randn(1536).astype(np.float32)
        reg.register("key1", emb)
        reg.register("key2", emb)
        assert reg.count == 2

    def test_register_none_skipped(self):
        reg = EmbeddingRegistry()
        reg.register("key1", None)
        assert reg.count == 0

    def test_save_and_load(self, tmp_persist_dir):
        reg = EmbeddingRegistry()
        emb1 = np.random.randn(1536).astype(np.float32)
        emb2 = np.random.randn(1536).astype(np.float32)
        reg.register("id_a", emb1)
        reg.register("id_b", emb2)

        count = reg.save(tmp_persist_dir)
        assert count == 2
        assert (tmp_persist_dir / EMBEDDINGS_FILENAME).exists()

        loaded = EmbeddingRegistry.load(tmp_persist_dir)
        assert len(loaded) == 2
        np.testing.assert_array_almost_equal(loaded["id_a"], emb1)
        np.testing.assert_array_almost_equal(loaded["id_b"], emb2)

    def test_save_empty_removes_stale(self, tmp_persist_dir):
        # Create a dummy file
        dummy = tmp_persist_dir / EMBEDDINGS_FILENAME
        dummy.touch()
        assert dummy.exists()

        reg = EmbeddingRegistry()
        reg.save(tmp_persist_dir)
        assert not dummy.exists()

    def test_load_missing_file(self, tmp_persist_dir):
        loaded = EmbeddingRegistry.load(tmp_persist_dir)
        assert loaded == {}


class TestSaveLoadMemoryState:
    """Tests for save_memory_state and load_memory_state."""

    def test_save_creates_files(self, populated_manager, tmp_persist_dir):
        result = save_memory_state(populated_manager, tmp_persist_dir)
        assert (tmp_persist_dir / STATE_FILENAME).exists()
        assert (tmp_persist_dir / EMBEDDINGS_FILENAME).exists()
        assert result["memories"] > 0
        assert result["embeddings_saved"] > 0

    def test_round_trip_memories(self, populated_manager, mock_embedding_fn, tmp_persist_dir):
        save_memory_state(populated_manager, tmp_persist_dir)
        restored = load_memory_state(tmp_persist_dir, mock_embedding_fn)

        assert restored is not None
        assert len(restored.graph.memories) == len(populated_manager.graph.memories)

        # Verify memory content
        for mid, orig_mem in populated_manager.graph.memories.items():
            rest_mem = restored.graph.memories.get(mid)
            assert rest_mem is not None, f"Memory {mid} not found after restore"
            assert rest_mem.content == orig_mem.content
            assert rest_mem.speaker == orig_mem.speaker

    def test_round_trip_episodes(self, populated_manager, mock_embedding_fn, tmp_persist_dir):
        save_memory_state(populated_manager, tmp_persist_dir)
        restored = load_memory_state(tmp_persist_dir, mock_embedding_fn)

        assert len(restored.episodic_memory.episodes) == len(populated_manager.episodic_memory.episodes)

        for eid, orig_ep in populated_manager.episodic_memory.episodes.items():
            rest_ep = restored.episodic_memory.episodes.get(eid)
            assert rest_ep is not None
            assert rest_ep.summary == orig_ep.summary
            assert len(rest_ep.messages) == len(orig_ep.messages)

    def test_round_trip_facts(self, populated_manager, mock_embedding_fn, tmp_persist_dir):
        save_memory_state(populated_manager, tmp_persist_dir)
        restored = load_memory_state(tmp_persist_dir, mock_embedding_fn)

        assert len(restored.semantic_memory.facts) == len(populated_manager.semantic_memory.facts)

        for fid, orig_fact in populated_manager.semantic_memory.facts.items():
            rest_fact = restored.semantic_memory.facts.get(fid)
            assert rest_fact is not None
            assert rest_fact.subject == orig_fact.subject
            assert rest_fact.predicate == orig_fact.predicate
            assert rest_fact.object == orig_fact.object
            assert rest_fact.negated == orig_fact.negated

    def test_round_trip_entities(self, populated_manager, mock_embedding_fn, tmp_persist_dir):
        save_memory_state(populated_manager, tmp_persist_dir)
        restored = load_memory_state(tmp_persist_dir, mock_embedding_fn)

        assert len(restored.graph.entity_graph.nodes) == len(populated_manager.graph.entity_graph.nodes)

        for nid, orig_node in populated_manager.graph.entity_graph.nodes.items():
            rest_node = restored.graph.entity_graph.nodes.get(nid)
            assert rest_node is not None
            assert rest_node.name == orig_node.name
            assert rest_node.entity_type == orig_node.entity_type

    def test_round_trip_embeddings(self, populated_manager, mock_embedding_fn, tmp_persist_dir):
        save_memory_state(populated_manager, tmp_persist_dir)
        restored = load_memory_state(tmp_persist_dir, mock_embedding_fn)

        # Check that at least some memories have embeddings restored
        orig_with_emb = [m for m in populated_manager.graph.memories.values() if m.embedding is not None]
        rest_with_emb = [m for m in restored.graph.memories.values() if m.embedding is not None]
        assert len(rest_with_emb) == len(orig_with_emb)

    def test_round_trip_config(self, populated_manager, mock_embedding_fn, tmp_persist_dir):
        save_memory_state(populated_manager, tmp_persist_dir)
        restored = load_memory_state(tmp_persist_dir, mock_embedding_fn)

        assert restored.config.working_memory_capacity == populated_manager.config.working_memory_capacity
        assert restored.config.embedding_dim == populated_manager.config.embedding_dim

    def test_empty_state_round_trip(self, mock_embedding_fn, tmp_persist_dir):
        mm = MemoryManager()
        mm.set_embedding_function(mock_embedding_fn)

        save_memory_state(mm, tmp_persist_dir)
        restored = load_memory_state(tmp_persist_dir, mock_embedding_fn)

        assert restored is not None
        assert len(restored.graph.memories) == 0
        assert len(restored.episodic_memory.episodes) == 0
        assert len(restored.semantic_memory.facts) == 0

    def test_load_missing_files_returns_none(self, tmp_persist_dir):
        result = load_memory_state(tmp_persist_dir)
        assert result is None

    def test_backup_created_on_save(self, populated_manager, tmp_persist_dir):
        # First save
        save_memory_state(populated_manager, tmp_persist_dir)
        assert not (tmp_persist_dir / f"{STATE_FILENAME}.bak").exists()

        # Second save should create backup
        save_memory_state(populated_manager, tmp_persist_dir)
        assert (tmp_persist_dir / f"{STATE_FILENAME}.bak").exists()

    def test_creates_directory(self, populated_manager, tmp_path):
        new_dir = tmp_path / "new" / "nested" / "dir"
        assert not new_dir.exists()
        save_memory_state(populated_manager, new_dir)
        assert new_dir.exists()
        assert (new_dir / STATE_FILENAME).exists()

    def test_load_without_embedding_fn(self, populated_manager, tmp_persist_dir):
        save_memory_state(populated_manager, tmp_persist_dir)
        restored = load_memory_state(tmp_persist_dir)
        assert restored is not None
        assert restored._embed_fn is None

    def test_round_trip_temporal_nodes(self, populated_manager, mock_embedding_fn, tmp_persist_dir):
        save_memory_state(populated_manager, tmp_persist_dir)
        restored = load_memory_state(tmp_persist_dir, mock_embedding_fn)

        assert len(restored.graph.temporal_graph.nodes) == len(populated_manager.graph.temporal_graph.nodes)

    def test_round_trip_semantic_graph_nodes(self, populated_manager, mock_embedding_fn, tmp_persist_dir):
        save_memory_state(populated_manager, tmp_persist_dir)
        restored = load_memory_state(tmp_persist_dir, mock_embedding_fn)

        assert len(restored.graph.semantic_graph.nodes) == len(populated_manager.graph.semantic_graph.nodes)

    def test_round_trip_preserves_indexes(self, populated_manager, mock_embedding_fn, tmp_persist_dir):
        save_memory_state(populated_manager, tmp_persist_dir)
        restored = load_memory_state(tmp_persist_dir, mock_embedding_fn)

        # Entity graph indexes
        assert len(restored.graph.entity_graph._name_index) > 0
        assert len(restored.graph.entity_graph._type_index) > 0

        # Semantic memory indexes
        assert len(restored.semantic_memory._subject_index) > 0


class TestGraphFromDict:
    """Tests for individual graph from_dict methods."""

    def test_temporal_graph_round_trip(self, mock_embedding_fn):
        from zerogmem.graph.temporal import TemporalGraph, TemporalNode, TimeInterval

        g = TemporalGraph()
        n1 = TemporalNode(content="morning jog", event_time=TimeInterval(
            start=datetime(2024, 6, 1, 8, 0), end=datetime(2024, 6, 1, 9, 0)
        ), entities=["alice"])
        n2 = TemporalNode(content="lunch", event_time=TimeInterval(
            start=datetime(2024, 6, 1, 12, 0)
        ))
        g.add_node(n1)
        g.add_node(n2)

        data = g.to_dict()
        restored = TemporalGraph.from_dict(data)

        assert len(restored.nodes) == 2
        assert restored.nodes[n1.id].content == "morning jog"
        assert restored.nodes[n1.id].entities == ["alice"]
        # Edges should be recomputed
        assert len(restored.edges) > 0

    def test_entity_graph_round_trip(self):
        from zerogmem.graph.entity import EntityGraph, EntityNode, EntityEdge, EntityType

        g = EntityGraph()
        alice = EntityNode(name="Alice", entity_type=EntityType.PERSON, aliases=["Ali"])
        bob = EntityNode(name="Bob", entity_type=EntityType.PERSON)
        g.add_node(alice)
        g.add_node(bob)
        g.add_edge(EntityEdge(source_id=alice.id, target_id=bob.id, relation="knows"))

        data = g.to_dict()
        restored = EntityGraph.from_dict(data)

        assert len(restored.nodes) == 2
        assert restored.nodes[alice.id].name == "Alice"
        assert restored.nodes[alice.id].aliases == ["Ali"]
        assert len(restored.edges) == 1

    def test_causal_graph_round_trip(self):
        from zerogmem.graph.causal import CausalGraph, CausalNode, CausalEdge

        g = CausalGraph()
        n1 = CausalNode(content="rain", event_type="event")
        n2 = CausalNode(content="flooding", event_type="event")
        g.add_node(n1)
        g.add_node(n2)
        g.add_edge(CausalEdge(cause_id=n1.id, effect_id=n2.id, strength=0.9))

        data = g.to_dict()
        restored = CausalGraph.from_dict(data)

        assert len(restored.nodes) == 2
        assert len(restored.edges) == 1

    def test_semantic_graph_round_trip(self, mock_embedding_fn):
        from zerogmem.graph.semantic import SemanticGraph, SemanticNode

        g = SemanticGraph(embedding_dim=1536)
        emb = mock_embedding_fn("hiking")
        n = SemanticNode(content="hiking", embedding=emb, concepts=["outdoors"])
        g.add_node(n)

        data = g.to_dict()
        emb_map = g.get_embeddings_map()
        restored = SemanticGraph.from_dict(data, emb_map)

        assert len(restored.nodes) == 1
        assert restored.nodes[n.id].content == "hiking"
        assert restored.nodes[n.id].concepts == ["outdoors"]
        np.testing.assert_array_almost_equal(
            restored.nodes[n.id].embedding, emb
        )


class TestCorruptionRecovery:
    """Tests for error recovery on corrupt persistence files."""

    def test_corrupt_json_returns_none(self, tmp_persist_dir):
        """Corrupt primary JSON with no .bak -> returns None."""
        state_file = tmp_persist_dir / STATE_FILENAME
        state_file.write_text("{invalid json!!")
        result = load_memory_state(tmp_persist_dir)
        assert result is None

    def test_corrupt_json_falls_back_to_bak(
        self, populated_manager, mock_embedding_fn, tmp_persist_dir
    ):
        """Corrupt primary JSON but valid .bak -> restores from .bak."""
        # First save, then second save to create .bak
        save_memory_state(populated_manager, tmp_persist_dir)
        save_memory_state(populated_manager, tmp_persist_dir)

        # Corrupt the primary
        state_file = tmp_persist_dir / STATE_FILENAME
        state_file.write_text("CORRUPT!")

        restored = load_memory_state(tmp_persist_dir, mock_embedding_fn)
        assert restored is not None
        assert len(restored.graph.memories) == len(populated_manager.graph.memories)

    def test_both_json_and_bak_corrupt(self, tmp_persist_dir):
        """Both primary and .bak corrupt -> returns None."""
        (tmp_persist_dir / STATE_FILENAME).write_text("BAD")
        (tmp_persist_dir / f"{STATE_FILENAME}.bak").write_text("ALSO BAD")
        result = load_memory_state(tmp_persist_dir)
        assert result is None

    def test_corrupt_npz_still_loads_json(
        self, populated_manager, mock_embedding_fn, tmp_persist_dir
    ):
        """Corrupt NPZ -> JSON loads, embeddings empty, no crash."""
        save_memory_state(populated_manager, tmp_persist_dir)

        # Corrupt the NPZ
        npz_path = tmp_persist_dir / EMBEDDINGS_FILENAME
        npz_path.write_bytes(b"this is not a valid npz file")

        restored = load_memory_state(tmp_persist_dir, mock_embedding_fn)
        assert restored is not None
        assert len(restored.graph.memories) == len(populated_manager.graph.memories)

    def test_missing_npz_still_loads(
        self, populated_manager, mock_embedding_fn, tmp_persist_dir
    ):
        """Missing NPZ (deleted after save) -> loads without embeddings."""
        save_memory_state(populated_manager, tmp_persist_dir)
        (tmp_persist_dir / EMBEDDINGS_FILENAME).unlink()

        restored = load_memory_state(tmp_persist_dir, mock_embedding_fn)
        assert restored is not None
        assert len(restored.graph.memories) == len(populated_manager.graph.memories)

    def test_malformed_json_structure_does_not_crash(self, tmp_persist_dir, mock_embedding_fn):
        """Valid JSON but wrong structure -> should not crash."""
        state_file = tmp_persist_dir / STATE_FILENAME
        state_file.write_text('{"not": "a valid state"}')
        # from_dict uses .get() with defaults, so this produces an empty manager
        # rather than crashing — either outcome (None or empty manager) is acceptable
        result = load_memory_state(tmp_persist_dir, mock_embedding_fn)
        assert result is None or isinstance(result, MemoryManager)
