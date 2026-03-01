"""
Integration tests for 0GMem.

These tests verify the end-to-end workflow of the memory system.
"""

import pytest
from zerogmem import (
    MemoryManager,
    Encoder,
    Retriever,
    MemoryConfig,
    RetrievalResponse,
)


class TestMemoryWorkflow:
    """Test the complete memory workflow."""

    def test_session_lifecycle(self, memory_with_embeddings):
        """Test starting and ending sessions."""
        memory = memory_with_embeddings

        # Start session
        session_id = memory.start_session()
        assert session_id is not None
        assert memory.current_session_id == session_id

        # End session
        episode_id = memory.end_session()
        assert memory.current_session_id is None

    def test_add_messages(self, memory_with_embeddings):
        """Test adding messages to a session."""
        memory = memory_with_embeddings

        memory.start_session()

        # Add messages
        msg_id1 = memory.add_message("Alice", "Hello!")
        msg_id2 = memory.add_message("Bob", "Hi there!")

        assert msg_id1 is not None
        assert msg_id2 is not None

        memory.end_session()

    def test_multi_session(self, memory_with_embeddings):
        """Test multiple conversation sessions."""
        memory = memory_with_embeddings

        # First session
        memory.start_session()
        memory.add_message("Alice", "I love coffee.")
        memory.end_session()

        # Second session
        memory.start_session()
        memory.add_message("Alice", "I prefer tea now.")
        memory.end_session()

        # Check episodic memory has both episodes
        stats = memory.get_stats()
        episodic_stats = stats.get("episodic_memory", {})
        assert episodic_stats.get("total_episodes", 0) >= 2

    def test_memory_statistics(self, populated_memory):
        """Test getting memory statistics."""
        stats = populated_memory.get_stats()

        assert isinstance(stats, dict)
        assert "total_episodes" in stats or "episodic_memory" in stats


class TestRetrieval:
    """Test memory retrieval functionality."""

    def test_basic_retrieval(self, retriever):
        """Test basic retrieval returns results."""
        result = retriever.retrieve("hiking")

        assert isinstance(result, RetrievalResponse)
        assert result.composed_context is not None

    def test_retrieval_with_entity(self, retriever):
        """Test retrieval with entity mention."""
        result = retriever.retrieve("What did Alice say about hiking?")

        assert result is not None
        assert len(result.composed_context) > 0

    def test_retrieval_empty_query(self, retriever):
        """Test retrieval handles empty query gracefully."""
        result = retriever.retrieve("")

        # Should return some response, not crash
        assert result is not None


class TestImports:
    """Test that all expected imports work."""

    def test_core_imports(self):
        """Test core class imports."""
        from zerogmem import MemoryManager, Encoder, Retriever
        assert MemoryManager is not None
        assert Encoder is not None
        assert Retriever is not None

    def test_config_imports(self):
        """Test configuration class imports."""
        from zerogmem import MemoryConfig, EncoderConfig, RetrieverConfig
        assert MemoryConfig is not None
        assert EncoderConfig is not None
        assert RetrieverConfig is not None

    def test_data_type_imports(self):
        """Test data type imports."""
        from zerogmem import RetrievalResult, RetrievalResponse, QueryAnalysis
        assert RetrievalResult is not None
        assert RetrievalResponse is not None
        assert QueryAnalysis is not None

    def test_version(self):
        """Test version is accessible."""
        from zerogmem import __version__
        assert __version__ is not None
        assert isinstance(__version__, str)


class TestConfiguration:
    """Test configuration classes."""

    def test_memory_config_defaults(self):
        """Test MemoryConfig has sensible defaults."""
        from zerogmem import MemoryConfig

        config = MemoryConfig()
        assert config.working_memory_capacity > 0
        assert config.embedding_dim > 0

    def test_memory_config_custom(self):
        """Test custom MemoryConfig values."""
        from zerogmem import MemoryConfig

        config = MemoryConfig(
            working_memory_capacity=50,
            embedding_dim=768,
        )
        assert config.working_memory_capacity == 50
        assert config.embedding_dim == 768
