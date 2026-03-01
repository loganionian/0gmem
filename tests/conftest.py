"""
Shared pytest fixtures for 0GMem tests.
"""

import pytest
import numpy as np
from typing import Callable

from zerogmem import MemoryManager, Encoder, Retriever, MemoryConfig


@pytest.fixture
def mock_embedding_fn() -> Callable[[str], np.ndarray]:
    """Create a mock embedding function for testing without API calls."""
    def embed(text: str) -> np.ndarray:
        # Create deterministic embeddings based on text hash
        np.random.seed(hash(text) % (2**32))
        return np.random.randn(1536).astype(np.float32)
    return embed


@pytest.fixture
def memory_config() -> MemoryConfig:
    """Create a memory configuration for testing."""
    return MemoryConfig(
        working_memory_capacity=10,
        working_memory_decay_rate=0.1,
        embedding_dim=1536,
    )


@pytest.fixture
def memory_manager(memory_config: MemoryConfig) -> MemoryManager:
    """Create a MemoryManager instance for testing."""
    return MemoryManager(config=memory_config)


@pytest.fixture
def memory_with_embeddings(memory_manager: MemoryManager, mock_embedding_fn) -> MemoryManager:
    """Create a MemoryManager with mock embedding function."""
    memory_manager.set_embedding_function(mock_embedding_fn)
    return memory_manager


@pytest.fixture
def populated_memory(memory_with_embeddings: MemoryManager) -> MemoryManager:
    """Create a MemoryManager populated with test data."""
    memory = memory_with_embeddings

    # Add a test conversation
    memory.start_session()
    memory.add_message("Alice", "I love hiking in the mountains.")
    memory.add_message("Bob", "Which mountains have you visited?")
    memory.add_message("Alice", "I went to the Alps last summer. The Matterhorn was incredible!")
    memory.add_message("Bob", "That sounds amazing!")
    memory.end_session()

    return memory


@pytest.fixture
def retriever(populated_memory: MemoryManager, mock_embedding_fn) -> Retriever:
    """Create a Retriever instance with populated memory."""
    return Retriever(populated_memory, embedding_fn=mock_embedding_fn)
