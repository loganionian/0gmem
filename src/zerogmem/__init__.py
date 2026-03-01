"""
0GMem (Zero Gravity Memory): A next-generation AI memory system
designed to achieve SOTA performance on the LoCoMo benchmark.

Key innovations:
- Temporal-first design with explicit temporal reasoning
- Graph-native architecture for multi-hop reasoning
- Hierarchical memory (working -> episodic -> semantic -> meta)
- Negative fact storage for adversarial robustness
- Position-aware composition to combat "lost-in-the-middle"

Quick Start:
    from zerogmem import MemoryManager, Encoder, Retriever

    memory = MemoryManager()
    encoder = Encoder()
    memory.set_embedding_function(encoder.get_embedding)
    retriever = Retriever(memory, embedding_fn=encoder.get_embedding)

    memory.start_session()
    memory.add_message("Alice", "I love hiking in the mountains.")
    memory.end_session()

    result = retriever.retrieve("What does Alice enjoy?")
"""

__version__ = "0.1.0"
__author__ = "0G Labs"

# Core classes
from zerogmem.memory.manager import MemoryManager, MemoryConfig
from zerogmem.encoder.encoder import Encoder, EncoderConfig
from zerogmem.retriever.retriever import (
    Retriever,
    RetrieverConfig,
    RetrievalResult,
    RetrievalResponse,
)
from zerogmem.retriever.query_analyzer import QueryAnalyzer, QueryAnalysis

__all__ = [
    # Core orchestrators
    "MemoryManager",
    "Encoder",
    "Retriever",
    # Configuration
    "MemoryConfig",
    "EncoderConfig",
    "RetrieverConfig",
    # Data types
    "RetrievalResult",
    "RetrievalResponse",
    "QueryAnalysis",
    "QueryAnalyzer",
    # Meta
    "__version__",
]
