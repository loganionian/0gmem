# 0GMem: Zero Gravity Memory

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)

A next-generation AI memory system designed to achieve state-of-the-art performance on the LoCoMo benchmark for long-term conversational memory.

## Key Innovations

0GMem addresses the fundamental limitations of existing memory systems through:

### 1. Unified Memory Graph (UMG)
Four orthogonal graph views for comprehensive memory representation:
- **Temporal Graph**: Allen's interval algebra for precise temporal reasoning
- **Semantic Graph**: Embedding-based similarity with concept relationships
- **Causal Graph**: Cause-effect tracking for "why" questions
- **Entity Graph**: Entity relationships with **negative relation support**

### 2. Memory Hierarchy
Inspired by cognitive science:
- **Working Memory**: Active reasoning workspace with attention-based decay
- **Episodic Memory**: Personal history with lossless compression
- **Semantic Memory**: Accumulated facts with confidence and contradiction tracking

### 3. Advanced Retrieval
- **Multi-hop graph traversal** for complex reasoning
- **Temporal chain reasoning** for time-based questions
- **Position-aware composition** to combat "lost-in-the-middle"
- **Negation checking** for adversarial robustness

## Installation

```bash
# Clone the repository
git clone https://github.com/loganionian/0gmem.git
cd 0gmem

# Install dependencies
pip install -e .

# For development
pip install -e ".[dev]"

# For evaluation
pip install -e ".[eval]"
```

## Quick Start

```python
from zerogmem import MemoryManager, Encoder, Retriever

# Initialize components
memory = MemoryManager()
encoder = Encoder()
memory.set_embedding_function(encoder.get_embedding)
retriever = Retriever(memory, embedding_fn=encoder.get_embedding)

# Start a conversation session
memory.start_session()

# Add messages
memory.add_message("Alice", "I love hiking in the mountains.")
memory.add_message("Bob", "Which mountains have you visited?")
memory.add_message("Alice", "I've been to the Alps last summer and Rocky Mountains in 2022.")

# End session
memory.end_session()

# Query the memory
result = retriever.retrieve("When did Alice visit the Alps?")
print(result.composed_context)
```

## API Reference

### Core Classes

| Class | Description |
|-------|-------------|
| `MemoryManager` | Central orchestrator for memory operations |
| `Encoder` | Converts text to memory representations |
| `Retriever` | Queries memories with multi-strategy retrieval |

### Configuration

| Class | Description |
|-------|-------------|
| `MemoryConfig` | Configure memory capacity, decay rates |
| `EncoderConfig` | Configure embedding model, extraction options |
| `RetrieverConfig` | Configure retrieval strategies, weights |

### Data Types

| Class | Description |
|-------|-------------|
| `RetrievalResult` | Single retrieval result with score and source |
| `RetrievalResponse` | Complete retrieval response with context |
| `QueryAnalysis` | Query understanding and intent classification |

## Running LoCoMo Evaluation

```bash
# Download/create sample data
python scripts/download_locomo.py --sample-only

# Run evaluation (without LLM)
python scripts/run_evaluation.py --data-path data/locomo/sample_locomo.json

# Run evaluation with LLM (requires OPENAI_API_KEY)
export OPENAI_API_KEY="your-key-here"
python scripts/run_evaluation.py --data-path data/locomo/sample_locomo.json --use-llm
```

## Architecture

```
0GMem Architecture
==================

┌─────────────────────────────────────────────────────────────────┐
│                         0GMem System                             │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  ┌──────────┐    ┌──────────┐    ┌──────────┐    ┌──────────┐  │
│  │ Encoder  │───▶│ Memory   │───▶│Consolidate│───▶│ Retriever│  │
│  │ Layer    │    │ Manager  │    │  Layer   │    │  Layer   │  │
│  └──────────┘    └──────────┘    └──────────┘    └──────────┘  │
│       │               │               │               │         │
│       ▼               ▼               ▼               ▼         │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │                 Unified Memory Graph                     │   │
│  │  ┌────────┐  ┌────────┐  ┌────────┐  ┌────────┐        │   │
│  │  │Temporal│  │Semantic│  │ Causal │  │ Entity │        │   │
│  │  │ Graph  │  │ Graph  │  │ Graph  │  │ Graph  │        │   │
│  │  └────────┘  └────────┘  └────────┘  └────────┘        │   │
│  └─────────────────────────────────────────────────────────┘   │
│                                                                  │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │                   Memory Hierarchy                       │   │
│  │  ┌─────────┐   ┌─────────┐   ┌─────────┐               │   │
│  │  │ Working │   │Episodic │   │Semantic │               │   │
│  │  │ Memory  │   │ Memory  │   │ Memory  │               │   │
│  │  └─────────┘   └─────────┘   └─────────┘               │   │
│  └─────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────┘
```

## Performance

Current LoCoMo benchmark results:

| Metric | Score |
|--------|-------|
| 3-conversation | 95.57% |
| 10-conversation | 80.41% |

## Project Structure

```
0gmem/
├── src/zerogmem/
│   ├── graph/              # Unified Memory Graph
│   │   ├── temporal.py     # Temporal reasoning
│   │   ├── semantic.py     # Semantic similarity
│   │   ├── causal.py       # Causal reasoning
│   │   ├── entity.py       # Entity relationships
│   │   └── unified.py      # Combined graph
│   ├── memory/             # Memory hierarchy
│   │   ├── working.py      # Working memory
│   │   ├── episodic.py     # Episodic memory
│   │   ├── semantic.py     # Semantic facts
│   │   └── manager.py      # Memory orchestration
│   ├── encoder/            # Memory encoding
│   │   ├── encoder.py      # Main encoder
│   │   ├── entity_extractor.py
│   │   └── temporal_extractor.py
│   ├── retriever/          # Memory retrieval
│   │   ├── retriever.py    # Main retriever
│   │   └── query_analyzer.py
│   └── evaluation/         # Benchmarking
│       └── locomo.py       # LoCoMo evaluator
├── examples/               # Usage examples
├── tests/                  # Test suite
├── docs/                   # Documentation
└── scripts/                # Utility scripts
```

## Key Differentiators from EverMemOS

| Aspect | EverMemOS | 0GMem |
|--------|-----------|-------|
| Temporal Reasoning | Timestamp-based | Allen's Interval Algebra |
| Graph Structure | Flat documents | Multi-graph (4 views) |
| Multi-hop | Multiple LLM calls | Graph traversal |
| Negations | Not explicit | First-class support |
| Consolidation | LLM summarization (lossy) | Lossless with indexing |
| Context Composition | Linear | Position-aware |

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for development setup and guidelines.

## References

- [EverMemOS Paper](https://arxiv.org/abs/2601.02163)
- [LoCoMo Benchmark](https://snap-research.github.io/locomo/)
- [LoCoMo Paper (ACL 2024)](https://arxiv.org/abs/2402.17753)

## License

MIT License - see [LICENSE](LICENSE) for details.
