# 0GMem: Zero Gravity Memory

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)

A next-generation AI memory system designed to achieve state-of-the-art performance on the LoCoMo benchmark for long-term conversational memory.

## Why 0GMem?

Most AI memory systems treat memories as flat text chunks in a vector store — they embed, retrieve, and hope for the best. This works for simple recall but falls apart when conversations grow long and questions get harder: *"When did Alice visit the Alps?"*, *"What does Bob NOT like?"*, *"Who did Alice meet after her trip to Japan?"*

0GMem takes a fundamentally different approach: **structure at write time, intelligence at read time.**

### The Problem with Flat Memory

| Challenge | Flat Vector Store | 0GMem |
|-----------|------------------|-------|
| "What does she NOT like?" | Retrieves mentions of "like" — returns both likes and dislikes, often hallucinating | Stores negations as first-class facts; retrieves the correct polarity |
| "When did X happen?" | Finds the right event but returns the wrong session's date | Event-Date Index resolves dates at ingestion, not retrieval |
| "Who did A meet after B?" | Single-hop retrieval can't chain temporal + entity reasoning | Multi-graph BFS traverses entity, temporal, and semantic edges simultaneously |
| Long conversations (900+ messages) | Retrieves too much — LLM accuracy degrades from context noise | Attention filter performs "precise forgetting," the single biggest accuracy driver (+5% on 10-conv) |
| "Did she say X or Y?" | No contradiction tracking; LLM guesses | Entity graph tracks contradictions and negative relations explicitly |

### Design Principles

- **Encode structure, not just text.** Every message is decomposed into entities, temporal anchors, causal links, and negations at ingestion time — not deferred to retrieval.
- **Multiple views of the same memory.** Four orthogonal graphs (Temporal, Semantic, Causal, Entity) capture different dimensions of meaning, enabling multi-hop reasoning across all of them.
- **Cognitive-science-inspired hierarchy.** Working memory (attention-decayed scratchpad), episodic memory (lossless conversation storage), and semantic memory (accumulated facts with confidence tracking) mirror how human memory actually works.
- **Precise forgetting matters as much as precise remembering.** The attention filter removes redundant and low-relevance context before it reaches the LLM — over-retrieval actively hurts accuracy.
- **Query-aware retrieval.** Every query is classified by intent, reasoning type, and temporal scope before retrieval begins. A temporal question activates different strategies than an adversarial or multi-hop question.

### How It Compares

| | Mem0 | Zep | MemGPT/Letta | **0GMem** |
|---|---|---|---|---|
| Memory structure | Flat facts in vector store | Knowledge graph | Agent-managed paging | **Four orthogonal graphs + three-tier hierarchy** |
| Temporal reasoning | None | Basic | None | **Allen's Interval Algebra (13 relations) + bitemporal modeling** |
| Negation handling | None | None | None | **First-class negation storage and retrieval** |
| Multi-hop reasoning | Single retrieval | Entity traversal | Agent decides | **Simultaneous BFS across entity, temporal, and semantic graphs** |
| Context quality | Top-k similarity | Top-k similarity | Agent-selected | **Attention-filtered with redundancy removal and diversity enforcement** |
| LoCoMo accuracy | 66.9–68.5% | 58–75% | 48–74% | **80.4–95.6%** |

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

## Claude Code Integration

0GMem can be used as an MCP server to give Claude Code persistent, intelligent memory:

```bash
# Install and add to Claude Code
pip install -e .
python -m spacy download en_core_web_sm
claude mcp add --transport stdio 0gmem -- python -m zerogmem.mcp_server

# Verify
claude mcp list
```

Once configured, Claude Code gains access to memory tools:
- `store_memory` - Remember important information
- `retrieve_memories` - Recall relevant context
- `search_memories_by_entity` - Find info about people/places/things
- `search_memories_by_time` - Find memories from specific times

See [docs/MCP_SERVER.md](docs/MCP_SERVER.md) for detailed setup and usage.

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

### LoCoMo Benchmark Results

The [LoCoMo benchmark](https://snap-research.github.io/locomo/) evaluates long-term conversational memory across multi-session dialogues.

**0GMem Results:**

| Subset | Accuracy |
|--------|----------|
| 3-conversation | 95.57% |
| 10-conversation | 80.41% |

### Comparison with Other Systems

Based on published results from various sources:

| System | Score | Notes |
|--------|-------|-------|
| Human Performance | 87.9 F1 | Upper bound ([LoCoMo Paper](https://arxiv.org/abs/2402.17753)) |
| GPT-4-turbo (4K) | ~32 F1 | Baseline LLM |
| GPT-3.5-turbo-16K | 37.8 F1 | Extended context window |
| Best RAG Baseline | 41.4 F1 | Retrieval-augmented generation |
| MemGPT/Letta | 48-74% | Varies by configuration ([Letta Blog](https://www.letta.com/blog/benchmarking-ai-agent-memory)) |
| OpenAI Memory | 52.9% | Built-in memory feature |
| Zep | 58-75% | Results disputed across studies |
| Mem0 | 66.9-68.5% | Graph-enhanced variant ([Mem0 Research](https://mem0.ai/research)) |

*Note: Metrics vary across studies (F1 vs accuracy, different evaluation protocols). Direct comparisons should be interpreted with caution.*

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

## Key Architectural Features

| Feature | 0GMem Approach |
|---------|----------------|
| Temporal Reasoning | Allen's Interval Algebra for precise time relationships |
| Graph Structure | Multi-graph with 4 orthogonal views |
| Multi-hop Reasoning | Graph traversal (no iterative LLM calls) |
| Negation Handling | First-class support for contradictions |
| Memory Consolidation | Lossless compression with semantic indexing |
| Context Composition | Position-aware to combat "lost-in-the-middle" |

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for development setup and guidelines.

## References

- [LoCoMo Benchmark](https://snap-research.github.io/locomo/) - Long-term conversational memory evaluation
- [LoCoMo Paper (ACL 2024)](https://arxiv.org/abs/2402.17753) - "Evaluating Very Long-Term Conversational Memory of LLM Agents"

## License

MIT License - see [LICENSE](LICENSE) for details.
