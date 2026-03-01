# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.1.0] - 2024-03-01

### Added

- Initial open-source release of 0GMem
- **Unified Memory Graph** with four orthogonal views:
  - Temporal Graph using Allen's interval algebra
  - Semantic Graph with embedding-based similarity
  - Causal Graph for cause-effect relationships
  - Entity Graph with negative relation support
- **Memory Hierarchy**:
  - Working Memory with attention-based decay
  - Episodic Memory with lossless compression
  - Semantic Memory with confidence tracking
- **Core Components**:
  - `MemoryManager` - Central orchestrator
  - `Encoder` - Text to memory encoding
  - `Retriever` - Multi-strategy retrieval
  - `QueryAnalyzer` - Query understanding
- **LoCoMo Benchmark** evaluation support
- Position-aware context composition
- Multi-hop graph traversal for complex queries
- Examples and documentation

### Performance

- 3-conversation benchmark: 95.57% accuracy
- 10-conversation benchmark: 80.41% accuracy

[Unreleased]: https://github.com/loganionian/0gmem/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/loganionian/0gmem/releases/tag/v0.1.0
