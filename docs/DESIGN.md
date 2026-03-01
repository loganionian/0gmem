# 0GMem: Zero-Gap Memory System Design

## Vision

0GMem (Zero-Gap Memory) is a next-generation AI memory system designed to achieve SOTA performance on the LoCoMo benchmark by addressing the fundamental limitations of existing systems like EverMemOS. The name reflects our goal of **zero gap** between AI and human-level conversational memory.

## Design Principles

1. **Temporal-First**: Explicit temporal reasoning as a core capability, not an afterthought
2. **Graph-Native**: Knowledge graph as the primary memory substrate, not documents
3. **Hierarchical**: Working → Episodic → Semantic → Meta memory layers
4. **Lossless-by-Default**: Compress intelligently, but never lose retrievable information
5. **Adversarial-Aware**: Track negative facts and contradictions explicitly
6. **Position-Aware**: Combat "lost-in-the-middle" through strategic composition

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              0GMem System                                    │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐    ┌─────────────┐  │
│  │   Ingress   │───▶│  Encoder    │───▶│Consolidator │───▶│  Retriever  │  │
│  │   Gateway   │    │   Layer     │    │   Layer     │    │   Layer     │  │
│  └─────────────┘    └─────────────┘    └─────────────┘    └─────────────┘  │
│         │                 │                  │                  │           │
│         ▼                 ▼                  ▼                  ▼           │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                    Unified Memory Graph (UMG)                        │   │
│  │  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐            │   │
│  │  │ Temporal │  │ Semantic │  │  Causal  │  │  Entity  │            │   │
│  │  │  Graph   │  │  Graph   │  │  Graph   │  │  Graph   │            │   │
│  │  └──────────┘  └──────────┘  └──────────┘  └──────────┘            │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                    │                                        │
│  ┌─────────────────────────────────┴───────────────────────────────────┐   │
│  │                    Memory Hierarchy                                  │   │
│  │  ┌─────────┐   ┌─────────┐   ┌─────────┐   ┌─────────┐             │   │
│  │  │ Working │   │Episodic │   │Semantic │   │  Meta   │             │   │
│  │  │ Memory  │   │ Memory  │   │ Memory  │   │ Memory  │             │   │
│  │  │(recent) │   │(events) │   │ (facts) │   │(schemas)│             │   │
│  │  └─────────┘   └─────────┘   └─────────┘   └─────────┘             │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Core Components

### 1. Unified Memory Graph (UMG)

The UMG is the heart of 0GMem - a multi-relational knowledge graph that represents all memory through four orthogonal graph views:

#### 1.1 Temporal Graph
```python
class TemporalNode:
    id: str
    timestamp: datetime           # When event occurred (event time)
    ingestion_time: datetime      # When recorded (system time) - bitemporal
    duration: Optional[timedelta] # Event duration
    temporal_relations: List[TemporalEdge]  # before, after, during, overlaps

class TemporalEdge:
    source: str
    target: str
    relation: TemporalRelation    # BEFORE, AFTER, DURING, OVERLAPS, MEETS, STARTS, FINISHES
    confidence: float

class TemporalRelation(Enum):
    BEFORE = "before"             # A ends before B starts
    AFTER = "after"               # A starts after B ends
    DURING = "during"             # A occurs within B's timespan
    OVERLAPS = "overlaps"         # A and B partially overlap
    MEETS = "meets"               # A ends exactly when B starts
    STARTS = "starts"             # A starts at same time as B
    FINISHES = "finishes"         # A ends at same time as B
    CONCURRENT = "concurrent"     # A and B occur simultaneously
```

**Key Innovation**: Bitemporal modeling (event time + system time) allows both "when did this happen?" and "when did we learn this?" queries.

#### 1.2 Semantic Graph
```python
class SemanticNode:
    id: str
    content: str                  # The actual memory content
    embedding: np.ndarray         # Dense vector representation
    concepts: List[str]           # Linked concepts/topics
    importance: float             # Attention-weighted importance score

class SemanticEdge:
    source: str
    target: str
    relation: str                 # is_a, part_of, related_to, similar_to
    weight: float
```

#### 1.3 Causal Graph
```python
class CausalNode:
    id: str
    event: str
    preconditions: List[str]      # What must be true for this to happen
    effects: List[str]            # What becomes true after this happens

class CausalEdge:
    cause: str
    effect: str
    strength: float               # Causal strength [0, 1]
    evidence: List[str]           # Supporting memory IDs
```

**Key Innovation**: Explicit causal reasoning enables "why did X happen?" and "what will happen if Y?" queries.

#### 1.4 Entity Graph
```python
class EntityNode:
    id: str
    name: str
    type: EntityType              # PERSON, PLACE, ORGANIZATION, OBJECT, CONCEPT
    attributes: Dict[str, Any]    # Dynamic attribute storage
    aliases: List[str]            # Alternative names

class EntityEdge:
    source: str
    target: str
    relation: str                 # knows, works_at, lives_in, owns, etc.
    temporal_scope: Optional[TimeRange]  # When was this relation valid?
    negated: bool                 # Is this a negative relation?
```

**Key Innovation**: `negated` flag enables explicit negative fact storage for adversarial robustness.

---

### 2. Memory Hierarchy

#### 2.1 Working Memory
```python
class WorkingMemory:
    """
    Active reasoning workspace - analogous to human working memory.
    Holds currently relevant context for ongoing conversation.
    """
    capacity: int = 20            # Limited slots (like human 7±2)
    items: List[MemoryItem]       # Currently active memories
    attention_weights: np.ndarray # Importance of each item
    decay_rate: float = 0.1       # Items decay over time

    def update(self, new_items: List[MemoryItem]) -> None:
        """Add new items, decay old ones, evict lowest attention."""

    def get_context(self, query: str) -> List[MemoryItem]:
        """Retrieve relevant working memory items for query."""
```

**Key Innovation**: Explicit working memory prevents information overload and maintains focus.

#### 2.2 Episodic Memory
```python
class EpisodicMemory:
    """
    Personal history - specific events and experiences.
    Answers "what happened when?"
    """
    episodes: List[Episode]

class Episode:
    id: str
    summary: str                  # High-level description
    detailed_trace: str           # Full conversation trace (lossless)
    participants: List[str]
    location: Optional[str]
    time_range: TimeRange
    emotional_valence: float      # Positive/negative sentiment
    importance: float
    retrieval_count: int          # How often accessed (for consolidation)
```

**Key Innovation**: Keep both summary AND detailed_trace - lossless with intelligent compression.

#### 2.3 Semantic Memory
```python
class SemanticMemory:
    """
    Accumulated knowledge - facts divorced from specific experiences.
    Answers "what do I know about X?"
    """
    facts: Dict[str, Fact]
    concepts: Dict[str, Concept]
    relations: List[Relation]

class Fact:
    id: str
    content: str
    confidence: float             # How certain are we?
    sources: List[str]            # Episode IDs that support this
    contradictions: List[str]     # Episode IDs that contradict this
    first_learned: datetime
    last_confirmed: datetime
    negated: bool                 # Is this explicitly NOT true?
```

**Key Innovation**: Track both supporting evidence AND contradictions for robustness.

#### 2.4 Meta Memory
```python
class MetaMemory:
    """
    Memory about memory - schemas, patterns, and self-knowledge.
    Enables reflection and learning-to-learn.
    """
    user_schemas: Dict[str, UserSchema]    # User behavior patterns
    topic_schemas: Dict[str, TopicSchema]  # Domain knowledge structures
    retrieval_strategies: List[Strategy]   # What works for what queries

class UserSchema:
    user_id: str
    communication_style: str
    typical_topics: List[str]
    preference_patterns: Dict[str, Any]
    temporal_patterns: Dict[str, Any]      # When do they discuss what?
```

---

### 3. Encoder Layer

The encoder transforms raw conversations into structured memory representations.

```python
class EncoderLayer:
    def encode(self, conversation: Conversation) -> EncodingResult:
        """
        Multi-pass encoding pipeline:
        1. Entity extraction and linking
        2. Temporal marker detection
        3. Causal relation extraction
        4. Semantic embedding generation
        5. Importance scoring
        """

    def extract_entities(self, text: str) -> List[Entity]:
        """NER + entity linking to existing graph."""

    def extract_temporal_markers(self, text: str) -> List[TemporalMarker]:
        """Detect timestamps, relative time expressions, durations."""

    def extract_causal_relations(self, text: str) -> List[CausalRelation]:
        """Detect because, therefore, leads to, causes, etc."""

    def compute_importance(self, item: MemoryItem) -> float:
        """
        Attention-based importance scoring:
        - Entity salience
        - Temporal recency
        - Emotional intensity
        - Information density
        - User engagement signals
        """
```

#### Temporal Marker Extraction (Critical for LoCoMo)
```python
class TemporalExtractor:
    """
    Specialized component for temporal reasoning - addresses
    LoCoMo's hardest question type.
    """

    TEMPORAL_PATTERNS = [
        r"(?P<relative>yesterday|today|tomorrow|last week|next month)",
        r"(?P<absolute>\d{4}-\d{2}-\d{2})",
        r"(?P<duration>for \d+ (hours?|days?|weeks?|months?))",
        r"(?P<sequence>before|after|then|first|finally|meanwhile)",
        r"(?P<frequency>always|never|sometimes|usually|every \w+)",
    ]

    def extract(self, text: str, reference_time: datetime) -> TemporalInfo:
        """
        Extract and normalize temporal expressions.
        Resolve relative times against reference_time.
        """

    def build_temporal_graph(self, events: List[Event]) -> TemporalGraph:
        """
        Construct explicit temporal relations between events
        using Allen's interval algebra.
        """
```

---

### 4. Consolidator Layer

Manages the transformation of memory across hierarchy levels.

```python
class ConsolidatorLayer:
    """
    Inspired by memory consolidation in sleep - background process
    that organizes and compresses memories while preserving retrievability.
    """

    def consolidate(self) -> None:
        """
        Periodic consolidation process:
        1. Episodic → Semantic: Extract facts from repeated experiences
        2. Compress old episodic memories (keep summaries, archive details)
        3. Update importance scores based on retrieval patterns
        4. Detect and resolve contradictions
        5. Build/update meta schemas
        """

    def episodic_to_semantic(self, episodes: List[Episode]) -> List[Fact]:
        """
        When similar information appears across multiple episodes,
        extract it as a semantic fact with provenance tracking.
        """

    def detect_contradictions(self) -> List[Contradiction]:
        """
        Find facts that contradict each other.
        Flag for human resolution or confidence downgrade.
        """

    def compress_episode(self, episode: Episode) -> CompressedEpisode:
        """
        Lossless compression: Keep detailed trace in cold storage,
        maintain summary + index for retrieval.
        """
```

#### Intelligent Compression (Lossless-by-Default)
```python
class LosslessCompressor:
    """
    Unlike EverMemOS which may lose details in summarization,
    we compress but maintain retrievability.
    """

    def compress(self, episode: Episode) -> CompressedEpisode:
        return CompressedEpisode(
            summary=self.generate_summary(episode),
            key_entities=self.extract_key_entities(episode),
            key_facts=self.extract_key_facts(episode),
            temporal_markers=self.extract_temporal_markers(episode),
            detailed_trace_ref=self.archive_to_cold_storage(episode),  # Never deleted!
            retrieval_index=self.build_retrieval_index(episode),
        )

    def decompress(self, compressed: CompressedEpisode) -> Episode:
        """Retrieve full episode from cold storage when needed."""
```

---

### 5. Retriever Layer

The most critical component for benchmark performance.

```python
class RetrieverLayer:
    """
    Multi-strategy retrieval with graph traversal and
    position-aware composition.
    """

    def retrieve(self, query: str, context: Context) -> RetrievalResult:
        """
        Pipeline:
        1. Query understanding (intent, entities, temporal scope)
        2. Multi-graph traversal (parallel on 4 graphs)
        3. Working memory integration
        4. Candidate fusion and ranking
        5. Position-aware composition
        6. Confidence scoring
        """
```

#### 5.1 Query Understanding
```python
class QueryUnderstanding:
    def analyze(self, query: str) -> QueryAnalysis:
        return QueryAnalysis(
            intent=self.classify_intent(query),           # factual, temporal, causal, etc.
            entities=self.extract_entities(query),
            temporal_scope=self.extract_temporal_scope(query),
            expected_answer_type=self.predict_answer_type(query),
            reasoning_type=self.classify_reasoning(query), # single-hop, multi-hop, temporal
        )
```

#### 5.2 Graph-Based Multi-Hop Retrieval
```python
class GraphRetriever:
    """
    Key innovation: Graph traversal instead of flat document retrieval.
    Addresses multi-hop reasoning weakness.
    """

    def retrieve_multi_hop(self, query: QueryAnalysis, max_hops: int = 3) -> List[MemoryItem]:
        """
        1. Start from query entities
        2. Traverse relevant edges based on query intent
        3. Collect evidence along paths
        4. Return paths, not just endpoints
        """

        # Start nodes
        start_nodes = self.entity_graph.find_nodes(query.entities)

        # BFS with intent-guided edge selection
        paths = []
        for node in start_nodes:
            paths.extend(self.guided_bfs(
                node,
                max_depth=max_hops,
                edge_filter=self.get_edge_filter(query.intent),
            ))

        return self.paths_to_memories(paths)

    def guided_bfs(self, start: Node, max_depth: int, edge_filter: Callable) -> List[Path]:
        """
        Breadth-first search guided by query intent.
        For temporal queries, prioritize temporal edges.
        For causal queries, prioritize causal edges.
        """
```

#### 5.3 Temporal Retrieval (Critical for LoCoMo)
```python
class TemporalRetriever:
    """
    Specialized retrieval for temporal questions.
    Addresses LoCoMo's hardest question type.
    """

    def retrieve_temporal(self, query: QueryAnalysis) -> List[MemoryItem]:
        """
        Handle temporal queries:
        - "What happened before X?"
        - "When did Y occur?"
        - "What was happening during Z?"
        """

        temporal_scope = query.temporal_scope

        if temporal_scope.type == "point":
            # Find events at specific time
            return self.temporal_graph.events_at(temporal_scope.timestamp)

        elif temporal_scope.type == "range":
            # Find events in time range
            return self.temporal_graph.events_in_range(
                temporal_scope.start,
                temporal_scope.end
            )

        elif temporal_scope.type == "relative":
            # Find events relative to anchor
            anchor_events = self.find_anchor_events(query.entities)
            return self.temporal_graph.events_relative_to(
                anchor_events,
                temporal_scope.relation  # BEFORE, AFTER, DURING
            )

    def temporal_chain_reasoning(self, query: QueryAnalysis) -> List[MemoryItem]:
        """
        For complex temporal questions requiring chain reasoning:
        "What happened after A but before B?"
        """
        constraints = self.parse_temporal_constraints(query)
        candidates = self.temporal_graph.find_constrained(constraints)
        return self.rank_by_temporal_relevance(candidates, query)
```

#### 5.4 Position-Aware Composition
```python
class PositionAwareComposer:
    """
    Combat "lost-in-the-middle" problem.
    Strategically position important information.
    """

    def compose(self, memories: List[MemoryItem], query: str) -> str:
        """
        Compose retrieved memories into prompt, positioning
        most relevant information at beginning AND end.
        """

        # Sort by relevance
        ranked = self.rank_by_relevance(memories, query)

        # Multi-scale composition
        composed = []

        # High relevance at start
        composed.append("## Most Relevant Context")
        composed.extend(self.format_memories(ranked[:3]))

        # Medium relevance in middle (with structure)
        composed.append("## Additional Context")
        for i, mem in enumerate(ranked[3:-3]):
            composed.append(f"[{i+1}] {self.format_memory(mem)}")

        # High relevance repeated at end (reinforcement)
        composed.append("## Key Points (Summary)")
        composed.extend(self.summarize_key_points(ranked[:3]))

        return "\n\n".join(composed)
```

#### 5.5 Confidence Scoring and Provenance
```python
class ConfidenceScorer:
    """
    Track confidence and provenance for adversarial robustness.
    """

    def score(self, memory: MemoryItem, query: QueryAnalysis) -> ConfidenceScore:
        return ConfidenceScore(
            relevance=self.compute_relevance(memory, query),
            recency=self.compute_recency_score(memory),
            corroboration=self.compute_corroboration(memory),  # Multiple sources?
            contradiction_risk=self.compute_contradiction_risk(memory),
            provenance=self.get_provenance_chain(memory),
        )

    def compute_corroboration(self, memory: MemoryItem) -> float:
        """
        Higher score if multiple independent sources confirm this.
        """
        supporting_episodes = self.get_supporting_episodes(memory)
        return min(1.0, len(supporting_episodes) / 3)  # Max out at 3 sources

    def compute_contradiction_risk(self, memory: MemoryItem) -> float:
        """
        Check if any stored memories contradict this.
        """
        contradictions = self.find_contradictions(memory)
        return len(contradictions) / (len(contradictions) + 1)
```

---

### 6. Negative Fact Storage

```python
class NegativeFactStore:
    """
    Explicitly store what is NOT true.
    Critical for adversarial question robustness.
    """

    def store_negative(self, fact: str, evidence: List[str]) -> None:
        """
        Store explicit negation:
        - "User does NOT like coffee" (preference negation)
        - "Event X did NOT happen on Tuesday" (temporal negation)
        - "Person A does NOT know Person B" (relation negation)
        """

    def check_for_negation(self, query: str) -> Optional[NegativeFact]:
        """
        Before answering, check if we have explicit negation stored.
        """

    def detect_and_store_negations(self, conversation: Conversation) -> List[NegativeFact]:
        """
        Automatically detect negations in conversation:
        - "I don't like X"
        - "That's not true"
        - "Actually, it was Y, not X"
        """
```

---

## Implementation Roadmap

### Phase 1: Core Infrastructure (Week 1-2)
- [ ] Unified Memory Graph data structures
- [ ] Basic encoder with entity extraction
- [ ] Simple embedding-based retrieval
- [ ] LoCoMo evaluation harness integration

### Phase 2: Temporal Reasoning (Week 3-4)
- [ ] Temporal marker extraction
- [ ] Temporal graph construction
- [ ] Allen's interval algebra implementation
- [ ] Temporal retrieval strategies

### Phase 3: Multi-Hop & Graph Retrieval (Week 5-6)
- [ ] Graph-based multi-hop traversal
- [ ] Causal graph construction
- [ ] Intent-guided edge selection
- [ ] Path-based evidence collection

### Phase 4: Memory Hierarchy (Week 7-8)
- [ ] Working memory implementation
- [ ] Episodic → Semantic consolidation
- [ ] Lossless compression
- [ ] Meta memory schemas

### Phase 5: Robustness (Week 9-10)
- [ ] Negative fact storage
- [ ] Contradiction detection
- [ ] Confidence scoring
- [ ] Position-aware composition

### Phase 6: Optimization (Week 11-12)
- [ ] Parallel graph traversal
- [ ] Caching strategies
- [ ] Speculative prefetch
- [ ] Benchmark optimization

---

## Expected Performance Improvements

| Component | Target Improvement | LoCoMo Question Type |
|-----------|-------------------|---------------------|
| Temporal Graph | +15-20% | Temporal |
| Multi-hop Retrieval | +10-15% | Multi-hop |
| Negative Facts | +5-10% | Adversarial |
| Position-Aware Composition | +3-5% | All |
| Lossless Consolidation | +3-5% | Single-hop (recall) |
| Working Memory | +2-3% | All |

**Target**: 95%+ overall accuracy on LoCoMo (vs EverMemOS 92.3%)

---

## Technology Stack

- **Language**: Python 3.10+
- **Graph Database**: Neo4j or NetworkX (for prototyping)
- **Vector Store**: FAISS or Qdrant
- **LLM**: Claude/GPT-4 for extraction, smaller model for classification
- **Embedding**: text-embedding-3-large or similar
- **Framework**: LangChain/LlamaIndex for orchestration

---

## Conclusion

0GMem addresses the fundamental limitations of EverMemOS through:
1. **Temporal-first design** for LoCoMo's hardest questions
2. **Graph-native architecture** for multi-hop reasoning
3. **Lossless compression** to never lose retrievable details
4. **Negative fact storage** for adversarial robustness
5. **Position-aware composition** to combat lost-in-the-middle

The system maintains EverMemOS's strengths (efficient token usage, agentic retrieval) while adding the capabilities needed to achieve SOTA on LoCoMo.
