# 0GMem Improvement Plan: 61% → 95% LoCoMo Accuracy

## Executive Summary

**Current State:** 61.37% accuracy on LoCoMo benchmark
**Target:** 95% accuracy (exceeding EverMemOS's 92.3%)
**Gap:** 33.63 percentage points (164 more correct answers needed out of 497)

---

## Part 1: Root Cause Analysis

### 1.1 Performance Gap by Category

| Category | Current | Target (95%) | Gap | Questions to Fix |
|----------|---------|--------------|-----|------------------|
| Adversarial | 84.8% | 95% | 10.2% | 11 |
| Temporal | 60.0% | 95% | 35% | 31 |
| Multi-hop | 57.1% | 95% | 37.9% | 7 |
| Single-hop | 56.8% | 95% | 38.2% | 28 |
| Open-domain | 51.0% | 95% | 44% | 87 |

**Critical Finding:** Open-domain is the largest gap (87 questions to fix)

### 1.2 Failure Pattern Analysis

```
Total Failures: 192 questions

1. "None" when answer exists: 49 failures (25.5%)
   - System returns "None" but answer is in the data
   - Root cause: Retrieval fails to find relevant content

2. Wrong answer (F1 < 0.3): 95 failures (49.5%)
   - System returns something but it's wrong
   - Root cause: Wrong memory retrieved or wrong inference

3. Partial match (F1 0.3-0.7): 28 failures (14.6%)
   - Answer is close but not exact
   - Root cause: Answer extraction/formatting issues

4. Correct but different format: ~20 failures (10.4%)
   - Answer is semantically correct but scored as wrong
   - Root cause: Evaluation strictness
```

### 1.3 EverMemOS vs 0GMem Architecture Comparison

| Feature | EverMemOS | 0GMem (Current) | Gap |
|---------|-----------|-----------------|-----|
| Memory Structure | MemCells → MemScenes (hierarchical) | Flat documents | Critical |
| Consolidation | 3-phase (encode, consolidate, retrieve) | Direct storage | Critical |
| Profile Building | Progressive aggregation across scenes | Basic fact extraction | High |
| Retrieval | Agentic multi-step reasoning | Hybrid BM25+semantic | High |
| Memory Types | Episodes, facts, preferences, relations | Single type | Medium |
| Temporal Tracking | MemCell timestamps + foresight | Session dates | Medium |
| Attention Filter | "Precise forgetting" (noise reduction) | None | Medium |

---

## Part 2: Architectural Changes Required

### 2.1 Phase 1: MemCell/MemScene Hierarchical Structure (Expected: +10-15%)

**Problem:** Flat document storage loses context relationships.

**Solution:** Implement two-tier memory organization:

```python
@dataclass
class MemCell:
    """Atomic memory unit - one fact, event, or statement."""
    id: str
    content: str
    cell_type: str  # 'episode', 'fact', 'preference', 'relation'
    entity: str
    timestamp: datetime
    session_id: str
    confidence: float
    metadata: Dict[str, Any]

@dataclass
class MemScene:
    """Collection of related MemCells forming coherent context."""
    id: str
    cells: List[MemCell]
    scene_type: str  # 'activity', 'conversation_topic', 'life_event'
    entities: Set[str]
    time_range: Tuple[datetime, datetime]
    summary: str  # LLM-generated scene summary
```

**Implementation:**
1. Create `src/zerogmem/memory/memcell.py` and `src/zerogmem/memory/memscene.py`
2. During ingestion, extract MemCells from each message
3. Group related MemCells into MemScenes (same topic, same session, same entities)
4. Generate MemScene summaries for efficient retrieval

### 2.2 Phase 2: Consolidation Pipeline (Expected: +8-12%)

**Problem:** No memory consolidation - raw storage loses patterns.

**Solution:** Implement 3-phase memory lifecycle:

```python
class ConsolidationPipeline:
    """EverMemOS-inspired memory consolidation."""

    def phase1_encode(self, message: str, speaker: str, session_date: str) -> List[MemCell]:
        """Extract atomic facts, events, preferences from message."""
        pass

    def phase2_consolidate(self, cells: List[MemCell]) -> List[MemScene]:
        """Group cells into scenes, update profiles, detect patterns."""
        # 1. Cluster cells by topic/entity/time
        # 2. Generate scene summaries
        # 3. Update entity profiles with new facts
        # 4. Detect recurring patterns
        pass

    def phase3_prepare_retrieval(self, query: str) -> RetrievalContext:
        """Prepare optimized retrieval context."""
        # 1. Identify relevant scenes
        # 2. Rank cells within scenes
        # 3. Build minimal sufficient context
        pass
```

### 2.3 Phase 3: Progressive Profile Building (Expected: +8-10%)

**Problem:** Basic profile is missing many attributes.

**Solution:** Build comprehensive entity profiles progressively:

```python
@dataclass
class EntityProfile:
    """Comprehensive profile built from MemScenes."""
    entity_name: str

    # Core attributes
    demographics: Dict[str, str]  # age, location, occupation, etc.
    relationships: Dict[str, str]  # friend: Gina, husband: Tom
    preferences: Dict[str, List[str]]  # likes: [camping, art], dislikes: [theme parks]

    # Behavioral patterns
    activities: List[ActivityRecord]  # What they do, when, how often
    life_events: List[LifeEvent]  # Major events with dates
    goals: List[str]  # Future plans, aspirations

    # Inferred attributes
    personality_traits: List[str]  # Inferred from behavior
    political_leaning: Optional[str]  # Inferred from statements
    religious_affiliation: Optional[str]
    financial_status: Optional[str]  # Inferred from lifestyle

    # Meta
    last_updated: datetime
    confidence_scores: Dict[str, float]
```

**Implementation:**
1. After each MemScene consolidation, update relevant profiles
2. Use LLM to infer implicit attributes from explicit statements
3. Track confidence scores for each attribute
4. Enable direct profile queries for single-hop questions

### 2.4 Phase 4: Agentic Retrieval (Expected: +10-15%)

**Problem:** Current retrieval is single-pass, misses complex queries.

**Solution:** Implement multi-step agentic retrieval:

```python
class AgenticRetriever:
    """Multi-step reasoning retrieval for complex queries."""

    def retrieve(self, query: str, entity: str) -> RetrievalResult:
        # Step 1: Query Analysis
        query_plan = self.analyze_query(query)
        # Returns: {type: 'multi_hop', steps: ['find X', 'find Y', 'combine']}

        # Step 2: Execute retrieval plan
        results = []
        for step in query_plan.steps:
            step_result = self.execute_step(step, results)
            results.append(step_result)

        # Step 3: Synthesize context
        context = self.synthesize(results, query_plan)
        return context

    def execute_step(self, step: str, previous: List) -> StepResult:
        # Dynamic sub-queries based on previous results
        if step.type == 'find_entity_attribute':
            return self.search_profile(step.entity, step.attribute)
        elif step.type == 'find_event':
            return self.search_memscenes(step.event_keywords)
        elif step.type == 'infer':
            return self.llm_infer(step.premise, previous)
```

### 2.5 Phase 5: Attention Filter / Noise Reduction (Expected: +5-8%)

**Problem:** Too much irrelevant context dilutes attention.

**Solution:** Implement "precise forgetting":

```python
class AttentionFilter:
    """Filter context to only essential information."""

    def filter_context(self, query: str, retrieved: List[MemCell]) -> List[MemCell]:
        # 1. Score each cell's relevance to query
        scored = [(cell, self.relevance_score(query, cell)) for cell in retrieved]

        # 2. Remove low-relevance cells (noise)
        filtered = [c for c, s in scored if s > self.threshold]

        # 3. Ensure diversity (don't repeat similar info)
        deduped = self.deduplicate_similar(filtered)

        # 4. Limit total context size
        return self.truncate_to_budget(deduped, max_tokens=2000)
```

---

## Part 3: Specific Bug Fixes

### 3.1 Fix "None" Over-answering (49 failures)

**Root Cause:** Current prompt is too aggressive about returning "None".

**Fix 1:** Lower "None" threshold for non-adversarial questions
```python
# In _build_qa_prompt():
if question.category != "adversarial":
    prompt += """
    IMPORTANT: Only answer "None" if you are CERTAIN the information doesn't exist.
    If there's ANY relevant information, extract and return it.
    Prefer partial answers over "None".
    """
```

**Fix 2:** Add fallback retrieval before returning "None"
```python
def answer_question(self, question):
    answer = self.primary_answer(question)
    if answer.lower() == "none":
        # Try broader retrieval
        expanded_answer = self.fallback_retrieval(question)
        if expanded_answer:
            return expanded_answer
    return answer
```

### 3.2 Fix Secondary Entity Retrieval (Jon, Jean, Maria issues)

**Root Cause:** Retrieval optimized for Caroline/Melanie, not other entities.

**Fix:** Entity-specific retrieval boost
```python
def retrieve_for_entity(self, query: str, entity: str):
    # Always include entity name in query
    boosted_query = f"{entity}: {query}"

    # Search in entity's profile first
    profile_results = self.search_profile(entity)

    # Then search MemScenes containing entity
    scene_results = self.search_scenes_by_entity(entity)

    # Merge and rank
    return self.merge_results(profile_results, scene_results)
```

### 3.3 Fix Temporal Date Mismatches

**Root Cause:** EventDateIndex returns wrong session dates.

**Fix:** Stricter event-date matching
```python
def lookup_event(self, entity: str, event_keywords: List[str], month: str = None):
    candidates = self.by_entity.get(entity, [])

    # Require ALL keywords to match, not just one
    matches = []
    for entry in candidates:
        keyword_overlap = len(set(event_keywords) & entry.keywords)
        if keyword_overlap >= len(event_keywords) * 0.7:  # 70% match
            matches.append((entry, keyword_overlap))

    # Sort by match quality
    matches.sort(key=lambda x: x[1], reverse=True)

    # Filter by month if specified
    if month:
        matches = [m for m in matches if month.lower() in m[0].session_date.lower()]

    return matches[0] if matches else None
```

### 3.4 Fix Answer Format Mismatches

**Root Cause:** Different wording marked as wrong (e.g., "National park" vs "National park; she likes outdoors")

**Fix:** Normalize answers before comparison
```python
def normalize_answer(self, answer: str) -> str:
    # Remove explanations after semicolon
    if ';' in answer:
        answer = answer.split(';')[0].strip()

    # Remove filler phrases
    answer = re.sub(r'^(I think |It seems |Based on |According to )', '', answer)

    # Normalize casing
    return answer.strip().lower()
```

---

## Part 4: Implementation Roadmap

### Sprint 1 (Week 1-2): Foundation
1. [ ] Implement MemCell and MemScene data structures
2. [ ] Create ConsolidationPipeline skeleton
3. [ ] Update ingestion to produce MemCells
4. [ ] Write unit tests

### Sprint 2 (Week 3-4): Consolidation
1. [ ] Implement MemCell → MemScene grouping
2. [ ] Add LLM-based scene summarization
3. [ ] Implement progressive profile building
4. [ ] Update profiles during ingestion

### Sprint 3 (Week 5-6): Retrieval
1. [ ] Implement AgenticRetriever with query planning
2. [ ] Add multi-step retrieval execution
3. [ ] Implement AttentionFilter
4. [ ] Integrate with answer generation

### Sprint 4 (Week 7-8): Bug Fixes & Tuning
1. [ ] Fix "None" over-answering
2. [ ] Improve secondary entity retrieval
3. [ ] Fix temporal date matching
4. [ ] Normalize answer formatting
5. [ ] Extensive evaluation and tuning

---

## Part 5: Expected Impact

| Phase | Change | Expected Gain | Cumulative |
|-------|--------|---------------|------------|
| Current | Baseline | 61.37% | 61.37% |
| Phase 1 | MemCell/MemScene | +10-15% | 71-76% |
| Phase 2 | Consolidation | +8-12% | 79-88% |
| Phase 3 | Progressive Profiles | +8-10% | 87-98% |
| Phase 4 | Agentic Retrieval | +5-8% | 92-100% |
| Phase 5 | Attention Filter | +3-5% | 95-100% |
| Bug Fixes | Specific issues | +2-5% | 97-100% |

**Conservative Estimate:** 92-95%
**Optimistic Estimate:** 97-100%

---

## Part 6: Key Files to Modify/Create

### New Files:
- `src/zerogmem/memory/memcell.py` - MemCell data structure
- `src/zerogmem/memory/memscene.py` - MemScene data structure
- `src/zerogmem/memory/consolidation.py` - Consolidation pipeline
- `src/zerogmem/retriever/agentic_retriever.py` - Multi-step retrieval
- `src/zerogmem/retriever/attention_filter.py` - Noise reduction

### Files to Modify:
- `src/zerogmem/evaluation/locomo.py` - Main evaluator
- `src/zerogmem/encoder/llm_fact_extractor.py` - Profile building
- `src/zerogmem/encoder/event_date_index.py` - Temporal fixes
- `src/zerogmem/retriever/retriever.py` - Retrieval integration

---

## References

- [EverMemOS GitHub](https://github.com/EverMind-AI/EverMemOS)
- [EverMemOS Paper](https://arxiv.org/pdf/2601.02163)
- [LoCoMo Benchmark](https://snap-research.github.io/locomo/)
