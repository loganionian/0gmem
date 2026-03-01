# Optimization Strategy: v82+ Roadmap

## Current State Analysis

### Performance Summary
| Metric | Value | Target | Gap |
|--------|-------|--------|-----|
| **10-conv accuracy** | 80.51% (1599/1986) | 95% (1886/1986) | **287 more correct** |
| **3-conv accuracy** | 95.57% (475/497) | 95% | ✅ Achieved |

### Category Breakdown (10-conv)
| Category | Correct | Total | Accuracy | Gap to 95% | Priority |
|----------|---------|-------|----------|------------|----------|
| Multi-hop | 55 | 96 | 57.29% | 36 | HIGH |
| Single-hop | 195 | 282 | 69.15% | 73 | HIGH |
| Open-domain | 641 | 841 | 76.22% | 158 | CRITICAL |
| Temporal | 267 | 321 | 83.18% | 38 | MEDIUM |
| Adversarial | 441 | 446 | 98.88% | 0 | - |

### Conversation Complexity Analysis
| Conversation | Messages | Error Rate | Notes |
|--------------|----------|------------|-------|
| conv-26 | 622 | 1.5% | Best performer |
| conv-30 | 557 | 5.7% | Good |
| conv-41 | 1019 | 8.3% | Good despite length |
| conv-44 | 980 | 18.6% | Problem |
| conv-49 | 774 | 22.1% | Problem |
| conv-47 | 988 | 24.4% | Problem |
| conv-48 | 1002 | 24.0% | Problem |
| conv-43 | 976 | 26.0% | Severe |
| conv-50 | 853 | 28.5% | Severe |
| conv-42 | 924 | 29.2% | Worst |

**Key Insight**: conv-41 (1019 msgs, 8.3%) vs conv-42 (924 msgs, 29.2%) - length isn't the only factor.

### Root Cause Classification
- **92.3%** Wrong fact/inference (retrieves but wrong answer)
- **4.4%** Retrieval failures (nothing found)
- **3.2%** Partial matches

**Critical Insight**: The problem is NOT retrieval - it's selecting the WRONG information from retrieved context.

---

## What's Been Tried (Lessons Learned)

### ✅ Successful Techniques
1. **Attention Filter (v74)**: +5 on 10-conv - "precise forgetting" works
2. **Cross-entity detection (v68e)**: Handles "did X and Y both" patterns
3. **Entity-isolated retrieval**: Reduces cross-entity confusion

### ❌ Failed Techniques
1. **Semantic expansion (v71-73)**: Helps 3-conv, hurts 10-conv (false positives)
2. **World knowledge inference (v78)**: No improvement
3. **Utility scoring (v81)**: Static heuristics don't capture "usefulness"
4. **City-to-state patterns (v79)**: Too broad, causes noise

### 🔄 Pattern Identified
- Techniques that add MORE context → hurt long conversations
- Techniques that FILTER context → help long conversations

---

## Optimization Strategy v82+

### Phase 1: Conversation-Adaptive Filtering (v82) ❌ FAILED
**Target: +20-30 correct | Actual: -11 (REGRESSION)**

The key insight was that conv-42/43/50 fail while conv-26/41 succeed. We tried adapting based on conversation length.

**Implementation:**
```python
if num_messages > 900:
    score_threshold = 0.20  # More aggressive filtering
    max_chars = 4000  # Tighter context budget
elif num_messages > 700:
    score_threshold = 0.18
    max_chars = 5000
else:
    score_threshold = 0.15
    max_chars = 6000
```

**Results:**
- 10-conv: 79.96% (1588/1986) - **11 fewer correct than v74**
- Open-domain category hurt most: -10 correct
- Multi-hop slightly improved: +3 correct (not enough to offset)

**Post-mortem:**
- Aggressive filtering removes TOO MUCH context, not just noise
- Open-domain questions need broader context - tighter budgets hurt
- Length alone doesn't predict which conversations need aggressive filtering
- REVERTED to v74 uniform thresholds (0.15, 6000 chars)

### Phase 2: Answer Verification (v83) ❌ FAILED
**Target: +30-40 correct | Actual: -5 (REGRESSION)**

Post-generation verification to catch wrong answers.

**Implementation:**
- `_v83_verify_answer_entities`: Extract entities from answer, check if in context
- `_v83_verify_counting_answer`: Verify count answers against evidence
- If entity confidence < 0.3, regenerate with stricter prompt

**Results:**
- 3-conv: 94.57% (470/497) - **-5 from v80 (95.57%)**
- Open-domain: 92.00% (184/200) - worst regression

**Post-mortem:**
- Entity verification triggers too aggressively on valid answers
- Regeneration prompt too restrictive ("only use facts from context")
- Checking entities is wrong signal - LLM might use synonyms/paraphrases
- REVERTED - methods kept but disabled

### Phase 3: Question-Type Specialization (v84)
**Target: +40-50 correct**

Dedicated handlers for problematic question types.

1. **Counting questions** (32 errors):
   - Pattern: "how many times", "how many X has Y"
   - Dedicated counting logic with deduplication
   - Return number words ("two", "three") not digits

2. **Description questions** (31 errors):
   - Pattern: "how does X describe Y", "what does X think of Y"
   - Extract adjectives/opinions from entity's messages
   - Track sentiment toward other entities

3. **Preference questions** (10 errors):
   - Pattern: "favorite X", "preferred Y"
   - Weight recent mentions higher
   - Track preference changes over time

### Phase 4: Retrieval Re-ranking (v85)
**Target: +30-40 correct**

Two-stage retrieval with question-aware re-ranking.

1. **Stage 1**: Broad retrieval (current approach)
2. **Stage 2**: Re-rank based on question type:
   - Temporal → boost date-containing content
   - Counting → boost activity-specific content
   - Multi-hop → boost entity-connection content

3. **Question-context alignment scoring**:
   - Score how well retrieved context answers the question
   - Discard low-alignment results

### Phase 5: Multi-hop Decomposition (v86)
**Target: +25-30 correct**

Break multi-hop into explicit steps.

1. **Question decomposition**:
   - "What console does X own?" →
     - Step 1: "What games does X play?"
     - Step 2: "What console runs [game from step 1]?"

2. **Sequential retrieval**:
   - Retrieve for step 1, get answer
   - Use answer in step 2 query
   - Combine evidence for final answer

3. **World knowledge rules**:
   - Explicit game→console mappings
   - City→state mappings (already in v80)

---

## Implementation Priorities

### Immediate (v82): Conversation-Adaptive Filtering
- Low risk, builds on proven v74 attention filter
- Expected: +15-25 correct

### Short-term (v83): Answer Verification
- Moderate complexity
- Catches obvious errors
- Expected: +20-30 correct

### Medium-term (v84-85): Question Specialization + Re-ranking
- Higher complexity
- Targets specific error categories
- Expected: +40-60 correct

### Long-term (v86): Multi-hop Decomposition
- Complex implementation
- Addresses hardest category
- Expected: +20-25 correct

---

## Success Criteria

1. **Phase 1 success**: 10-conv > 82% (1628+)
2. **Phase 2 success**: 10-conv > 85% (1688+)
3. **Phase 3-4 success**: 10-conv > 90% (1787+)
4. **Phase 5 success**: 10-conv > 93% (1847+)

---

## Risk Mitigation

1. **Test on 3-conv first** - verify no regression before 10-conv
2. **Incremental changes** - one technique per version
3. **A/B comparison** - always compare to v74 baseline
4. **Category tracking** - monitor per-category impact
5. **Conversation tracking** - monitor per-conversation impact

---

## Alternative Approaches (If Above Fails)

1. **Ensemble methods**: Run multiple retrieval strategies, vote
2. **Fine-tuned re-ranker**: Train a model on LoCoMo to rank results
3. **Conversation summarization**: Compress long conversations
4. **Explicit fact extraction**: Build structured KB from conversations
