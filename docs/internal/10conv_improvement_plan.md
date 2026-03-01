# 10-Conv Improvement Plan: 79.61% → 95% Target

## Current State
- **Accuracy**: 79.61% (1581/1986 correct)
- **Target**: 95% (1886/1986)
- **Gap**: 305 more correct answers needed

## Error Distribution Analysis

### By Conversation (Error Rate)
| Conversation | Errors | Total | Error Rate |
|--------------|--------|-------|------------|
| conv-26 | 3 | 199 | 1.5% |
| conv-30 | 6 | 105 | 5.7% |
| conv-41 | 18 | 217 | 8.3% |
| conv-44 | 38 | 204 | 18.6% |
| conv-49 | 44 | 199 | 22.1% |
| conv-47 | 48 | 197 | 24.4% |
| conv-48 | 50 | 208 | 24.0% |
| conv-50 | 59 | 207 | 28.5% |
| conv-43 | 63 | 242 | 26.0% |
| conv-42 | 76 | 260 | 29.2% |

**Key Insight**: Conversations with ~200+ questions have 5-30x higher error rates than shorter conversations.

### By Category (Errors Needed for 95%)
| Category | Correct | Total | Current | Target | Gap |
|----------|---------|-------|---------|--------|-----|
| Multi-hop | 51 | 96 | 53.1% | 95% | 40 |
| Single-hop | 192 | 282 | 68.1% | 95% | 76 |
| Open-domain | 639 | 841 | 76.0% | 95% | 160 |
| Temporal | 258 | 321 | 80.4% | 95% | 47 |
| Adversarial | 441 | 446 | 98.9% | - | 0 |

### Error Type Classification
- **Wrong fact/inference**: 374 (92.3%) - Retrieves something but wrong answer
- **Retrieval failures**: 18 (4.4%) - "No information found"
- **Partial matches**: 13 (3.2%) - F1 >= 0.3 but not correct

**Key Insight**: The problem is NOT retrieval - it's getting the WRONG information or WRONG inference.

## Failing Question Patterns

### 1. Counting Questions (32 errors)
```
Q: How many times has Joanna found new hiking trails?
Expected: twice
Got: 18

Q: How many dogs has Maria adopted?
Expected: two
Got: Maria has not mentioned adopting any dogs
```
**Root cause**: Counting logic doesn't find explicit counts in conversation context

### 2. Temporal Questions (26 errors)
```
Q: When was Jon in Rome?
Expected: June 2023
Got: last week

Q: When did Gina go to a dance class?
Expected: 21 July 2023
Got: No information
```
**Root cause**: Relative date conversion failing; date-event association weak

### 3. Description/How Questions (31 errors)
```
Q: How does Gina describe the studio that Jon has opened?
Expected: amazing
Got: No description
```
**Root cause**: Missing adjective/description extraction from context

### 4. Favorite/Preference Questions (10 errors)
```
Q: What is one of Joanna's favorite movies?
Expected: "Eternal Sunshine of the Spotless Mind"
Got: 'Little Women'
```
**Root cause**: Multiple favorites exist; retrieves wrong one

### 5. Location Questions (10 errors)
```
Q: Which city have both Jean and John visited?
Expected: Rome
Got: None
```
**Root cause**: Cross-entity intersection not working for some pairs

## Root Cause Analysis

### Why conv-42/43 have high error rates:
1. **Longer conversations** (557-1019 messages) dilute relevant context
2. **More entities** mentioned with similar activities
3. **More temporal relationships** to track
4. **More implicit facts** requiring inference

### Why Multi-hop fails (53% accuracy):
1. Requires connecting 2+ facts
2. Implicit inference (e.g., Xenoblade 2 → Nintendo Switch)
3. World knowledge needed alongside memory

### Why Single-hop fails (68% accuracy):
1. Specific fact buried in long conversation
2. Multiple similar facts exist (e.g., multiple hiking trips)
3. Counting requires aggregating across messages

## Improvement Strategy

### Phase 1: Counting Logic Improvement (Target: +30 correct)
**Impact**: Fix 32 counting question errors

1. **Pattern matching for explicit counts**:
   - "found new hiking trails twice"
   - "rejected three times"
   - "adopted two dogs"

2. **Aggregation from conversation**:
   - Track unique instances across messages
   - Deduplicate same events mentioned multiple times

3. **Implementation**:
   ```python
   def extract_count(question, context):
       # 1. Check for explicit count statements
       # 2. If not found, count distinct event mentions
       # 3. Handle ordinals (first, second, third)
   ```

### Phase 2: Temporal Enhancement (Target: +25 correct)
**Impact**: Fix temporal date resolution

1. **Relative-to-absolute date conversion**:
   - Track message dates in conversation
   - "last week" → calculate from message timestamp
   - "last month" → convert based on context

2. **Event-date association strengthening**:
   - Build explicit event→date index per conversation
   - Multiple passes for date extraction

### Phase 3: Context Precision (Target: +50 correct)
**Impact**: Reduce wrong fact retrieval

1. **Entity-specific context windows**:
   - When asking about entity X, boost context from X's messages
   - Reduce noise from unrelated entity messages

2. **Recency weighting for "favorite" questions**:
   - Most recent mention of preference = likely current
   - Track preference changes over time

3. **Multi-entity intersection queries**:
   - "Both X and Y" → find common ground
   - Already partially implemented in v68e, needs expansion

### Phase 4: Multi-hop Reasoning (Target: +35 correct)
**Impact**: Address 53% accuracy in multi-hop

1. **Decomposed retrieval**:
   - Split multi-hop into sequential single-hop
   - Q: "What console does Nate own?" →
     - Step 1: What games does Nate play?
     - Step 2: What console runs those games?

2. **Chain-of-thought prompting**:
   - Encourage LLM to show reasoning steps
   - Validate each step has evidence

3. **World knowledge integration**:
   - For implicit facts (game→console), allow external inference
   - But flag when using world knowledge vs memory

### Phase 5: Description/Adjective Extraction (Target: +25 correct)
**Impact**: Fix "how does X describe Y" questions

1. **Sentiment/adjective extraction**:
   - Track adjectives used for entities/events
   - "amazing studio" → store studio:amazing

2. **Opinion mining from messages**:
   - Pattern: "[entity] is [adjective]"
   - Pattern: "the [entity] was [adjective]"

## Implementation Priority

| Phase | Expected Gain | Complexity | Priority |
|-------|--------------|------------|----------|
| Phase 1: Counting | +30 | Medium | HIGH |
| Phase 3: Context Precision | +50 | High | HIGH |
| Phase 2: Temporal | +25 | Medium | MEDIUM |
| Phase 4: Multi-hop | +35 | High | MEDIUM |
| Phase 5: Descriptions | +25 | Low | LOW |

**Total expected improvement**: ~165 additional correct answers
**Projected accuracy**: 87.9% (still 7% gap to 95%)

## Additional Techniques (from EverMemOS paper)

### Techniques not yet implemented:
1. **Attention filter** - "Precise forgetting" to reduce noise
2. **Sufficiency threshold** - Self-check if retrieved memories answer the question
3. **Agentic retrieval** - Multi-round with query rewriting
4. **Episode segmentation** - Divide conversations into meaningful chunks

### Implementation ideas:
- Add self-evaluation: "Is this context sufficient to answer the question?"
- If not, expand search with different keywords
- Track confidence scores and abstain when low

## Success Metrics

1. **Conv-42 error rate**: 29.2% → <10%
2. **Conv-43 error rate**: 26.0% → <10%
3. **Multi-hop accuracy**: 53.1% → 85%
4. **Single-hop accuracy**: 68.1% → 90%
5. **Overall accuracy**: 79.6% → 95%

## Implementation Results

### v69b: Phase 1 (Counting Logic)
- Added action verb extraction for "how many times" questions
- Patterns for: found, discovered, received, written, won, taken, adopted
- **Results**: Minimal improvement on counting accuracy

### v70: Phase 2 (Temporal Enhancement)
- Added EVENT_PATTERNS in event_date_index.py for: beach visits, road trips, movie watching, started activities, awards, pet adoption
- Added specific_event_mapping entries for common event phrases
- **Results**:
  - **10-conv: 80.26% (1594/1986)** - +6 from v69b baseline
  - Category: Temporal 262→262, Multi-hop 54, Open-domain 649, Single-hop 188

### v71: Phase 3 Attempt (Expanded Cross-Entity Patterns) - REVERTED
- Expanded cross-entity patterns and query expansions
- **Results**:
  - **3-conv: 96.18% (478/497)** - NEW HIGH!
  - **10-conv: 80.11% (1591/1986)** - WORSE than v70 (-3)
- **Lesson**: Broader patterns help short conversations but hurt long ones due to false positives

### v72: Phase 4 Attempt (World Knowledge Inference) - REVERTED
- Added world knowledge inference patterns for multi-hop
- **Results**:
  - **3-conv: 96.18% (478/497)** - Same as v71
  - **10-conv: 80.16% (1592/1986)** - WORSE than v70 (-2)
- **Lesson**: World knowledge hints add variability that hurts other categories

### v73: Phase 5 Attempt (Description/Feeling Equivalences) - REVERTED
- Added 50+ semantic equivalents for description/feeling words
- Covered: calming/peaceful/relaxing, supportive/encouraging, great/wonderful/amazing, difficult/hard/challenging, etc.
- **Results**:
  - **3-conv: 96.58% (480/497)** - NEW HIGH! (+4 from v70)
  - **10-conv: 80.16% (1592/1986)** - WORSE than v70 (-2)
  - Improvements: Temporal +5, Single-hop +3
  - Regressions: Open-domain -9, Multi-hop -1
- **Lesson**: Description word equivalences cause false positives in longer conversations where more diverse text increases incorrect matches

### Key Pattern Identified
All three reverted attempts (v71, v72, v73) show the same pattern:
- **3-conv improves** (up to +4 correct)
- **10-conv regresses** (typically -2 to -3 correct)
- **Root cause**: Semantic expansion that helps short conversations causes false positive matches in longer conversations with more diverse content

### v74: Attention Filter - SUCCESS!
- Added `_attention_filter_hybrid_results` method in locomo.py
- Filter removes: low-scoring results (< 15% of max score), semantic duplicates, enforces 6000 char budget
- **Results**:
  - **10-conv: 80.51% (1599/1986)** - +5 from v70! First 10-conv improvement!
  - **3-conv: 95.37% (474/497)** - slight regression from v70 (-2)
  - Temporal: +5, Single-hop: +7, Multi-hop: +1
  - Open-domain: -8 (trade-off)
- **Key insight**: "Precise forgetting" works better than semantic expansion for longer conversations

### Current Best: v74
- **10-conv: 80.51% (1599/1986)** - 287 more needed for 95%
- **3-conv: 95.37% (474/497)**

## Next Steps

1. ~~Start with Phase 1 (Counting) - most straightforward~~ DONE
2. ~~Phase 2 (Temporal) - implemented~~ DONE
3. **Need different approach**: Instead of semantic expansion, focus on:
   - **Attention filter** (EverMemOS technique): Reduce noise before LLM processing
   - **Sufficiency checking**: Self-evaluation of retrieved context
   - **Entity-specific context windows**: Boost relevance of entity-specific messages
4. Analyze specific conv-42/conv-43 errors in detail
5. Consider agentic retrieval with query rewriting
