# Root Cause Analysis: Multi-hop and Temporal Failures

## Executive Summary

Based on analysis of 497 questions across 3 conversations:
- **Multi-hop**: 38.1% accuracy (8/21 correct)
- **Temporal**: 41.1% accuracy (37/90 correct)

The failures reveal systemic issues in how information is extracted, indexed, and reasoned over.

---

## Multi-hop Reasoning Failures

### Current Performance
- 21 questions, 38.1% accuracy
- 13 failures analyzed

### Failure Categories

| Category | Count | Example |
|----------|-------|---------|
| Inferential reasoning gap | 5 | Q50: Expected "Liberal" from LGBTQ+ support, got "LGBTQ+ supportive" |
| Secondary entity neglect | 4 | Q8, Q14: Questions about John return "Unknown" |
| Cross-session aggregation | 2 | Q69: Personality traits need multiple sessions |
| Answer format issues | 2 | Q42: Said "Yes, national park" instead of "National park" |

### Root Cause 1: Inferential Reasoning Gap

**Symptom**: System extracts facts but doesn't make necessary inferences.

**Example**:
```
Q: What would Caroline's political leaning likely be?
Expected: Liberal
Got: LGBTQ+ supportive
```

**Analysis**: The system found Caroline is LGBTQ+ supportive but didn't infer the political implication. Multi-hop questions require:
1. Fact A: Caroline is transgender and LGBTQ+ activist
2. Fact B: She supports LGBTQ+ rights, attends pride events
3. Inference: LGBTQ+ support correlates with liberal political views
4. Answer: Liberal

**Root Cause**: The LLM prompt asks to "answer based on context" but doesn't prompt for inferential reasoning chains.

### Root Cause 2: Secondary Entity Neglect

**Symptom**: Questions about non-speaker entities (John, Gina, Maria) fail with "Unknown".

**Example**:
```
Q: What might John's degree be in?
Expected: Political science, Public administration
Got: Unknown
```

**Analysis**: The system extracts facts only for SPEAKERS:
- Maria says: "John wants to run for office someday"
- Extracted: Nothing (John isn't speaking)
- Should extract: John → goal → political office

**Root Cause**: Profile extraction (`llm_fact_extractor.py`) only captures `speaker` facts, ignoring mentioned entities.

### Root Cause 3: Cross-Session Aggregation Failure

**Symptom**: Questions requiring information from multiple sessions fail.

**Example**:
```
Q: What personality traits might Melanie say Caroline has?
Expected: Thoughtful, authentic, driven
Got: Creative, thankful, supportive, emotional
```

**Analysis**:
- Session 5: Melanie says Caroline is "thoughtful"
- Session 12: Melanie says Caroline is "authentic"
- Session 20: Melanie says Caroline is "driven"
- Retrieval returns only 1-2 sessions, missing the full picture

**Root Cause**: Retrieval (`_hybrid_retrieve`) returns top-k documents but doesn't ensure coverage across sessions.

---

## Temporal Generalization Failures

### Current Performance
- 90 questions, 41.1% accuracy
- 53 failures analyzed

### Failure Categories

| Category | Count | Example |
|----------|-------|---------|
| None/Unknown answers | 16 | Q26: "When did Melanie read the book?" → None (expected: 2022) |
| Format mismatch | 13 | Expected "The week before 27 June" → Got "8 May 2023" |
| Wrong session date | 21 | Events get FIRST session date, not ACTUAL event date |
| Cross-conversation | 3 | Gina/Jon questions get January dates from wrong parsing |

### Root Cause 1: Session-Event Mismatch

**Symptom**: Events return the FIRST session date where they're mentioned, not the date they occurred.

**Example**:
```
Q: When did Caroline go to the LGBTQ conference?
Expected: 10 July 2023
Got: 8 May 2023 (Session 1)
```

**Analysis**:
- Session 1 (8 May): Caroline mentions "I'm planning to go to the conference in July"
- Session 10 (10 July): Caroline says "I went to the conference today"
- System returns Session 1's date because it's the first mention

**Root Cause**: `answer_temporal_from_profile()` stores event dates by first occurrence, not by when event actually happened.

**Fix needed**: Distinguish between:
- Planning mentions (future tense): "I'm going to..."
- Completion mentions (past tense): "I went to..."

### Root Cause 2: No Event-Specific Indexing

**Symptom**: Similar events share dates, causing confusion.

**Example**:
```
Q: When did Melanie go camping in July?
Expected: two weekends before 17 July 2023
Got: 8 May 2023 (camping trip 1, not trip 2)
```

**Analysis**: Melanie goes camping twice:
- June camping trip (mentioned in May)
- July camping trip (mentioned in July)

The system stores: `{"camping": "8 May 2023"}` → only one date

**Root Cause**: Profile stores `event_type → date` but should store `event_type → [(date1, context1), (date2, context2)]`

### Root Cause 3: Year-Only Facts Missing

**Symptom**: Questions asking for years return "None".

**Example**:
```
Q: When did Melanie read the book "Nothing is Impossible"?
Expected: 2022
Got: None

Q: How long has Melanie been practicing art?
Expected: Since 2016
Got: None
```

**Analysis**: The conversations mention:
- "I read that book back in 2022"
- "I've been doing art since 2016, about 7 years now"

**Root Cause**: Extraction patterns in `llm_fact_extractor.py` focus on session timestamps (e.g., "8 May 2023") but don't capture year-only references.

### Root Cause 4: Relative Date Format Mismatch

**Symptom**: Expected relative dates but returned absolute dates.

**Example**:
```
Q: When did Caroline draw a self-portrait?
Expected: The week before 23 August 2023
Got: 27 June 2023
```

**Analysis**: The expected answer preserves the RELATIVE format from the conversation ("the week before"). But the system converts everything to absolute dates.

**Root Cause**: `_extract_date_from_timestamp()` normalizes to absolute dates, losing relative context that some questions expect.

---

## Improvement Schemes

### For Multi-hop Reasoning

#### Scheme 1: Chain-of-Thought Prompting

**What**: Add explicit reasoning steps to the QA prompt.

**Why**: Current prompt is single-step ("answer based on context"). Multi-hop requires multi-step reasoning.

**Implementation**:
```python
MULTI_HOP_PROMPT = """
Question: {question}
Context: {context}

Reasoning Steps:
1. What facts are directly stated in the context?
2. What can be inferred from these facts?
3. How do the facts connect to answer the question?

Based on this reasoning, the answer is:
"""
```

**Expected Impact**: +10-15% on inferential questions (Q50, Q59, Q45)

#### Scheme 2: Secondary Entity Extraction

**What**: Extract facts about MENTIONED entities, not just speakers.

**Why**: Questions about John, Gina, Maria fail because their facts aren't captured.

**Implementation**:
```python
def _extract_facts_regex(self, text: str, speaker: str):
    # Current: Only extract for speaker
    # New: Also extract for mentioned entities

    mentioned_entities = self._find_mentioned_people(text)
    for entity in mentioned_entities:
        # Extract facts about this entity from the text
        self._extract_entity_facts(text, entity)
```

**Pattern examples**:
- "John wants to run for office" → John: goal=political office
- "Maria's brother is a doctor" → Maria's brother: profession=doctor

**Expected Impact**: +15-20% on secondary entity questions (Q8, Q14, Q17, etc.)

#### Scheme 3: Session Coverage Retrieval

**What**: Ensure retrieval covers multiple sessions for multi-hop questions.

**Why**: Current retrieval may return 5 documents from 2 sessions, missing information from session 10.

**Implementation**:
```python
def _retrieve_with_session_coverage(self, query: str, min_sessions: int = 5):
    results = self._hybrid_retrieve(query, top_k=50)

    # Ensure coverage across sessions
    sessions_covered = {}
    final_results = []
    for doc in results:
        session = doc.metadata['session_idx']
        if session not in sessions_covered or len(sessions_covered[session]) < 2:
            final_results.append(doc)
            sessions_covered.setdefault(session, []).append(doc)

    return final_results[:20]
```

**Expected Impact**: +5-10% on cross-session questions

---

### For Temporal Generalization

#### Scheme 1: Tense-Aware Event Indexing

**What**: Only index events from PAST TENSE mentions (completed events).

**Why**: "I'm going camping next week" shouldn't set the event date to current session.

**Implementation**:
```python
def _is_completed_event(self, text: str, event: str) -> bool:
    """Check if event is mentioned as completed (past tense)."""
    past_indicators = [
        f"went {event}", f"attended {event}", f"did {event}",
        f"finished {event}", f"completed {event}", f"had {event}",
        f"{event} yesterday", f"{event} last week"
    ]
    future_indicators = [
        f"going to {event}", f"will {event}", f"plan to {event}",
        f"want to {event}", f"{event} next", f"{event} tomorrow"
    ]

    text_lower = text.lower()
    has_past = any(p in text_lower for p in past_indicators)
    has_future = any(p in text_lower for p in future_indicators)

    return has_past and not has_future
```

**Expected Impact**: +15-20% on wrong-date failures (Q25, Q31, Q33, etc.)

#### Scheme 2: Multi-Instance Event Storage

**What**: Store multiple dates for recurring events.

**Why**: Melanie goes camping twice; pottery class happens weekly.

**Implementation**:
```python
# Current: {"camping": "8 May 2023"}
# New: {"camping": [
#   {"date": "June 2023", "context": "camping in June"},
#   {"date": "July 2023", "context": "camping in July"}
# ]}

def _add_to_profile_temporal(self, fact: PersonFact):
    key = f"{fact.fact_type}_dates"
    entry = {
        "date": fact.session_date,
        "context": fact.source_text[:100],
        "session": fact.session_id
    }
    self.person_profiles[person][key].append(entry)
```

Then for answering:
```python
def answer_temporal_from_profile(self, question: str, person: str):
    # Match question context to find the right instance
    # "camping in July" → find entry with "July" in context
```

**Expected Impact**: +10% on recurring event questions

#### Scheme 3: Year-Only Pattern Extraction

**What**: Extract year references and "since X" patterns.

**Why**: "2022", "Since 2016" questions fail with None.

**Implementation**:
```python
# Add to _extract_facts_regex:
year_patterns = [
    (r'(?:in|back in)\s+(\d{4})', 'year_mentioned'),
    (r'since\s+(\d{4})', 'since_year'),
    (r'(\d+)\s+years?\s+(?:ago|now)', 'years_duration'),
]

for pattern, fact_type in year_patterns:
    match = re.search(pattern, text_lower)
    if match:
        # Associate with nearby activity/topic
        context = self._get_surrounding_context(text, match.start())
        facts.append(PersonFact(speaker, fact_type, match.group(1), context))
```

**Expected Impact**: +10-15% on year-based questions (Q26, Q49, Q68, Q72)

#### Scheme 4: Preserve Relative Date Format

**What**: Store both absolute and relative date formats.

**Why**: Some expected answers are relative ("the week before X").

**Implementation**:
```python
def _extract_temporal_info(self, text: str, session_date: str):
    relative_patterns = [
        r'(the week before \w+)',
        r'(the friday before \w+)',
        r'(last \w+)',
        r'(yesterday)',
    ]

    for pattern in relative_patterns:
        match = re.search(pattern, text_lower)
        if match:
            return {
                "relative": match.group(1),
                "absolute": self._resolve_to_absolute(match.group(1), session_date)
            }
```

**Expected Impact**: +5% on relative date questions

---

## Priority Order

Based on impact and effort:

| Priority | Scheme | Expected Impact | Effort |
|----------|--------|-----------------|--------|
| 1 | Tense-Aware Event Indexing | +15-20% temporal | Medium |
| 2 | Secondary Entity Extraction | +15-20% multi-hop | Medium |
| 3 | Year-Only Pattern Extraction | +10-15% temporal | Low |
| 4 | Chain-of-Thought Prompting | +10-15% multi-hop | Low |
| 5 | Multi-Instance Event Storage | +10% temporal | Medium |
| 6 | Session Coverage Retrieval | +5-10% multi-hop | Low |
| 7 | Preserve Relative Date Format | +5% temporal | Low |

**Recommended implementation order**: 1 → 3 → 4 → 2 → 5 → 6 → 7
