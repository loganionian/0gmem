"""
Fact Extractor: Extracts structured facts from conversations.

Critical for single-hop questions that require specific factual lookups.
Example: "Caroline is a transgender woman" → (Caroline, identity, transgender woman)
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any, Tuple
from enum import Enum


class FactType(Enum):
    """Types of facts that can be extracted."""
    IDENTITY = "identity"           # X is a Y
    ATTRIBUTE = "attribute"         # X has property Y
    RELATIONSHIP = "relationship"   # X is related to Y
    LOCATION = "location"           # X is from/in Y
    ACTIVITY = "activity"           # X does/likes Y
    PREFERENCE = "preference"       # X likes/dislikes Y
    EVENT = "event"                 # X did Y
    STATUS = "status"               # X is Y (status)


@dataclass
class ExtractedFact:
    """A structured fact extracted from text."""
    subject: str
    predicate: str
    object: str
    fact_type: FactType
    confidence: float
    source_text: str
    negated: bool = False
    valid_from: Optional[int] = None
    valid_to: Optional[int] = None
    polarity: str = "positive"
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_text(self) -> str:
        """Convert to natural language."""
        neg = "not " if self.negated else ""
        return f"{self.subject} {neg}{self.predicate} {self.object}"

    def matches_query(self, query: str) -> float:
        """Score how well this fact matches a query."""
        query_lower = query.lower()
        score = 0.0

        # Subject match
        if self.subject.lower() in query_lower:
            score += 0.4

        # Predicate match (fuzzy)
        pred_words = self.predicate.lower().split()
        for word in pred_words:
            if word in query_lower:
                score += 0.2

        # Object match
        if self.object.lower() in query_lower:
            score += 0.3

        return min(1.0, score)


class FactExtractor:
    """
    Extracts structured facts from conversation text.

    Patterns extracted:
    - Identity: "X is a Y", "X identifies as Y"
    - Attributes: "X has Y", "X's Y is Z"
    - Location: "X is from Y", "X moved from Y", "X lives in Y"
    - Activities: "X likes/enjoys Y", "X does Y"
    - Relationships: "X is Y's Z", "X and Y are Z"
    - Status: "X is single/married/employed"
    """

    # Pattern templates for fact extraction
    PATTERNS = [
        # Identity patterns
        (r"(?:i|he|she|they|(\w+))\s+(?:am|is|are)\s+(?:a|an)\s+(\w+(?:\s+\w+)?(?:\s+woman|\s+man|\s+person)?)",
         "is", FactType.IDENTITY),
        (r"(?:i|he|she|they|(\w+))\s+identif(?:y|ies)\s+as\s+(?:a|an)?\s*(\w+(?:\s+\w+)?)",
         "identifies as", FactType.IDENTITY),
        (r"(\w+)\s+is\s+(?:a|an)\s+(transgender|trans|cisgender|cis|gay|lesbian|bisexual|queer)(?:\s+\w+)?",
         "is", FactType.IDENTITY),

        # Location patterns
        (r"(?:i|he|she|they|(\w+))\s+(?:am|is|are)\s+from\s+(\w+(?:\s+\w+)?)",
         "is from", FactType.LOCATION),
        (r"(?:i|he|she|they|(\w+))\s+moved\s+(?:from|to)\s+(\w+(?:\s+\w+)?)",
         "moved from", FactType.LOCATION),
        (r"(?:i|he|she|they|(\w+))\s+(?:live|lives)\s+in\s+(\w+(?:\s+\w+)?)",
         "lives in", FactType.LOCATION),
        (r"(?:i|he|she|they|(\w+))\s+(?:came|come)\s+from\s+(\w+(?:\s+\w+)?)",
         "came from", FactType.LOCATION),

        # Status patterns
        (r"(?:i|he|she|they|(\w+))\s+(?:am|is|are)\s+(single|married|divorced|engaged|in a relationship|employed|unemployed)",
         "is", FactType.STATUS),
        (r"(\w+)'s\s+relationship\s+status\s+is\s+(\w+)",
         "relationship status is", FactType.STATUS),

        # Activity patterns
        (r"(?:i|he|she|they|(\w+))\s+(?:like|likes|enjoy|enjoys|love|loves)\s+(\w+(?:ing)?(?:\s+\w+)?)",
         "likes", FactType.PREFERENCE),
        (r"(?:i|he|she|they|(\w+))\s+(?:do|does|did)\s+(\w+(?:ing)?(?:\s+\w+)?)",
         "does", FactType.ACTIVITY),
        (r"(?:i|he|she|they|(\w+))\s+(?:go|goes|went)\s+(\w+(?:ing)?)",
         "goes", FactType.ACTIVITY),
        (r"(?:i|he|she|they|(\w+))\s+signed\s+up\s+for\s+(?:a\s+)?(\w+(?:\s+\w+)?)",
         "signed up for", FactType.EVENT),
        (r"(?:i|he|she|they|(\w+))\s+(?:research|researched)\s+(\w+(?:\s+\w+)?)",
         "researched", FactType.EVENT),

        # Preference patterns
        (r"(?:i|he|she|they|(\w+))\s+(?:hate|hates|dislike|dislikes)\s+(\w+(?:\s+\w+)?)",
         "dislikes", FactType.PREFERENCE),
        (r"(?:my|his|her|their|(\w+)'s)\s+(?:favorite|favourite)\s+(\w+)\s+is\s+(\w+(?:\s+\w+)?)",
         "favorite is", FactType.PREFERENCE),

        # Career/Education patterns
        (r"(?:i|he|she|they|(\w+))\s+(?:want|wants|decided)\s+to\s+(?:pursue|study|become)\s+(?:a\s+)?(\w+(?:\s+\w+)?)",
         "wants to pursue", FactType.EVENT),
        (r"(?:i|he|she|they|(\w+))\s+(?:work|works)\s+(?:as|in)\s+(?:a\s+)?(\w+(?:\s+\w+)?)",
         "works as", FactType.ATTRIBUTE),

        # Attribute patterns
        (r"(\w+)'s\s+(\w+)\s+(?:is|are)\s+(\w+(?:\s+\w+)?)",
         "has", FactType.ATTRIBUTE),
        (r"(?:my|his|her|their|(\w+)'s)\s+kids?\s+(?:like|likes|love|loves)\s+(\w+(?:\s+\w+)?)",
         "kids like", FactType.PREFERENCE),
    ]

    # Keywords that indicate the subject when pronouns are used
    SUBJECT_KEYWORDS = {
        "i": "speaker",
        "my": "speaker",
        "me": "speaker",
        "he": "he",
        "she": "she",
        "they": "they",
        "his": "he",
        "her": "she",
        "their": "they",
    }

    def __init__(self):
        self.compiled_patterns = [
            (re.compile(pattern, re.IGNORECASE), predicate, fact_type)
            for pattern, predicate, fact_type in self.PATTERNS
        ]

    def extract_facts(
        self,
        text: str,
        speaker: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> List[ExtractedFact]:
        """
        Extract structured facts from text.

        Args:
            text: Text to extract facts from
            speaker: Speaker name to resolve pronouns
            metadata: Additional metadata

        Returns:
            List of extracted facts
        """
        facts = []
        text_lower = text.lower()

        for pattern, predicate, fact_type in self.compiled_patterns:
            for match in pattern.finditer(text):
                groups = match.groups()

                # Extract subject
                subject = None
                obj = None

                if len(groups) >= 1:
                    # First capture group is often subject
                    subject = groups[0]
                    if not subject and speaker:
                        # Pronoun case - use speaker
                        subject = speaker

                if len(groups) >= 2:
                    obj = groups[1]

                if len(groups) >= 3:
                    # For patterns like "X's Y is Z"
                    obj = groups[2]

                if not subject or not obj:
                    continue

                # Clean up extracted values
                subject = subject.strip().title() if subject else speaker
                obj = obj.strip()

                # Check for negation
                negated = self._is_negated(text, match.start())

                fact = ExtractedFact(
                    subject=subject,
                    predicate=predicate,
                    object=obj,
                    fact_type=fact_type,
                    confidence=0.8 if not negated else 0.7,
                    source_text=text,
                    negated=negated,
                    metadata=metadata or {},
                )
                facts.append(fact)

        # Also extract key-value pairs from specific patterns
        facts.extend(self._extract_kv_facts(text, speaker))

        return facts

    def _is_negated(self, text: str, position: int) -> bool:
        """Check if the text around position contains negation."""
        # Look for negation words before the position
        before_text = text[max(0, position - 50):position].lower()
        negation_words = ["not", "never", "don't", "doesn't", "didn't", "won't", "can't", "couldn't", "no"]
        return any(word in before_text for word in negation_words)

    def _extract_kv_facts(self, text: str, speaker: Optional[str]) -> List[ExtractedFact]:
        """Extract key-value style facts."""
        facts = []

        # Pattern: "X is from Y" style
        kv_patterns = [
            # Activities with specific places
            (r"(?:i|we|he|she|they|(\w+))\s+(?:went|go|goes)\s+(?:to\s+)?(?:the\s+)?(\w+(?:\s+\w+)?)\s+(?:on|at|in)\s+(\w+(?:\s+\w+)?)",
             "went to", FactType.EVENT),
            # Camping locations
            (r"(?:i|we|he|she|they|(\w+))\s+(?:camp|camped|camping)\s+(?:at|in|on)\s+(?:the\s+)?(\w+(?:\s+\w+)?)",
             "camped at", FactType.LOCATION),
            # Beach, mountains, forest mentions
            (r"(?:camp|camped|camping)\s+(?:at|in|on|near)\s+(?:the\s+)?(beach|mountain|mountains|forest|lake|river)",
             "camped at", FactType.LOCATION),
        ]

        for pattern, predicate, fact_type in kv_patterns:
            compiled = re.compile(pattern, re.IGNORECASE)
            for match in compiled.finditer(text):
                groups = match.groups()
                subject = groups[0] if groups[0] else speaker
                obj = groups[-1] if groups[-1] else groups[-2] if len(groups) > 1 else None

                if subject and obj:
                    facts.append(ExtractedFact(
                        subject=subject.title() if subject else speaker,
                        predicate=predicate,
                        object=obj,
                        fact_type=fact_type,
                        confidence=0.7,
                        source_text=text,
                    ))

        return facts


class FactStore:
    """
    Storage for extracted facts with efficient lookup.

    Supports:
    - Lookup by subject
    - Lookup by predicate
    - Query matching
    """

    def __init__(self):
        self.facts: List[ExtractedFact] = []
        self.by_subject: Dict[str, List[ExtractedFact]] = {}
        self.by_predicate: Dict[str, List[ExtractedFact]] = {}
        self.by_type: Dict[FactType, List[ExtractedFact]] = {}
        self.by_subject_predicate: Dict[Tuple[str, str], ExtractedFact] = {}
        self.max_turn: int = 0

        self._supersedable_types = {
            FactType.IDENTITY,
            FactType.ATTRIBUTE,
            FactType.LOCATION,
            FactType.STATUS,
            FactType.PREFERENCE,
        }

    def add_fact(self, fact: ExtractedFact) -> None:
        """Add a fact to the store."""
        # Track turn/session for recency
        turn = None
        if fact.metadata:
            turn = fact.metadata.get("turn")
        if isinstance(turn, int) and turn > self.max_turn:
            self.max_turn = turn

        # Mark polarity explicitly for negations
        if fact.negated:
            fact.polarity = "negative"

        # Belief versioning for supersedable facts
        key = (fact.subject.lower(), fact.predicate.lower())
        if fact.fact_type in self._supersedable_types and isinstance(turn, int):
            latest = self.by_subject_predicate.get(key)
            if latest and latest is not fact:
                # Only supersede older non-negated facts
                latest_turn = latest.metadata.get("turn") if latest.metadata else None
                if isinstance(latest_turn, int) and latest_turn <= turn:
                    # New positive fact updates the belief
                    if not fact.negated and not latest.negated and latest.object != fact.object:
                        latest.valid_to = turn
                        latest.metadata["valid_to"] = turn
                    # New negated fact invalidates matching prior belief
                    if fact.negated and not latest.negated and latest.object == fact.object:
                        latest.valid_to = turn
                        latest.metadata["valid_to"] = turn

            fact.valid_from = fact.valid_from if fact.valid_from is not None else turn
            fact.metadata["valid_from"] = fact.valid_from

            # Update latest pointer for positive facts
            if not fact.negated:
                self.by_subject_predicate[key] = fact

        self.facts.append(fact)

        # Index by subject
        subject_key = fact.subject.lower()
        if subject_key not in self.by_subject:
            self.by_subject[subject_key] = []
        self.by_subject[subject_key].append(fact)

        # Index by predicate
        pred_key = fact.predicate.lower()
        if pred_key not in self.by_predicate:
            self.by_predicate[pred_key] = []
        self.by_predicate[pred_key].append(fact)

        # Index by type
        if fact.fact_type not in self.by_type:
            self.by_type[fact.fact_type] = []
        self.by_type[fact.fact_type].append(fact)

    def add_facts(self, facts: List[ExtractedFact]) -> None:
        """Add multiple facts."""
        for fact in facts:
            self.add_fact(fact)

    def get_facts_about(self, subject: str) -> List[ExtractedFact]:
        """Get all facts about a subject."""
        return self.by_subject.get(subject.lower(), [])

    def get_facts_by_type(self, fact_type: FactType) -> List[ExtractedFact]:
        """Get all facts of a specific type."""
        return self.by_type.get(fact_type, [])

    def search(
        self,
        query: str,
        top_k: int = 10,
        prefer_latest: bool = False,
        prefer_positive: bool = False,
        require_negated: bool = False,
    ) -> List[Tuple[ExtractedFact, float]]:
        """Search for facts matching a query."""
        scored = []
        for fact in self.facts:
            if prefer_latest:
                valid_to = fact.metadata.get("valid_to") if fact.metadata else None
                if valid_to is not None:
                    continue
            if require_negated and not fact.negated:
                continue

            score = fact.matches_query(query)
            if score > 0:
                # Recency boost if turn available
                turn = fact.metadata.get("turn") if fact.metadata else None
                if isinstance(turn, int) and self.max_turn > 0:
                    score *= 1.0 + (turn / self.max_turn) * 0.15
                if prefer_positive and fact.negated:
                    score *= 0.8
                scored.append((fact, score))

        # Sort by score
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[:top_k]

    def get_stats(self) -> Dict[str, Any]:
        """Get store statistics."""
        return {
            "total_facts": len(self.facts),
            "unique_subjects": len(self.by_subject),
            "unique_predicates": len(self.by_predicate),
            "facts_by_type": {t.value: len(f) for t, f in self.by_type.items()},
        }
