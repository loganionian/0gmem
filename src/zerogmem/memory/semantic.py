"""
Semantic Memory: Accumulated factual knowledge.

Stores facts divorced from specific episodes, with support for
confidence scoring, contradiction detection, and provenance tracking.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional, List, Dict, Any, Tuple, Set
import numpy as np


@dataclass
class Fact:
    """
    A semantic fact with provenance and confidence tracking.

    Key design: Track BOTH supporting evidence AND contradictions
    for adversarial robustness.
    """
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    content: str = ""  # The fact statement
    subject: str = ""  # Subject entity
    predicate: str = ""  # Relation/property
    object: str = ""  # Object entity or value

    # Confidence and provenance
    confidence: float = 1.0  # How certain are we?
    sources: List[str] = field(default_factory=list)  # Episode IDs supporting this
    contradictions: List[str] = field(default_factory=list)  # Episode IDs contradicting this

    # Temporal tracking
    first_learned: datetime = field(default_factory=datetime.now)
    last_confirmed: datetime = field(default_factory=datetime.now)
    confirmation_count: int = 1

    # Negation handling
    negated: bool = False  # Is this explicitly NOT true?
    negation_source: Optional[str] = None  # Episode that negated this

    # Embedding for similarity
    embedding: Optional[np.ndarray] = None

    # Categorization
    category: str = ""  # preference, attribute, relation, event_fact, etc.
    tags: List[str] = field(default_factory=list)

    metadata: Dict[str, Any] = field(default_factory=dict)

    def confirm(self, source_id: str) -> None:
        """Confirm this fact with additional evidence."""
        if source_id not in self.sources:
            self.sources.append(source_id)
        self.last_confirmed = datetime.now()
        self.confirmation_count += 1
        # Increase confidence with more confirmations
        self.confidence = min(1.0, self.confidence + 0.1)

    def contradict(self, source_id: str) -> None:
        """Record a contradiction to this fact."""
        if source_id not in self.contradictions:
            self.contradictions.append(source_id)
        # Decrease confidence with contradictions
        self.confidence = max(0.1, self.confidence - 0.2)

    def negate(self, source_id: str) -> None:
        """Mark this fact as negated."""
        self.negated = True
        self.negation_source = source_id
        self.confidence = 0.0

    @property
    def is_reliable(self) -> bool:
        """Check if fact is considered reliable."""
        return (
            self.confidence >= 0.5 and
            not self.negated and
            len(self.contradictions) < len(self.sources)
        )


class SemanticMemoryStore:
    """
    Semantic memory store managing factual knowledge.

    Key capabilities:
    - Store facts with provenance
    - Detect and track contradictions
    - Handle negations explicitly
    - Search by entity, predicate, or similarity
    """

    # Fact categories
    CATEGORIES = [
        "preference",     # User likes/dislikes
        "attribute",      # Entity properties
        "relation",       # Entity relationships
        "event_fact",     # Facts about events
        "belief",         # Opinions/beliefs
        "skill",          # Abilities/capabilities
        "habit",          # Behavioral patterns
        "biographical",   # Life facts
    ]

    def __init__(self):
        self.facts: Dict[str, Fact] = {}

        # Indexes
        self._subject_index: Dict[str, Set[str]] = {}  # subject -> fact_ids
        self._predicate_index: Dict[str, Set[str]] = {}  # predicate -> fact_ids
        self._object_index: Dict[str, Set[str]] = {}  # object -> fact_ids
        self._category_index: Dict[str, Set[str]] = {}  # category -> fact_ids
        self._negated_facts: Set[str] = set()  # fact_ids that are negated

        # Embeddings
        self._embeddings: List[np.ndarray] = []
        self._embedding_ids: List[str] = []

    def add_fact(self, fact: Fact) -> Tuple[str, bool]:
        """
        Add a fact to semantic memory.

        Returns: (fact_id, is_new) - is_new is False if fact was merged with existing
        """
        # Check for existing similar fact
        existing = self._find_similar_fact(fact)
        if existing:
            # Merge with existing fact
            existing.confirm(fact.sources[0] if fact.sources else "unknown")
            return existing.id, False

        # Check for contradicting facts
        contradictions = self._find_contradicting_facts(fact)
        for contra_fact in contradictions:
            fact.contradictions.append(contra_fact.id)
            contra_fact.contradict(fact.id)

        # Add new fact
        self.facts[fact.id] = fact
        self._index_fact(fact)

        # Track if negated
        if fact.negated:
            self._negated_facts.add(fact.id)

        return fact.id, True

    def _index_fact(self, fact: Fact) -> None:
        """Index a fact for efficient retrieval."""
        # Subject index
        if fact.subject:
            if fact.subject not in self._subject_index:
                self._subject_index[fact.subject] = set()
            self._subject_index[fact.subject].add(fact.id)

        # Predicate index
        if fact.predicate:
            if fact.predicate not in self._predicate_index:
                self._predicate_index[fact.predicate] = set()
            self._predicate_index[fact.predicate].add(fact.id)

        # Object index
        if fact.object:
            if fact.object not in self._object_index:
                self._object_index[fact.object] = set()
            self._object_index[fact.object].add(fact.id)

        # Category index
        if fact.category:
            if fact.category not in self._category_index:
                self._category_index[fact.category] = set()
            self._category_index[fact.category].add(fact.id)

        # Embedding index
        if fact.embedding is not None:
            self._embeddings.append(fact.embedding)
            self._embedding_ids.append(fact.id)

    def _find_similar_fact(self, fact: Fact) -> Optional[Fact]:
        """Find an existing fact that is essentially the same."""
        # Check by subject-predicate-object match
        if fact.subject and fact.predicate:
            subject_facts = self._subject_index.get(fact.subject, set())
            for fid in subject_facts:
                existing = self.facts.get(fid)
                if existing and existing.predicate == fact.predicate:
                    if existing.object == fact.object:
                        return existing
        return None

    def _find_contradicting_facts(self, fact: Fact) -> List[Fact]:
        """Find facts that contradict this fact."""
        contradictions = []

        # If this is a negation, find the positive version
        if fact.negated:
            subject_facts = self._subject_index.get(fact.subject, set())
            for fid in subject_facts:
                existing = self.facts.get(fid)
                if existing and not existing.negated:
                    if existing.predicate == fact.predicate and existing.object == fact.object:
                        contradictions.append(existing)

        # If this is positive, find negated versions
        else:
            subject_facts = self._subject_index.get(fact.subject, set())
            for fid in subject_facts:
                existing = self.facts.get(fid)
                if existing and existing.negated:
                    if existing.predicate == fact.predicate and existing.object == fact.object:
                        contradictions.append(existing)

        return contradictions

    def get_fact(self, fact_id: str) -> Optional[Fact]:
        """Get a fact by ID."""
        return self.facts.get(fact_id)

    def get_facts_about(
        self,
        subject: str,
        predicate: Optional[str] = None,
        include_negated: bool = False,
        min_confidence: float = 0.0
    ) -> List[Fact]:
        """Get facts about a subject."""
        fact_ids = self._subject_index.get(subject, set())

        results = []
        for fid in fact_ids:
            fact = self.facts.get(fid)
            if not fact:
                continue

            # Apply filters
            if predicate and fact.predicate != predicate:
                continue
            if not include_negated and fact.negated:
                continue
            if fact.confidence < min_confidence:
                continue

            results.append(fact)

        # Sort by confidence
        results.sort(key=lambda f: f.confidence, reverse=True)
        return results

    def get_facts_by_predicate(
        self,
        predicate: str,
        include_negated: bool = False
    ) -> List[Fact]:
        """Get all facts with a specific predicate."""
        fact_ids = self._predicate_index.get(predicate, set())

        results = []
        for fid in fact_ids:
            fact = self.facts.get(fid)
            if fact and (include_negated or not fact.negated):
                results.append(fact)

        return results

    def get_facts_by_category(
        self,
        category: str,
        subject: Optional[str] = None
    ) -> List[Fact]:
        """Get facts in a category."""
        fact_ids = self._category_index.get(category, set())

        results = []
        for fid in fact_ids:
            fact = self.facts.get(fid)
            if fact:
                if subject and fact.subject != subject:
                    continue
                results.append(fact)

        return results

    def search_similar(
        self,
        query_embedding: np.ndarray,
        top_k: int = 10,
        threshold: float = 0.0
    ) -> List[Tuple[Fact, float]]:
        """Search for similar facts by embedding."""
        if not self._embeddings:
            return []

        results = []
        for i, emb in enumerate(self._embeddings):
            sim = self._cosine_similarity(query_embedding, emb)
            if sim >= threshold:
                fact_id = self._embedding_ids[i]
                fact = self.facts.get(fact_id)
                if fact:
                    results.append((fact, sim))

        results.sort(key=lambda x: x[1], reverse=True)
        return results[:top_k]

    def check_negation(
        self,
        subject: str,
        predicate: str,
        obj: str
    ) -> Tuple[bool, Optional[Fact]]:
        """
        Check if a fact is explicitly negated.

        Returns: (is_negated, negating_fact)
        """
        subject_facts = self._subject_index.get(subject, set())

        for fid in subject_facts:
            fact = self.facts.get(fid)
            if fact and fact.negated:
                if fact.predicate == predicate and fact.object == obj:
                    return True, fact

        return False, None

    def add_negation(
        self,
        subject: str,
        predicate: str,
        obj: str,
        source_id: str
    ) -> str:
        """Explicitly add a negated fact."""
        fact = Fact(
            content=f"{subject} does NOT {predicate} {obj}",
            subject=subject,
            predicate=predicate,
            object=obj,
            negated=True,
            negation_source=source_id,
            sources=[source_id],
            confidence=1.0,
        )
        fact_id, _ = self.add_fact(fact)
        return fact_id

    def get_reliable_facts(
        self,
        subject: Optional[str] = None,
        min_confirmations: int = 1
    ) -> List[Fact]:
        """Get facts that are considered reliable."""
        facts = self.facts.values()

        if subject:
            fact_ids = self._subject_index.get(subject, set())
            facts = [self.facts[fid] for fid in fact_ids if fid in self.facts]

        return [
            f for f in facts
            if f.is_reliable and f.confirmation_count >= min_confirmations
        ]

    def get_contradicted_facts(self) -> List[Fact]:
        """Get facts that have contradictions."""
        return [f for f in self.facts.values() if f.contradictions]

    def get_negated_facts(self) -> List[Fact]:
        """Get all negated facts."""
        return [
            self.facts[fid] for fid in self._negated_facts
            if fid in self.facts
        ]

    def get_user_profile(self, user_id: str) -> Dict[str, Any]:
        """Get compiled profile for a user from their facts."""
        facts = self.get_facts_about(user_id, include_negated=True)

        profile = {
            "user_id": user_id,
            "preferences": [],
            "attributes": [],
            "relations": [],
            "dislikes": [],  # Negated preferences
        }

        for fact in facts:
            fact_info = {
                "predicate": fact.predicate,
                "value": fact.object,
                "confidence": fact.confidence,
            }

            if fact.negated:
                profile["dislikes"].append(fact_info)
            elif fact.category == "preference":
                profile["preferences"].append(fact_info)
            elif fact.category == "attribute":
                profile["attributes"].append(fact_info)
            elif fact.category == "relation":
                profile["relations"].append(fact_info)

        return profile

    def _cosine_similarity(self, a: np.ndarray, b: np.ndarray) -> float:
        """Compute cosine similarity."""
        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return float(np.dot(a, b) / (norm_a * norm_b))

    def get_stats(self) -> Dict[str, Any]:
        """Get statistics about semantic memory."""
        return {
            "total_facts": len(self.facts),
            "negated_facts": len(self._negated_facts),
            "contradicted_facts": len(self.get_contradicted_facts()),
            "unique_subjects": len(self._subject_index),
            "unique_predicates": len(self._predicate_index),
            "categories": {
                cat: len(fids)
                for cat, fids in self._category_index.items()
            },
        }

    def to_dict(self) -> Dict[str, Any]:
        """Serialize semantic memory.

        Note: Embeddings are NOT included. They must be saved separately
        and passed to from_dict().
        """
        return {
            "facts": [
                {
                    "id": f.id,
                    "content": f.content,
                    "subject": f.subject,
                    "predicate": f.predicate,
                    "object": f.object,
                    "confidence": f.confidence,
                    "sources": f.sources,
                    "contradictions": f.contradictions,
                    "first_learned": f.first_learned.isoformat(),
                    "last_confirmed": f.last_confirmed.isoformat(),
                    "confirmation_count": f.confirmation_count,
                    "negated": f.negated,
                    "negation_source": f.negation_source,
                    "category": f.category,
                    "tags": f.tags,
                    "metadata": f.metadata,
                }
                for f in self.facts.values()
            ]
        }

    @classmethod
    def from_dict(
        cls,
        data: Dict[str, Any],
        embeddings_map: Optional[Dict[str, np.ndarray]] = None,
    ) -> "SemanticMemoryStore":
        """Deserialize semantic memory from dictionary.

        Args:
            data: Output of to_dict().
            embeddings_map: Map of fact_id -> embedding.
        """
        embeddings_map = embeddings_map or {}
        store = cls()

        for fd in data.get("facts", []):
            fact = Fact(
                id=fd["id"],
                content=fd.get("content", ""),
                subject=fd.get("subject", ""),
                predicate=fd.get("predicate", ""),
                object=fd.get("object", ""),
                confidence=fd.get("confidence", 1.0),
                sources=fd.get("sources", []),
                contradictions=fd.get("contradictions", []),
                first_learned=(
                    datetime.fromisoformat(fd["first_learned"])
                    if fd.get("first_learned")
                    else datetime.now()
                ),
                last_confirmed=(
                    datetime.fromisoformat(fd["last_confirmed"])
                    if fd.get("last_confirmed")
                    else datetime.now()
                ),
                confirmation_count=fd.get("confirmation_count", 1),
                negated=fd.get("negated", False),
                negation_source=fd.get("negation_source"),
                embedding=embeddings_map.get(fd["id"]),
                category=fd.get("category", ""),
                tags=fd.get("tags", []),
                metadata=fd.get("metadata", {}),
            )
            # Directly populate to avoid duplicate-detection via add_fact
            store.facts[fact.id] = fact
            store._index_fact(fact)
            if fact.negated:
                store._negated_facts.add(fact.id)

        return store
