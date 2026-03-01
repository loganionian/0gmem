"""
Semantic Profile Matcher: Use embeddings to match questions to profile facts.

INNOVATION: Unlike keyword-based profile lookup, we use semantic similarity
to find relevant profile facts even when the wording differs.

Example:
- Question: "What creative activities does Caroline enjoy?"
- Profile has: {"art": ["abstract art", "pottery"], "activity": ["painting"]}
- Keyword match would miss "creative activities" -> "art"
- Semantic match finds the connection

This dramatically improves single-hop accuracy.
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass
from typing import Optional, List, Dict, Any, Tuple, Callable


@dataclass
class ProfileMatch:
    """A match between a question and a profile fact."""
    fact_type: str
    fact_values: List[str]
    similarity_score: float
    matched_text: str


class SemanticProfileMatcher:
    """
    Matches questions to profile facts using semantic similarity.

    INNOVATION: Precomputes embeddings for profile fact types and their
    associated question patterns, enabling fast semantic matching.
    """

    # Mapping of fact types to question patterns they answer
    FACT_TYPE_PATTERNS = {
        "identity": [
            "what is identity",
            "who is",
            "gender identity",
            "transgender",
        ],
        "relationship_status": [
            "relationship status",
            "is married",
            "is single",
            "dating status",
        ],
        "location": [
            "where from",
            "where live",
            "country",
            "moved from",
            "home country",
        ],
        "activity": [
            "what activities",
            "hobbies",
            "what does do",
            "leisure activities",
            "free time",
        ],
        "art": [
            "what art",
            "creative activities",
            "artistic",
            "painting",
            "pottery",
            "what kind of art",
        ],
        "career": [
            "what career",
            "job",
            "profession",
            "work",
            "pursue",
        ],
        "book": [
            "what books",
            "reading",
            "favorite book",
            "read",
        ],
        "instrument": [
            "what instrument",
            "play music",
            "musical",
        ],
        "favorite_musician": [
            "favorite musician",
            "music taste",
            "listen to",
        ],
        "pets": [
            "what pets",
            "animals",
            "pet type",
        ],
        "pet_name": [
            "pet name",
            "what is pet called",
        ],
        "num_children": [
            "how many children",
            "how many kids",
            "number of kids",
        ],
        "kids_like": [
            "what kids like",
            "children interests",
            "kids enjoy",
        ],
        "lgbtq_participation": [
            "lgbtq community",
            "participating lgbtq",
            "pride",
            "activist",
        ],
        "friends_years": [
            "how long friends",
            "years friendship",
            "known for",
        ],
        "married_years": [
            "how long married",
            "years married",
            "marriage duration",
        ],
        "researched": [
            "what researched",
            "researching",
            "looking into",
        ],
        "camped_at": [
            "where camp",
            "camping location",
            "went camping",
        ],
        "painted": [
            "what painted",
            "painting of",
        ],
    }

    def __init__(self, embedding_fn: Optional[Callable[[str], np.ndarray]] = None):
        """
        Initialize the matcher.

        Args:
            embedding_fn: Function to compute embeddings
        """
        self.embedding_fn = embedding_fn
        self.fact_type_embeddings: Dict[str, np.ndarray] = {}
        self._pattern_cache: Dict[str, np.ndarray] = {}

    def precompute_embeddings(self) -> None:
        """Precompute embeddings for all fact type patterns."""
        if not self.embedding_fn:
            return

        for fact_type, patterns in self.FACT_TYPE_PATTERNS.items():
            # Compute embedding as average of pattern embeddings
            pattern_embeds = []
            for pattern in patterns:
                if pattern not in self._pattern_cache:
                    embed = self.embedding_fn(pattern)
                    self._pattern_cache[pattern] = embed
                pattern_embeds.append(self._pattern_cache[pattern])

            if pattern_embeds:
                self.fact_type_embeddings[fact_type] = np.mean(pattern_embeds, axis=0)

    def match_question_to_profile(
        self,
        question: str,
        profile: Dict[str, List[str]],
        top_k: int = 3,
    ) -> List[ProfileMatch]:
        """
        Match a question to relevant profile facts using semantic similarity.

        Returns list of ProfileMatch objects sorted by similarity.
        """
        if not profile:
            return []

        # If we have embeddings, use semantic matching
        if self.embedding_fn and self.fact_type_embeddings:
            return self._semantic_match(question, profile, top_k)

        # Fallback to keyword matching
        return self._keyword_match(question, profile, top_k)

    def _semantic_match(
        self,
        question: str,
        profile: Dict[str, List[str]],
        top_k: int,
    ) -> List[ProfileMatch]:
        """Match using semantic similarity."""
        # Get question embedding
        q_embed = self.embedding_fn(question)

        # Score each fact type
        matches = []
        for fact_type, values in profile.items():
            if fact_type not in self.fact_type_embeddings:
                continue

            fact_embed = self.fact_type_embeddings[fact_type]

            # Cosine similarity
            similarity = np.dot(q_embed, fact_embed) / (
                np.linalg.norm(q_embed) * np.linalg.norm(fact_embed) + 1e-8
            )

            if similarity > 0.3:  # Threshold
                matches.append(ProfileMatch(
                    fact_type=fact_type,
                    fact_values=values,
                    similarity_score=float(similarity),
                    matched_text=", ".join(values),
                ))

        # Sort by similarity
        matches.sort(key=lambda m: m.similarity_score, reverse=True)
        return matches[:top_k]

    def _keyword_match(
        self,
        question: str,
        profile: Dict[str, List[str]],
        top_k: int,
    ) -> List[ProfileMatch]:
        """Match using keyword overlap."""
        q_lower = question.lower()
        q_words = set(q_lower.split())

        matches = []
        for fact_type, values in profile.items():
            # Check patterns for this fact type
            patterns = self.FACT_TYPE_PATTERNS.get(fact_type, [])
            best_score = 0

            for pattern in patterns:
                pattern_words = set(pattern.split())
                overlap = len(q_words & pattern_words)
                score = overlap / max(len(pattern_words), 1)
                best_score = max(best_score, score)

            # Also check if fact type words appear in question
            fact_words = set(fact_type.replace("_", " ").split())
            direct_overlap = len(q_words & fact_words)
            if direct_overlap > 0:
                best_score = max(best_score, direct_overlap / len(fact_words))

            if best_score > 0.2:
                matches.append(ProfileMatch(
                    fact_type=fact_type,
                    fact_values=values,
                    similarity_score=best_score,
                    matched_text=", ".join(values),
                ))

        matches.sort(key=lambda m: m.similarity_score, reverse=True)
        return matches[:top_k]

    def answer_from_semantic_match(
        self,
        question: str,
        profile: Dict[str, List[str]],
    ) -> Optional[str]:
        """
        INNOVATION: Answer a question by finding semantically matching profile facts.

        This handles questions that don't exactly match profile field names.
        """
        matches = self.match_question_to_profile(question, profile, top_k=1)

        if not matches:
            return None

        best_match = matches[0]

        # Only return if confidence is high enough
        if best_match.similarity_score < 0.4:
            return None

        return best_match.matched_text


class AdaptiveProfileAnswerer:
    """
    INNOVATION: Combines multiple profile matching strategies.

    1. Exact keyword match (fastest, highest precision)
    2. Semantic similarity match (handles paraphrasing)
    3. Inference from related facts (handles indirect questions)
    """

    def __init__(
        self,
        embedding_fn: Optional[Callable[[str], np.ndarray]] = None,
        llm_client: Optional[Any] = None,
    ):
        self.semantic_matcher = SemanticProfileMatcher(embedding_fn)
        self._client = llm_client
        self._inference_rules = self._build_inference_rules()

    def _build_inference_rules(self) -> Dict[str, List[Tuple[str, str]]]:
        """Build rules for inferring facts from other facts."""
        return {
            # If we know X, we can infer Y
            "single_parent": [("relationship_status", "single")],
            "transgender": [("identity", "transgender")],
            "plays_instrument": [("activity", "music")],
        }

    def answer(
        self,
        question: str,
        profile: Dict[str, List[str]],
        target_entity: str,
    ) -> Optional[str]:
        """
        Answer a question using adaptive profile matching.

        Tries strategies in order of precision:
        1. Direct keyword match
        2. Semantic similarity match
        3. Inference from related facts
        """
        # Strategy 1: Direct keyword match (existing logic)
        direct_answer = self._direct_match(question, profile)
        if direct_answer:
            return direct_answer

        # Strategy 2: Semantic similarity match
        semantic_answer = self.semantic_matcher.answer_from_semantic_match(question, profile)
        if semantic_answer:
            return semantic_answer

        # Strategy 3: Inference from related facts
        inferred_answer = self._infer_answer(question, profile)
        if inferred_answer:
            return inferred_answer

        return None

    def _direct_match(self, question: str, profile: Dict[str, List[str]]) -> Optional[str]:
        """Direct keyword matching (existing logic from llm_fact_extractor)."""
        q_lower = question.lower()

        # Identity
        if "identity" in q_lower and "identity" in profile:
            return ", ".join(profile["identity"])

        # Relationship
        if ("relationship" in q_lower or "status" in q_lower) and "relationship_status" in profile:
            return ", ".join(profile["relationship_status"])

        # Location
        if ("from" in q_lower or "country" in q_lower) and "location" in profile:
            return ", ".join(profile["location"])

        # Activities
        if any(w in q_lower for w in ["activities", "hobbies", "partake"]):
            activities = []
            for key in ["activity", "preference"]:
                if key in profile:
                    activities.extend(profile[key])
            if activities:
                return ", ".join(set(activities))

        # Art
        if "art" in q_lower:
            for key in ["art", "painted"]:
                if key in profile:
                    return ", ".join(profile[key])

        # Career
        if any(w in q_lower for w in ["career", "pursue", "work"]):
            if "career" in profile:
                return ", ".join(profile["career"])

        # Children
        if "how many" in q_lower and ("children" in q_lower or "kids" in q_lower):
            if "num_children" in profile:
                return profile["num_children"][0]

        # Pets
        if "pet" in q_lower and "name" in q_lower:
            if "pet_name" in profile:
                return ", ".join(profile["pet_name"])

        # Books
        if "book" in q_lower or "read" in q_lower:
            if "book" in profile:
                return ", ".join(profile["book"])

        # Instrument
        if "instrument" in q_lower:
            if "instrument" in profile:
                return ", ".join(profile["instrument"])

        # LGBTQ
        if "lgbtq" in q_lower and "community" in q_lower:
            if "lgbtq_participation" in profile:
                return ", ".join(profile["lgbtq_participation"])

        return None

    def _infer_answer(self, question: str, profile: Dict[str, List[str]]) -> Optional[str]:
        """Infer answer from related facts."""
        q_lower = question.lower()

        # Example inference: "creative activities" -> look at art + activity
        if "creative" in q_lower:
            creative = []
            for key in ["art", "painted", "activity"]:
                if key in profile:
                    for val in profile[key]:
                        if any(w in val.lower() for w in ["paint", "pottery", "art", "draw", "create"]):
                            creative.append(val)
            if creative:
                return ", ".join(set(creative))

        # Example inference: "music related" -> instrument + musician
        if "music" in q_lower:
            music = []
            for key in ["instrument", "favorite_musician", "activity"]:
                if key in profile:
                    for val in profile[key]:
                        music.append(val)
            if music:
                return ", ".join(set(music))

        return None
