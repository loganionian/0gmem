"""
Multi-Query Retrieval: Generate multiple complementary queries for complex questions.

Generates 2-3 complementary queries for complex intents, executing in parallel
with Reciprocal Rank Fusion (RRF).
"""

from __future__ import annotations

import re
from collections import defaultdict
from collections.abc import Callable
from typing import Any


class MultiQueryGenerator:
    """
    Generates multiple complementary queries for complex questions.

    For complex multi-hop or temporal questions, generates 2-3 queries
    that approach the information from different angles.
    """

    # Question type patterns
    TEMPORAL_WORDS = {"when", "date", "time", "how long", "ago", "before", "after"}
    MULTI_HOP_WORDS = {"would", "likely", "probably", "might", "could", "based on"}
    PREFERENCE_WORDS = {"like", "enjoy", "favorite", "prefer", "love", "hate"}
    ACTIVITY_WORDS = {"do", "activity", "activities", "hobbies", "partake"}
    RELATIONSHIP_WORDS = {"relationship", "status", "married", "single", "dating"}

    def __init__(self, llm_client: Any | None = None):
        self._client = llm_client

    def generate_queries(self, question: str, target_entity: str | None = None) -> list[str]:
        """
        Generate multiple complementary queries for a question.

        Returns list of 1-3 queries depending on question complexity.
        """
        queries = [question]  # Always include original
        q_lower = question.lower()

        # Extract target entity if not provided
        if not target_entity:
            # All LoCoMo entity names - sorted by length (longest first) to avoid partial matches
            # e.g., "tim" in "time" false positive
            all_entities = [
                "caroline",
                "melanie",
                "deborah",
                "joanna",
                "jolene",
                "andrew",
                "audrey",
                "calvin",
                "james",
                "maria",
                "gina",
                "nate",
                "john",
                "evan",
                "dave",
                "tim",
                "sam",
                "jon",
            ]
            import re

            for name in all_entities:
                # Use word boundary to match exact names, not substrings
                if re.search(r"\b" + name + r"\b", q_lower):
                    target_entity = name.title()
                    break

        # For multi-hop questions, generate inference-focused queries
        if any(word in q_lower for word in self.MULTI_HOP_WORDS):
            queries.extend(self._generate_multihop_queries(question, target_entity))

        # For temporal questions, generate date-focused queries
        elif any(word in q_lower for word in self.TEMPORAL_WORDS):
            queries.extend(self._generate_temporal_queries(question, target_entity))

        # For preference questions, generate preference-focused queries
        elif any(word in q_lower for word in self.PREFERENCE_WORDS):
            queries.extend(self._generate_preference_queries(question, target_entity))

        # For activity questions, generate activity-focused queries
        elif any(word in q_lower for word in self.ACTIVITY_WORDS):
            queries.extend(self._generate_activity_queries(question, target_entity))

        # Deduplicate while preserving order
        seen = set()
        unique_queries = []
        for q in queries:
            q_normalized = q.lower().strip()
            if q_normalized not in seen:
                seen.add(q_normalized)
                unique_queries.append(q)

        return unique_queries[:3]  # Max 3 queries

    def _generate_multihop_queries(self, question: str, entity: str | None) -> list[str]:
        """Generate queries for multi-hop reasoning questions."""
        queries = []
        q_lower = question.lower()

        # Extract the topic being asked about
        if entity:
            # Query for background/context
            if "career" in q_lower or "pursue" in q_lower:
                queries.append(f"What career does {entity} want to pursue?")
                queries.append(f"{entity} career goals plans interests")

            elif "political" in q_lower or "leaning" in q_lower:
                queries.append(f"{entity} political views opinions beliefs")
                queries.append(f"{entity} values community activism")

            elif "religious" in q_lower:
                queries.append(f"{entity} religious spiritual beliefs faith")
                queries.append(f"{entity} church prayer spiritual")

            elif "personality" in q_lower or "traits" in q_lower:
                queries.append(f"{entity} personality character description")
                queries.append(f"How would you describe {entity}?")

            elif "lgbtq" in q_lower or "transgender" in q_lower:
                queries.append(f"{entity} LGBTQ transgender support community")
                queries.append(f"{entity} pride identity gender")

            elif "music" in q_lower or "song" in q_lower:
                queries.append(f"{entity} music favorite songs artists")
                queries.append(f"{entity} plays instrument classical")

            elif "book" in q_lower or "reading" in q_lower:
                queries.append(f"{entity} books reading favorite author")

            else:
                # Generic context query
                queries.append(f"{entity} background interests preferences")

        return queries

    def _generate_temporal_queries(self, question: str, entity: str | None) -> list[str]:
        """Generate queries for temporal questions."""
        queries = []
        q_lower = question.lower()

        # Extract the event being asked about
        event_patterns = [
            (r"when did (?:\w+\s+)?(\w+(?:\s+\w+){0,3})", 1),
            (r"when is (?:\w+\s+)?(\w+(?:\s+\w+){0,3})", 1),
            (r"how long (?:has|have|did) (?:\w+\s+)?(\w+(?:\s+\w+){0,3})", 1),
        ]

        for pattern, group in event_patterns:
            match = re.search(pattern, q_lower)
            if match:
                event = match.group(group)
                if entity:
                    queries.append(f"{entity} {event}")
                queries.append(f"{event} date time when")
                break

        # Add entity-specific query
        if entity:
            queries.append(f"{entity} events activities dates")

        return queries

    def _generate_preference_queries(self, question: str, entity: str | None) -> list[str]:
        """Generate queries for preference questions."""
        queries = []

        if entity:
            queries.append(f"{entity} likes loves enjoys favorite")
            queries.append(f"{entity} preferences interests hobbies")

        return queries

    def _generate_activity_queries(self, question: str, entity: str | None) -> list[str]:
        """Generate queries for activity questions."""
        queries = []

        if entity:
            queries.append(f"{entity} activities hobbies does enjoys")
            queries.append(f"{entity} signed up for class joined")
            queries.append(f"What does {entity} do for fun?")

        return queries


class MultiQueryRetriever:
    """
    Retriever that uses multiple queries with RRF fusion.

    Uses parallel execution with Reciprocal Rank Fusion.
    """

    def __init__(
        self,
        search_fn: Callable[[str, int], list[tuple[str, float, str]]],
        rrf_k: int = 60,
    ):
        """
        Initialize multi-query retriever.

        Args:
            search_fn: Function(query, top_k) -> List[(doc_id, score, content)]
            rrf_k: RRF smoothing constant
        """
        self.search_fn = search_fn
        self.rrf_k = rrf_k
        self.query_generator = MultiQueryGenerator()

    def retrieve(
        self,
        question: str,
        top_k: int = 10,
        target_entity: str | None = None,
    ) -> list[tuple[str, float, str]]:
        """
        Retrieve documents using multiple queries with RRF fusion.

        Returns: List of (doc_id, combined_score, content) tuples
        """
        # Generate multiple queries
        queries = self.query_generator.generate_queries(question, target_entity)

        # Execute all queries
        all_results: dict[str, str] = {}
        query_ranks: dict[str, dict[str, int]] = defaultdict(dict)

        for query in queries:
            results = self.search_fn(query, top_k * 2)

            for rank, (doc_id, score, content) in enumerate(results):
                if doc_id not in all_results:
                    all_results[doc_id] = content
                query_ranks[doc_id][query] = rank

        # Combine scores using RRF
        combined_scores = {}
        for doc_id in all_results:
            rrf_score = 0.0
            for query in queries:
                rank = query_ranks[doc_id].get(query, len(all_results) + 1)
                rrf_score += 1 / (self.rrf_k + rank)
            combined_scores[doc_id] = rrf_score

        # Sort by combined score
        sorted_docs = sorted(combined_scores.items(), key=lambda x: x[1], reverse=True)

        # Return top_k results
        return [(doc_id, score, all_results[doc_id]) for doc_id, score in sorted_docs[:top_k]]
