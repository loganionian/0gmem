"""
Attention Filter: Implements "precise forgetting" for context selection.

High-quality memory requires not only precise remembering but also precise forgetting.
This filter removes redundant and low-relevance content that dilutes LLM attention.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass
class FilterConfig:
    """Configuration for the attention filter."""

    relevance_threshold: float = 0.3  # Min relevance score to keep
    max_context_tokens: int = 2000  # Target context size
    diversity_weight: float = 0.3  # How much to prioritize diversity vs relevance
    semantic_similarity_threshold: float = 0.85  # Threshold for deduplication
    chars_per_token: int = 4  # Rough estimate for token counting


class AttentionFilter:
    """
    Filters retrieved context to only essential information.

    Applies "precise forgetting" - acting as an intelligent attention
    filter to reduce cognitive load and direct focus to critical information.

    Features:
    - Relevance scoring per item
    - Redundancy removal (semantic deduplication)
    - Diversity preservation (don't return 5 similar results)
    - Token budget management
    """

    def __init__(
        self,
        config: FilterConfig | None = None,
        embedding_fn: Callable[..., Any] | None = None,
    ):
        self.config = config or FilterConfig()
        self._embedding_fn = embedding_fn

    def filter_context(
        self,
        query: str,
        results: list[Any],  # RetrievalResult objects
        query_analysis: Any | None = None,
    ) -> list[Any]:
        """
        Filter retrieved results to only essential information.

        Args:
            query: The original query
            results: List of RetrievalResult objects
            query_analysis: Optional QueryAnalysis for additional context

        Returns:
            Filtered list of results, respecting token budget
        """
        if not results:
            return []

        # Step 1: Score relevance for each result
        scored_results = self._score_relevance(query, results, query_analysis)

        # Step 2: Remove low-relevance items
        filtered = [
            (r, score) for r, score in scored_results if score >= self.config.relevance_threshold
        ]

        # Step 3: Remove semantic duplicates
        deduped = self._remove_semantic_duplicates(filtered)

        # Step 4: Ensure diversity (avoid too many similar items)
        diverse = self._ensure_diversity(deduped, query_analysis)

        # Step 5: Apply token budget
        budgeted = self._apply_token_budget(diverse)

        # Return just the results (not scores)
        return [r for r, _ in budgeted]

    def _score_relevance(
        self,
        query: str,
        results: list[Any],
        query_analysis: Any | None,
    ) -> list[tuple[Any, float]]:
        """Score each result's relevance to the query."""
        scored = []
        query_lower = query.lower()
        query_words = set(query_lower.split())

        # Get entities and keywords from analysis if available
        entities = set()
        keywords = set()
        if query_analysis:
            if hasattr(query_analysis, "entities"):
                entities = set(e.lower() for e in query_analysis.entities)
            if hasattr(query_analysis, "keywords"):
                keywords = set(k.lower() for k in query_analysis.keywords)

        for result in results:
            score = result.score  # Start with retrieval score

            content_lower = result.content.lower()
            content_words = set(content_lower.split())

            # Boost for entity overlap
            if entities:
                entity_hits = sum(1 for e in entities if e in content_lower)
                score *= 1 + 0.2 * entity_hits

            # Boost for keyword overlap
            if keywords:
                keyword_hits = sum(1 for k in keywords if k in content_lower)
                score *= 1 + 0.1 * keyword_hits

            # Boost for query word overlap
            word_overlap = len(query_words & content_words) / max(len(query_words), 1)
            score *= 1 + 0.15 * word_overlap

            # Penalty for very long content (might be noise)
            content_len = len(result.content)
            if content_len > 500:
                length_penalty = max(0.7, 1 - (content_len - 500) / 2000)
                score *= length_penalty

            # Boost for negation-related content if query is preference-related
            if result.negated and any(
                w in query_lower for w in ["like", "love", "prefer", "enjoy"]
            ):
                score *= 1.3

            scored.append((result, score))

        return scored

    def _remove_semantic_duplicates(
        self, results: list[tuple[Any, float]]
    ) -> list[tuple[Any, float]]:
        """Remove semantically similar results, keeping highest scored."""
        if not results or not self._embedding_fn:
            return results

        # Sort by score descending
        sorted_results = sorted(results, key=lambda x: x[1], reverse=True)
        kept = []
        kept_embeddings: list[np.ndarray] = []

        for result, score in sorted_results:
            # Get embedding for this result
            try:
                emb = self._embedding_fn(result.content[:500])  # Limit for efficiency
                if emb is None:
                    kept.append((result, score))
                    continue

                # Check similarity with kept embeddings
                is_duplicate = False
                for kept_emb in kept_embeddings:
                    similarity = self._cosine_similarity(emb, kept_emb)
                    if similarity > self.config.semantic_similarity_threshold:
                        is_duplicate = True
                        break

                if not is_duplicate:
                    kept.append((result, score))
                    kept_embeddings.append(emb)

            except Exception:
                # If embedding fails, keep the result
                kept.append((result, score))

        return kept

    def _ensure_diversity(
        self,
        results: list[tuple[Any, float]],
        query_analysis: Any | None,
    ) -> list[tuple[Any, float]]:
        """Ensure diversity in results - don't return too many similar items."""
        if len(results) <= 3:
            return results

        # Group by source
        by_source: dict[str, list[tuple[Any, float]]] = {}
        for result, score in results:
            source = getattr(result, "source", "unknown")
            if source not in by_source:
                by_source[source] = []
            by_source[source].append((result, score))

        # If one source dominates (>70%), diversify
        total = len(results)
        diverse = []

        for source, source_results in by_source.items():
            if len(source_results) / total > 0.7:
                # Take top 50% from this source
                max_from_source = max(3, len(source_results) // 2)
                diverse.extend(
                    sorted(source_results, key=lambda x: x[1], reverse=True)[:max_from_source]
                )
            else:
                diverse.extend(source_results)

        # Sort by score
        diverse.sort(key=lambda x: x[1], reverse=True)
        return diverse

    def _apply_token_budget(self, results: list[tuple[Any, float]]) -> list[tuple[Any, float]]:
        """Apply token budget to limit context size."""
        max_chars = self.config.max_context_tokens * self.config.chars_per_token
        total_chars = 0
        budgeted = []

        for result, score in results:
            content_len = len(result.content)
            if total_chars + content_len <= max_chars:
                budgeted.append((result, score))
                total_chars += content_len
            elif total_chars < max_chars:
                # Truncate this result to fit
                # (handled during composition, not here)
                budgeted.append((result, score))
                break

        return budgeted

    def _cosine_similarity(self, a: np.ndarray, b: np.ndarray) -> float:
        """Compute cosine similarity between two vectors."""
        if a is None or b is None:
            return 0.0
        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return float(np.dot(a, b) / (norm_a * norm_b))

    def compute_sufficiency_score(
        self,
        query: str,
        results: list[Any],
        query_analysis: Any | None = None,
    ) -> float:
        """
        Compute a score indicating how sufficient the context is for answering the query.

        Returns a score between 0 and 1:
        - 0: No relevant context found
        - 0.5: Partial context
        - 1.0: High-confidence sufficient context

        This can be used for agentic retrieval to determine if more rounds are needed.
        """
        if not results:
            return 0.0

        scores = []

        # Factor 1: Entity coverage
        if query_analysis and hasattr(query_analysis, "entities") and query_analysis.entities:
            entities = set(e.lower() for e in query_analysis.entities)
            entity_hits = 0
            for result in results[:10]:
                content_lower = result.content.lower()
                for entity in entities:
                    if entity in content_lower:
                        entity_hits += 1
                        break
            entity_coverage = entity_hits / max(len(results[:10]), 1)
            scores.append(entity_coverage)

        # Factor 2: Average relevance score of top results
        top_scores = [r.score for r in results[:5]]
        if top_scores:
            avg_score = sum(top_scores) / len(top_scores)
            # Normalize to 0-1 range (assuming scores are typically 0-2)
            normalized_score = min(1.0, avg_score / 2.0)
            scores.append(normalized_score)

        # Factor 3: Result count adequacy
        result_count_score = min(1.0, len(results) / 10)
        scores.append(result_count_score)

        # Factor 4: Keyword coverage
        if query_analysis and hasattr(query_analysis, "keywords") and query_analysis.keywords:
            keywords = set(k.lower() for k in query_analysis.keywords[:5])
            keyword_hits = 0
            for result in results[:10]:
                content_lower = result.content.lower()
                for kw in keywords:
                    if kw in content_lower:
                        keyword_hits += 1
                        break
            keyword_coverage = keyword_hits / max(len(results[:10]), 1)
            scores.append(keyword_coverage)

        return sum(scores) / len(scores) if scores else 0.5
