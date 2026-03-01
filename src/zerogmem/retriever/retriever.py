"""
Retriever: Main retrieval pipeline for 0GMem.

Implements multi-strategy retrieval with graph traversal and
position-aware composition to combat lost-in-the-middle.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

import numpy as np

from zerogmem.memory.manager import MemoryManager
from zerogmem.retriever.attention_filter import AttentionFilter, FilterConfig
from zerogmem.retriever.query_analyzer import (
    QueryAnalysis,
    QueryAnalyzer,
    QueryIntent,
    ReasoningType,
    TemporalScope,
)
from zerogmem.retriever.reranker import CrossEncoderReranker


@dataclass
class RetrievalResult:
    """A single retrieval result."""

    id: str
    content: str
    score: float
    source: str  # semantic, temporal, entity, fact, episode, working_memory
    reasoning_path: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    # Additional context
    entities: list[str] = field(default_factory=list)
    timestamp: str | None = None
    confidence: float = 1.0
    negated: bool = False


@dataclass
class RetrievalResponse:
    """Complete retrieval response."""

    query: str
    query_analysis: QueryAnalysis
    results: list[RetrievalResult]
    composed_context: str
    strategy_used: dict[str, Any]
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class RetrieverConfig:
    """Configuration for the retriever."""

    top_k: int = 20  # Increased for larger conversations
    semantic_weight: float = 0.4
    temporal_weight: float = 0.3
    entity_weight: float = 0.2
    recency_weight: float = 0.1
    max_context_tokens: int = 8000  # Increased for larger conversations
    use_position_aware_composition: bool = True
    check_negations: bool = True
    use_reranker: bool = False
    rerank_top_n: int = 30
    rerank_weight: float = 0.6
    reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    # RRF (Reciprocal Rank Fusion) settings
    use_rrf: bool = True
    rrf_k: int = 60  # Standard RRF constant
    # Agentic retrieval settings
    use_agentic_retrieval: bool = True
    max_retrieval_rounds: int = 3
    sufficiency_threshold: float = 0.6  # Min confidence to stop retrieval
    # Attention filter settings - "precise forgetting"
    use_attention_filter: bool = True
    attention_relevance_threshold: float = 0.3
    attention_diversity_weight: float = 0.3


class Retriever:
    """
    Main retrieval pipeline for 0GMem.

    Implements:
    - Query understanding and intent classification
    - Multi-strategy retrieval (semantic, temporal, entity, graph)
    - Multi-hop reasoning for complex queries
    - Negation checking for adversarial robustness
    - Position-aware composition for lost-in-the-middle mitigation
    """

    def __init__(
        self,
        memory_manager: MemoryManager,
        config: RetrieverConfig | None = None,
        embedding_fn: Callable[[str], np.ndarray] | None = None,
    ):
        self.memory = memory_manager
        self.config = config or RetrieverConfig()
        self._embedding_fn = embedding_fn or memory_manager._embed_fn

        self.query_analyzer = QueryAnalyzer()
        self.reranker = None
        if self.config.use_reranker:
            try:
                self.reranker = CrossEncoderReranker(model_name=self.config.reranker_model)
            except Exception as e:
                print(f"Reranker init failed: {e}")

        # Initialize attention filter for "precise forgetting"
        self.attention_filter = None
        if self.config.use_attention_filter:
            filter_config = FilterConfig(
                relevance_threshold=self.config.attention_relevance_threshold,
                diversity_weight=self.config.attention_diversity_weight,
                max_context_tokens=self.config.max_context_tokens // 2,  # Leave room for formatting
            )
            self.attention_filter = AttentionFilter(
                config=filter_config,
                embedding_fn=self._embedding_fn,
            )

    def retrieve(
        self,
        query: str,
        context: dict[str, Any] | None = None,
        override_strategy: dict[str, Any] | None = None,
    ) -> RetrievalResponse:
        """
        Main retrieval method.

        Args:
            query: User query
            context: Optional context (conversation history, etc.)
            override_strategy: Override automatic strategy selection

        Returns:
            RetrievalResponse with results and composed context
        """
        # Use agentic retrieval if enabled
        if self.config.use_agentic_retrieval:
            return self._agentic_retrieve(query, context, override_strategy)

        return self._single_pass_retrieve(query, context, override_strategy)

    def _single_pass_retrieve(
        self,
        query: str,
        context: dict[str, Any] | None = None,
        override_strategy: dict[str, Any] | None = None,
    ) -> RetrievalResponse:
        """Single-pass retrieval (original implementation)."""
        # Analyze query
        analysis = self.query_analyzer.analyze(query, context)

        # Determine strategy
        strategy = self.query_analyzer.get_retrieval_strategy(analysis)
        if override_strategy:
            strategy.update(override_strategy)

        # Execute retrieval based on strategy
        results = self._execute_retrieval(query, analysis, strategy)

        # Check negations if needed
        if strategy.get("check_negations") and analysis.is_negation_check:
            results = self._verify_negations(results, analysis)

        # Optional cross-encoder reranking
        if self.reranker and self.config.use_reranker and results:
            results = self._apply_reranker(query, results)

        # Score and rank results
        results = self._rank_results(results, analysis, strategy)

        # Apply attention filter for "precise forgetting"
        # This removes redundant/low-relevance content that dilutes LLM attention
        pre_filter_count = len(results)
        if self.attention_filter and self.config.use_attention_filter:
            results = self.attention_filter.filter_context(query, results, query_analysis=analysis)

        # Compose context with position-awareness
        composed_context = self._compose_context(results, analysis)

        return RetrievalResponse(
            query=query,
            query_analysis=analysis,
            results=results[: self.config.top_k],
            composed_context=composed_context,
            strategy_used=strategy,
            metadata={
                "total_candidates": pre_filter_count,
                "filtered_candidates": len(results),
                "reasoning_type": analysis.reasoning_type.value,
                "intent": analysis.intent.value,
                "attention_filter_applied": self.config.use_attention_filter,
            },
        )

    def _agentic_retrieve(
        self,
        query: str,
        context: dict[str, Any] | None = None,
        override_strategy: dict[str, Any] | None = None,
    ) -> RetrievalResponse:
        """
        Agentic retrieval with sufficiency checking and query rewriting.

        Multi-step retrieval that checks if context is sufficient
        and rewrites queries when needed.
        """
        all_results: list[RetrievalResult] = []
        queries_tried: list[str] = [query]
        round_metadata: list[dict[str, Any]] = []

        # First pass with original query
        response = self._single_pass_retrieve(query, context, override_strategy)
        all_results.extend(response.results)

        # Check sufficiency
        sufficiency = self._check_sufficiency(query, response)
        round_metadata.append(
            {
                "round": 1,
                "query": query,
                "results_count": len(response.results),
                "sufficiency": sufficiency,
            }
        )

        # Multi-round retrieval if insufficient
        for round_num in range(2, self.config.max_retrieval_rounds + 1):
            if sufficiency >= self.config.sufficiency_threshold:
                break

            # Generate rewritten query
            rewritten_query = self._rewrite_query(
                original_query=query,
                previous_queries=queries_tried,
                current_results=all_results,
                analysis=response.query_analysis,
            )

            if not rewritten_query or rewritten_query in queries_tried:
                break

            queries_tried.append(rewritten_query)

            # Retrieve with rewritten query
            new_response = self._single_pass_retrieve(rewritten_query, context, override_strategy)

            # Add new results (avoiding duplicates)
            seen_ids = {r.id for r in all_results}
            for r in new_response.results:
                if r.id not in seen_ids:
                    all_results.append(r)
                    seen_ids.add(r.id)

            # Re-check sufficiency
            sufficiency = self._check_sufficiency(
                query,
                RetrievalResponse(
                    query=query,
                    query_analysis=response.query_analysis,
                    results=all_results,
                    composed_context="",
                    strategy_used=response.strategy_used,
                ),
            )

            round_metadata.append(
                {
                    "round": round_num,
                    "query": rewritten_query,
                    "new_results_count": len(new_response.results),
                    "total_results_count": len(all_results),
                    "sufficiency": sufficiency,
                }
            )

        # Final ranking and composition with all accumulated results
        final_results = self._rank_results(
            all_results,
            response.query_analysis,
            response.strategy_used,
        )[: self.config.top_k]

        composed_context = self._compose_context(final_results, response.query_analysis)

        return RetrievalResponse(
            query=query,
            query_analysis=response.query_analysis,
            results=final_results,
            composed_context=composed_context,
            strategy_used=response.strategy_used,
            metadata={
                "total_candidates": len(all_results),
                "reasoning_type": response.query_analysis.reasoning_type.value,
                "intent": response.query_analysis.intent.value,
                "agentic_retrieval": True,
                "rounds": len(round_metadata),
                "queries_tried": queries_tried,
                "round_details": round_metadata,
                "final_sufficiency": sufficiency,
            },
        )

    def _check_sufficiency(self, query: str, response: RetrievalResponse) -> float:
        """
        Check if retrieved context is sufficient to answer the query.

        Returns a score between 0 and 1.
        """
        if not response.results:
            return 0.0

        # Multiple heuristics for sufficiency
        scores = []

        # 1. Entity coverage - are query entities in results?
        if response.query_analysis.entities:
            entity_hits = 0
            for entity in response.query_analysis.entities:
                for r in response.results[:10]:
                    if entity.lower() in r.content.lower():
                        entity_hits += 1
                        break
            entity_coverage = entity_hits / len(response.query_analysis.entities)
            scores.append(entity_coverage)

        # 2. Keyword coverage
        if response.query_analysis.keywords:
            keyword_hits = 0
            for kw in response.query_analysis.keywords[:5]:
                for r in response.results[:10]:
                    if kw.lower() in r.content.lower():
                        keyword_hits += 1
                        break
            keyword_coverage = keyword_hits / min(5, len(response.query_analysis.keywords))
            scores.append(keyword_coverage)

        # 3. Score quality - average score of top results
        top_scores = [r.score for r in response.results[:5]]
        if top_scores:
            avg_score = sum(top_scores) / len(top_scores)
            score_quality = min(1.0, avg_score)  # Normalize
            scores.append(score_quality)

        # 4. Result count
        result_count_score = min(1.0, len(response.results) / 10)
        scores.append(result_count_score)

        # Weighted average
        if not scores:
            return 0.5  # Default
        return sum(scores) / len(scores)

    def _rewrite_query(
        self,
        original_query: str,
        previous_queries: list[str],
        current_results: list[RetrievalResult],
        analysis: QueryAnalysis,
    ) -> str | None:
        """
        Rewrite query to find missing information.

        Simple rule-based rewriting. Can be enhanced with LLM later.
        """
        # Extract what we might be missing
        query_lower = original_query.lower()

        # Strategy 1: Focus on specific entities not found
        for entity in analysis.entities:
            entity_found = any(entity.lower() in r.content.lower() for r in current_results[:10])
            if not entity_found:
                # Try query focused on this entity
                new_query = f"{entity} {' '.join(analysis.keywords[:3])}"
                if new_query not in previous_queries:
                    return new_query

        # Strategy 2: Use synonyms/related terms for keywords
        keyword_expansions = {
            "like": ["enjoy", "love", "prefer", "favorite"],
            "when": ["date", "time", "year", "month"],
            "where": ["location", "place", "city", "country"],
            "what": ["which", "type", "kind"],
            "hobby": ["activity", "interest", "pastime"],
            "work": ["job", "career", "profession", "occupation"],
            "live": ["reside", "stay", "home", "address"],
        }

        for kw in analysis.keywords[:3]:
            if kw.lower() in keyword_expansions:
                for expansion in keyword_expansions[kw.lower()]:
                    new_query = original_query.replace(kw, expansion)
                    if new_query not in previous_queries:
                        return new_query

        # Strategy 3: Decompose multi-hop queries
        if analysis.reasoning_type == ReasoningType.MULTI_HOP:
            # Try searching for entities separately
            if len(analysis.entities) > 1:
                for entity in analysis.entities:
                    simple_query = f"{entity}"
                    if simple_query not in previous_queries:
                        return simple_query

        # Strategy 4: Add temporal context
        if "when" in query_lower or "date" in query_lower or "time" in query_lower:
            if analysis.entities:
                new_query = f"{analysis.entities[0]} timeline events dates"
                if new_query not in previous_queries:
                    return new_query

        return None

    def _execute_retrieval(
        self,
        query: str,
        analysis: QueryAnalysis,
        strategy: dict[str, Any],
    ) -> list[RetrievalResult]:
        """Execute retrieval based on strategy."""
        # Get query embedding
        query_embedding = None
        if self._embedding_fn and strategy.get("use_semantic_search", True):
            query_embedding = self._embedding_fn(query)

        # Collect results from each strategy separately for RRF
        strategy_results: dict[str, list[RetrievalResult]] = {}

        # 1. Semantic search - use higher limit for larger conversations
        if query_embedding is not None:
            strategy_results["semantic"] = self._semantic_search(
                query_embedding, strategy.get("top_k", 30)
            )

        # 2. Entity-based search
        if strategy.get("use_entity_search") and analysis.entities:
            strategy_results["entity"] = self._entity_search(analysis.entities)

        # 3. Temporal search
        if strategy.get("use_temporal_search"):
            strategy_results["temporal"] = self._temporal_search(analysis)

        # 4. Multi-hop graph traversal
        if strategy.get("use_graph_traversal"):
            strategy_results["graph"] = self._graph_traversal(
                analysis, query_embedding, max_hops=strategy.get("max_hops", 2)
            )

        # 5. Fact search
        strategy_results["fact"] = self._fact_search(query_embedding, analysis)

        # 6. Working memory
        strategy_results["working_memory"] = self._working_memory_search(query_embedding)

        # 7. Keyword-based search (for multi-hop and commonsense)
        if analysis.reasoning_type in [ReasoningType.MULTI_HOP, ReasoningType.COMMONSENSE]:
            strategy_results["keyword"] = self._keyword_search(
                analysis.keywords + analysis.entities
            )

        # Use RRF fusion if enabled, otherwise simple merge
        if self.config.use_rrf:
            all_results = self._rrf_fusion(strategy_results, analysis)
        else:
            all_results = []
            for results in strategy_results.values():
                all_results.extend(results)
            all_results = self._deduplicate(all_results)

        # 8. Dialogue context (get surrounding messages for multi-hop)
        if analysis.reasoning_type == ReasoningType.MULTI_HOP:
            dialogue_results = self._get_dialogue_context(all_results)
            all_results.extend(dialogue_results)
            all_results = self._deduplicate(all_results)

        return all_results

    def _rrf_fusion(
        self,
        strategy_results: dict[str, list[RetrievalResult]],
        analysis: QueryAnalysis,
    ) -> list[RetrievalResult]:
        """
        Reciprocal Rank Fusion (RRF) to combine results from multiple strategies.

        RRF(d) = Σ 1/(k + rank_i(d))

        This method is proven to work better than weighted score combination
        because it normalizes across different scoring scales.
        """
        k = self.config.rrf_k

        # Strategy weights based on query type
        strategy_weights = self._get_strategy_weights(analysis)

        # Build rank mappings for each strategy
        strategy_ranks: dict[str, dict[str, int]] = {}
        for strategy_name, results in strategy_results.items():
            # Sort by score within strategy
            sorted_results = sorted(results, key=lambda r: r.score, reverse=True)
            strategy_ranks[strategy_name] = {r.id: rank for rank, r in enumerate(sorted_results)}

        # Collect all unique document IDs
        all_doc_ids = set()
        doc_to_result: dict[str, RetrievalResult] = {}
        for results in strategy_results.values():
            for r in results:
                all_doc_ids.add(r.id)
                # Keep the result with highest original score
                if r.id not in doc_to_result or r.score > doc_to_result[r.id].score:
                    doc_to_result[r.id] = r

        # Compute RRF scores
        rrf_scores: dict[str, float] = {}
        for doc_id in all_doc_ids:
            rrf_score = 0.0
            for strategy_name, ranks in strategy_ranks.items():
                weight = strategy_weights.get(strategy_name, 1.0)
                if doc_id in ranks:
                    rank = ranks[doc_id]
                    rrf_score += weight * (1.0 / (k + rank))
                else:
                    # Document not in this strategy - use large rank
                    max_rank = len(ranks) + 100
                    rrf_score += weight * (1.0 / (k + max_rank))
            rrf_scores[doc_id] = rrf_score

        # Create final results with RRF scores
        final_results = []
        for doc_id, rrf_score in rrf_scores.items():
            result = doc_to_result[doc_id]
            # Replace score with RRF score (normalized to 0-1 range)
            max_possible = sum(strategy_weights.values()) / k
            result.score = rrf_score / max_possible if max_possible > 0 else rrf_score
            result.metadata["rrf_score"] = rrf_score
            result.metadata["original_score"] = doc_to_result[doc_id].score
            final_results.append(result)

        # Sort by RRF score
        final_results.sort(key=lambda r: r.score, reverse=True)
        return final_results

    def _get_strategy_weights(self, analysis: QueryAnalysis) -> dict[str, float]:
        """Get strategy weights based on query type."""
        # Base weights
        weights = {
            "semantic": 1.0,
            "entity": 0.8,
            "temporal": 0.7,
            "graph": 0.7,
            "fact": 1.0,
            "working_memory": 1.2,
            "keyword": 0.6,
        }

        # Adjust based on query type
        if analysis.reasoning_type == ReasoningType.TEMPORAL:
            weights["temporal"] = 1.3
            weights["semantic"] = 0.8
        elif analysis.reasoning_type == ReasoningType.MULTI_HOP:
            weights["graph"] = 1.2
            weights["entity"] = 1.0
            weights["keyword"] = 0.8
        elif analysis.intent == QueryIntent.PREFERENCE:
            weights["fact"] = 1.2
            weights["entity"] = 1.0

        return weights

    def _get_dialogue_context(self, results: list[RetrievalResult]) -> list[RetrievalResult]:
        """Get surrounding messages for dialogue context."""
        context_results = []
        seen_ids = {r.id for r in results}

        # Sort memories by turn number to find adjacent messages
        sorted_memories = sorted(
            self.memory.graph.memories.values(),
            key=lambda m: (m.session_id or "", m.turn_number or 0),
        )
        memory_list = list(sorted_memories)
        id_to_idx = {m.id: i for i, m in enumerate(memory_list)}

        for result in results[:5]:  # Only expand top results
            if result.id in id_to_idx:
                idx = id_to_idx[result.id]
                # Get previous and next messages
                for offset in [-1, 1]:
                    adj_idx = idx + offset
                    if 0 <= adj_idx < len(memory_list):
                        adj_memory = memory_list[adj_idx]
                        if adj_memory.id not in seen_ids:
                            seen_ids.add(adj_memory.id)
                            context_results.append(
                                RetrievalResult(
                                    id=adj_memory.id,
                                    content=adj_memory.content,
                                    score=result.score * 0.8,  # Slightly lower score
                                    source="dialogue_context",
                                    entities=adj_memory.entity_names,
                                    timestamp=(
                                        adj_memory.event_time.start.isoformat()
                                        if adj_memory.event_time
                                        else None
                                    ),
                                )
                            )

        return context_results

    def _keyword_search(self, keywords: list[str]) -> list[RetrievalResult]:
        """Search for memories containing specific keywords."""
        results = []
        seen_ids = set()

        for memory in self.memory.graph.memories.values():
            content_lower = memory.content.lower()
            # Check if memory contains any keyword
            matches = sum(1 for kw in keywords if kw.lower() in content_lower)
            if matches > 0:
                if memory.id not in seen_ids:
                    seen_ids.add(memory.id)
                    results.append(
                        RetrievalResult(
                            id=memory.id,
                            content=memory.content,
                            score=0.5 + (matches * 0.1),  # Base score + keyword match bonus
                            source="keyword",
                            entities=memory.entity_names,
                            timestamp=(
                                memory.event_time.start.isoformat() if memory.event_time else None
                            ),
                        )
                    )

        return results

    def _semantic_search(
        self, query_embedding: np.ndarray, top_k: int = 20
    ) -> list[RetrievalResult]:
        """Semantic similarity search."""
        results = []

        # Search unified graph
        similar = self.memory.graph.query_by_similarity(query_embedding, top_k=top_k)

        for memory, score in similar:
            results.append(
                RetrievalResult(
                    id=memory.id,
                    content=memory.content,
                    score=score,
                    source="semantic",
                    entities=memory.entity_names,
                    timestamp=memory.event_time.start.isoformat() if memory.event_time else None,
                    metadata={"original_score": score},
                )
            )

        # Search episodic memory
        episodes = self.memory.episodic_memory.search_similar(query_embedding, top_k=top_k // 2)

        for episode, score in episodes:
            results.append(
                RetrievalResult(
                    id=episode.id,
                    content=episode.summary or episode.get_full_text()[:500],
                    score=score * 0.9,  # Slightly lower weight for episodes
                    source="episode",
                    entities=episode.participant_names,
                    timestamp=episode.start_time.isoformat(),
                    metadata={"episode_id": episode.id, "message_count": episode.message_count},
                )
            )

        return results

    def _entity_search(self, entities: list[str]) -> list[RetrievalResult]:
        """Search by entity mentions."""
        results = []

        for entity_name in entities:
            # Find entity in graph
            entity_nodes = self.memory.graph.entity_graph.find_by_name(entity_name)

            for entity in entity_nodes:
                # Get memories involving this entity
                memories = self.memory.graph.query_by_entity(entity.id)

                for memory in memories[:5]:  # Limit per entity
                    results.append(
                        RetrievalResult(
                            id=memory.id,
                            content=memory.content,
                            score=0.7,  # Base score for entity match
                            source="entity",
                            entities=[entity_name],
                            timestamp=(
                                memory.event_time.start.isoformat() if memory.event_time else None
                            ),
                            reasoning_path=[f"entity:{entity_name}"],
                        )
                    )

                # Get entity profile as context
                profile = self.memory.graph.entity_graph.get_entity_profile(entity.id)
                if profile.get("relations"):
                    profile_text = f"{entity_name}: " + ", ".join(
                        f"{r['relation']} {r['entity']}" for r in profile["relations"][:5]
                    )
                    results.append(
                        RetrievalResult(
                            id=f"profile_{entity.id}",
                            content=profile_text,
                            score=0.6,
                            source="entity_profile",
                            entities=[entity_name],
                        )
                    )

        return results

    def _temporal_search(self, analysis: QueryAnalysis) -> list[RetrievalResult]:
        """Search based on temporal constraints."""
        results = []

        # Extract time constraints from temporal expressions
        for expr in analysis.temporal_expressions:
            if expr.normalized_start:
                # Search by time
                memories = self.memory.graph.query_by_time(
                    expr.normalized_start,
                    expr.normalized_end,
                    entities=analysis.entities if analysis.entities else None,
                )

                for memory in memories[:10]:
                    results.append(
                        RetrievalResult(
                            id=memory.id,
                            content=memory.content,
                            score=0.8,
                            source="temporal",
                            entities=memory.entity_names,
                            timestamp=(
                                memory.event_time.start.isoformat() if memory.event_time else None
                            ),
                            reasoning_path=[f"temporal:{expr.text}"],
                        )
                    )

        # Handle relative temporal queries
        if analysis.temporal_scope == TemporalScope.RELATIVE and analysis.entities:
            for entity in analysis.entities:
                chain = self.memory.graph.find_temporal_chain(entity)
                for memory in chain[:10]:
                    results.append(
                        RetrievalResult(
                            id=memory.id,
                            content=memory.content,
                            score=0.75,
                            source="temporal_chain",
                            entities=memory.entity_names,
                            timestamp=(
                                memory.event_time.start.isoformat() if memory.event_time else None
                            ),
                            reasoning_path=[f"temporal_chain:{entity}"],
                        )
                    )

        return results

    def _graph_traversal(
        self, analysis: QueryAnalysis, query_embedding: np.ndarray | None, max_hops: int = 2
    ) -> list[RetrievalResult]:
        """Multi-hop graph traversal for complex queries."""
        results: list[RetrievalResult] = []

        if not analysis.entities:
            return results

        # Use multi-hop query from unified graph
        multi_hop_results = self.memory.graph.multi_hop_query(
            start_entities=analysis.entities,
            query_embedding=query_embedding,
            max_hops=max_hops,
            top_k=15,
        )

        for memory, score, path in multi_hop_results:
            results.append(
                RetrievalResult(
                    id=memory.id,
                    content=memory.content,
                    score=score,
                    source="graph_traversal",
                    entities=memory.entity_names,
                    timestamp=memory.event_time.start.isoformat() if memory.event_time else None,
                    reasoning_path=path,
                    metadata={"hop_count": len(path)},
                )
            )

        return results

    def _fact_search(
        self, query_embedding: np.ndarray | None, analysis: QueryAnalysis
    ) -> list[RetrievalResult]:
        """Search semantic facts."""
        results = []

        # Semantic search on facts
        if query_embedding is not None:
            similar_facts = self.memory.semantic_memory.search_similar(query_embedding, top_k=10)

            for fact, score in similar_facts:
                results.append(
                    RetrievalResult(
                        id=fact.id,
                        content=fact.content,
                        score=score * fact.confidence,
                        source="fact",
                        confidence=fact.confidence,
                        negated=fact.negated,
                        metadata={
                            "subject": fact.subject,
                            "predicate": fact.predicate,
                            "object": fact.object,
                            "confirmations": fact.confirmation_count,
                        },
                    )
                )

        # Search facts about entities
        for entity in analysis.entities:
            facts = self.memory.semantic_memory.get_facts_about(entity, include_negated=True)

            for fact in facts[:5]:
                results.append(
                    RetrievalResult(
                        id=fact.id,
                        content=fact.content,
                        score=0.7 * fact.confidence,
                        source="entity_fact",
                        entities=[entity],
                        confidence=fact.confidence,
                        negated=fact.negated,
                    )
                )

        return results

    def _working_memory_search(self, query_embedding: np.ndarray | None) -> list[RetrievalResult]:
        """Search working memory for recent context."""
        results = []

        wm_items = self.memory.working_memory.get_context(query_embedding, top_k=5)

        for item in wm_items:
            results.append(
                RetrievalResult(
                    id=item.id,
                    content=item.content,
                    score=0.9 * item.attention_weight,  # High weight for working memory
                    source="working_memory",
                    metadata={"attention": item.attention_weight},
                )
            )

        return results

    def _verify_negations(
        self, results: list[RetrievalResult], analysis: QueryAnalysis
    ) -> list[RetrievalResult]:
        """Verify and flag negations for adversarial robustness."""
        import re

        # Check for explicit negations in semantic memory
        for result in results:
            # If this is a fact, check for contradictions
            if result.source == "fact" and result.metadata.get("subject"):
                is_negated, neg_fact = self.memory.semantic_memory.check_negation(
                    result.metadata["subject"],
                    result.metadata.get("predicate", ""),
                    result.metadata.get("object", ""),
                )

                if is_negated:
                    result.negated = True
                    result.confidence *= 0.3  # Reduce confidence for negated facts

        # Add explicit negation results from semantic memory
        negated_facts = self.memory.semantic_memory.get_negated_facts()
        for fact in negated_facts[:10]:
            # Check if relevant to query
            query_lower = analysis.original_query.lower()
            fact_relevant = (
                any(e.lower() in fact.subject.lower() for e in analysis.entities)
                or any(e.lower() in fact.object.lower() for e in analysis.entities)
                or fact.object.lower() in query_lower
                or fact.subject.lower() in query_lower
            )

            if fact_relevant:
                results.append(
                    RetrievalResult(
                        id=fact.id,
                        content=f"IMPORTANT: {fact.content}",
                        score=1.5,  # High score for relevant negations
                        source="negation",
                        negated=True,
                        confidence=1.0,
                    )
                )

        # CRITICAL: Search raw memories for negation patterns
        # This catches "I could never eat X" type statements
        negation_patterns = [
            r"(?:could|would|can|will)\s+never",
            r"(?:don't|doesn't|do not|does not)\s+(?:like|love|enjoy|eat|want)",
            r"(?:hate|detest|loathe|can't stand)",
            r"never\s+(?:eat|like|enjoy|want)",
            r"not\s+(?:a\s+)?fan\s+of",
        ]

        query_keywords = set(analysis.keywords)

        for memory in self.memory.graph.memories.values():
            content_lower = memory.content.lower()

            # Check if content contains negation patterns
            for pattern in negation_patterns:
                if re.search(pattern, content_lower):
                    # Check relevance to query
                    content_keywords = set(content_lower.split())
                    if query_keywords.intersection(content_keywords) or any(
                        kw in content_lower for kw in analysis.keywords
                    ):
                        results.append(
                            RetrievalResult(
                                id=f"neg_{memory.id}",
                                content=f"[NEGATION DETECTED] {memory.content}",
                                score=1.8,  # Very high score for detected negations
                                source="negation_detected",
                                negated=True,
                                confidence=1.0,
                                metadata={"pattern_matched": pattern},
                            )
                        )
                        break  # Only add once per memory

        # Also check negated_facts stored in memories
        for memory in self.memory.graph.memories.values():
            if memory.negated_facts:
                for neg_text in memory.negated_facts:
                    if any(kw in neg_text.lower() for kw in analysis.keywords):
                        results.append(
                            RetrievalResult(
                                id=f"memfact_{memory.id}",
                                content=f"[STATED AS FALSE] {neg_text}",
                                score=1.6,
                                source="memory_negation",
                                negated=True,
                                confidence=1.0,
                            )
                        )

        return results

    def _rank_results(
        self,
        results: list[RetrievalResult],
        analysis: QueryAnalysis,
        strategy: dict[str, Any],
    ) -> list[RetrievalResult]:
        """Rank results based on multiple factors."""
        import re

        # Check if this is a preference/like question (adversarial-prone)
        query_lower = analysis.original_query.lower()

        # More precise detection of preference questions
        is_preference_question = bool(
            re.search(r"\b(like|love|enjoy|hate|prefer|favorite|fan of)\b", query_lower)
        ) and query_lower.startswith(("do ", "does ", "did ", "is ", "are ", "can ", "could "))

        # Factual questions should NOT be treated as yes/no
        is_factual = query_lower.startswith(("what ", "where ", "when ", "who ", "which ", "how "))

        for result in results:
            # Start with base score
            final_score = result.score

            # Adjust based on source weights
            source_weights = {
                "working_memory": 1.2,
                "semantic": 1.0,
                "fact": 1.0,
                "temporal": 0.9 if analysis.reasoning_type == ReasoningType.TEMPORAL else 0.7,
                "entity": 0.8,
                "graph_traversal": (
                    0.9 if analysis.reasoning_type == ReasoningType.MULTI_HOP else 0.7
                ),
                "episode": 0.8,
                "entity_profile": 0.6,
                "negation": 1.2,  # Moderate boost
                "negation_detected": 1.3,
                "memory_negation": 1.2,
            }
            source_weight = source_weights.get(result.source, 0.7)
            final_score *= source_weight

            # Boost for entity matches
            if result.entities:
                entity_overlap = len(set(result.entities) & set(analysis.entities))
                final_score *= 1 + 0.1 * entity_overlap

            # Boost for recency (if timestamp available)
            if result.timestamp:
                try:
                    ts = datetime.fromisoformat(result.timestamp)
                    age_days = (datetime.now() - ts).days
                    recency_factor = max(0.5, 1 - age_days / 365)
                    final_score *= 1 + self.config.recency_weight * recency_factor
                except Exception:
                    pass

            # Penalty for low confidence
            final_score *= result.confidence

            # Handle negations based on question type
            if result.negated:
                if is_preference_question or analysis.is_negation_check:
                    # Strong boost only for true preference questions
                    final_score *= 1.5
                elif is_factual:
                    # For factual questions, negations are less relevant
                    final_score *= 0.6
                else:
                    # Neutral for other questions
                    final_score *= 0.9

            # Temporal alignment boost for temporal queries
            if (
                analysis.temporal_scope != TemporalScope.NONE
                or analysis.reasoning_type == ReasoningType.TEMPORAL
            ):
                has_time_signal = bool(result.timestamp)
                if result.metadata:
                    has_time_signal = has_time_signal or any(
                        k in result.metadata
                        for k in ["session_idx", "session_timestamp", "resolved_dates"]
                    )
                if has_time_signal:
                    final_score *= 1.15
                else:
                    final_score *= 0.85

            result.score = final_score

        # Sort by final score
        results.sort(key=lambda r: r.score, reverse=True)
        return results

    def _apply_reranker(self, query: str, results: list[RetrievalResult]) -> list[RetrievalResult]:
        """Re-rank top candidates with a cross-encoder."""
        if not self.reranker or not results:
            return results

        # Use current score ordering for candidate selection
        candidates = sorted(results, key=lambda r: r.score, reverse=True)
        top_n = min(self.config.rerank_top_n, len(candidates))
        top = candidates[:top_n]

        texts = [r.content for r in top]
        scores = self.reranker.score_pairs(query, texts)
        if not scores:
            return results

        min_s = min(scores)
        max_s = max(scores)
        span = max_s - min_s
        if span == 0:
            norm_scores = [0.5 for _ in scores]
        else:
            norm_scores = [(s - min_s) / span for s in scores]

        weight = max(0.0, min(1.0, self.config.rerank_weight))
        for r, ns in zip(top, norm_scores):
            r.metadata["rerank_score"] = ns
            # Scale existing score by reranker signal
            r.score *= 1.0 + weight * (ns - 0.5) * 2.0

        return results

    def _deduplicate(self, results: list[RetrievalResult]) -> list[RetrievalResult]:
        """Remove duplicate results."""
        seen_ids = set()
        unique = []

        for result in results:
            if result.id not in seen_ids:
                seen_ids.add(result.id)
                unique.append(result)

        return unique

    def _compose_context(
        self,
        results: list[RetrievalResult],
        analysis: QueryAnalysis,
    ) -> str:
        """
        Compose retrieval results into context string.

        Uses position-aware composition to combat lost-in-the-middle.
        Negations are emphasized only for preference/adversarial questions.
        """
        import re

        if not results:
            return ""

        if not self.config.use_position_aware_composition:
            # Simple composition
            return "\n\n".join(r.content for r in results[: self.config.top_k])

        # Determine if this is a preference/adversarial question
        query_lower = analysis.original_query.lower()
        is_preference_question = bool(
            re.search(
                r"\b(like|love|enjoy|hate|prefer|favorite|fan of|escargot|snail)\b", query_lower
            )
        )
        query_lower.startswith(("what ", "where ", "when ", "who ", "which ", "how "))

        # Position-aware composition
        parts = []
        negation_results = [r for r in results if r.negated]
        non_negation_results = [r for r in results if not r.negated]

        # For preference questions: put negations at top
        # For factual questions: put factual content first, negations only if relevant
        if is_preference_question and negation_results:
            parts.append("## IMPORTANT - Stated Preferences/Dislikes:")
            for r in negation_results[:3]:
                content = (
                    r.content.replace("[NEGATION DETECTED]", "")
                    .replace("[STATED AS FALSE]", "")
                    .strip()
                )
                parts.append(f"- {content}")
            parts.append("")

        # For multi-hop/commonsense, include more context
        is_multi_hop = analysis.reasoning_type == ReasoningType.MULTI_HOP
        is_commonsense = "common" in query_lower or "both" in query_lower
        max_results = 10 if (is_multi_hop or is_commonsense) else 5

        # For multi-hop, sort results by turn number to preserve dialogue order
        if is_multi_hop:
            sorted_results = sorted(
                non_negation_results,
                key=lambda r: self.memory.graph.memories.get(
                    r.id, type("obj", (object,), {"turn_number": 9999})()
                ).turn_number
                or 9999,
            )
            non_negation_results = sorted_results

        # SANDWICH COMPOSITION: Put high-relevance at beginning AND end
        # This combats "lost in the middle" problem where LLMs forget middle content
        all_results = non_negation_results[: max_results + 5]  # Get more for sandwiching

        if len(all_results) >= 5:
            # Top 3 results at beginning (most important)
            high_priority = all_results[:3]
            # Middle content (may be ignored by LLM)
            middle = all_results[3:-2] if len(all_results) > 5 else []
            # Last 2 results at end (recency bias helps recall)
            end_priority = all_results[-2:] if len(all_results) > 3 else []

            # Build sandwich structure
            parts.append("## Key Information")
            for r in high_priority:
                content = self._format_result_content(r)
                parts.append(f"- {content}")

            if middle:
                parts.append("\n## Supporting Context")
                for r in middle:
                    content = self._format_result_content(r)
                    # Truncate middle content slightly to save space
                    parts.append(f"- {content[:250]}")

            if end_priority:
                parts.append("\n## Additional Key Facts")
                for r in end_priority:
                    content = self._format_result_content(r)
                    parts.append(f"- {content}")
        else:
            # Not enough results for sandwich, use simple list
            parts.append("## Relevant Information")
            for r in all_results:
                content = self._format_result_content(r)
                parts.append(f"- {content}")

        # Additional context beyond sandwich
        remaining = non_negation_results[max_results + 5 :]
        if remaining:
            parts.append("\n## More Context")
            for i, r in enumerate(remaining[:3], 1):
                content = self._format_result_content(r)[:200]
                parts.append(f"[{i}] {content}")

        # For NON-preference factual questions, only include
        # negations if they seem directly relevant
        if not is_preference_question and negation_results:
            # Check if negation mentions keywords from the question
            relevant_negations = []
            for r in negation_results:
                # Check if any non-trivial query word appears in the negation
                query_words = [
                    w
                    for w in query_lower.split()
                    if len(w) > 3
                    and w
                    not in {"what", "where", "when", "does", "were", "that", "this", "with", "from"}
                ]
                if any(word in r.content.lower() for word in query_words):
                    relevant_negations.append(r)

            if relevant_negations:
                parts.append("\n## Note")
                for r in relevant_negations[:2]:
                    content = self._format_result_content(r)
                    parts.append(f"- {content}")

        # Add temporal context if relevant
        if analysis.reasoning_type == ReasoningType.TEMPORAL:
            temporal_results = [r for r in results if r.source == "temporal"]
            if temporal_results:
                parts.append("\n## Timeline")
                for r in sorted(temporal_results, key=lambda x: x.timestamp or "")[:5]:
                    if r.timestamp:
                        parts.append(f"- [{r.timestamp}] {r.content[:100]}")

        context = "\n".join(parts)

        # Truncate if too long (rough estimate: 4 chars per token)
        max_chars = self.config.max_context_tokens * 4
        if len(context) > max_chars:
            context = context[:max_chars] + "\n...[truncated]"

        return context

    def _format_result_content(self, result: RetrievalResult) -> str:
        """Format retrieval result with optional speaker/time metadata."""
        content = (
            result.content.replace("[NEGATION DETECTED]", "")
            .replace("[STATED AS FALSE]", "")
            .strip()
        )

        memory = self.memory.graph.memories.get(result.id)
        if not memory:
            return content

        prefix_parts = []
        if memory.speaker:
            prefix_parts.append(memory.speaker)
        if memory.metadata:
            session_ts = memory.metadata.get("session_timestamp")
            if session_ts:
                prefix_parts.append(session_ts)

        if prefix_parts:
            return f"[{', '.join(prefix_parts)}] {content}"

        return content

    def retrieve_for_question(
        self,
        question: str,
        conversation_history: list[dict[str, str]] | None = None,
    ) -> str:
        """
        Convenience method for question answering.

        Returns composed context ready for LLM.
        """
        context = {}
        if conversation_history:
            context["history"] = conversation_history

        response = self.retrieve(question, context=context)
        return response.composed_context
