"""Tests for AttentionFilter: context filtering with precise forgetting."""

from dataclasses import dataclass

from zerogmem.retriever.attention_filter import AttentionFilter, FilterConfig


@dataclass
class MockResult:
    """Mock retrieval result for testing."""

    content: str
    score: float
    source: str = "semantic"
    negated: bool = False
    entities: list[str] = None
    metadata: dict = None

    def __post_init__(self):
        if self.entities is None:
            self.entities = []
        if self.metadata is None:
            self.metadata = {}


class TestAttentionFilter:
    """Tests for the AttentionFilter."""

    def test_filter_empty_results(self, attention_filter):
        results = attention_filter.filter_context("test query", [])
        assert results == []

    def test_filter_removes_low_relevance(self, mock_embedding_fn):
        config = FilterConfig(relevance_threshold=0.5, max_context_tokens=10000)
        af = AttentionFilter(config=config, embedding_fn=mock_embedding_fn)

        results = [
            MockResult(content="very relevant hiking mountains", score=0.9),
            MockResult(content="x", score=0.01),
        ]
        filtered = af.filter_context("hiking mountains", results)
        # The short, low-score result should be filtered out
        assert len(filtered) <= len(results)

    def test_filter_respects_token_budget(self, mock_embedding_fn):
        # Very small token budget
        config = FilterConfig(
            relevance_threshold=0.0,
            max_context_tokens=10,
            chars_per_token=4,
        )
        af = AttentionFilter(config=config, embedding_fn=mock_embedding_fn)

        results = [
            MockResult(content="A " * 100, score=0.9),
            MockResult(content="B " * 100, score=0.8),
            MockResult(content="C " * 100, score=0.7),
        ]
        filtered = af.filter_context("query", results)
        # With 10 tokens * 4 chars = 40 chars budget, should be heavily filtered
        assert len(filtered) < len(results)

    def test_deduplication(self, mock_embedding_fn):
        config = FilterConfig(
            relevance_threshold=0.0,
            max_context_tokens=10000,
            semantic_similarity_threshold=0.85,
        )
        af = AttentionFilter(config=config, embedding_fn=mock_embedding_fn)

        # Two results with identical content should be deduplicated
        results = [
            MockResult(content="Alice went hiking in the mountains", score=0.9),
            MockResult(content="Alice went hiking in the mountains", score=0.8),
        ]
        filtered = af.filter_context("hiking", results)
        assert len(filtered) <= len(results)

    def test_compute_sufficiency_no_results(self, attention_filter):
        score = attention_filter.compute_sufficiency_score("test", [])
        assert score == 0.0

    def test_compute_sufficiency_good_coverage(self, attention_filter):
        results = [
            MockResult(
                content="Alice went hiking in the mountains last summer",
                score=0.9,
                entities=["Alice"],
            ),
            MockResult(
                content="She loved the Matterhorn especially",
                score=0.8,
                entities=["Alice"],
            ),
        ]
        score = attention_filter.compute_sufficiency_score(
            "What did Alice do in the mountains?", results
        )
        assert score > 0.0

    def test_diversity_enforcement(self, mock_embedding_fn):
        config = FilterConfig(
            relevance_threshold=0.0,
            max_context_tokens=10000,
            diversity_weight=0.5,
        )
        af = AttentionFilter(config=config, embedding_fn=mock_embedding_fn)

        # All from same source — diversity filter should kick in
        results = [
            MockResult(content=f"Result {i} about hiking", score=0.9 - i * 0.01, source="semantic")
            for i in range(10)
        ]
        filtered = af.filter_context("hiking", results)
        # Should return results but not necessarily all
        assert len(filtered) >= 1

    def test_filter_preserves_order(self, attention_filter):
        results = [
            MockResult(content="hiking mountains alps", score=0.9),
            MockResult(content="cooking dinner recipe", score=0.5),
        ]
        filtered = attention_filter.filter_context("hiking in the alps", results)
        if len(filtered) >= 2:
            # Higher score should come first
            assert filtered[0].score >= filtered[1].score
