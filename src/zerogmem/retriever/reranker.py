"""
Cross-encoder reranker for retrieval candidates.
"""

from __future__ import annotations

from typing import List, Optional


class CrossEncoderReranker:
    """Thin wrapper around sentence-transformers CrossEncoder."""

    def __init__(self, model_name: str, device: Optional[str] = None):
        try:
            from sentence_transformers import CrossEncoder
        except Exception as e:
            raise RuntimeError("sentence-transformers is required for reranking") from e

        self.model_name = model_name
        self.model = CrossEncoder(model_name, device=device)

    def score_pairs(self, query: str, texts: List[str]) -> List[float]:
        if not texts:
            return []

        pairs = [(query, t) for t in texts]
        scores = self.model.predict(pairs)
        return [float(s) for s in scores]
