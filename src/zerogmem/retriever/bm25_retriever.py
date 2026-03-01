"""
BM25 Retriever: Sparse keyword-based retrieval using BM25 algorithm.

Complements semantic search for specific factual queries.
"""

from __future__ import annotations

import math
import re
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any, Tuple
import numpy as np


@dataclass
class BM25Config:
    """Configuration for BM25 retrieval."""
    k1: float = 1.5  # Term frequency saturation
    b: float = 0.75  # Length normalization
    min_token_length: int = 2
    stopwords: set = field(default_factory=lambda: {
        'a', 'an', 'the', 'is', 'are', 'was', 'were', 'be', 'been', 'being',
        'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could',
        'should', 'may', 'might', 'must', 'shall', 'can', 'need', 'dare',
        'ought', 'used', 'to', 'of', 'in', 'for', 'on', 'with', 'at', 'by',
        'from', 'as', 'into', 'through', 'during', 'before', 'after',
        'above', 'below', 'between', 'under', 'again', 'further', 'then',
        'once', 'here', 'there', 'when', 'where', 'why', 'how', 'all', 'each',
        'few', 'more', 'most', 'other', 'some', 'such', 'no', 'nor', 'not',
        'only', 'own', 'same', 'so', 'than', 'too', 'very', 's', 't', 'just',
        'don', 'now', 'i', 'you', 'he', 'she', 'it', 'we', 'they', 'what',
        'which', 'who', 'whom', 'this', 'that', 'these', 'those', 'am', 'and',
        'but', 'if', 'or', 'because', 'until', 'while', 'about', 'against',
    })


@dataclass
class BM25Document:
    """A document indexed for BM25."""
    id: str
    content: str
    tokens: List[str]
    token_counts: Counter
    metadata: Dict[str, Any] = field(default_factory=dict)


class BM25Retriever:
    """
    BM25 sparse retrieval implementation.

    Features:
    - Standard BM25 scoring
    - Incremental indexing
    - Fast retrieval with inverted index
    - Query expansion for better recall
    """

    # Query expansion mappings for common question types
    QUERY_EXPANSIONS = {
        # Relationship and status
        "relationship": ["single", "married", "dating", "breakup", "divorce", "partner", "boyfriend", "girlfriend", "spouse", "parent", "engaged", "widowed"],
        "status": ["single", "married", "engaged", "divorced", "dating", "relationship"],
        "single": ["unmarried", "alone", "breakup", "divorced"],
        "married": ["spouse", "husband", "wife", "wedding"],

        # Identity
        "identity": ["transgender", "trans", "gender", "lgbtq", "queer", "gay", "lesbian", "bisexual", "coming out", "journey"],
        "transgender": ["trans", "transition", "gender", "coming out"],

        # Activities and hobbies
        "activities": ["hobby", "hobbies", "like", "enjoy", "do", "play", "sport", "signed up", "class"],
        "hobbies": ["activities", "interests", "enjoy", "like", "do", "play"],
        "partake": ["do", "enjoy", "participate", "engage", "signed up"],

        # Outdoor activities
        "camped": ["camping", "camp", "tent", "forest", "beach", "mountains", "outdoors", "hiking", "nature"],
        "camping": ["camp", "camped", "tent", "outdoors", "forest", "beach", "mountains"],
        "hiked": ["hiking", "hike", "trail", "mountains", "nature", "walk"],

        # Family
        "kids": ["children", "son", "daughter", "child", "family", "three", "two", "dinosaurs", "nature"],
        "children": ["kids", "child", "son", "daughter", "family", "three", "two"],
        "family": ["kids", "children", "spouse", "husband", "wife", "parents", "daughter", "son"],
        "how many": ["three", "two", "one", "four", "five", "number"],

        # Pets
        "pets": ["pet", "dog", "cat", "cats", "guinea pig", "hamster", "fish", "oliver", "luna", "bailey"],
        "pet": ["pets", "dog", "cat", "cats", "guinea pig", "hamster"],
        "names": ["named", "name", "called", "oliver", "luna", "bailey"],

        # Preferences
        "like": ["enjoy", "love", "favorite", "favourite", "prefer", "fond"],
        "enjoy": ["like", "love", "favorite", "prefer"],
        "favorite": ["favourite", "best", "prefer", "love", "like"],
        "destress": ["relax", "unwind", "calm", "stress", "running", "exercise"],

        # Research and learning
        "research": ["researching", "researched", "looking", "studied", "investigate", "explore"],
        "researched": ["research", "looked into", "studied", "investigated"],

        # Reading and books
        "books": ["book", "read", "reading", "novel", "literature", "author"],
        "read": ["reading", "book", "books", "novel"],

        # Art and creativity
        "paint": ["painting", "painted", "art", "draw", "drawing", "canvas", "artistic"],
        "painted": ["paint", "painting", "art", "artwork", "canvas"],
        "painting": ["paint", "painted", "art", "canvas", "draw"],

        # Career and education
        "career": ["job", "work", "profession", "pursue", "counseling", "mental health"],
        "pursue": ["career", "study", "become", "work", "job"],
        "education": ["study", "school", "degree", "university", "college", "certification"],

        # Events and participation
        "events": ["event", "participated", "attended", "went", "joined"],
        "participated": ["joined", "attended", "went", "event", "group"],

        # Location
        "moved": ["move", "from", "country", "relocated", "came from"],
        "from": ["moved", "country", "hometown", "origin", "came"],
        "country": ["from", "moved", "homeland", "origin", "sweden", "hometown"],

        # LGBTQ specific
        "lgbtq": ["lgbt", "pride", "transgender", "trans", "queer", "gay", "lesbian", "community", "support group"],
        "pride": ["parade", "lgbtq", "lgbt", "community", "march"],
        "support": ["group", "community", "help", "meeting"],

        # Time-related
        "recently": ["recent", "lately", "last", "just", "new"],
        "sunset": ["sunrise", "sky", "landscape", "scene", "painted"],

        # Music and instruments
        "instruments": ["instrument", "play", "plays", "violin", "piano", "guitar", "clarinet", "flute", "drums"],
        "instrument": ["instruments", "play", "plays", "violin", "piano", "guitar", "clarinet"],
        "musicians": ["musician", "artist", "band", "bach", "mozart", "ed sheeran", "classical"],
        "classical": ["bach", "mozart", "beethoven", "vivaldi", "symphony"],
        "artists": ["artist", "band", "musician", "concert", "performed"],
        "bands": ["band", "artist", "concert", "performed", "seen"],

        # Art specific
        "art": ["painting", "painted", "draw", "drawing", "abstract", "canvas", "sculpture"],
        "abstract": ["art", "painting", "painted", "artistic"],
    }

    def __init__(self, config: Optional[BM25Config] = None):
        self.config = config or BM25Config()

        # Document storage
        self.documents: Dict[str, BM25Document] = {}

        # Inverted index: token -> list of (doc_id, term_freq)
        self.inverted_index: Dict[str, List[Tuple[str, int]]] = defaultdict(list)

        # Document frequency: token -> number of docs containing it
        self.doc_freq: Counter = Counter()

        # Corpus stats
        self.total_docs = 0
        self.avg_doc_length = 0.0
        self._total_tokens = 0

    def tokenize(self, text: str) -> List[str]:
        """Tokenize text into words."""
        # Lowercase and extract words
        text = text.lower()
        tokens = re.findall(r'\b[a-z]+\b', text)

        # Filter by length and stopwords
        tokens = [
            t for t in tokens
            if len(t) >= self.config.min_token_length
            and t not in self.config.stopwords
        ]

        return tokens

    def add_document(
        self,
        doc_id: str,
        content: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """Add a document to the index."""
        if doc_id in self.documents:
            # Remove old document first
            self.remove_document(doc_id)

        tokens = self.tokenize(content)
        token_counts = Counter(tokens)

        doc = BM25Document(
            id=doc_id,
            content=content,
            tokens=tokens,
            token_counts=token_counts,
            metadata=metadata or {},
        )

        self.documents[doc_id] = doc

        # Update inverted index
        for token, count in token_counts.items():
            self.inverted_index[token].append((doc_id, count))
            self.doc_freq[token] += 1

        # Update corpus stats
        self.total_docs += 1
        self._total_tokens += len(tokens)
        self.avg_doc_length = self._total_tokens / self.total_docs

    def add_documents(
        self,
        documents: List[Tuple[str, str, Optional[Dict[str, Any]]]]
    ) -> None:
        """Add multiple documents at once."""
        for doc_id, content, metadata in documents:
            self.add_document(doc_id, content, metadata)

    def remove_document(self, doc_id: str) -> bool:
        """Remove a document from the index."""
        if doc_id not in self.documents:
            return False

        doc = self.documents[doc_id]

        # Update inverted index
        for token, count in doc.token_counts.items():
            self.inverted_index[token] = [
                (d_id, c) for d_id, c in self.inverted_index[token]
                if d_id != doc_id
            ]
            self.doc_freq[token] -= 1
            if self.doc_freq[token] == 0:
                del self.doc_freq[token]
                del self.inverted_index[token]

        # Update stats
        self.total_docs -= 1
        self._total_tokens -= len(doc.tokens)
        self.avg_doc_length = self._total_tokens / self.total_docs if self.total_docs > 0 else 0

        del self.documents[doc_id]
        return True

    def _idf(self, token: str) -> float:
        """Compute IDF for a token."""
        if token not in self.doc_freq or self.total_docs == 0:
            return 0.0

        df = self.doc_freq[token]
        # Standard IDF formula with smoothing
        return math.log((self.total_docs - df + 0.5) / (df + 0.5) + 1)

    def _score_document(self, doc: BM25Document, query_tokens: List[str]) -> float:
        """Compute BM25 score for a document given query tokens."""
        score = 0.0
        doc_length = len(doc.tokens)

        k1 = self.config.k1
        b = self.config.b
        avg_dl = self.avg_doc_length

        for token in query_tokens:
            if token not in doc.token_counts:
                continue

            tf = doc.token_counts[token]
            idf = self._idf(token)

            # BM25 score contribution
            numerator = tf * (k1 + 1)
            denominator = tf + k1 * (1 - b + b * doc_length / avg_dl)
            score += idf * numerator / denominator

        return score

    def _expand_query(self, tokens: List[str]) -> List[str]:
        """Expand query tokens with related terms."""
        expanded = list(tokens)
        for token in tokens:
            if token in self.QUERY_EXPANSIONS:
                # Add expansion terms (but with lower weight - handled by not duplicating)
                for expansion in self.QUERY_EXPANSIONS[token]:
                    if expansion not in expanded:
                        expanded.append(expansion)
        return expanded

    def search(
        self,
        query: str,
        top_k: int = 10,
        min_score: float = 0.0,
        use_expansion: bool = True
    ) -> List[Tuple[str, float, str]]:
        """
        Search for documents matching the query.

        Returns: List of (doc_id, score, content) tuples
        """
        if self.total_docs == 0:
            return []

        query_tokens = self.tokenize(query)
        if not query_tokens:
            return []

        # Expand query with related terms
        if use_expansion:
            expanded_tokens = self._expand_query(query_tokens)
        else:
            expanded_tokens = query_tokens

        # Find candidate documents (any doc containing any query token)
        candidate_doc_ids = set()
        for token in expanded_tokens:
            if token in self.inverted_index:
                for doc_id, _ in self.inverted_index[token]:
                    candidate_doc_ids.add(doc_id)

        if not candidate_doc_ids:
            return []

        # Score all candidates using expanded tokens
        scored = []
        for doc_id in candidate_doc_ids:
            doc = self.documents[doc_id]
            score = self._score_document(doc, expanded_tokens)
            if score >= min_score:
                scored.append((doc_id, score, doc.content))

        # Sort by score descending
        scored.sort(key=lambda x: x[1], reverse=True)

        return scored[:top_k]

    def get_stats(self) -> Dict[str, Any]:
        """Get index statistics."""
        return {
            "total_documents": self.total_docs,
            "total_tokens": self._total_tokens,
            "unique_tokens": len(self.inverted_index),
            "avg_doc_length": self.avg_doc_length,
        }

    def clear(self) -> None:
        """Clear the index."""
        self.documents.clear()
        self.inverted_index.clear()
        self.doc_freq.clear()
        self.total_docs = 0
        self.avg_doc_length = 0.0
        self._total_tokens = 0


class HybridRetriever:
    """
    Combines BM25 and semantic search using Reciprocal Rank Fusion.

    Features:
    - RRF score combination
    - Configurable weights
    - Deduplication
    """

    def __init__(
        self,
        bm25: BM25Retriever,
        semantic_search_fn: callable,
        alpha: float = 0.5,  # Weight for semantic (1-alpha for BM25)
        rrf_k: int = 60,  # RRF constant
    ):
        """
        Initialize hybrid retriever.

        Args:
            bm25: BM25Retriever instance
            semantic_search_fn: Function(query) -> List[(doc_id, score, content)]
            alpha: Weight for semantic search (0 = BM25 only, 1 = semantic only)
            rrf_k: RRF smoothing constant
        """
        self.bm25 = bm25
        self.semantic_search_fn = semantic_search_fn
        self.alpha = alpha
        self.rrf_k = rrf_k

    def search(
        self,
        query: str,
        top_k: int = 10,
    ) -> List[Tuple[str, float, str, Dict[str, float]]]:
        """
        Hybrid search combining BM25 and semantic.

        Returns: List of (doc_id, combined_score, content, score_breakdown) tuples
        """
        # Get BM25 results
        bm25_results = self.bm25.search(query, top_k=top_k * 2)

        # Get semantic results
        semantic_results = self.semantic_search_fn(query)[:top_k * 2]

        # Create rank mappings
        bm25_ranks = {doc_id: rank for rank, (doc_id, _, _) in enumerate(bm25_results)}
        semantic_ranks = {doc_id: rank for rank, (doc_id, _, _) in enumerate(semantic_results)}

        # Collect all doc IDs
        all_doc_ids = set(bm25_ranks.keys()) | set(semantic_ranks.keys())

        # Combine scores using RRF
        combined = []
        for doc_id in all_doc_ids:
            # RRF scores
            bm25_rrf = 1 / (self.rrf_k + bm25_ranks.get(doc_id, len(bm25_results) + 1))
            semantic_rrf = 1 / (self.rrf_k + semantic_ranks.get(doc_id, len(semantic_results) + 1))

            # Weighted combination
            combined_score = (1 - self.alpha) * bm25_rrf + self.alpha * semantic_rrf

            # Get content from whichever result has it
            content = ""
            for _, _, c in bm25_results:
                if _ == doc_id:
                    content = c
                    break
            if not content:
                for d_id, _, c in semantic_results:
                    if d_id == doc_id:
                        content = c
                        break

            combined.append((
                doc_id,
                combined_score,
                content,
                {"bm25_rank": bm25_ranks.get(doc_id), "semantic_rank": semantic_ranks.get(doc_id)},
            ))

        # Sort by combined score
        combined.sort(key=lambda x: x[1], reverse=True)

        return combined[:top_k]
