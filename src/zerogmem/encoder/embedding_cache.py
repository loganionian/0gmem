"""
Embedding Cache: Fast caching layer for embeddings with batch support.

Key features:
- In-memory LRU cache
- Disk persistence for cross-session caching
- Batch embedding support for OpenAI API
"""

from __future__ import annotations

import hashlib
import json
import os
import pickle
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional, List, Dict, Callable, Tuple
import numpy as np


@dataclass
class CacheEntry:
    """A cached embedding entry."""
    text_hash: str
    embedding: np.ndarray
    model: str
    created_at: str
    access_count: int = 0


@dataclass
class EmbeddingCacheConfig:
    """Configuration for embedding cache."""
    cache_dir: str = ".cache/embeddings"
    max_memory_entries: int = 10000
    persist_to_disk: bool = True
    batch_size: int = 100  # Max texts per API call
    model: str = "text-embedding-3-small"


class EmbeddingCache:
    """
    Caching layer for embeddings with batch support.

    Features:
    - Hash-based deduplication
    - In-memory cache with LRU eviction
    - Disk persistence
    - Batch API calls for speed
    """

    def __init__(
        self,
        config: Optional[EmbeddingCacheConfig] = None,
        openai_client: Optional[any] = None,
    ):
        self.config = config or EmbeddingCacheConfig()
        self._client = openai_client

        # In-memory cache: hash -> embedding
        self._cache: Dict[str, np.ndarray] = {}
        self._access_order: List[str] = []  # For LRU

        # Stats
        self.stats = {
            "hits": 0,
            "misses": 0,
            "api_calls": 0,
            "texts_embedded": 0,
        }

        # Load from disk if available
        if self.config.persist_to_disk:
            self._load_cache()

    def _get_client(self):
        """Get or create OpenAI client."""
        if self._client is None:
            try:
                import openai
                self._client = openai.OpenAI()
            except Exception as e:
                print(f"Warning: Could not initialize OpenAI client: {e}")
        return self._client

    def _hash_text(self, text: str) -> str:
        """Create a stable hash for text."""
        return hashlib.sha256(text.encode('utf-8')).hexdigest()[:16]

    def get_embedding(self, text: str) -> np.ndarray:
        """Get embedding for a single text (uses batch internally)."""
        results = self.get_embeddings([text])
        return results[0]

    def get_embeddings(self, texts: List[str]) -> List[np.ndarray]:
        """
        Get embeddings for multiple texts with caching and batching.

        This is the main entry point for batch embedding.
        """
        if not texts:
            return []

        # Deduplicate and check cache
        unique_texts = []
        text_to_hash = {}
        cached_results = {}

        for text in texts:
            text_hash = self._hash_text(text)
            text_to_hash[text] = text_hash

            if text_hash in self._cache:
                self.stats["hits"] += 1
                cached_results[text_hash] = self._cache[text_hash]
                self._update_access(text_hash)
            elif text not in unique_texts:
                self.stats["misses"] += 1
                unique_texts.append(text)

        # Embed uncached texts in batches
        if unique_texts:
            new_embeddings = self._batch_embed(unique_texts)

            # Store in cache
            for text, embedding in zip(unique_texts, new_embeddings):
                text_hash = self._hash_text(text)
                self._add_to_cache(text_hash, embedding)
                cached_results[text_hash] = embedding

        # Reconstruct results in original order
        results = []
        for text in texts:
            text_hash = text_to_hash[text]
            results.append(cached_results[text_hash])

        return results

    def _batch_embed(self, texts: List[str]) -> List[np.ndarray]:
        """Embed texts in batches using OpenAI API."""
        client = self._get_client()
        if client is None:
            # Fallback to random embeddings
            return [self._random_embedding(text) for text in texts]

        all_embeddings = []
        batch_size = self.config.batch_size

        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]

            try:
                response = client.embeddings.create(
                    model=self.config.model,
                    input=batch,
                )
                self.stats["api_calls"] += 1
                self.stats["texts_embedded"] += len(batch)

                # Extract embeddings in order
                embeddings = [None] * len(batch)
                for item in response.data:
                    embeddings[item.index] = np.array(item.embedding, dtype=np.float32)

                all_embeddings.extend(embeddings)

            except Exception as e:
                print(f"Batch embedding error: {e}")
                # Fallback to individual calls or random
                for text in batch:
                    try:
                        response = client.embeddings.create(
                            model=self.config.model,
                            input=text,
                        )
                        self.stats["api_calls"] += 1
                        self.stats["texts_embedded"] += 1
                        all_embeddings.append(np.array(response.data[0].embedding, dtype=np.float32))
                    except:
                        all_embeddings.append(self._random_embedding(text))

        return all_embeddings

    def _random_embedding(self, text: str, dim: int = 1536) -> np.ndarray:
        """Generate deterministic random embedding (for testing)."""
        np.random.seed(hash(text) % (2**32))
        return np.random.randn(dim).astype(np.float32)

    def _add_to_cache(self, text_hash: str, embedding: np.ndarray) -> None:
        """Add embedding to cache with LRU eviction."""
        if text_hash in self._cache:
            return

        # Evict if at capacity
        while len(self._cache) >= self.config.max_memory_entries:
            oldest = self._access_order.pop(0)
            self._cache.pop(oldest, None)

        self._cache[text_hash] = embedding
        self._access_order.append(text_hash)

    def _update_access(self, text_hash: str) -> None:
        """Update access order for LRU."""
        if text_hash in self._access_order:
            self._access_order.remove(text_hash)
        self._access_order.append(text_hash)

    def _load_cache(self) -> None:
        """Load cache from disk."""
        cache_path = Path(self.config.cache_dir) / "embeddings.pkl"

        if cache_path.exists():
            try:
                with open(cache_path, 'rb') as f:
                    data = pickle.load(f)
                    self._cache = data.get("cache", {})
                    self._access_order = data.get("access_order", list(self._cache.keys()))
                print(f"Loaded {len(self._cache)} cached embeddings from disk")
            except Exception as e:
                print(f"Warning: Could not load embedding cache: {e}")

    def save_cache(self) -> None:
        """Save cache to disk."""
        if not self.config.persist_to_disk:
            return

        cache_dir = Path(self.config.cache_dir)
        cache_dir.mkdir(parents=True, exist_ok=True)
        cache_path = cache_dir / "embeddings.pkl"

        try:
            with open(cache_path, 'wb') as f:
                pickle.dump({
                    "cache": self._cache,
                    "access_order": self._access_order,
                }, f)
            print(f"Saved {len(self._cache)} embeddings to disk")
        except Exception as e:
            print(f"Warning: Could not save embedding cache: {e}")

    def get_stats(self) -> Dict[str, any]:
        """Get cache statistics."""
        total_requests = self.stats["hits"] + self.stats["misses"]
        hit_rate = self.stats["hits"] / total_requests if total_requests > 0 else 0

        return {
            "cache_size": len(self._cache),
            "hits": self.stats["hits"],
            "misses": self.stats["misses"],
            "hit_rate": hit_rate,
            "api_calls": self.stats["api_calls"],
            "texts_embedded": self.stats["texts_embedded"],
        }

    def clear(self) -> None:
        """Clear the cache."""
        self._cache.clear()
        self._access_order.clear()


def create_cached_embedding_fn(
    cache: Optional[EmbeddingCache] = None,
    config: Optional[EmbeddingCacheConfig] = None,
) -> Callable[[str], np.ndarray]:
    """
    Factory function to create a cached embedding function.

    Returns a function compatible with the Encoder interface.
    """
    if cache is None:
        cache = EmbeddingCache(config)

    def embed(text: str) -> np.ndarray:
        return cache.get_embedding(text)

    return embed
