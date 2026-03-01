"""
Working Memory: Active reasoning workspace.

Maintains currently relevant context with limited capacity
and attention-based decay, analogous to human working memory.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Optional, List, Dict, Any
import numpy as np


@dataclass
class WorkingMemoryItem:
    """An item in working memory."""
    id: str
    content: str
    embedding: Optional[np.ndarray] = None
    source_memory_id: Optional[str] = None  # Reference to long-term memory
    attention_weight: float = 1.0
    added_at: datetime = field(default_factory=datetime.now)
    last_accessed: datetime = field(default_factory=datetime.now)
    access_count: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)

    def decay(self, rate: float = 0.1, time_factor: float = 1.0) -> None:
        """Apply attention decay based on time and rate."""
        elapsed = (datetime.now() - self.last_accessed).total_seconds()
        decay_amount = rate * time_factor * (elapsed / 60.0)  # Per minute
        self.attention_weight = max(0.01, self.attention_weight - decay_amount)

    def boost(self, amount: float = 0.2) -> None:
        """Boost attention when accessed."""
        self.attention_weight = min(1.0, self.attention_weight + amount)
        self.last_accessed = datetime.now()
        self.access_count += 1


class WorkingMemory:
    """
    Working memory with limited capacity and attention-based management.

    Inspired by cognitive science models:
    - Limited capacity (~7±2 items, we use 20)
    - Items decay over time
    - Attention determines what stays
    - Recent and frequently accessed items persist
    """

    def __init__(
        self,
        capacity: int = 20,
        decay_rate: float = 0.05,
        eviction_threshold: float = 0.1
    ):
        self.capacity = capacity
        self.decay_rate = decay_rate
        self.eviction_threshold = eviction_threshold
        self.items: List[WorkingMemoryItem] = []
        self._item_index: Dict[str, WorkingMemoryItem] = {}

    def add(self, item: WorkingMemoryItem, force: bool = False) -> bool:
        """
        Add item to working memory.

        Args:
            item: The item to add
            force: If True, force add even if at capacity

        Returns:
            True if added successfully
        """
        # Check if already exists
        if item.id in self._item_index:
            # Boost existing item
            existing = self._item_index[item.id]
            existing.boost()
            return True

        # Apply decay to existing items
        self._apply_decay()

        # Evict if at capacity
        if len(self.items) >= self.capacity:
            if not self._evict_lowest():
                if not force:
                    return False

        # Add new item
        self.items.append(item)
        self._item_index[item.id] = item
        return True

    def get(self, item_id: str) -> Optional[WorkingMemoryItem]:
        """Get an item by ID, boosting its attention."""
        item = self._item_index.get(item_id)
        if item:
            item.boost()
        return item

    def get_all(self, min_attention: float = 0.0) -> List[WorkingMemoryItem]:
        """Get all items above minimum attention threshold."""
        return [
            item for item in self.items
            if item.attention_weight >= min_attention
        ]

    def get_context(
        self,
        query_embedding: Optional[np.ndarray] = None,
        top_k: int = 10
    ) -> List[WorkingMemoryItem]:
        """
        Get relevant working memory items for current context.

        If query_embedding provided, combines attention with similarity.
        """
        if not self.items:
            return []

        # Score items
        scored_items = []
        for item in self.items:
            score = item.attention_weight

            # Add similarity component if query provided
            if query_embedding is not None and item.embedding is not None:
                similarity = self._cosine_similarity(query_embedding, item.embedding)
                score = 0.5 * score + 0.5 * similarity

            scored_items.append((item, score))

        # Sort by score descending
        scored_items.sort(key=lambda x: x[1], reverse=True)

        # Boost accessed items
        result = []
        for item, _ in scored_items[:top_k]:
            item.boost(0.1)  # Small boost for being retrieved
            result.append(item)

        return result

    def update_attention(self, item_id: str, delta: float) -> None:
        """Manually adjust attention for an item."""
        item = self._item_index.get(item_id)
        if item:
            item.attention_weight = max(0.01, min(1.0, item.attention_weight + delta))

    def remove(self, item_id: str) -> bool:
        """Remove an item from working memory."""
        item = self._item_index.get(item_id)
        if item:
            self.items.remove(item)
            del self._item_index[item_id]
            return True
        return False

    def clear(self) -> None:
        """Clear all items from working memory."""
        self.items.clear()
        self._item_index.clear()

    def _apply_decay(self) -> None:
        """Apply decay to all items."""
        for item in self.items:
            item.decay(self.decay_rate)

    def _evict_lowest(self) -> bool:
        """Evict the item with lowest attention."""
        if not self.items:
            return False

        # Find lowest attention item
        lowest = min(self.items, key=lambda x: x.attention_weight)

        if lowest.attention_weight < self.eviction_threshold or len(self.items) >= self.capacity:
            self.items.remove(lowest)
            del self._item_index[lowest.id]
            return True

        return False

    def _cosine_similarity(self, a: np.ndarray, b: np.ndarray) -> float:
        """Compute cosine similarity between two vectors."""
        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return float(np.dot(a, b) / (norm_a * norm_b))

    def get_summary(self) -> str:
        """Get a text summary of working memory contents."""
        if not self.items:
            return "Working memory is empty."

        # Sort by attention
        sorted_items = sorted(self.items, key=lambda x: x.attention_weight, reverse=True)

        lines = ["Current working memory:"]
        for i, item in enumerate(sorted_items[:10], 1):
            attention_bar = "█" * int(item.attention_weight * 10)
            lines.append(f"  {i}. [{attention_bar:<10}] {item.content[:50]}...")

        return "\n".join(lines)

    def to_context_string(self, max_items: int = 10) -> str:
        """Convert working memory to a context string for prompts."""
        items = self.get_context(top_k=max_items)
        if not items:
            return ""

        context_parts = []
        for item in items:
            context_parts.append(item.content)

        return "\n\n".join(context_parts)

    @property
    def size(self) -> int:
        """Current number of items in working memory."""
        return len(self.items)

    @property
    def is_full(self) -> bool:
        """Check if working memory is at capacity."""
        return len(self.items) >= self.capacity

    def get_stats(self) -> Dict[str, Any]:
        """Get statistics about working memory."""
        if not self.items:
            return {
                "size": 0,
                "capacity": self.capacity,
                "avg_attention": 0.0,
                "min_attention": 0.0,
                "max_attention": 0.0,
            }

        attentions = [item.attention_weight for item in self.items]
        return {
            "size": len(self.items),
            "capacity": self.capacity,
            "avg_attention": np.mean(attentions),
            "min_attention": min(attentions),
            "max_attention": max(attentions),
        }
