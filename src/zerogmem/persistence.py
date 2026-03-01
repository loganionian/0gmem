"""
Persistence layer for 0GMem: save and load memory state.

Saves the full memory state as JSON (structure) + NPZ (embeddings).
The JSON file is human-readable and debuggable. Embeddings are stored
in a separate compressed numpy file for efficiency.

File layout:
    <path>/memory_state.json      # All non-embedding data
    <path>/memory_embeddings.npz  # {id: embedding} numpy arrays
    <path>/memory_state.json.bak  # Previous save (backup)
"""

from __future__ import annotations

import json
import logging
import os
import shutil
import tempfile
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np

from zerogmem.memory.manager import MemoryManager

logger = logging.getLogger("0gmem-persistence")

STATE_FILENAME = "memory_state.json"
EMBEDDINGS_FILENAME = "memory_embeddings.npz"


class EmbeddingRegistry:
    """Collects embeddings during serialization for bulk NPZ save."""

    def __init__(self):
        self._store: Dict[str, np.ndarray] = {}

    def register(self, key: str, embedding: Optional[np.ndarray]) -> None:
        """Register an embedding under a key. Skips None."""
        if embedding is not None:
            self._store[key] = embedding

    def save(self, path: Path) -> int:
        """Save all registered embeddings to an NPZ file.

        Returns the number of embeddings saved.
        """
        if not self._store:
            # Remove stale file if present
            npz_path = path / EMBEDDINGS_FILENAME
            if npz_path.exists():
                npz_path.unlink()
            return 0

        # Atomic write: save to temp file, then rename.
        # np.savez_compressed appends .npz if the name doesn't end with it,
        # so we use a .npz suffix to keep the output path predictable.
        npz_path = path / EMBEDDINGS_FILENAME
        fd, tmp_path = tempfile.mkstemp(dir=path, suffix=".npz")
        try:
            os.close(fd)
            np.savez_compressed(tmp_path, **self._store)
            os.replace(tmp_path, npz_path)
        except BaseException:
            # Clean up temp file on any failure
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)
            raise
        return len(self._store)

    @staticmethod
    def load(path: Path) -> Dict[str, np.ndarray]:
        """Load embeddings from NPZ file.

        Returns a dict of {key: np.ndarray}.
        """
        npz_path = path / EMBEDDINGS_FILENAME
        if not npz_path.exists():
            return {}

        data = np.load(npz_path, allow_pickle=False)
        return {key: data[key] for key in data.files}

    @property
    def count(self) -> int:
        return len(self._store)


def _collect_embeddings(manager: MemoryManager, registry: EmbeddingRegistry) -> None:
    """Walk the memory manager and register all embeddings."""
    # Unified graph memories
    for mid, mem in manager.graph.memories.items():
        registry.register(mid, mem.embedding)

    # Semantic graph nodes (keyed by node ID, e.g. "semantic_<memory_id>")
    for nid, node in manager.graph.semantic_graph.nodes.items():
        registry.register(nid, node.embedding)

    # Episodic memory embeddings
    for eid, episode in manager.episodic_memory.episodes.items():
        registry.register(f"episode_summary_{eid}", episode.summary_embedding)
        registry.register(f"episode_full_{eid}", episode.full_embedding)

    # Semantic facts embeddings
    for fid, fact in manager.semantic_memory.facts.items():
        registry.register(f"fact_{fid}", fact.embedding)


def save_memory_state(manager: MemoryManager, path: str | Path) -> Dict[str, Any]:
    """Save the full memory state to disk.

    Args:
        manager: The MemoryManager to serialize.
        path: Directory path to save state files.

    Returns:
        Summary dict with counts of saved items.
    """
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)

    state_file = path / STATE_FILENAME

    # Backup existing state
    if state_file.exists():
        backup_file = path / f"{STATE_FILENAME}.bak"
        shutil.copy2(state_file, backup_file)

    # Serialize structure
    state = manager.to_dict()

    # Collect and save embeddings
    registry = EmbeddingRegistry()
    _collect_embeddings(manager, registry)
    emb_count = registry.save(path)

    # Atomic write: JSON to temp file, then rename
    fd, tmp_path = tempfile.mkstemp(dir=path, suffix=".tmp")
    try:
        with os.fdopen(fd, "w") as f:
            json.dump(state, f, indent=2, default=str)
        os.replace(tmp_path, state_file)
    except BaseException:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)
        raise

    summary = {
        "state_file": str(state_file),
        "embeddings_saved": emb_count,
        "memories": len(manager.graph.memories),
        "episodes": len(manager.episodic_memory.episodes),
        "facts": len(manager.semantic_memory.facts),
        "entities": len(manager.graph.entity_graph.nodes),
    }
    logger.info(f"Saved memory state: {summary}")
    return summary


def load_memory_state(
    path: str | Path,
    embedding_fn: Optional[callable] = None,
) -> Optional[MemoryManager]:
    """Load memory state from disk.

    Args:
        path: Directory path containing state files.
        embedding_fn: Embedding function to set on restored manager.

    Returns:
        Restored MemoryManager, or None if no state file exists.
    """
    path = Path(path)
    state_file = path / STATE_FILENAME

    if not state_file.exists():
        logger.info(f"No state file at {state_file}, returning None")
        return None

    # Load JSON state
    with open(state_file) as f:
        state = json.load(f)

    # Load embeddings
    raw_embeddings = EmbeddingRegistry.load(path)

    # Build embeddings map for MemoryManager.from_dict
    # We need to route embeddings to the right components:
    # - "memory_id" -> unified graph memory embedding
    # - "semantic_<id>" -> semantic graph node embedding
    # - "episode_summary_<id>" / "episode_full_<id>" -> episodic
    # - "fact_<id>" -> semantic memory fact embedding
    #
    # MemoryManager.from_dict passes the full map to each sub-component.
    # Each sub-component's from_dict picks the keys it needs.

    embeddings_map = {}

    for key, emb in raw_embeddings.items():
        if key.startswith("episode_summary_"):
            eid = key[len("episode_summary_"):]
            embeddings_map[eid] = emb  # EpisodicMemory.from_dict keys by episode_id
        elif key.startswith("episode_full_"):
            pass  # full_embedding not restored currently (optional)
        elif key.startswith("fact_"):
            fid = key[len("fact_"):]
            embeddings_map[fid] = emb  # SemanticMemoryStore.from_dict keys by fact_id
        elif key.startswith("semantic_"):
            embeddings_map[key] = emb  # UnifiedMemoryGraph.from_dict routes these
        else:
            # Memory ID (unified graph)
            embeddings_map[key] = emb

    manager = MemoryManager.from_dict(state, embedding_fn, embeddings_map)

    logger.info(
        f"Loaded memory state: "
        f"{len(manager.graph.memories)} memories, "
        f"{len(manager.episodic_memory.episodes)} episodes, "
        f"{len(manager.semantic_memory.facts)} facts"
    )
    return manager
