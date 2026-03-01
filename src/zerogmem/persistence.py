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
import zipfile
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, Optional

import numpy as np

from zerogmem.memory.manager import MemoryManager

logger = logging.getLogger("0gmem-persistence")

STATE_FILENAME = "memory_state.json"
EMBEDDINGS_FILENAME = "memory_embeddings.npz"
SCHEMA_VERSION = 1

# Migration registry: version N -> function that upgrades state dict to version N+1.
_MIGRATIONS: Dict[int, Callable[[dict], dict]] = {
    # Example for future use:
    # 1: _migrate_v1_to_v2,
}


def _migrate(state: dict, from_version: int) -> Optional[dict]:
    """Run migration chain from from_version to SCHEMA_VERSION.

    Returns the migrated state dict, or None if a migration step is missing.
    """
    current = from_version
    while current < SCHEMA_VERSION:
        fn = _MIGRATIONS.get(current)
        if fn is None:
            logger.warning(
                f"No migration from v{current} to v{current + 1}. "
                f"Starting with fresh memory state."
            )
            return None
        state = fn(state)
        current += 1
        logger.info(f"Migrated state from v{current - 1} to v{current}")
    return state


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

        Returns a dict of {key: np.ndarray}. Returns empty dict if
        the file is missing or corrupt.
        """
        npz_path = path / EMBEDDINGS_FILENAME
        if not npz_path.exists():
            return {}

        try:
            data = np.load(npz_path, allow_pickle=False)
            return {key: data[key] for key in data.files}
        except Exception as e:
            logger.warning(
                f"Corrupt embeddings file {npz_path}, "
                f"proceeding without embeddings: {e}"
            )
            return {}

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
    state["schema_version"] = SCHEMA_VERSION

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

    Recovery cascade:
    1. Try primary JSON file. On failure, try .bak backup.
    2. If both corrupt, return None (caller creates fresh state).
    3. NPZ corruption is independent: proceed without embeddings.
    4. from_dict failure: return None (malformed JSON structure).

    Args:
        path: Directory path containing state files.
        embedding_fn: Embedding function to set on restored manager.

    Returns:
        Restored MemoryManager, or None if no valid state file exists.
    """
    path = Path(path)
    state_file = path / STATE_FILENAME
    backup_file = path / f"{STATE_FILENAME}.bak"

    # --- Step 1: Load JSON state with fallback to .bak ---
    state = None
    source_label = None

    if state_file.exists():
        try:
            with open(state_file) as f:
                state = json.load(f)
            source_label = "primary"
        except (json.JSONDecodeError, OSError) as e:
            logger.warning(f"Corrupt primary state file {state_file}: {e}")

    if state is None and backup_file.exists():
        try:
            with open(backup_file) as f:
                state = json.load(f)
            source_label = "backup"
            logger.warning(
                f"Recovered from backup file {backup_file}. "
                f"Primary state file was corrupt or missing."
            )
        except (json.JSONDecodeError, OSError) as e:
            logger.warning(f"Backup state file also corrupt {backup_file}: {e}")

    if state is None:
        if state_file.exists() or backup_file.exists():
            logger.warning(
                "All state files are corrupt. Starting with fresh memory state."
            )
        else:
            logger.info(f"No state file at {state_file}, returning None")
        return None

    # --- Step 1b: Schema version check and migration ---
    version = state.get("schema_version", 1)

    if version > SCHEMA_VERSION:
        logger.warning(
            f"State file version {version} is newer than supported "
            f"version {SCHEMA_VERSION}. Starting with fresh memory state."
        )
        return None

    if version < SCHEMA_VERSION:
        state = _migrate(state, version)
        if state is None:
            return None

    # Remove schema_version before passing to from_dict
    state.pop("schema_version", None)

    # --- Step 2: Load embeddings (independent, non-fatal) ---
    raw_embeddings = EmbeddingRegistry.load(path)

    # Build embeddings map for MemoryManager.from_dict
    embeddings_map = {}

    for key, emb in raw_embeddings.items():
        if key.startswith("episode_summary_"):
            eid = key[len("episode_summary_"):]
            embeddings_map[eid] = emb
        elif key.startswith("episode_full_"):
            pass  # full_embedding not restored currently (optional)
        elif key.startswith("fact_"):
            fid = key[len("fact_"):]
            embeddings_map[fid] = emb
        elif key.startswith("semantic_"):
            embeddings_map[key] = emb
        else:
            # Memory ID (unified graph)
            embeddings_map[key] = emb

    # --- Step 3: Deserialize (guard against malformed JSON structure) ---
    try:
        manager = MemoryManager.from_dict(state, embedding_fn, embeddings_map)
    except Exception as e:
        logger.warning(
            f"Failed to deserialize state from {source_label} file: {e}. "
            f"Starting with fresh memory state."
        )
        return None

    logger.info(
        f"Loaded memory state (from {source_label}): "
        f"{len(manager.graph.memories)} memories, "
        f"{len(manager.episodic_memory.episodes)} episodes, "
        f"{len(manager.semantic_memory.facts)} facts"
    )
    return manager


# ---------------------------------------------------------------------------
# Export / Import
# ---------------------------------------------------------------------------

EXPORT_FORMAT_VERSION = 1
METADATA_FILENAME = "export_metadata.json"


def export_memory_archive(
    manager: MemoryManager,
    archive_path: str | Path,
    data_dir: str | Path,
) -> Dict[str, Any]:
    """Export memory state to a portable ZIP archive.

    The archive contains:
    - memory_state.json   (structural data)
    - memory_embeddings.npz (vector data)
    - export_metadata.json (version, counts, timestamp)

    Args:
        manager: The MemoryManager to export.
        archive_path: Destination path for the .zip file.
        data_dir: Working directory used by save_memory_state().

    Returns:
        Summary dict with archive path and counts.
    """
    archive_path = Path(archive_path)
    data_dir = Path(data_dir)

    # Ensure latest state is written to data_dir
    save_summary = save_memory_state(manager, data_dir)

    # Build metadata
    from zerogmem import __version__

    emb_count = save_summary["embeddings_saved"]
    metadata = {
        "format_version": EXPORT_FORMAT_VERSION,
        "exported_at": datetime.now().isoformat(),
        "source_dir": str(data_dir),
        "zerogmem_version": __version__,
        "counts": {
            "memories": save_summary["memories"],
            "episodes": save_summary["episodes"],
            "facts": save_summary["facts"],
            "entities": save_summary["entities"],
            "embeddings": emb_count,
        },
    }

    # Create archive
    archive_path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive_path, "w", zipfile.ZIP_DEFLATED) as zf:
        # State JSON
        state_file = data_dir / STATE_FILENAME
        if state_file.exists():
            zf.write(state_file, STATE_FILENAME)

        # Embeddings NPZ
        emb_file = data_dir / EMBEDDINGS_FILENAME
        if emb_file.exists():
            zf.write(emb_file, EMBEDDINGS_FILENAME)

        # Metadata
        zf.writestr(METADATA_FILENAME, json.dumps(metadata, indent=2))

    logger.info(f"Exported memory archive to {archive_path}")
    return {
        "archive_path": str(archive_path),
        **metadata["counts"],
    }


def import_memory_archive(
    archive_path: str | Path,
    embedding_fn: Optional[callable] = None,
) -> Optional[MemoryManager]:
    """Import memory state from a ZIP archive.

    Args:
        archive_path: Path to the .zip archive.
        embedding_fn: Embedding function to set on restored manager.

    Returns:
        Restored MemoryManager, or None on failure.
    """
    archive_path = Path(archive_path)

    if not archive_path.exists():
        logger.warning(f"Archive not found: {archive_path}")
        return None

    if not zipfile.is_zipfile(archive_path):
        logger.warning(f"Not a valid ZIP file: {archive_path}")
        return None

    with zipfile.ZipFile(archive_path, "r") as zf:
        names = zf.namelist()

        # Validate required file
        if STATE_FILENAME not in names:
            logger.warning(
                f"Archive missing required file '{STATE_FILENAME}': {archive_path}"
            )
            return None

        # Check format version if metadata present
        if METADATA_FILENAME in names:
            try:
                meta = json.loads(zf.read(METADATA_FILENAME))
                version = meta.get("format_version", 1)
                if version > EXPORT_FORMAT_VERSION:
                    logger.warning(
                        f"Archive format version {version} is newer than "
                        f"supported version {EXPORT_FORMAT_VERSION}."
                    )
                    return None
            except (json.JSONDecodeError, KeyError) as e:
                logger.warning(f"Corrupt export metadata, proceeding anyway: {e}")

        # Extract to temp directory and load
        tmp_dir = tempfile.mkdtemp(prefix="0gmem_import_")
        try:
            zf.extractall(tmp_dir)
            manager = load_memory_state(tmp_dir, embedding_fn)
        finally:
            shutil.rmtree(tmp_dir, ignore_errors=True)

    if manager is not None:
        logger.info(f"Imported memory archive from {archive_path}")
    return manager
