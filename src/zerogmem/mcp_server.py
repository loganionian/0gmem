"""
0GMem MCP Server - Model Context Protocol server for Claude Code integration.

This server exposes 0GMem's memory capabilities as MCP tools that can be used
by Claude Code for persistent, intelligent conversation memory.

Usage:
    # Run directly
    python -m zerogmem.mcp_server

    # Or add to Claude Code
    claude mcp add --transport stdio 0gmem -- python -m zerogmem.mcp_server
"""

import asyncio
import atexit
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional

# MCP server must not write to stdout - use stderr for logging
import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    stream=sys.stderr,
)
logger = logging.getLogger("0gmem-mcp")

from mcp.server.fastmcp import FastMCP

# Initialize the MCP server
mcp = FastMCP("0gmem")

# Global state for the memory system
_memory_manager = None
_encoder = None
_retriever = None
_initialized = False
_memory_dir = None
_store_count = 0  # Counter for autosave
_autosave_interval = int(os.environ.get("ZEROGMEM_AUTOSAVE_INTERVAL", "5"))

# Serializes all tool handler access to shared state
_lock = asyncio.Lock()


class OperationMetrics:
    """Lightweight per-operation metrics tracker."""

    def __init__(self):
        self._ops: Dict[str, dict] = {}
        self._start_time = datetime.now()

    def record(self, operation: str, duration_ms: float, error: bool = False):
        """Record a completed operation."""
        if operation not in self._ops:
            self._ops[operation] = {
                "count": 0, "errors": 0,
                "total_ms": 0.0, "last_ms": 0.0,
                "min_ms": float("inf"), "max_ms": 0.0,
            }
        op = self._ops[operation]
        op["count"] += 1
        op["total_ms"] += duration_ms
        op["last_ms"] = duration_ms
        op["min_ms"] = min(op["min_ms"], duration_ms)
        op["max_ms"] = max(op["max_ms"], duration_ms)
        if error:
            op["errors"] += 1

    def get_summary(self) -> dict:
        """Get metrics summary for all operations."""
        result = {
            "uptime_seconds": (datetime.now() - self._start_time).total_seconds(),
        }
        for name, op in self._ops.items():
            avg = op["total_ms"] / op["count"] if op["count"] > 0 else 0
            result[name] = {
                "count": op["count"],
                "errors": op["errors"],
                "avg_ms": round(avg, 1),
                "min_ms": round(op["min_ms"], 1) if op["min_ms"] != float("inf") else 0,
                "max_ms": round(op["max_ms"], 1),
                "last_ms": round(op["last_ms"], 1),
            }
        return result


_metrics = OperationMetrics()


def _env_int(name: str, default: int) -> int:
    """Read an integer from an environment variable with fallback."""
    val = os.environ.get(name)
    if val is None:
        return default
    try:
        return int(val)
    except ValueError:
        logger.warning(f"Invalid integer for {name}={val!r}, using default {default}")
        return default


def _build_configs():
    """Build config objects from environment variables.

    Returns (encoder_config, memory_config, retriever_config).
    """
    from zerogmem.encoder.encoder import EncoderConfig
    from zerogmem.memory.manager import MemoryConfig
    from zerogmem.retriever.retriever import RetrieverConfig

    encoder_config = EncoderConfig(
        embedding_model=os.environ.get(
            "ZEROGMEM_EMBEDDING_MODEL", "text-embedding-3-small"
        ),
        max_retries=_env_int("ZEROGMEM_API_MAX_RETRIES", 3),
    )
    memory_config = MemoryConfig(
        max_episodes=_env_int("ZEROGMEM_MAX_EPISODES", 500),
        max_facts=_env_int("ZEROGMEM_MAX_FACTS", 5000),
        working_memory_capacity=_env_int("ZEROGMEM_WORKING_MEMORY_CAPACITY", 20),
    )
    retriever_config = RetrieverConfig(
        max_context_tokens=_env_int("ZEROGMEM_MAX_CONTEXT_TOKENS", 8000),
    )
    return encoder_config, memory_config, retriever_config

# --- Input validation limits ---
MAX_CONTENT_LENGTH = 50_000
MAX_SPEAKER_LENGTH = 200
MAX_QUERY_LENGTH = 2_000
MAX_TOPIC_LENGTH = 500
MAX_ENTITY_LENGTH = 500
MAX_TIME_DESC_LENGTH = 500
MAX_METADATA_LENGTH = 10_000
MAX_PATH_LENGTH = 1_000
MIN_MAX_RESULTS = 1
MAX_MAX_RESULTS = 100


def _validate_string(value: str, field_name: str, max_length: int) -> Optional[str]:
    """Validate a string input. Returns error message if invalid, None if valid."""
    if not value or not value.strip():
        return f"Error: '{field_name}' must not be empty."
    if len(value) > max_length:
        return (
            f"Error: '{field_name}' exceeds maximum length of {max_length:,} "
            f"characters (got {len(value):,})."
        )
    return None


def _clamp_max_results(value: int) -> int:
    """Clamp max_results to valid range."""
    return max(MIN_MAX_RESULTS, min(MAX_MAX_RESULTS, value))


def _get_memory_dir() -> Path:
    """Get the directory for storing memory data."""
    global _memory_dir
    if _memory_dir is None:
        # Default to ~/.0gmem for persistent storage
        _memory_dir = Path(os.environ.get("ZEROGMEM_DATA_DIR", Path.home() / ".0gmem"))
        _memory_dir.mkdir(parents=True, exist_ok=True)
    return _memory_dir


def _save_state():
    """Save current memory state to disk."""
    if _memory_manager is None:
        return
    try:
        from zerogmem.persistence import save_memory_state
        save_memory_state(_memory_manager, _get_memory_dir())
    except Exception as e:
        logger.error(f"Failed to save memory state: {e}")


def _initialize_memory():
    """Initialize the memory system lazily."""
    global _memory_manager, _encoder, _retriever, _initialized

    if _initialized:
        return

    logger.info("Initializing 0GMem memory system...")
    t0 = time.monotonic()
    error = False

    try:
        from zerogmem import MemoryManager, Encoder, Retriever
        from zerogmem.persistence import load_memory_state

        encoder_config, memory_config, retriever_config = _build_configs()

        # Initialize encoder first (needed for both fresh and restored)
        _encoder = Encoder(config=encoder_config)

        # Try to load existing memory state
        memory_dir = _get_memory_dir()
        restored = load_memory_state(memory_dir, _encoder.get_embedding)

        if restored is not None:
            _memory_manager = restored
            logger.info("Restored memory state from disk")
        else:
            _memory_manager = MemoryManager(config=memory_config)
            _memory_manager.set_embedding_function(_encoder.get_embedding)
            logger.info("Starting with fresh memory state")

        _retriever = Retriever(
            _memory_manager, embedding_fn=_encoder.get_embedding,
            config=retriever_config,
        )

        # Register atexit handler to save state on shutdown
        atexit.register(_save_state)

        _initialized = True
        logger.info("0GMem initialized successfully")

    except Exception as e:
        error = True
        logger.error(f"Failed to initialize 0GMem: {e}")
        raise
    finally:
        elapsed_ms = (time.monotonic() - t0) * 1000
        _metrics.record("initialize", elapsed_ms, error=error)


def _ensure_session():
    """Ensure there's an active session, creating one if needed."""
    _initialize_memory()
    if _memory_manager.current_session_id is None:
        _memory_manager.start_session()
        logger.info("Started new memory session")


@mcp.tool()
async def store_memory(
    speaker: str,
    content: str,
    metadata: Optional[str] = None,
) -> str:
    """Store a conversation message or fact in long-term memory.

    Use this to remember important information from conversations that should
    be recalled later. The memory system will automatically extract entities,
    temporal information, and relationships.

    Args:
        speaker: Who said this (e.g., "user", "assistant", person's name)
        content: The message content or fact to remember
        metadata: Optional JSON string with additional context (e.g., {"topic": "work", "importance": "high"})

    Returns:
        Confirmation message with memory ID
    """
    async with _lock:
        err = _validate_string(speaker, "speaker", MAX_SPEAKER_LENGTH)
        if err:
            return err
        err = _validate_string(content, "content", MAX_CONTENT_LENGTH)
        if err:
            return err
        if metadata is not None and len(metadata) > MAX_METADATA_LENGTH:
            return f"Error: 'metadata' exceeds maximum length of {MAX_METADATA_LENGTH:,} characters."

        t0 = time.monotonic()
        error = False
        try:
            _ensure_session()

            # Parse metadata if provided
            meta_dict = {}
            if metadata:
                try:
                    meta_dict = json.loads(metadata)
                except json.JSONDecodeError:
                    meta_dict = {"raw_metadata": metadata}

            # Add timestamp
            meta_dict["stored_at"] = datetime.now().isoformat()

            # Store the message
            msg_id = _memory_manager.add_message(speaker, content, metadata=meta_dict)

            # Autosave periodically
            global _store_count
            _store_count += 1
            if _store_count % _autosave_interval == 0:
                _save_state()
                logger.info(f"Autosaved after {_store_count} stores")

            logger.info(f"Stored memory: {msg_id} from {speaker}")
            return f"Memory stored successfully (ID: {msg_id})"

        except Exception as e:
            error = True
            logger.error(f"Error storing memory: {e}")
            return f"Error storing memory: {str(e)}"
        finally:
            elapsed_ms = (time.monotonic() - t0) * 1000
            _metrics.record("store_memory", elapsed_ms, error=error)


@mcp.tool()
async def retrieve_memories(
    query: str,
    max_results: int = 5,
) -> str:
    """Retrieve relevant memories for a given query.

    Use this to recall information from past conversations. The system uses
    semantic search, temporal reasoning, and entity relationships to find
    the most relevant memories.

    Args:
        query: The question or topic to search for (e.g., "What did the user say about their job?")
        max_results: Maximum number of memory chunks to return (default: 5)

    Returns:
        Relevant memory context formatted for use in responses
    """
    async with _lock:
        err = _validate_string(query, "query", MAX_QUERY_LENGTH)
        if err:
            return err
        max_results = _clamp_max_results(max_results)

        t0 = time.monotonic()
        error = False
        try:
            _initialize_memory()

            # Retrieve relevant memories
            result = _retriever.retrieve(query)
            if result.results:
                result.results = result.results[:max_results]

            if not result.composed_context or result.composed_context.strip() == "":
                return "No relevant memories found for this query."

            # Format the response
            response_parts = [
                f"## Retrieved Memories for: \"{query}\"\n",
                result.composed_context,
            ]

            if result.results:
                response_parts.append(f"\n---\n*Found {len(result.results)} relevant memory segments*")

            logger.info(f"Retrieved {len(result.results)} memories for query: {query[:50]}...")
            return "\n".join(response_parts)

        except Exception as e:
            error = True
            logger.error(f"Error retrieving memories: {e}")
            return f"Error retrieving memories: {str(e)}"
        finally:
            elapsed_ms = (time.monotonic() - t0) * 1000
            _metrics.record("retrieve_memories", elapsed_ms, error=error)


@mcp.tool()
async def search_memories_by_entity(
    entity_name: str,
    max_results: int = 10,
) -> str:
    """Search memories related to a specific entity (person, place, thing).

    Use this to find all information about a particular entity mentioned
    in past conversations.

    Args:
        entity_name: Name of the entity to search for (e.g., "Alice", "New York", "the project")
        max_results: Maximum number of results to return

    Returns:
        All memories mentioning or related to this entity
    """
    async with _lock:
        err = _validate_string(entity_name, "entity_name", MAX_ENTITY_LENGTH)
        if err:
            return err
        max_results = _clamp_max_results(max_results)

        t0 = time.monotonic()
        error = False
        try:
            _initialize_memory()

            # Use the retriever with an entity-focused query
            query = f"What do I know about {entity_name}?"
            result = _retriever.retrieve(query)
            if result.results:
                result.results = result.results[:max_results]

            if not result.composed_context or result.composed_context.strip() == "":
                return f"No memories found about '{entity_name}'."

            response = f"## Memories about: {entity_name}\n\n{result.composed_context}"
            logger.info(f"Found memories for entity: {entity_name}")
            return response

        except Exception as e:
            error = True
            logger.error(f"Error searching by entity: {e}")
            return f"Error searching memories: {str(e)}"
        finally:
            elapsed_ms = (time.monotonic() - t0) * 1000
            _metrics.record("search_by_entity", elapsed_ms, error=error)


@mcp.tool()
async def search_memories_by_time(
    time_description: str,
    max_results: int = 10,
) -> str:
    """Search memories from a specific time period.

    Use this to find memories from a particular time, like "yesterday",
    "last week", "in March", or "before the meeting".

    Args:
        time_description: Natural language time reference (e.g., "yesterday", "last month", "in 2024")
        max_results: Maximum number of results to return

    Returns:
        Memories from the specified time period
    """
    async with _lock:
        err = _validate_string(time_description, "time_description", MAX_TIME_DESC_LENGTH)
        if err:
            return err
        max_results = _clamp_max_results(max_results)

        t0 = time.monotonic()
        error = False
        try:
            _initialize_memory()

            # Use temporal query
            query = f"What happened {time_description}?"
            result = _retriever.retrieve(query)
            if result.results:
                result.results = result.results[:max_results]

            if not result.composed_context or result.composed_context.strip() == "":
                return f"No memories found for time period: '{time_description}'."

            response = f"## Memories from: {time_description}\n\n{result.composed_context}"
            logger.info(f"Found memories for time: {time_description}")
            return response

        except Exception as e:
            error = True
            logger.error(f"Error searching by time: {e}")
            return f"Error searching memories: {str(e)}"
        finally:
            elapsed_ms = (time.monotonic() - t0) * 1000
            _metrics.record("search_by_time", elapsed_ms, error=error)


@mcp.tool()
async def get_memory_summary() -> str:
    """Get a summary of the current memory state.

    Use this to understand what's currently stored in memory, including
    statistics about episodes, entities, and facts.

    Returns:
        Summary of memory contents and statistics
    """
    async with _lock:
        t0 = time.monotonic()
        error = False
        try:
            _initialize_memory()

            stats = _memory_manager.get_stats()

            # Format the summary
            summary_parts = ["## 0GMem Memory Summary\n"]

            # Episodic memory stats
            if "episodic_memory" in stats:
                ep = stats["episodic_memory"]
                summary_parts.append("### Episodic Memory")
                summary_parts.append(f"- Total episodes: {ep.get('total_episodes', 0)}")
                summary_parts.append(f"- Total messages: {ep.get('total_messages', 0)}")
                summary_parts.append(f"- Unique participants: {ep.get('unique_participants', 0)}")
                summary_parts.append("")

            # Semantic memory stats
            if "semantic_memory" in stats:
                sem = stats["semantic_memory"]
                summary_parts.append("### Semantic Memory")
                summary_parts.append(f"- Total facts: {sem.get('total_facts', 0)}")
                summary_parts.append(f"- Categories: {len(sem.get('categories', {}))}")
                summary_parts.append("")

            # Graph stats
            if "graph" in stats:
                g = stats["graph"]
                summary_parts.append("### Memory Graph")
                summary_parts.append(f"- Entity nodes: {g.get('entity_nodes', 0)}")
                summary_parts.append(f"- Semantic nodes: {g.get('semantic_nodes', 0)}")
                summary_parts.append(f"- Temporal nodes: {g.get('temporal_nodes', 0)}")
                summary_parts.append("")

            # Capacity
            if "capacity" in stats:
                cap = stats["capacity"]
                ep_stats = stats.get("episodic_memory", {})
                sem_stats = stats.get("semantic_memory", {})
                ep_count = ep_stats.get("total_episodes", 0)
                fact_count = sem_stats.get("total_facts", 0)
                summary_parts.append("### Capacity")
                summary_parts.append(
                    f"- Episodes: {ep_count}/{cap['max_episodes']} "
                    f"({cap['episode_utilization']:.0%})"
                )
                summary_parts.append(
                    f"- Facts: {fact_count}/{cap['max_facts']} "
                    f"({cap['fact_utilization']:.0%})"
                )
                summary_parts.append("")

            # Performance metrics
            metrics = _metrics.get_summary()
            uptime_s = metrics.pop("uptime_seconds", 0)
            if metrics:
                if uptime_s >= 3600:
                    uptime_str = f"{uptime_s / 3600:.1f}h"
                elif uptime_s >= 60:
                    uptime_str = f"{uptime_s / 60:.0f}m"
                else:
                    uptime_str = f"{uptime_s:.0f}s"
                summary_parts.append("### Performance")
                summary_parts.append(f"- Uptime: {uptime_str}")
                for op_name, op_stats in metrics.items():
                    err_str = f", {op_stats['errors']} errors" if op_stats["errors"] else ""
                    summary_parts.append(
                        f"- {op_name}: {op_stats['count']} calls, "
                        f"avg {op_stats['avg_ms']:.0f}ms{err_str}"
                    )
                summary_parts.append("")

            # Current session
            if stats.get("current_session"):
                summary_parts.append(f"### Active Session\n- Session ID: {stats['current_session']}")

            logger.info("Generated memory summary")
            return "\n".join(summary_parts)

        except Exception as e:
            error = True
            logger.error(f"Error getting summary: {e}")
            return f"Error getting memory summary: {str(e)}"
        finally:
            elapsed_ms = (time.monotonic() - t0) * 1000
            _metrics.record("get_summary", elapsed_ms, error=error)


@mcp.tool()
async def end_conversation_session() -> str:
    """End the current conversation session and consolidate memories.

    Call this when a conversation topic or context is complete. This helps
    the memory system organize and index the memories for better retrieval.

    Returns:
        Confirmation that the session was ended
    """
    async with _lock:
        t0 = time.monotonic()
        error = False
        try:
            _initialize_memory()

            if _memory_manager.current_session_id is None:
                return "No active session to end."

            session_id = _memory_manager.current_session_id
            _memory_manager.end_session()

            # Save state on session end
            _save_state()

            logger.info(f"Ended session: {session_id}")
            return f"Session ended and memories consolidated (Session ID: {session_id})"

        except Exception as e:
            error = True
            logger.error(f"Error ending session: {e}")
            return f"Error ending session: {str(e)}"
        finally:
            elapsed_ms = (time.monotonic() - t0) * 1000
            _metrics.record("end_session", elapsed_ms, error=error)


@mcp.tool()
async def start_new_session(topic: Optional[str] = None) -> str:
    """Start a new conversation session.

    Call this when starting a new conversation topic. Sessions help organize
    memories into coherent episodes.

    Args:
        topic: Optional topic/title for this session

    Returns:
        Confirmation with the new session ID
    """
    async with _lock:
        if topic is not None:
            err = _validate_string(topic, "topic", MAX_TOPIC_LENGTH)
            if err:
                return err

        t0 = time.monotonic()
        error = False
        try:
            _initialize_memory()

            # End existing session if any
            if _memory_manager.current_session_id is not None:
                _memory_manager.end_session()

            # Start new session
            session_id = _memory_manager.start_session()

            # Store topic as first message if provided
            if topic:
                _memory_manager.add_message("system", f"Session topic: {topic}")

            logger.info(f"Started new session: {session_id}, topic: {topic}")
            return f"New session started (ID: {session_id})" + (f" - Topic: {topic}" if topic else "")

        except Exception as e:
            error = True
            logger.error(f"Error starting session: {e}")
            return f"Error starting session: {str(e)}"
        finally:
            elapsed_ms = (time.monotonic() - t0) * 1000
            _metrics.record("start_session", elapsed_ms, error=error)


@mcp.tool()
async def clear_all_memories() -> str:
    """Clear all stored memories and start fresh.

    WARNING: This permanently deletes all memories. Use with caution.

    Returns:
        Confirmation that memories were cleared
    """
    async with _lock:
        t0 = time.monotonic()
        error = False
        try:
            global _memory_manager, _encoder, _retriever, _initialized

            # Reset the memory system with env-var configs
            from zerogmem import MemoryManager, Encoder, Retriever

            encoder_config, memory_config, retriever_config = _build_configs()

            _encoder = Encoder(config=encoder_config)
            _memory_manager = MemoryManager(config=memory_config)
            _memory_manager.set_embedding_function(_encoder.get_embedding)
            _retriever = Retriever(
                _memory_manager, embedding_fn=_encoder.get_embedding,
                config=retriever_config,
            )

            # Clear any persisted state files
            memory_dir = _get_memory_dir()
            for fname in ["memory_state.json", "memory_state.json.bak", "memory_embeddings.npz"]:
                fpath = memory_dir / fname
                if fpath.exists():
                    fpath.unlink()

            logger.info("All memories cleared")
            return "All memories have been cleared. Starting fresh."

        except Exception as e:
            error = True
            logger.error(f"Error clearing memories: {e}")
            return f"Error clearing memories: {str(e)}"
        finally:
            elapsed_ms = (time.monotonic() - t0) * 1000
            _metrics.record("clear_all", elapsed_ms, error=error)


@mcp.tool()
async def export_memory(output_path: Optional[str] = None) -> str:
    """Export all memories to a portable ZIP archive for backup or migration.

    Creates a single .zip file containing the full memory state (structure,
    embeddings, and metadata). This file can be imported on another machine
    or used to restore after clearing memories.

    Args:
        output_path: Optional destination path for the archive. Defaults to
                     <data_dir>/exports/0gmem_export_<timestamp>.zip

    Returns:
        Path to the created archive and summary counts
    """
    async with _lock:
        if output_path is not None:
            if len(output_path) > MAX_PATH_LENGTH:
                return f"Error: 'output_path' exceeds maximum length of {MAX_PATH_LENGTH} characters."
            if not output_path.endswith(".zip"):
                return "Error: 'output_path' must end with '.zip'."

        t0 = time.monotonic()
        error = False
        try:
            _initialize_memory()

            memory_dir = _get_memory_dir()

            if output_path is None:
                exports_dir = memory_dir / "exports"
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                archive_path = exports_dir / f"0gmem_export_{timestamp}.zip"
            else:
                archive_path = Path(output_path)

            from zerogmem.persistence import export_memory_archive

            summary = export_memory_archive(_memory_manager, archive_path, memory_dir)

            parts = [
                f"Memory exported successfully to: {summary['archive_path']}",
                f"- Memories: {summary['memories']}",
                f"- Episodes: {summary['episodes']}",
                f"- Facts: {summary['facts']}",
                f"- Entities: {summary['entities']}",
                f"- Embeddings: {summary['embeddings']}",
            ]
            logger.info(f"Exported memory to {summary['archive_path']}")
            return "\n".join(parts)

        except Exception as e:
            error = True
            logger.error(f"Error exporting memory: {e}")
            return f"Error exporting memory: {str(e)}"
        finally:
            elapsed_ms = (time.monotonic() - t0) * 1000
            _metrics.record("export_memory", elapsed_ms, error=error)


@mcp.tool()
async def import_memory(archive_path: str, merge: bool = False) -> str:
    """Import memories from a previously exported ZIP archive.

    Replaces the current memory state with the contents of the archive.
    Use this to restore from a backup or migrate memories from another machine.

    Args:
        archive_path: Path to the .zip archive to import
        merge: If True, merge with existing memories (not yet supported)

    Returns:
        Confirmation with imported memory counts
    """
    async with _lock:
        err = _validate_string(archive_path, "archive_path", MAX_PATH_LENGTH)
        if err:
            return err
        if not archive_path.endswith(".zip"):
            return "Error: 'archive_path' must end with '.zip'."

        if merge:
            return (
                "Error: Merge import is not yet supported. "
                "Use merge=False (default) to replace current memory state."
            )

        t0 = time.monotonic()
        error = False
        try:
            _initialize_memory()

            from zerogmem.persistence import import_memory_archive, save_memory_state

            global _memory_manager, _retriever

            restored = import_memory_archive(archive_path, _encoder.get_embedding)
            if restored is None:
                return "Error: Failed to import archive. Check that the file is a valid 0GMem export."

            # Replace current state
            _memory_manager = restored

            # Re-create retriever with new manager
            from zerogmem import Retriever
            _, _, retriever_config = _build_configs()
            _retriever = Retriever(
                _memory_manager, embedding_fn=_encoder.get_embedding,
                config=retriever_config,
            )

            # Save imported state to data dir
            save_memory_state(_memory_manager, _get_memory_dir())

            stats = _memory_manager.get_stats()
            ep = stats.get("episodic_memory", {})
            sem = stats.get("semantic_memory", {})
            g = stats.get("graph", {})

            parts = [
                "Memory imported successfully.",
                f"- Memories: {len(_memory_manager.graph.memories)}",
                f"- Episodes: {ep.get('total_episodes', 0)}",
                f"- Facts: {sem.get('total_facts', 0)}",
                f"- Entities: {g.get('entity_nodes', 0)}",
            ]
            logger.info(f"Imported memory from {archive_path}")
            return "\n".join(parts)

        except Exception as e:
            error = True
            logger.error(f"Error importing memory: {e}")
            return f"Error importing memory: {str(e)}"
        finally:
            elapsed_ms = (time.monotonic() - t0) * 1000
            _metrics.record("import_memory", elapsed_ms, error=error)


def main():
    """Run the MCP server."""
    logger.info("Starting 0GMem MCP server...")
    mcp.run(transport="stdio")


if __name__ == "__main__":
    main()
