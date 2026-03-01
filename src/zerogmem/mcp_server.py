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
from datetime import datetime
from pathlib import Path
from typing import Optional

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

    try:
        from zerogmem import MemoryManager, Encoder, Retriever
        from zerogmem.persistence import load_memory_state

        # Initialize encoder first (needed for both fresh and restored)
        _encoder = Encoder()

        # Try to load existing memory state
        memory_dir = _get_memory_dir()
        restored = load_memory_state(memory_dir, _encoder.get_embedding)

        if restored is not None:
            _memory_manager = restored
            logger.info("Restored memory state from disk")
        else:
            _memory_manager = MemoryManager()
            _memory_manager.set_embedding_function(_encoder.get_embedding)
            logger.info("Starting with fresh memory state")

        _retriever = Retriever(_memory_manager, embedding_fn=_encoder.get_embedding)

        # Register atexit handler to save state on shutdown
        atexit.register(_save_state)

        _initialized = True
        logger.info("0GMem initialized successfully")

    except Exception as e:
        logger.error(f"Failed to initialize 0GMem: {e}")
        raise


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
            logger.error(f"Error storing memory: {e}")
            return f"Error storing memory: {str(e)}"


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
            logger.error(f"Error retrieving memories: {e}")
            return f"Error retrieving memories: {str(e)}"


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
            logger.error(f"Error searching by entity: {e}")
            return f"Error searching memories: {str(e)}"


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
            logger.error(f"Error searching by time: {e}")
            return f"Error searching memories: {str(e)}"


@mcp.tool()
async def get_memory_summary() -> str:
    """Get a summary of the current memory state.

    Use this to understand what's currently stored in memory, including
    statistics about episodes, entities, and facts.

    Returns:
        Summary of memory contents and statistics
    """
    async with _lock:
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

            # Current session
            if stats.get("current_session"):
                summary_parts.append(f"### Active Session\n- Session ID: {stats['current_session']}")

            logger.info("Generated memory summary")
            return "\n".join(summary_parts)

        except Exception as e:
            logger.error(f"Error getting summary: {e}")
            return f"Error getting memory summary: {str(e)}"


@mcp.tool()
async def end_conversation_session() -> str:
    """End the current conversation session and consolidate memories.

    Call this when a conversation topic or context is complete. This helps
    the memory system organize and index the memories for better retrieval.

    Returns:
        Confirmation that the session was ended
    """
    async with _lock:
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
            logger.error(f"Error ending session: {e}")
            return f"Error ending session: {str(e)}"


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
            logger.error(f"Error starting session: {e}")
            return f"Error starting session: {str(e)}"


@mcp.tool()
async def clear_all_memories() -> str:
    """Clear all stored memories and start fresh.

    WARNING: This permanently deletes all memories. Use with caution.

    Returns:
        Confirmation that memories were cleared
    """
    async with _lock:
        try:
            global _memory_manager, _encoder, _retriever, _initialized

            # Reset the memory system
            from zerogmem import MemoryManager, Encoder, Retriever

            _memory_manager = MemoryManager()
            _encoder = Encoder()
            _memory_manager.set_embedding_function(_encoder.get_embedding)
            _retriever = Retriever(_memory_manager, embedding_fn=_encoder.get_embedding)

            # Clear any persisted state files
            memory_dir = _get_memory_dir()
            for fname in ["memory_state.json", "memory_state.json.bak", "memory_embeddings.npz"]:
                fpath = memory_dir / fname
                if fpath.exists():
                    fpath.unlink()

            logger.info("All memories cleared")
            return "All memories have been cleared. Starting fresh."

        except Exception as e:
            logger.error(f"Error clearing memories: {e}")
            return f"Error clearing memories: {str(e)}"


def main():
    """Run the MCP server."""
    logger.info("Starting 0GMem MCP server...")
    mcp.run(transport="stdio")


if __name__ == "__main__":
    main()
