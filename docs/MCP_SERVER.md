# 0GMem MCP Server

This guide explains how to integrate 0GMem with Claude Code using the Model Context Protocol (MCP).

## Overview

The 0GMem MCP server exposes intelligent memory capabilities to Claude Code, enabling:

- **Persistent memory** across conversations
- **Semantic search** over past interactions
- **Entity tracking** for people, places, and things
- **Temporal reasoning** for time-based queries
- **Multi-hop retrieval** for complex questions

## Quick Setup

### 1. Install 0GMem

```bash
# Clone and install
git clone https://github.com/loganionian/0gmem.git
cd 0gmem
pip install -e .

# Download spaCy model (required for entity extraction)
python -m spacy download en_core_web_sm
```

### 2. Add to Claude Code

```bash
# Add the MCP server
claude mcp add --transport stdio 0gmem -- python -m zerogmem.mcp_server

# Or use the console script
claude mcp add --transport stdio 0gmem -- 0gmem-mcp
```

### 3. Verify Setup

```bash
# List configured servers
claude mcp list

# Should show:
# 0gmem (stdio): python -m zerogmem.mcp_server
```

## Available Tools

Once configured, Claude Code will have access to these tools:

| Tool | Description |
|------|-------------|
| `store_memory` | Store a conversation message or fact |
| `retrieve_memories` | Retrieve relevant memories for a query |
| `search_memories_by_entity` | Find all memories about an entity |
| `search_memories_by_time` | Find memories from a time period |
| `get_memory_summary` | Get statistics about stored memories |
| `start_new_session` | Start a new conversation session |
| `end_conversation_session` | End session and consolidate memories |
| `export_memory` | Export memories to a portable ZIP archive |
| `import_memory` | Import memories from a ZIP archive |
| `clear_all_memories` | Delete all stored memories |

## Usage Examples

### Storing Memories

Claude can automatically store important information:

```
User: "Remember that my meeting with Sarah is next Tuesday at 2pm"
Claude: [calls store_memory("user", "Meeting with Sarah is next Tuesday at 2pm")]
       "I've stored that - your meeting with Sarah is scheduled for next Tuesday at 2pm."
```

### Retrieving Memories

Query past conversations naturally:

```
User: "What do you know about my meetings?"
Claude: [calls retrieve_memories("user's meetings")]
       "Based on my memory, you have a meeting with Sarah next Tuesday at 2pm..."
```

### Entity-Based Search

Find all information about a person/place/thing:

```
User: "Tell me everything you know about Sarah"
Claude: [calls search_memories_by_entity("Sarah")]
       "Here's what I know about Sarah: ..."
```

### Temporal Search

Find memories from specific times:

```
User: "What did we talk about yesterday?"
Claude: [calls search_memories_by_time("yesterday")]
       "Yesterday we discussed: ..."
```

## Configuration

### Environment Variables

All settings can be configured via environment variables:

| Variable | Type | Default | Description |
|----------|------|---------|-------------|
| `ZEROGMEM_DATA_DIR` | str | `~/.0gmem` | Directory for persistent memory storage |
| `ZEROGMEM_AUTOSAVE_INTERVAL` | int | `5` | Auto-save after every N store operations |
| `ZEROGMEM_MAX_EPISODES` | int | `500` | Maximum number of episodic memory episodes |
| `ZEROGMEM_MAX_FACTS` | int | `5000` | Maximum number of semantic memory facts |
| `ZEROGMEM_WORKING_MEMORY_CAPACITY` | int | `20` | Working memory capacity (recent items kept active) |
| `ZEROGMEM_EMBEDDING_MODEL` | str | `text-embedding-3-small` | OpenAI embedding model to use |
| `ZEROGMEM_MAX_CONTEXT_TOKENS` | int | `8000` | Maximum tokens in retrieved context |
| `ZEROGMEM_API_MAX_RETRIES` | int | `3` | Max retry attempts for OpenAI API calls |

> **Note:** When loading saved memory state from disk, the capacity settings (`max_episodes`, `max_facts`, `working_memory_capacity`) from the saved state take precedence over environment variables. Environment variable values are used for fresh starts only.

### Data Directory

By default, memories are stored in `~/.0gmem/`. Override with:

```bash
claude mcp add --transport stdio --env ZEROGMEM_DATA_DIR=/path/to/data 0gmem -- python -m zerogmem.mcp_server
```

### Project-Scoped Configuration

For team sharing, create `.mcp.json` in your project:

```json
{
  "mcpServers": {
    "0gmem": {
      "type": "stdio",
      "command": "python",
      "args": ["-m", "zerogmem.mcp_server"],
      "env": {
        "ZEROGMEM_DATA_DIR": "./.0gmem",
        "ZEROGMEM_MAX_EPISODES": "1000",
        "ZEROGMEM_EMBEDDING_MODEL": "text-embedding-3-large"
      }
    }
  }
}
```

## How It Works

```
┌─────────────────┐         MCP          ┌─────────────────┐
│   Claude Code   │◄───────────────────►│  0GMem Server   │
│     (Client)    │    JSON-RPC/stdio    │                 │
└─────────────────┘                      └────────┬────────┘
                                                  │
                                                  ▼
                                         ┌────────────────┐
                                         │     0GMem      │
                                         ├────────────────┤
                                         │ Memory Manager │
                                         │    Encoder     │
                                         │   Retriever    │
                                         └────────┬───────┘
                                                  │
                                                  ▼
                                         ┌────────────────┐
                                         │ Unified Graph  │
                                         │ • Temporal     │
                                         │ • Semantic     │
                                         │ • Entity       │
                                         │ • Causal       │
                                         └────────────────┘
```

## Troubleshooting

### Server Not Starting

Check logs in stderr:
```bash
python -m zerogmem.mcp_server 2>&1
```

### Missing Dependencies

```bash
pip install -e ".[dev]"
python -m spacy download en_core_web_sm
```

### Memory Not Persisting

Ensure the data directory is writable:
```bash
ls -la ~/.0gmem/
```

## Performance Notes

- First query may be slow (model loading)
- Subsequent queries: ~100-500ms
- Memory usage: ~500MB-1GB (embedding model)

## Roadmap

- [ ] Full state persistence/serialization
- [ ] Memory export/import
- [ ] Multi-user support
- [ ] Remote server mode (HTTP transport)
