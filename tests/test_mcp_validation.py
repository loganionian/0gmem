"""Tests for MCP server input validation."""

import pytest

from zerogmem.mcp_server import (
    _validate_string,
    _clamp_max_results,
    MAX_CONTENT_LENGTH,
    MAX_SPEAKER_LENGTH,
    MAX_QUERY_LENGTH,
    MAX_TOPIC_LENGTH,
    MAX_ENTITY_LENGTH,
    MAX_TIME_DESC_LENGTH,
    MAX_METADATA_LENGTH,
    MIN_MAX_RESULTS,
    MAX_MAX_RESULTS,
)


class TestValidateString:
    """Tests for the _validate_string helper."""

    def test_valid_string(self):
        assert _validate_string("hello", "field", 100) is None

    def test_empty_string(self):
        result = _validate_string("", "speaker", 100)
        assert result is not None
        assert "must not be empty" in result

    def test_whitespace_only(self):
        result = _validate_string("   \n\t  ", "speaker", 100)
        assert result is not None
        assert "must not be empty" in result

    def test_exceeds_max_length(self):
        result = _validate_string("a" * 201, "speaker", 200)
        assert result is not None
        assert "exceeds maximum length" in result
        assert "200" in result

    def test_at_max_length(self):
        assert _validate_string("a" * 200, "speaker", 200) is None

    def test_none_treated_as_empty(self):
        result = _validate_string(None, "field", 100)
        assert result is not None
        assert "must not be empty" in result

    def test_field_name_in_error(self):
        result = _validate_string("", "my_field", 100)
        assert "my_field" in result


class TestClampMaxResults:
    """Tests for the _clamp_max_results helper."""

    def test_within_range(self):
        assert _clamp_max_results(5) == 5
        assert _clamp_max_results(50) == 50

    def test_below_min(self):
        assert _clamp_max_results(0) == MIN_MAX_RESULTS
        assert _clamp_max_results(-10) == MIN_MAX_RESULTS

    def test_above_max(self):
        assert _clamp_max_results(999) == MAX_MAX_RESULTS
        assert _clamp_max_results(100_000) == MAX_MAX_RESULTS

    def test_at_boundaries(self):
        assert _clamp_max_results(MIN_MAX_RESULTS) == MIN_MAX_RESULTS
        assert _clamp_max_results(MAX_MAX_RESULTS) == MAX_MAX_RESULTS


class TestStoreMemoryValidation:
    """Test store_memory handler rejects invalid input before touching memory."""

    @pytest.mark.asyncio
    async def test_empty_speaker(self):
        from zerogmem.mcp_server import store_memory
        result = await store_memory(speaker="", content="hello")
        assert "Error" in result
        assert "speaker" in result

    @pytest.mark.asyncio
    async def test_whitespace_speaker(self):
        from zerogmem.mcp_server import store_memory
        result = await store_memory(speaker="   ", content="hello")
        assert "Error" in result
        assert "speaker" in result

    @pytest.mark.asyncio
    async def test_empty_content(self):
        from zerogmem.mcp_server import store_memory
        result = await store_memory(speaker="user", content="   ")
        assert "Error" in result
        assert "content" in result

    @pytest.mark.asyncio
    async def test_oversized_content(self):
        from zerogmem.mcp_server import store_memory
        result = await store_memory(speaker="user", content="x" * (MAX_CONTENT_LENGTH + 1))
        assert "Error" in result
        assert "exceeds" in result

    @pytest.mark.asyncio
    async def test_oversized_speaker(self):
        from zerogmem.mcp_server import store_memory
        result = await store_memory(speaker="x" * (MAX_SPEAKER_LENGTH + 1), content="hello")
        assert "Error" in result
        assert "speaker" in result

    @pytest.mark.asyncio
    async def test_oversized_metadata(self):
        from zerogmem.mcp_server import store_memory
        result = await store_memory(
            speaker="user", content="hello", metadata="x" * (MAX_METADATA_LENGTH + 1)
        )
        assert "Error" in result
        assert "metadata" in result


class TestRetrieveMemoriesValidation:
    """Test retrieve_memories handler rejects invalid input."""

    @pytest.mark.asyncio
    async def test_empty_query(self):
        from zerogmem.mcp_server import retrieve_memories
        result = await retrieve_memories(query="")
        assert "Error" in result
        assert "query" in result

    @pytest.mark.asyncio
    async def test_oversized_query(self):
        from zerogmem.mcp_server import retrieve_memories
        result = await retrieve_memories(query="x" * (MAX_QUERY_LENGTH + 1))
        assert "Error" in result
        assert "exceeds" in result


class TestSearchByEntityValidation:
    """Test search_memories_by_entity handler rejects invalid input."""

    @pytest.mark.asyncio
    async def test_empty_entity(self):
        from zerogmem.mcp_server import search_memories_by_entity
        result = await search_memories_by_entity(entity_name="")
        assert "Error" in result
        assert "entity_name" in result

    @pytest.mark.asyncio
    async def test_oversized_entity(self):
        from zerogmem.mcp_server import search_memories_by_entity
        result = await search_memories_by_entity(entity_name="x" * (MAX_ENTITY_LENGTH + 1))
        assert "Error" in result
        assert "exceeds" in result


class TestSearchByTimeValidation:
    """Test search_memories_by_time handler rejects invalid input."""

    @pytest.mark.asyncio
    async def test_empty_time(self):
        from zerogmem.mcp_server import search_memories_by_time
        result = await search_memories_by_time(time_description="")
        assert "Error" in result
        assert "time_description" in result

    @pytest.mark.asyncio
    async def test_oversized_time(self):
        from zerogmem.mcp_server import search_memories_by_time
        result = await search_memories_by_time(time_description="x" * (MAX_TIME_DESC_LENGTH + 1))
        assert "Error" in result
        assert "exceeds" in result


class TestStartSessionValidation:
    """Test start_new_session handler rejects invalid topic."""

    @pytest.mark.asyncio
    async def test_oversized_topic(self):
        from zerogmem.mcp_server import start_new_session
        result = await start_new_session(topic="x" * (MAX_TOPIC_LENGTH + 1))
        assert "Error" in result
        assert "topic" in result

    @pytest.mark.asyncio
    async def test_empty_topic_rejected(self):
        from zerogmem.mcp_server import start_new_session
        result = await start_new_session(topic="")
        assert "Error" in result
        assert "topic" in result

    @pytest.mark.asyncio
    async def test_none_topic_allowed(self):
        """topic=None should not trigger validation (it's optional)."""
        # This will try to initialize memory (and may fail without encoder),
        # but it should NOT fail with a validation error
        from zerogmem.mcp_server import start_new_session
        result = await start_new_session(topic=None)
        # Should not be a validation error — either succeeds or fails on init
        assert "topic" not in result or "Error" not in result or "must not be empty" not in result
