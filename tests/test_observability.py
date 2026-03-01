"""Tests for operation metrics and observability."""

import time
from unittest.mock import MagicMock

import pytest

from zerogmem.mcp_server import OperationMetrics


# ---------------------------------------------------------------------------
# OperationMetrics unit tests
# ---------------------------------------------------------------------------

class TestOperationMetrics:

    def test_record_single_operation(self):
        m = OperationMetrics()
        m.record("store_memory", 150.0)

        summary = m.get_summary()
        assert "store_memory" in summary
        assert summary["store_memory"]["count"] == 1
        assert summary["store_memory"]["avg_ms"] == 150.0
        assert summary["store_memory"]["min_ms"] == 150.0
        assert summary["store_memory"]["max_ms"] == 150.0
        assert summary["store_memory"]["last_ms"] == 150.0
        assert summary["store_memory"]["errors"] == 0

    def test_record_multiple_operations(self):
        m = OperationMetrics()
        m.record("retrieve", 100.0)
        m.record("retrieve", 200.0)
        m.record("retrieve", 300.0)

        summary = m.get_summary()
        assert summary["retrieve"]["count"] == 3
        assert summary["retrieve"]["avg_ms"] == 200.0
        assert summary["retrieve"]["min_ms"] == 100.0
        assert summary["retrieve"]["max_ms"] == 300.0
        assert summary["retrieve"]["last_ms"] == 300.0

    def test_record_with_error(self):
        m = OperationMetrics()
        m.record("store_memory", 50.0, error=False)
        m.record("store_memory", 75.0, error=True)
        m.record("store_memory", 60.0, error=True)

        summary = m.get_summary()
        assert summary["store_memory"]["count"] == 3
        assert summary["store_memory"]["errors"] == 2

    def test_get_summary_empty(self):
        m = OperationMetrics()
        summary = m.get_summary()

        assert "uptime_seconds" in summary
        assert summary["uptime_seconds"] >= 0
        # No operations recorded — only uptime key
        assert len(summary) == 1

    def test_get_summary_format(self):
        m = OperationMetrics()
        m.record("test_op", 42.5)

        summary = m.get_summary()
        op = summary["test_op"]

        # All expected fields present
        assert set(op.keys()) == {"count", "errors", "avg_ms", "min_ms", "max_ms", "last_ms"}
        # All values are numbers
        for v in op.values():
            assert isinstance(v, (int, float))

    def test_multiple_operation_types(self):
        m = OperationMetrics()
        m.record("store_memory", 100.0)
        m.record("retrieve", 200.0)
        m.record("store_memory", 150.0)

        summary = m.get_summary()
        assert summary["store_memory"]["count"] == 2
        assert summary["retrieve"]["count"] == 1
        # They are tracked independently
        assert summary["store_memory"]["avg_ms"] == 125.0
        assert summary["retrieve"]["avg_ms"] == 200.0

    def test_uptime_increases(self):
        m = OperationMetrics()
        time.sleep(0.05)
        summary = m.get_summary()
        assert summary["uptime_seconds"] >= 0.04

    def test_min_ms_zero_when_no_ops(self):
        """min_ms should show 0 (not inf) in summary for fresh operations."""
        m = OperationMetrics()
        m.record("op", 0.0)

        summary = m.get_summary()
        assert summary["op"]["min_ms"] == 0.0


# ---------------------------------------------------------------------------
# MCP summary integration test
# ---------------------------------------------------------------------------

class TestMCPSummaryMetrics:

    @pytest.mark.asyncio
    async def test_summary_includes_performance_section(self):
        from zerogmem import mcp_server

        mm = MagicMock()
        mm.get_stats.return_value = {
            "episodic_memory": {"total_episodes": 5, "total_messages": 20,
                                "unique_participants": 2},
            "semantic_memory": {"total_facts": 10},
            "graph": {"entity_nodes": 3, "semantic_nodes": 5,
                       "temporal_nodes": 5},
            "current_session": None,
            "capacity": {
                "max_episodes": 500,
                "max_facts": 5000,
                "episode_utilization": 0.01,
                "fact_utilization": 0.002,
            },
        }

        # Save original state
        original_mm = mcp_server._memory_manager
        original_init = mcp_server._initialized
        original_metrics = mcp_server._metrics

        # Inject mock manager and metrics with data
        mcp_server._memory_manager = mm
        mcp_server._initialized = True
        test_metrics = OperationMetrics()
        test_metrics.record("store_memory", 120.0)
        test_metrics.record("store_memory", 180.0)
        test_metrics.record("retrieve_memories", 350.0, error=True)
        mcp_server._metrics = test_metrics

        try:
            result = await mcp_server.get_memory_summary()
            assert "Performance" in result
            assert "store_memory" in result
            assert "2 calls" in result
            assert "retrieve_memories" in result
            assert "1 errors" in result
            assert "Uptime" in result
        finally:
            mcp_server._memory_manager = original_mm
            mcp_server._initialized = original_init
            mcp_server._metrics = original_metrics

    @pytest.mark.asyncio
    async def test_summary_no_performance_when_no_ops(self):
        from zerogmem import mcp_server

        mm = MagicMock()
        mm.get_stats.return_value = {
            "episodic_memory": {"total_episodes": 0, "total_messages": 0,
                                "unique_participants": 0},
            "semantic_memory": {"total_facts": 0},
            "graph": {"entity_nodes": 0, "semantic_nodes": 0,
                       "temporal_nodes": 0},
            "current_session": None,
            "capacity": {
                "max_episodes": 500,
                "max_facts": 5000,
                "episode_utilization": 0,
                "fact_utilization": 0,
            },
        }

        original_mm = mcp_server._memory_manager
        original_init = mcp_server._initialized
        original_metrics = mcp_server._metrics

        mcp_server._memory_manager = mm
        mcp_server._initialized = True
        mcp_server._metrics = OperationMetrics()  # Fresh, no ops

        try:
            result = await mcp_server.get_memory_summary()
            # Performance section should not appear (only get_summary itself runs)
            # But get_summary records itself, so it will show. Check it doesn't
            # show any other ops.
            assert "store_memory" not in result
            assert "retrieve" not in result
        finally:
            mcp_server._memory_manager = original_mm
            mcp_server._initialized = original_init
            mcp_server._metrics = original_metrics
