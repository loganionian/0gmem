"""Tests for memory export/import (ZIP archive round-trip)."""

import json
import zipfile
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest

from zerogmem import MemoryManager, MemoryConfig
from zerogmem.persistence import (
    EMBEDDINGS_FILENAME,
    EXPORT_FORMAT_VERSION,
    METADATA_FILENAME,
    STATE_FILENAME,
    export_memory_archive,
    import_memory_archive,
    save_memory_state,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _mock_embed(text: str) -> np.ndarray:
    np.random.seed(hash(text) % (2**32))
    return np.random.randn(1536).astype(np.float32)


def _populated_manager() -> MemoryManager:
    """Create a MemoryManager with test data."""
    mm = MemoryManager(config=MemoryConfig(max_episodes=100, max_facts=100))
    mm.set_embedding_function(_mock_embed)
    mm.start_session()
    mm.add_message("Alice", "I love hiking in the mountains.")
    mm.add_message("Bob", "Which mountains have you visited?")
    mm.add_message("Alice", "I went to the Alps last summer.")
    mm.end_session()
    mm.add_fact("Alice", "likes", "hiking", category="hobby")
    return mm


# ---------------------------------------------------------------------------
# persistence.export_memory_archive tests
# ---------------------------------------------------------------------------

class TestExportMemoryArchive:

    def test_creates_valid_zip(self, tmp_path):
        mm = _populated_manager()
        archive = tmp_path / "export.zip"
        data_dir = tmp_path / "data"

        export_memory_archive(mm, archive, data_dir)

        assert archive.exists()
        assert zipfile.is_zipfile(archive)
        with zipfile.ZipFile(archive) as zf:
            names = zf.namelist()
            assert STATE_FILENAME in names
            assert METADATA_FILENAME in names

    def test_metadata_format(self, tmp_path):
        mm = _populated_manager()
        archive = tmp_path / "export.zip"
        data_dir = tmp_path / "data"

        summary = export_memory_archive(mm, archive, data_dir)

        with zipfile.ZipFile(archive) as zf:
            meta = json.loads(zf.read(METADATA_FILENAME))

        assert meta["format_version"] == EXPORT_FORMAT_VERSION
        assert "exported_at" in meta
        assert "zerogmem_version" in meta
        assert "counts" in meta
        counts = meta["counts"]
        assert counts["memories"] == summary["memories"]
        assert counts["episodes"] == summary["episodes"]
        assert counts["facts"] == summary["facts"]
        assert counts["entities"] == summary["entities"]
        assert counts["embeddings"] == summary["embeddings"]

    def test_includes_embeddings_npz(self, tmp_path):
        mm = _populated_manager()
        archive = tmp_path / "export.zip"
        data_dir = tmp_path / "data"

        export_memory_archive(mm, archive, data_dir)

        with zipfile.ZipFile(archive) as zf:
            assert EMBEDDINGS_FILENAME in zf.namelist()

    def test_summary_counts(self, tmp_path):
        mm = _populated_manager()
        archive = tmp_path / "export.zip"
        data_dir = tmp_path / "data"

        summary = export_memory_archive(mm, archive, data_dir)

        assert summary["archive_path"] == str(archive)
        assert summary["memories"] > 0
        assert summary["episodes"] > 0
        assert summary["facts"] > 0


# ---------------------------------------------------------------------------
# persistence.import_memory_archive tests
# ---------------------------------------------------------------------------

class TestImportMemoryArchive:

    def test_round_trip(self, tmp_path):
        """Export then import restores same state."""
        mm = _populated_manager()
        archive = tmp_path / "export.zip"
        data_dir = tmp_path / "data"

        export_memory_archive(mm, archive, data_dir)
        restored = import_memory_archive(archive, _mock_embed)

        assert restored is not None
        assert len(restored.graph.memories) == len(mm.graph.memories)
        assert len(restored.episodic_memory.episodes) == len(mm.episodic_memory.episodes)
        assert len(restored.semantic_memory.facts) == len(mm.semantic_memory.facts)

    def test_preserves_embeddings(self, tmp_path):
        """Embeddings survive the round-trip."""
        mm = _populated_manager()
        archive = tmp_path / "export.zip"
        data_dir = tmp_path / "data"

        export_memory_archive(mm, archive, data_dir)
        restored = import_memory_archive(archive, _mock_embed)

        assert restored is not None
        # Check that at least some memories have embeddings
        has_embedding = any(
            m.embedding is not None for m in restored.graph.memories.values()
        )
        assert has_embedding

    def test_invalid_zip(self, tmp_path):
        """Graceful error on a non-ZIP file."""
        bad_file = tmp_path / "not_a_zip.zip"
        bad_file.write_text("this is not a zip")

        result = import_memory_archive(bad_file, _mock_embed)
        assert result is None

    def test_missing_state_file(self, tmp_path):
        """Graceful error when ZIP lacks memory_state.json."""
        archive = tmp_path / "incomplete.zip"
        with zipfile.ZipFile(archive, "w") as zf:
            zf.writestr("other_file.txt", "hello")

        result = import_memory_archive(archive, _mock_embed)
        assert result is None

    def test_future_version_rejected(self, tmp_path):
        """Rejects archives with format_version > current."""
        mm = _populated_manager()
        archive = tmp_path / "export.zip"
        data_dir = tmp_path / "data"

        # Create a valid export first
        export_memory_archive(mm, archive, data_dir)

        # Re-create ZIP with bumped format_version
        patched = tmp_path / "future.zip"
        with zipfile.ZipFile(archive) as src, zipfile.ZipFile(patched, "w") as dst:
            for name in src.namelist():
                if name == METADATA_FILENAME:
                    meta = json.loads(src.read(name))
                    meta["format_version"] = EXPORT_FORMAT_VERSION + 99
                    dst.writestr(name, json.dumps(meta))
                else:
                    dst.writestr(name, src.read(name))

        result = import_memory_archive(patched, _mock_embed)
        assert result is None

    def test_missing_metadata_still_works(self, tmp_path):
        """Import works even if export_metadata.json is missing."""
        mm = _populated_manager()
        data_dir = tmp_path / "data"
        save_memory_state(mm, data_dir)

        # Create ZIP without metadata
        archive = tmp_path / "no_meta.zip"
        with zipfile.ZipFile(archive, "w") as zf:
            zf.write(data_dir / STATE_FILENAME, STATE_FILENAME)
            emb_path = data_dir / EMBEDDINGS_FILENAME
            if emb_path.exists():
                zf.write(emb_path, EMBEDDINGS_FILENAME)

        restored = import_memory_archive(archive, _mock_embed)
        assert restored is not None
        assert len(restored.graph.memories) == len(mm.graph.memories)

    def test_nonexistent_archive(self, tmp_path):
        """Returns None for a path that doesn't exist."""
        result = import_memory_archive(tmp_path / "ghost.zip", _mock_embed)
        assert result is None


# ---------------------------------------------------------------------------
# MCP tool integration tests
# ---------------------------------------------------------------------------

class TestMCPExportImportTools:

    @pytest.mark.asyncio
    async def test_export_tool_returns_path(self, tmp_path):
        from zerogmem import mcp_server
        from zerogmem.mcp_server import OperationMetrics

        mm = _populated_manager()

        original_mm = mcp_server._memory_manager
        original_init = mcp_server._initialized
        original_metrics = mcp_server._metrics
        original_dir = mcp_server._memory_dir

        mcp_server._memory_manager = mm
        mcp_server._initialized = True
        mcp_server._metrics = OperationMetrics()
        mcp_server._memory_dir = tmp_path / "data"

        try:
            archive_path = str(tmp_path / "test_export.zip")
            result = await mcp_server.export_memory(output_path=archive_path)
            assert "exported successfully" in result
            assert archive_path in result
            assert Path(archive_path).exists()
        finally:
            mcp_server._memory_manager = original_mm
            mcp_server._initialized = original_init
            mcp_server._metrics = original_metrics
            mcp_server._memory_dir = original_dir

    @pytest.mark.asyncio
    async def test_import_tool_restores_state(self, tmp_path):
        from zerogmem import mcp_server, Encoder, Retriever
        from zerogmem.mcp_server import OperationMetrics
        from zerogmem.encoder.encoder import EncoderConfig
        from zerogmem.retriever.retriever import RetrieverConfig
        from zerogmem.persistence import export_memory_archive

        mm = _populated_manager()
        archive = tmp_path / "export.zip"
        data_dir = tmp_path / "data"
        export_memory_archive(mm, archive, data_dir)

        # Set up a fresh empty state
        empty_mm = MemoryManager()
        empty_mm.set_embedding_function(_mock_embed)
        encoder = Encoder(config=EncoderConfig(), embedding_fn=_mock_embed)
        retriever = Retriever(empty_mm, embedding_fn=_mock_embed)

        original_mm = mcp_server._memory_manager
        original_enc = mcp_server._encoder
        original_ret = mcp_server._retriever
        original_init = mcp_server._initialized
        original_metrics = mcp_server._metrics
        original_dir = mcp_server._memory_dir

        mcp_server._memory_manager = empty_mm
        mcp_server._encoder = encoder
        mcp_server._retriever = retriever
        mcp_server._initialized = True
        mcp_server._metrics = OperationMetrics()
        mcp_server._memory_dir = tmp_path / "import_data"

        try:
            result = await mcp_server.import_memory(archive_path=str(archive))
            assert "imported successfully" in result
            assert "Memories:" in result
            # After import, the manager should have data
            assert len(mcp_server._memory_manager.graph.memories) > 0
        finally:
            mcp_server._memory_manager = original_mm
            mcp_server._encoder = original_enc
            mcp_server._retriever = original_ret
            mcp_server._initialized = original_init
            mcp_server._metrics = original_metrics
            mcp_server._memory_dir = original_dir

    @pytest.mark.asyncio
    async def test_import_merge_not_supported(self):
        from zerogmem import mcp_server
        from zerogmem.mcp_server import OperationMetrics

        original_init = mcp_server._initialized
        original_metrics = mcp_server._metrics
        mcp_server._initialized = True
        mcp_server._metrics = OperationMetrics()

        try:
            result = await mcp_server.import_memory(
                archive_path="/tmp/any.zip", merge=True
            )
            assert "not yet supported" in result
        finally:
            mcp_server._initialized = original_init
            mcp_server._metrics = original_metrics

    @pytest.mark.asyncio
    async def test_export_invalid_path(self):
        from zerogmem import mcp_server
        from zerogmem.mcp_server import OperationMetrics

        original_init = mcp_server._initialized
        original_metrics = mcp_server._metrics
        mcp_server._initialized = True
        mcp_server._metrics = OperationMetrics()

        try:
            result = await mcp_server.export_memory(output_path="/tmp/bad.tar.gz")
            assert "Error" in result
            assert ".zip" in result
        finally:
            mcp_server._initialized = original_init
            mcp_server._metrics = original_metrics

    @pytest.mark.asyncio
    async def test_import_invalid_path(self):
        from zerogmem import mcp_server
        from zerogmem.mcp_server import OperationMetrics

        original_init = mcp_server._initialized
        original_metrics = mcp_server._metrics
        mcp_server._initialized = True
        mcp_server._metrics = OperationMetrics()

        try:
            result = await mcp_server.import_memory(archive_path="/tmp/bad.tar.gz")
            assert "Error" in result
            assert ".zip" in result
        finally:
            mcp_server._initialized = original_init
            mcp_server._metrics = original_metrics
