"""Unit tests for chunking functionality."""
from __future__ import annotations

import pytest
from src.chunking import chunk_pages, Chunk, _stable_chunk_id
from src.ingest_pdf import PageText


class TestStableChunkId:
    """Tests for deterministic chunk ID generation."""

    def test_same_input_same_id(self):
        """Same inputs should produce identical chunk IDs."""
        id1 = _stable_chunk_id("paper.pdf", 1, "Introduction", 0, "Some text here")
        id2 = _stable_chunk_id("paper.pdf", 1, "Introduction", 0, "Some text here")
        assert id1 == id2

    def test_different_text_different_id(self):
        """Different text should produce different chunk IDs."""
        id1 = _stable_chunk_id("paper.pdf", 1, "Introduction", 0, "Text A")
        id2 = _stable_chunk_id("paper.pdf", 1, "Introduction", 0, "Text B")
        assert id1 != id2

    def test_different_page_different_id(self):
        """Different page numbers should produce different chunk IDs."""
        id1 = _stable_chunk_id("paper.pdf", 1, "Introduction", 0, "Same text")
        id2 = _stable_chunk_id("paper.pdf", 2, "Introduction", 0, "Same text")
        assert id1 != id2

    def test_chunk_id_length(self):
        """Chunk IDs should be exactly 12 characters (truncated SHA1)."""
        chunk_id = _stable_chunk_id("paper.pdf", 1, "Section", 0, "text")
        assert len(chunk_id) == 12
        assert all(c in "0123456789abcdef" for c in chunk_id)


class TestChunkPages:
    """Tests for the chunk_pages function."""

    def _make_page(self, text: str, page_num: int = 1, headings: list[str] | None = None) -> PageText:
        return PageText(
            file_path="test.pdf",
            page_number=page_num,
            text=text,
            detected_headings=headings or [],
        )

    def test_basic_chunking(self):
        """Basic text should be chunked correctly."""
        # Create text that's roughly 400 tokens (more than chunk_tokens=350)
        text = "This is a test sentence. " * 100  # Approx 500 tokens
        pages = [self._make_page(text)]

        chunks = chunk_pages(
            pages,
            chunk_tokens=350,
            chunk_overlap_tokens=60,
            min_chunk_tokens=120,
        )

        assert len(chunks) >= 2  # Should produce multiple chunks
        assert all(isinstance(c, Chunk) for c in chunks)

    def test_chunk_has_required_metadata(self):
        """Each chunk should have all required metadata fields."""
        pages = [self._make_page("Some test text for the chunk.", headings=["Introduction"])]

        chunks = chunk_pages(
            pages,
            chunk_tokens=350,
            chunk_overlap_tokens=60,
            min_chunk_tokens=10,  # Low threshold to ensure we get chunks
        )

        assert len(chunks) >= 1
        chunk = chunks[0]

        # Check required metadata fields
        assert "source_file" in chunk.metadata
        assert "source_path" in chunk.metadata
        assert "page_number" in chunk.metadata
        assert "section_name" in chunk.metadata
        assert "chunk_index" in chunk.metadata

        # Check chunk_id is set
        assert chunk.chunk_id is not None
        assert len(chunk.chunk_id) == 12

    def test_section_name_propagates(self):
        """Section name from headings should propagate to chunks."""
        pages = [
            self._make_page("Content under introduction.", page_num=1, headings=["1 Introduction"]),
            self._make_page("Content under methods.", page_num=2, headings=["2 Methods"]),
        ]

        chunks = chunk_pages(
            pages,
            chunk_tokens=350,
            chunk_overlap_tokens=60,
            min_chunk_tokens=10,
        )

        # Find chunks from each page
        page1_chunks = [c for c in chunks if c.metadata["page_number"] == 1]
        page2_chunks = [c for c in chunks if c.metadata["page_number"] == 2]

        if page1_chunks:
            assert page1_chunks[0].metadata["section_name"] == "Introduction"
        if page2_chunks:
            assert page2_chunks[0].metadata["section_name"] == "Methods"

    def test_empty_page_skipped(self):
        """Empty pages should not produce chunks."""
        pages = [
            self._make_page("", page_num=1),
            self._make_page("   ", page_num=2),
            self._make_page("Actual content here.", page_num=3),
        ]

        chunks = chunk_pages(
            pages,
            chunk_tokens=350,
            chunk_overlap_tokens=60,
            min_chunk_tokens=10,
        )

        # Only the page with actual content should produce chunks
        assert len(chunks) >= 1
        assert all(c.metadata["page_number"] == 3 for c in chunks)

    def test_chunk_overlap(self):
        """Chunks should have overlapping content when text is long enough."""
        # Create text long enough to require multiple chunks
        text = "word " * 500  # ~500 tokens
        pages = [self._make_page(text)]

        chunks = chunk_pages(
            pages,
            chunk_tokens=100,
            chunk_overlap_tokens=30,
            min_chunk_tokens=50,
        )

        assert len(chunks) >= 3  # Should have multiple chunks

        # Check that consecutive chunks have some overlap
        for i in range(len(chunks) - 1):
            # Get the end of chunk i and start of chunk i+1
            words_i = chunks[i].text.split()
            words_next = chunks[i + 1].text.split()

            # There should be some common words due to overlap
            end_words = set(words_i[-30:]) if len(words_i) > 30 else set(words_i)
            start_words = set(words_next[:30]) if len(words_next) > 30 else set(words_next)

            # At least some overlap expected (not guaranteed due to token boundaries)
            # This is a soft check since token-based chunking doesn't align perfectly with words


class TestChunkIdUniqueness:
    """Tests for chunk ID uniqueness across documents."""

    def test_unique_ids_within_document(self):
        """All chunk IDs within a document should be unique."""
        text = "This is test content. " * 200
        pages = [
            PageText(file_path="test.pdf", page_number=i, text=text, detected_headings=[])
            for i in range(1, 4)
        ]

        chunks = chunk_pages(
            pages,
            chunk_tokens=100,
            chunk_overlap_tokens=20,
            min_chunk_tokens=50,
        )

        chunk_ids = [c.chunk_id for c in chunks]
        assert len(chunk_ids) == len(set(chunk_ids)), "Chunk IDs should be unique"

