"""Unit tests for metadata integrity."""
from __future__ import annotations

import pytest
from src.ingest_pdf import PageText, _is_heading_candidate, extract_pdf_pages
from src.chunking import chunk_pages
from src.vectorstore import chunks_to_documents


class TestHeadingDetection:
    """Tests for heading detection heuristics."""

    DEFAULT_CFG = {
        "max_len": 80,
        "min_len": 3,
        "max_words": 12,
        "require_no_period": True,
    }

    def test_numbered_heading(self):
        """Numbered headings like '1.2 Methods' should be detected."""
        assert _is_heading_candidate("1 Introduction", self.DEFAULT_CFG) is True
        assert _is_heading_candidate("2.1 Related Work", self.DEFAULT_CFG) is True
        assert _is_heading_candidate("3.2.1 Subsection", self.DEFAULT_CFG) is True

    def test_allcaps_heading(self):
        """ALL CAPS headings should be detected."""
        assert _is_heading_candidate("INTRODUCTION", self.DEFAULT_CFG) is True
        assert _is_heading_candidate("RELATED WORK", self.DEFAULT_CFG) is True
        assert _is_heading_candidate("EXPERIMENTAL RESULTS", self.DEFAULT_CFG) is True

    def test_title_case_heading(self):
        """Title Case headings should be detected."""
        assert _is_heading_candidate("Introduction", self.DEFAULT_CFG) is True
        assert _is_heading_candidate("Related Work", self.DEFAULT_CFG) is True

    def test_sentence_not_heading(self):
        """Regular sentences should not be detected as headings."""
        # Ends with period
        assert _is_heading_candidate("This is a sentence.", self.DEFAULT_CFG) is False

    def test_too_long_not_heading(self):
        """Lines that are too long should not be detected as headings."""
        long_line = "A" * 100
        assert _is_heading_candidate(long_line, self.DEFAULT_CFG) is False

    def test_too_short_not_heading(self):
        """Lines that are too short should not be detected as headings."""
        assert _is_heading_candidate("AB", self.DEFAULT_CFG) is False

    def test_too_many_words_not_heading(self):
        """Lines with too many words should not be detected as headings."""
        many_words = "Word " * 15
        assert _is_heading_candidate(many_words.strip(), self.DEFAULT_CFG) is False


class TestMetadataIntegrity:
    """Tests for metadata preservation through the pipeline."""

    def test_page_number_preserved(self):
        """Page numbers should be preserved in chunk metadata."""
        pages = [
            PageText(file_path="test.pdf", page_number=5, text="Content on page 5", detected_headings=[]),
            PageText(file_path="test.pdf", page_number=6, text="Content on page 6", detected_headings=[]),
        ]

        chunks = chunk_pages(pages, chunk_tokens=350, chunk_overlap_tokens=60, min_chunk_tokens=10)

        page_numbers = {c.metadata["page_number"] for c in chunks}
        assert 5 in page_numbers or 6 in page_numbers

    def test_source_file_preserved(self):
        """Source file name should be extracted and preserved."""
        pages = [
            PageText(
                file_path="/path/to/MyPaper.pdf",
                page_number=1,
                text="Some content",
                detected_headings=[],
            )
        ]

        chunks = chunk_pages(pages, chunk_tokens=350, chunk_overlap_tokens=60, min_chunk_tokens=10)

        assert len(chunks) >= 1
        assert chunks[0].metadata["source_file"] == "MyPaper.pdf"
        assert chunks[0].metadata["source_path"] == "/path/to/MyPaper.pdf"

    def test_section_name_preserved(self):
        """Section names should be preserved in metadata."""
        pages = [
            PageText(
                file_path="test.pdf",
                page_number=1,
                text="Introduction content",
                detected_headings=["1 Introduction"],
            )
        ]

        chunks = chunk_pages(pages, chunk_tokens=350, chunk_overlap_tokens=60, min_chunk_tokens=10)

        assert len(chunks) >= 1
        assert chunks[0].metadata["section_name"] == "Introduction"

    def test_chunks_to_documents_preserves_metadata(self):
        """Converting chunks to documents should preserve all metadata."""
        pages = [
            PageText(
                file_path="test.pdf",
                page_number=3,
                text="Test content for document",
                detected_headings=["Methods"],
            )
        ]

        chunks = chunk_pages(pages, chunk_tokens=350, chunk_overlap_tokens=60, min_chunk_tokens=10)
        docs = chunks_to_documents(chunks)

        assert len(docs) >= 1
        doc = docs[0]

        # Check all metadata is preserved
        assert doc.metadata["chunk_id"] == chunks[0].chunk_id
        assert doc.metadata["source_file"] == "test.pdf"
        assert doc.metadata["page_number"] == 3
        assert doc.metadata["section_name"] == "Methods"

        # Check content is preserved
        assert doc.page_content == chunks[0].text


class TestMetadataTypes:
    """Tests for correct metadata types."""

    def test_page_number_is_integer(self):
        """Page numbers should be integers."""
        pages = [
            PageText(file_path="test.pdf", page_number=1, text="Content", detected_headings=[])
        ]
        chunks = chunk_pages(pages, chunk_tokens=350, chunk_overlap_tokens=60, min_chunk_tokens=10)

        for chunk in chunks:
            assert isinstance(chunk.metadata["page_number"], int)

    def test_chunk_index_is_integer(self):
        """Chunk indices should be integers."""
        pages = [
            PageText(file_path="test.pdf", page_number=1, text="Content " * 100, detected_headings=[])
        ]
        chunks = chunk_pages(pages, chunk_tokens=50, chunk_overlap_tokens=10, min_chunk_tokens=20)

        for chunk in chunks:
            assert isinstance(chunk.metadata["chunk_index"], int)

    def test_section_name_is_string(self):
        """Section names should be strings."""
        pages = [
            PageText(file_path="test.pdf", page_number=1, text="Content", detected_headings=["Intro"])
        ]
        chunks = chunk_pages(pages, chunk_tokens=350, chunk_overlap_tokens=60, min_chunk_tokens=10)

        for chunk in chunks:
            assert isinstance(chunk.metadata["section_name"], str)

