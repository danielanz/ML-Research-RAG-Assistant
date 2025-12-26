"""Unit tests for citation extraction and formatting."""
from __future__ import annotations

import pytest
from src.citations import extract_citations, format_citation, citation_ids, CitedChunk


class TestCitationExtraction:
    """Tests for extracting citations from LLM responses."""

    def test_extract_single_citation(self):
        """Should extract a single citation."""
        text = "The Adam optimizer uses adaptive learning rates [abc123def456 p.5]."
        citations = extract_citations(text)

        assert len(citations) == 1
        assert citations[0].chunk_id == "abc123def456"
        assert citations[0].page_number == 5

    def test_extract_multiple_citations(self):
        """Should extract multiple citations from text."""
        text = (
            "First claim [abc123def456 p.5]. "
            "Second claim [def789abc012 p.10]. "
            "Third claim [111222333444 p.1]."
        )
        citations = extract_citations(text)

        assert len(citations) == 3
        assert citations[0].chunk_id == "abc123def456"
        assert citations[1].chunk_id == "def789abc012"
        assert citations[2].chunk_id == "111222333444"

    def test_extract_citation_with_large_page_number(self):
        """Should handle large page numbers."""
        text = "Reference from appendix [abcdef123456 p.999]."
        citations = extract_citations(text)

        assert len(citations) == 1
        assert citations[0].page_number == 999

    def test_no_citations(self):
        """Should return empty list when no citations present."""
        text = "This text has no citations at all."
        citations = extract_citations(text)

        assert len(citations) == 0

    def test_malformed_citations_ignored(self):
        """Should ignore malformed citation patterns."""
        text = (
            "Valid citation [abc123def456 p.5]. "
            "Invalid [abc p.5]. "  # chunk_id too short
            "Also invalid [abc123def456]. "  # missing page
            "Another invalid [abc123def456 page 5]. "  # wrong format
        )
        citations = extract_citations(text)

        assert len(citations) == 1
        assert citations[0].chunk_id == "abc123def456"

    def test_duplicate_citations(self):
        """Should extract duplicate citations (caller handles dedup)."""
        text = "First [abc123def456 p.5]. Repeated [abc123def456 p.5]."
        citations = extract_citations(text)

        assert len(citations) == 2
        assert citations[0].chunk_id == citations[1].chunk_id


class TestCitationFormatting:
    """Tests for formatting citations."""

    def test_format_basic_citation(self):
        """Should format citation correctly."""
        result = format_citation("abc123def456", 5)
        assert result == "[abc123def456 p.5]"

    def test_format_citation_large_page(self):
        """Should format citations with large page numbers."""
        result = format_citation("abc123def456", 100)
        assert result == "[abc123def456 p.100]"

    def test_format_citation_page_one(self):
        """Should format citations for page 1."""
        result = format_citation("abc123def456", 1)
        assert result == "[abc123def456 p.1]"


class TestCitationIds:
    """Tests for extracting unique citation IDs."""

    def test_unique_ids_from_citations(self):
        """Should extract unique chunk IDs."""
        citations = [
            CitedChunk(chunk_id="abc123def456", page_number=5),
            CitedChunk(chunk_id="def789abc012", page_number=10),
            CitedChunk(chunk_id="abc123def456", page_number=5),  # duplicate
        ]
        ids = citation_ids(citations)

        assert ids == {"abc123def456", "def789abc012"}

    def test_empty_citations(self):
        """Should return empty set for no citations."""
        ids = citation_ids([])
        assert ids == set()

    def test_single_citation(self):
        """Should handle single citation."""
        citations = [CitedChunk(chunk_id="abc123def456", page_number=5)]
        ids = citation_ids(citations)

        assert ids == {"abc123def456"}


class TestCitationRoundTrip:
    """Tests for format -> extract roundtrip."""

    def test_roundtrip_single(self):
        """Formatted citation should be extractable."""
        chunk_id = "abc123def456"
        page = 7

        formatted = format_citation(chunk_id, page)
        extracted = extract_citations(formatted)

        assert len(extracted) == 1
        assert extracted[0].chunk_id == chunk_id
        assert extracted[0].page_number == page

    def test_roundtrip_in_text(self):
        """Formatted citation embedded in text should be extractable."""
        chunk_id = "fedcba654321"
        page = 42

        text = f"The model achieves state-of-the-art results {format_citation(chunk_id, page)} on several benchmarks."
        extracted = extract_citations(text)

        assert len(extracted) == 1
        assert extracted[0].chunk_id == chunk_id
        assert extracted[0].page_number == page

