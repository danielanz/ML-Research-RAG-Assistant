"""Unit tests for query router."""
from __future__ import annotations

import pytest
from src.router import route_query, Route


class TestRouteQuery:
    """Tests for query routing logic."""

    def test_qa_mode_default(self):
        """Generic questions should route to QA mode."""
        assert route_query("What is machine learning?").mode == "qa"
        assert route_query("Explain the algorithm").mode == "qa"
        assert route_query("How does it work?").mode == "qa"

    def test_compare_mode(self):
        """Questions with comparison keywords should route to compare mode."""
        assert route_query("Compare Adam vs SGD").mode == "compare"
        assert route_query("What are the differences between method A and B?").mode == "compare"
        assert route_query("Similarities between the approaches").mode == "compare"
        assert route_query("Paper A versus Paper B").mode == "compare"

    def test_method_card_mode(self):
        """Questions about methods/architecture should route to method_card mode."""
        assert route_query("Summarize the method card").mode == "method_card"
        assert route_query("Describe the architecture").mode == "method_card"
        assert route_query("What is the training objective?").mode == "method_card"
        assert route_query("Explain the pipeline").mode == "method_card"

    def test_claim_verify_mode(self):
        """Questions verifying claims should route to claim_verify mode."""
        assert route_query("Is it true that Adam converges faster?").mode == "claim_verify"
        assert route_query("Verify: the model uses attention").mode == "claim_verify"
        assert route_query("Does the paper claim X?").mode == "claim_verify"
        assert route_query("Evidence for the hypothesis").mode == "claim_verify"
        assert route_query("Can you refute this claim?").mode == "claim_verify"

    def test_priority_verify_over_compare(self):
        """Claim verification should take priority over compare."""
        # "verify" keyword present - should be claim_verify even with "compare"
        result = route_query("Verify the comparison between A and B")
        assert result.mode == "claim_verify"

    def test_case_insensitive(self):
        """Routing should be case-insensitive."""
        assert route_query("COMPARE these methods").mode == "compare"
        assert route_query("Is It True that...").mode == "claim_verify"
        assert route_query("SUMMARIZE METHOD").mode == "method_card"

    def test_route_returns_route_object(self):
        """route_query should return a Route dataclass."""
        result = route_query("Any question")
        assert isinstance(result, Route)
        assert hasattr(result, "mode")

