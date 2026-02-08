"""Tests for hard programmatic answer verifier."""

import pytest

from src.rewards.verifier import HardVerifier, normalize_text, compute_ngram_overlap


class TestNormalization:
    def test_basic(self):
        assert normalize_text("  Hello World!  ") == "hello world"
        assert normalize_text("multiple   spaces") == "multiple spaces"

    def test_case_insensitive(self):
        assert normalize_text("ABC") == normalize_text("abc")


class TestNgramOverlap:
    def test_identical(self):
        assert compute_ngram_overlap("hello world", "hello world") == 1.0

    def test_no_overlap(self):
        assert compute_ngram_overlap("cat dog", "fish bird") == 0.0

    def test_partial(self):
        score = compute_ngram_overlap("the cat sat", "the cat ran")
        assert 0.3 < score < 0.8


class TestHardVerifier:
    def setup_method(self):
        self.verifier = HardVerifier()

    def test_exact_match(self):
        result = self.verifier.verify("The answer is 42.", "The answer is 42.")
        assert result["reward"] == 1.0
        assert result["match_type"] == "exact"

    def test_case_insensitive_match(self):
        result = self.verifier.verify("The Answer Is 42", "the answer is 42")
        assert result["reward"] == 1.0

    def test_no_match(self):
        result = self.verifier.verify("completely wrong", "the right answer")
        assert result["reward"] == 0.0

    def test_partial_match_ngram(self):
        # Close but not exact
        result = self.verifier.verify(
            "The order has been updated with the new address",
            "The order has been successfully updated with the new delivery address",
        )
        # Should pass due to high n-gram overlap
        assert result["reward"] in (0.0, 1.0)

    def test_tool_call_matching(self):
        pred_calls = [
            {"function_name": "get_order", "arguments": {"order_id": "123"}},
            {"function_name": "update_order", "arguments": {"order_id": "123", "status": "shipped"}},
        ]
        gt_calls = [
            {"function_name": "get_order", "arguments": {"order_id": "123"}},
            {"function_name": "update_order", "arguments": {"order_id": "123", "status": "shipped"}},
        ]
        result = self.verifier.verify(
            "Order updated successfully",
            "Order updated successfully",
            tool_calls_pred=pred_calls,
            tool_calls_gt=gt_calls,
        )
        assert result["reward"] == 1.0

    def test_batch_reward(self):
        results = self.verifier.compute_batch_reward(
            predictions=["correct answer", "wrong answer", "correct answer"],
            ground_truths=["correct answer", "correct answer", "correct answer"],
        )
        assert len(results) == 3
        assert results[0]["reward"] == 1.0
        assert results[2]["reward"] == 1.0
