"""
Hard programmatic answer verifier.

Strictly verifies model outputs against ground truth.
Multiple verification strategies:
  1. Exact match (after normalization)
  2. Key information extraction + match
  3. Tool call sequence verification
  4. Semantic similarity fallback (BLEU-based)
"""

import json
import logging
import re
import string
from typing import Dict, List, Any, Optional, Tuple
from collections import Counter

logger = logging.getLogger(__name__)


def normalize_text(text: str) -> str:
    """Normalize text for comparison."""
    text = text.lower().strip()
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text)
    # Remove punctuation that doesn't carry meaning
    text = text.translate(str.maketrans('', '', string.punctuation.replace('.', '').replace(',', '')))
    return text


def extract_key_values(text: str) -> Dict[str, str]:
    """Extract key-value pairs from structured text."""
    kvs = {}
    # Pattern: "key: value" or "key = value"
    for pattern in [r'(\w[\w\s]*?):\s*(.+?)(?:\n|$)', r'(\w[\w\s]*?)=\s*(.+?)(?:\n|$)']:
        for match in re.finditer(pattern, text):
            key = match.group(1).strip().lower()
            value = match.group(2).strip()
            kvs[key] = value

    # Try JSON parsing
    try:
        data = json.loads(text)
        if isinstance(data, dict):
            for k, v in data.items():
                kvs[k.lower()] = str(v)
    except (json.JSONDecodeError, TypeError):
        pass

    return kvs


def compute_ngram_overlap(pred: str, ref: str, n: int = 1) -> float:
    """Compute n-gram overlap (similar to BLEU precision)."""
    pred_tokens = normalize_text(pred).split()
    ref_tokens = normalize_text(ref).split()

    if not pred_tokens or not ref_tokens:
        return 0.0

    pred_ngrams = Counter(
        tuple(pred_tokens[i:i+n]) for i in range(len(pred_tokens) - n + 1)
    )
    ref_ngrams = Counter(
        tuple(ref_tokens[i:i+n]) for i in range(len(ref_tokens) - n + 1)
    )

    overlap = sum((pred_ngrams & ref_ngrams).values())
    total = sum(pred_ngrams.values())
    return overlap / total if total > 0 else 0.0


class HardVerifier:
    """
    Hard programmatic verifier for agentic task answers.

    Returns binary reward: 1.0 (correct) or 0.0 (incorrect).
    No partial credit — this is intentional for GRPO training.

    Verification stages (in order):
      1. Exact normalized match -> 1.0
      2. Key information overlap (>= threshold) -> 1.0
      3. N-gram overlap (>= threshold) -> 1.0
      4. Tool call sequence match -> 1.0
      5. Otherwise -> 0.0
    """

    def __init__(
        self,
        exact_match_weight: float = 1.0,
        key_overlap_threshold: float = 0.8,
        ngram_threshold: float = 0.7,
        tool_call_weight: float = 0.3,
        strict_mode: bool = True,
    ):
        self.exact_match_weight = exact_match_weight
        self.key_overlap_threshold = key_overlap_threshold
        self.ngram_threshold = ngram_threshold
        self.tool_call_weight = tool_call_weight
        self.strict_mode = strict_mode

    def verify(
        self,
        prediction: str,
        ground_truth: str,
        tool_calls_pred: Optional[List[Dict]] = None,
        tool_calls_gt: Optional[List[Dict]] = None,
    ) -> Dict[str, Any]:
        """
        Verify a prediction against ground truth.

        Returns:
            dict with:
              - 'reward': float, 0.0 or 1.0
              - 'match_type': str, how the match was determined
              - 'details': dict with sub-scores
        """
        details = {}

        # Stage 1: Exact match
        exact = self._exact_match(prediction, ground_truth)
        details["exact_match"] = exact
        if exact:
            return {"reward": 1.0, "match_type": "exact", "details": details}

        # Stage 2: Key information overlap
        key_score = self._key_overlap(prediction, ground_truth)
        details["key_overlap"] = key_score
        if key_score >= self.key_overlap_threshold:
            return {"reward": 1.0, "match_type": "key_overlap", "details": details}

        # Stage 3: N-gram overlap
        unigram = compute_ngram_overlap(prediction, ground_truth, n=1)
        bigram = compute_ngram_overlap(prediction, ground_truth, n=2)
        ngram_score = 0.6 * unigram + 0.4 * bigram
        details["ngram_score"] = ngram_score
        details["unigram_overlap"] = unigram
        details["bigram_overlap"] = bigram

        if ngram_score >= self.ngram_threshold:
            return {"reward": 1.0, "match_type": "ngram", "details": details}

        # Stage 4: Tool call sequence verification (if available)
        if tool_calls_pred and tool_calls_gt:
            tc_score = self._tool_call_match(tool_calls_pred, tool_calls_gt)
            details["tool_call_match"] = tc_score
            # Combined score with tool calls
            combined = ngram_score * (1 - self.tool_call_weight) + tc_score * self.tool_call_weight
            details["combined_score"] = combined
            if combined >= self.ngram_threshold:
                return {"reward": 1.0, "match_type": "combined", "details": details}

        # Stage 5: Failure
        return {"reward": 0.0, "match_type": "none", "details": details}

    def _exact_match(self, pred: str, gt: str) -> bool:
        """Normalized exact match."""
        return normalize_text(pred) == normalize_text(gt)

    def _key_overlap(self, pred: str, gt: str) -> float:
        """Compare key information extracted from both texts."""
        pred_kvs = extract_key_values(pred)
        gt_kvs = extract_key_values(gt)

        if not gt_kvs:
            # No structured data — fall back to word overlap
            pred_words = set(normalize_text(pred).split())
            gt_words = set(normalize_text(gt).split())
            if not gt_words:
                return 0.0
            return len(pred_words & gt_words) / len(gt_words)

        matches = 0
        for key, gt_val in gt_kvs.items():
            if key in pred_kvs:
                if normalize_text(pred_kvs[key]) == normalize_text(gt_val):
                    matches += 1
                elif normalize_text(gt_val) in normalize_text(pred_kvs[key]):
                    matches += 0.5

        return matches / len(gt_kvs) if gt_kvs else 0.0

    def _tool_call_match(
        self,
        pred_calls: List[Dict],
        gt_calls: List[Dict],
    ) -> float:
        """Compare tool call sequences."""
        if not gt_calls:
            return 1.0 if not pred_calls else 0.0

        # Compare function names in order
        pred_names = [c.get("function_name", c.get("function", "")) for c in pred_calls]
        gt_names = [c.get("function_name", c.get("function", "")) for c in gt_calls]

        # Longest common subsequence ratio
        lcs_len = self._lcs_length(pred_names, gt_names)
        name_score = lcs_len / len(gt_names)

        # Argument overlap for matching calls
        arg_scores = []
        for gt_call in gt_calls:
            gt_name = gt_call.get("function_name", gt_call.get("function", ""))
            gt_args = gt_call.get("arguments", {})
            # Find best matching pred call
            best = 0.0
            for pred_call in pred_calls:
                pred_name = pred_call.get("function_name", pred_call.get("function", ""))
                if pred_name == gt_name:
                    pred_args = pred_call.get("arguments", {})
                    arg_overlap = self._dict_overlap(pred_args, gt_args)
                    best = max(best, arg_overlap)
            arg_scores.append(best)

        arg_score = sum(arg_scores) / len(arg_scores) if arg_scores else 0.0

        return 0.6 * name_score + 0.4 * arg_score

    def _lcs_length(self, a: List, b: List) -> int:
        """Longest common subsequence length."""
        m, n = len(a), len(b)
        dp = [[0] * (n + 1) for _ in range(m + 1)]
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if a[i-1] == b[j-1]:
                    dp[i][j] = dp[i-1][j-1] + 1
                else:
                    dp[i][j] = max(dp[i-1][j], dp[i][j-1])
        return dp[m][n]

    def _dict_overlap(self, pred: Dict, gt: Dict) -> float:
        """Compute overlap between two dicts."""
        if not gt:
            return 1.0 if not pred else 0.0
        matches = 0
        for key, gt_val in gt.items():
            if key in pred:
                pred_val = pred[key]
                if str(pred_val).strip().lower() == str(gt_val).strip().lower():
                    matches += 1
                elif str(gt_val).strip().lower() in str(pred_val).strip().lower():
                    matches += 0.5
        return matches / len(gt)

    def compute_batch_reward(
        self,
        predictions: List[str],
        ground_truths: List[str],
        tool_calls_preds: Optional[List[List[Dict]]] = None,
        tool_calls_gts: Optional[List[List[Dict]]] = None,
    ) -> List[Dict[str, Any]]:
        """Verify a batch of predictions."""
        results = []
        for i, (pred, gt) in enumerate(zip(predictions, ground_truths)):
            tc_pred = tool_calls_preds[i] if tool_calls_preds else None
            tc_gt = tool_calls_gts[i] if tool_calls_gts else None
            result = self.verify(pred, gt, tc_pred, tc_gt)
            results.append(result)
        return results
