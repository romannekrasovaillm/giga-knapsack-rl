"""
SFT Dataset for tool_calling subset warmup.

Converts Nemotron tool_calling records into plain-text training data
for supervised fine-tuning on function calling.

Works with base models (no chat_template required) — formats conversations
using explicit role markers.
"""

import json
import logging
from typing import List, Dict, Any, Optional, Tuple

import torch
from torch.utils.data import Dataset

logger = logging.getLogger(__name__)

# Role markers for base model (no chat template)
ASSISTANT_MARKER = "<|assistant|>\n"
ROLE_MARKERS = {
    "system": "<|system|>\n{content}\n",
    "user": "<|user|>\n{content}\n",
    "assistant": ASSISTANT_MARKER + "{content}\n",
}
END_MARKER = "<|end|>"

# System prompt template
SYSTEM_TEMPLATE = (
    "You are a helpful assistant with access to tools. "
    "When you need to use a tool, output a tool call in the following JSON format:\n"
    '<tool_call>\n{{"name": "<function_name>", "arguments": {{...}}}}\n</tool_call>\n\n'
    "Available tools:\n{tools_json}"
)

TOOL_RESPONSE_TEMPLATE = "<tool_response>\n{content}\n</tool_response>"


def format_chat_plain(chat: List[Dict[str, str]]) -> str:
    """Format chat messages into plain text with role markers."""
    parts = []
    for msg in chat:
        role = msg["role"]
        content = msg["content"]
        marker = ROLE_MARKERS.get(role, ROLE_MARKERS["user"])
        parts.append(marker.format(content=content))
    parts.append(END_MARKER)
    return "".join(parts)


def find_assistant_spans(
    text: str,
    tokenizer,
) -> List[Tuple[int, int]]:
    """
    Find (start, end) token positions for assistant CONTENT only.

    Excludes the <|assistant|>\\n marker itself — loss is computed
    only on what the model should learn to generate.

    Uses offset_mapping for precise char→token alignment, with
    fallback to string-search based approach.
    """
    # Find all assistant content regions by character positions
    char_spans = []
    search_from = 0
    while True:
        marker_pos = text.find(ASSISTANT_MARKER, search_from)
        if marker_pos == -1:
            break
        content_start = marker_pos + len(ASSISTANT_MARKER)

        # Content ends at the next role marker or END_MARKER
        content_end = len(text)
        for next_marker in ["<|system|>\n", "<|user|>\n", "<|assistant|>\n", END_MARKER]:
            pos = text.find(next_marker, content_start)
            if pos != -1 and pos < content_end:
                content_end = pos

        if content_end > content_start:
            char_spans.append((content_start, content_end))
        search_from = content_end

    if not char_spans:
        return []

    # Try offset_mapping for precise char→token mapping
    try:
        encodings = tokenizer(
            text,
            return_offsets_mapping=True,
            add_special_tokens=True,
            truncation=False,
        )
        offset_mapping = encodings["offset_mapping"]

        token_spans = []
        for char_start, char_end in char_spans:
            tok_start = None
            tok_end = None
            for tok_idx, (cs, ce) in enumerate(offset_mapping):
                if cs == ce == 0 and tok_idx > 0:
                    continue  # skip special tokens
                if tok_start is None and ce > char_start:
                    tok_start = tok_idx
                if cs < char_end:
                    tok_end = tok_idx + 1
            if tok_start is not None and tok_end is not None:
                token_spans.append((tok_start, tok_end))

        return token_spans

    except Exception:
        # Fallback: encode prefix to find token boundaries
        # Less precise due to BPE merging, but works for all tokenizers
        token_spans = []
        for char_start, char_end in char_spans:
            prefix_ids = tokenizer.encode(text[:char_start], add_special_tokens=True)
            full_ids = tokenizer.encode(text[:char_end], add_special_tokens=True)
            token_spans.append((len(prefix_ids), len(full_ids)))

        return token_spans


class SFTToolCallingDataset(Dataset):
    """
    Dataset for SFT warmup on function calling.

    Each sample is a conversation formatted as plain text with role markers.
    Works with base models that have no chat_template.
    Loss is masked to only compute on assistant content tokens.
    """

    def __init__(
        self,
        records: List[Dict[str, Any]],
        tokenizer,
        max_length: int = 4096,
        mask_user_turns: bool = True,
    ):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.mask_user_turns = mask_user_turns

        # Pre-tokenize to avoid repeated work in __getitem__
        self.samples = self._prepare(records)
        logger.info(f"SFT dataset: {len(self.samples)} samples prepared")

    def _prepare(self, records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        samples = []
        skipped = 0
        total_loss_tokens = 0
        total_all_tokens = 0

        for record in records:
            messages = record.get("messages", [])
            tools = record.get("tools", [])
            if not messages:
                skipped += 1
                continue
            result = self._build_sample(messages, tools)
            if result is not None:
                samples.append(result)
                total_loss_tokens += result.get("n_loss_tokens", 0)
                total_all_tokens += result.get("n_total_tokens", 0)
            else:
                skipped += 1

        if skipped:
            logger.info(f"Skipped {skipped} records (no assistant turn or empty)")

        # Diagnostic: how many tokens are used for loss
        if samples and total_all_tokens > 0:
            pct = total_loss_tokens / total_all_tokens * 100
            avg_loss = total_loss_tokens / len(samples)
            avg_total = total_all_tokens / len(samples)
            logger.info(
                f"Loss token stats: {total_loss_tokens:,}/{total_all_tokens:,} "
                f"({pct:.1f}%) across {len(samples)} samples"
            )
            logger.info(
                f"Per sample avg: {avg_loss:.0f} loss tokens / "
                f"{avg_total:.0f} total tokens"
            )
            print(
                f"  [SFT Data] Loss tokens: {pct:.1f}% of non-padding tokens "
                f"(avg {avg_loss:.0f}/{avg_total:.0f} per sample)",
                flush=True,
            )

        return samples

    def _build_sample(
        self,
        messages: List[Dict[str, Any]],
        tools: List[Dict[str, Any]],
    ) -> Optional[Dict[str, Any]]:
        """Convert raw messages to formatted text + assistant spans."""
        chat = []
        tools_json = json.dumps(tools, indent=2, ensure_ascii=False) if tools else "None"

        for msg in messages:
            role = msg.get("role", "")
            content = msg.get("content", "") or ""
            tool_calls = msg.get("tool_calls")

            if role == "system":
                system_content = SYSTEM_TEMPLATE.format(tools_json=tools_json)
                if content:
                    system_content = content + "\n\n" + system_content
                chat.append({"role": "system", "content": system_content})

            elif role == "user":
                chat.append({"role": "user", "content": content})

            elif role == "assistant":
                assistant_text = content or ""
                if tool_calls:
                    tc_parts = []
                    for tc in tool_calls:
                        func = tc.get("function", {})
                        args_raw = func.get("arguments", "{}")
                        try:
                            args = json.loads(args_raw) if isinstance(args_raw, str) else args_raw
                        except json.JSONDecodeError:
                            args = {"_raw": args_raw}
                        tc_json = json.dumps(
                            {"name": func.get("name", ""), "arguments": args},
                            ensure_ascii=False,
                        )
                        tc_parts.append(f"<tool_call>\n{tc_json}\n</tool_call>")
                    assistant_text = (
                        (assistant_text + "\n" if assistant_text else "")
                        + "\n".join(tc_parts)
                    )
                chat.append({"role": "assistant", "content": assistant_text})

            elif role == "tool":
                tool_text = TOOL_RESPONSE_TEMPLATE.format(content=content)
                chat.append({"role": "user", "content": tool_text})

        # Must have at least one assistant turn
        if not any(m["role"] == "assistant" for m in chat):
            return None

        # Ensure system prompt
        if not chat or chat[0]["role"] != "system":
            system_content = SYSTEM_TEMPLATE.format(tools_json=tools_json)
            chat.insert(0, {"role": "system", "content": system_content})

        # Format as plain text
        text = format_chat_plain(chat)

        # Pre-compute spans and token stats
        spans = find_assistant_spans(text, self.tokenizer)
        n_loss_tokens = sum(e - s for s, e in spans) if spans else 0
        encoded = self.tokenizer.encode(text, add_special_tokens=True, truncation=False)
        n_total_tokens = min(len(encoded), self.max_length)

        return {
            "text": text,
            "chat": chat,
            "spans": spans,
            "n_loss_tokens": min(n_loss_tokens, self.max_length),
            "n_total_tokens": n_total_tokens,
        }

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sample = self.samples[idx]
        text = sample["text"]
        spans = sample["spans"]

        # Tokenize
        encodings = self.tokenizer(
            text,
            max_length=self.max_length,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )

        input_ids = encodings["input_ids"].squeeze(0)
        attention_mask = encodings["attention_mask"].squeeze(0)

        # Labels: clone input_ids, mask non-assistant tokens with -100
        labels = input_ids.clone()

        if self.mask_user_turns and spans:
            labels = self._apply_spans(spans, labels, attention_mask)
        elif self.mask_user_turns:
            labels[:] = -100

        # Also mask padding
        labels[attention_mask == 0] = -100

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }

    def _apply_spans(
        self,
        spans: List[Tuple[int, int]],
        labels: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Apply pre-computed assistant content spans to labels."""
        seq_len = labels.shape[0]

        # Start with everything masked
        mask = torch.ones(seq_len, dtype=torch.bool)

        for start, end in spans:
            s = min(start, seq_len)
            e = min(end, seq_len)
            mask[s:e] = False

        labels[mask] = -100
        return labels
