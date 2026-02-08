"""
SFT Dataset for tool_calling subset warmup.

Converts Nemotron tool_calling records into plain-text training data
for supervised fine-tuning on function calling.

Works with base models (no chat_template required) — formats conversations
using explicit role markers.
"""

import json
import logging
from typing import List, Dict, Any, Optional

import torch
from torch.utils.data import Dataset

logger = logging.getLogger(__name__)

# Role markers for base model (no chat template)
ROLE_MARKERS = {
    "system": "<|system|>\n{content}\n",
    "user": "<|user|>\n{content}\n",
    "assistant": "<|assistant|>\n{content}\n",
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
    chat: List[Dict[str, str]],
    tokenizer,
) -> List[tuple]:
    """
    Find (start, end) token positions for each assistant turn.
    Used to create labels mask: only compute loss on assistant tokens.
    """
    spans = []
    prefix = ""
    for msg in chat:
        role = msg["role"]
        content = msg["content"]
        marker = ROLE_MARKERS.get(role, ROLE_MARKERS["user"])
        formatted = marker.format(content=content)

        if role == "assistant":
            # Tokens before this message = prefix length
            prefix_ids = tokenizer.encode(prefix, add_special_tokens=False)
            start = len(prefix_ids)
            # Tokens including this message
            full_ids = tokenizer.encode(prefix + formatted, add_special_tokens=False)
            end = len(full_ids)
            spans.append((start, end))

        prefix += formatted

    return spans


class SFTToolCallingDataset(Dataset):
    """
    Dataset for SFT warmup on function calling.

    Each sample is a conversation formatted as plain text with role markers.
    Works with base models that have no chat_template.
    Loss is masked to only compute on assistant turns.
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
        for record in records:
            messages = record.get("messages", [])
            tools = record.get("tools", [])
            if not messages:
                skipped += 1
                continue
            result = self._build_sample(messages, tools)
            if result is not None:
                samples.append(result)
            else:
                skipped += 1
        if skipped:
            logger.info(f"Skipped {skipped} records (no assistant turn or empty)")
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

        return {"text": text, "chat": chat}

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sample = self.samples[idx]
        text = sample["text"]
        chat = sample["chat"]

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

        if self.mask_user_turns:
            labels = self._mask_non_assistant(chat, labels)

        # Also mask padding
        labels[attention_mask == 0] = -100

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }

    def _mask_non_assistant(
        self,
        chat: List[Dict[str, str]],
        labels: torch.Tensor,
    ) -> torch.Tensor:
        """Mask labels for non-assistant turns (set to -100)."""
        seq_len = labels.shape[0]

        # Find assistant spans
        spans = find_assistant_spans(chat, self.tokenizer)

        if not spans:
            # No assistant turns found — mask everything
            labels[:] = -100
            return labels

        # Start with everything masked
        mask = torch.ones(seq_len, dtype=torch.bool)

        # +1 offset for BOS token if tokenizer adds one
        bos_offset = 0
        if self.tokenizer.bos_token_id is not None:
            bos_offset = 1

        for start, end in spans:
            s = min(start + bos_offset, seq_len)
            e = min(end + bos_offset, seq_len)
            mask[s:e] = False

        labels[mask] = -100
        return labels
