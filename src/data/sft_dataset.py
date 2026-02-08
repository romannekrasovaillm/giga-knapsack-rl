"""
SFT Dataset for tool_calling subset warmup.

Converts Nemotron tool_calling records into chat-format training data
for supervised fine-tuning on function calling.
"""

import json
import logging
from typing import List, Dict, Any, Optional

import torch
from torch.utils.data import Dataset

logger = logging.getLogger(__name__)

# Chat template for GigaChat3 function calling
SYSTEM_TEMPLATE = (
    "You are a helpful assistant with access to tools. "
    "When you need to use a tool, output a tool call in the following JSON format:\n"
    '<tool_call>\n{{"name": "<function_name>", "arguments": {{...}}}}\n</tool_call>\n\n'
    "Available tools:\n{tools_json}"
)

TOOL_RESPONSE_TEMPLATE = "<tool_response>\n{content}\n</tool_response>"


class SFTToolCallingDataset(Dataset):
    """
    Dataset for SFT warmup on function calling.

    Each sample is a conversation with:
      - system prompt listing available tools
      - user messages
      - assistant messages (some with tool_call tags)
      - tool responses
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
        self.samples = self._prepare(records)
        logger.info(f"SFT dataset: {len(self.samples)} samples prepared")

    def _prepare(self, records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        samples = []
        for record in records:
            messages = record.get("messages", [])
            tools = record.get("tools", [])
            if not messages:
                continue
            chat = self._build_chat(messages, tools)
            if chat:
                samples.append({"chat": chat, "uuid": record.get("uuid", "")})
        return samples

    def _build_chat(
        self,
        messages: List[Dict[str, Any]],
        tools: List[Dict[str, Any]],
    ) -> Optional[List[Dict[str, str]]]:
        """Convert raw messages to chat format with tool_call/tool_response tags."""
        chat = []
        tools_json = json.dumps(tools, indent=2, ensure_ascii=False) if tools else "None"

        for msg in messages:
            role = msg.get("role", "")
            content = msg.get("content", "") or ""
            tool_calls = msg.get("tool_calls")
            tool_call_id = msg.get("tool_call_id")

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
                        tc_json = json.dumps({
                            "name": func.get("name", ""),
                            "arguments": json.loads(func.get("arguments", "{}"))
                            if isinstance(func.get("arguments"), str)
                            else func.get("arguments", {}),
                        }, ensure_ascii=False)
                        tc_parts.append(f"<tool_call>\n{tc_json}\n</tool_call>")
                    assistant_text = (assistant_text + "\n" if assistant_text else "") + "\n".join(tc_parts)
                chat.append({"role": "assistant", "content": assistant_text})

            elif role == "tool":
                # Encode tool response as a user message (common practice for SFT)
                tool_text = TOOL_RESPONSE_TEMPLATE.format(content=content)
                chat.append({"role": "user", "content": tool_text})

        # Ensure we have at least system + user + assistant
        roles = [m["role"] for m in chat]
        if "assistant" not in roles:
            return None

        # Ensure system prompt exists
        if not chat or chat[0]["role"] != "system":
            system_content = SYSTEM_TEMPLATE.format(tools_json=tools_json)
            chat.insert(0, {"role": "system", "content": system_content})

        return chat

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sample = self.samples[idx]
        chat = sample["chat"]

        # Apply chat template
        text = self.tokenizer.apply_chat_template(
            chat, tokenize=False, add_generation_prompt=False,
        )
        encodings = self.tokenizer(
            text,
            max_length=self.max_length,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )

        input_ids = encodings["input_ids"].squeeze(0)
        attention_mask = encodings["attention_mask"].squeeze(0)

        # Labels: same as input_ids but mask non-assistant tokens with -100
        labels = input_ids.clone()
        if self.mask_user_turns:
            labels = self._mask_non_assistant(chat, input_ids, labels)

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }

    def _mask_non_assistant(
        self,
        chat: List[Dict[str, str]],
        input_ids: torch.Tensor,
        labels: torch.Tensor,
    ) -> torch.Tensor:
        """Mask labels for non-assistant turns so loss is only on assistant outputs."""
        # Simple approach: find assistant response boundaries
        # Tokenize each message and track positions
        pos = 0
        in_assistant = False

        for msg in chat:
            role = msg["role"]
            content = msg["content"]
            # Approximate token count for this message (including special tokens)
            tokens = self.tokenizer.encode(content, add_special_tokens=False)
            # Role/template overhead ~ 3-5 tokens
            overhead = 4

            if role != "assistant":
                # Mask these positions
                end_pos = min(pos + len(tokens) + overhead, len(labels))
                labels[pos:end_pos] = -100
                pos = end_pos
            else:
                # Keep assistant tokens
                pos = min(pos + len(tokens) + overhead, len(labels))

        return labels
