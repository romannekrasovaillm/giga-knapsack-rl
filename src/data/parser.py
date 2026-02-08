"""
Parse Nemotron Agentic data into structured trajectories.

Each record has:
  - messages: List[{role, content, tool_calls, tool_call_id, reasoning_content, name}]
  - tools: List[{type, function: {name, description, parameters}}]
  - uuid, reasoning, license

We parse this into a structured trajectory:
  - system_prompt
  - tools (definitions)
  - turns: List[Turn] where each turn has user/assistant/tool exchanges
  - final_answer: the last assistant message without tool calls (ground truth)
"""

import json
import logging
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Tuple

logger = logging.getLogger(__name__)


@dataclass
class ToolCall:
    """A single tool invocation."""
    call_id: str
    function_name: str
    arguments: Dict[str, Any]


@dataclass
class ToolResponse:
    """Response from a tool."""
    call_id: str
    function_name: str
    content: str


@dataclass
class Turn:
    """One exchange in the trajectory."""
    user_message: Optional[str] = None
    assistant_message: Optional[str] = None
    assistant_reasoning: Optional[str] = None
    tool_calls: List[ToolCall] = field(default_factory=list)
    tool_responses: List[ToolResponse] = field(default_factory=list)


@dataclass
class AgenticTrajectory:
    """Fully parsed trajectory from a dataset record."""
    uuid: str
    system_prompt: str
    tools: List[Dict[str, Any]]
    turns: List[Turn]
    final_answer: str
    raw_messages: List[Dict[str, Any]]

    @property
    def num_turns(self) -> int:
        return len(self.turns)

    @property
    def num_tool_calls(self) -> int:
        return sum(len(t.tool_calls) for t in self.turns)

    @property
    def tool_names_used(self) -> List[str]:
        names = []
        for t in self.turns:
            for tc in t.tool_calls:
                if tc.function_name not in names:
                    names.append(tc.function_name)
        return names

    def get_prompt_for_rlvr(self) -> str:
        """Build the initial prompt for RLVR (system + tools + first user message)."""
        parts = []
        if self.system_prompt:
            parts.append(self.system_prompt)

        if self.tools:
            tool_desc = json.dumps(self.tools, indent=2, ensure_ascii=False)
            parts.append(f"\nAvailable tools:\n{tool_desc}")

        if self.turns and self.turns[0].user_message:
            parts.append(f"\nUser: {self.turns[0].user_message}")

        return "\n".join(parts)

    def get_tool_call_ground_truth(self) -> List[Dict[str, Any]]:
        """Extract all tool calls as ground truth for verification."""
        calls = []
        for turn in self.turns:
            for tc in turn.tool_calls:
                calls.append({
                    "function_name": tc.function_name,
                    "arguments": tc.arguments,
                })
        return calls

    def build_environment_log(self) -> List[Dict[str, Any]]:
        """Build a log of (tool_call -> tool_response) pairs for the simulated env."""
        log = []
        for turn in self.turns:
            for tc, tr in zip(turn.tool_calls, turn.tool_responses):
                log.append({
                    "call": {
                        "id": tc.call_id,
                        "function": tc.function_name,
                        "arguments": tc.arguments,
                    },
                    "response": {
                        "id": tr.call_id,
                        "function": tr.function_name,
                        "content": tr.content,
                    },
                })
        return log


class AgenticTrajectoryParser:
    """Parse raw Nemotron records into AgenticTrajectory objects."""

    def parse(self, record: Dict[str, Any]) -> Optional[AgenticTrajectory]:
        """Parse a single JSONL record."""
        try:
            uuid = record.get("uuid", "unknown")
            messages = record.get("messages", [])
            tools = record.get("tools", [])

            if not messages:
                return None

            system_prompt = ""
            turns = []
            current_turn = Turn()

            for msg in messages:
                role = msg.get("role", "")
                content = msg.get("content", "") or ""
                tool_calls_raw = msg.get("tool_calls")
                tool_call_id = msg.get("tool_call_id")
                reasoning = msg.get("reasoning_content")
                name = msg.get("name")

                if role == "system":
                    system_prompt = content

                elif role == "user":
                    # Start a new turn on user message
                    if current_turn.user_message is not None:
                        turns.append(current_turn)
                        current_turn = Turn()
                    current_turn.user_message = content

                elif role == "assistant":
                    current_turn.assistant_message = content
                    current_turn.assistant_reasoning = reasoning

                    if tool_calls_raw:
                        for tc in tool_calls_raw:
                            func = tc.get("function", {})
                            args_str = func.get("arguments", "{}")
                            try:
                                args = json.loads(args_str) if isinstance(args_str, str) else args_str
                            except json.JSONDecodeError:
                                args = {"_raw": args_str}

                            current_turn.tool_calls.append(ToolCall(
                                call_id=tc.get("id", ""),
                                function_name=func.get("name", ""),
                                arguments=args,
                            ))

                elif role == "tool":
                    current_turn.tool_responses.append(ToolResponse(
                        call_id=tool_call_id or "",
                        function_name=name or "",
                        content=content,
                    ))

            # Append last turn
            if current_turn.user_message is not None or current_turn.assistant_message is not None:
                turns.append(current_turn)

            # Extract final answer: last assistant message WITHOUT tool calls
            final_answer = self._extract_final_answer(turns)

            if not final_answer:
                return None

            return AgenticTrajectory(
                uuid=uuid,
                system_prompt=system_prompt,
                tools=tools,
                turns=turns,
                final_answer=final_answer,
                raw_messages=messages,
            )

        except Exception as e:
            logger.warning(f"Failed to parse record {record.get('uuid', '?')}: {e}")
            return None

    def _extract_final_answer(self, turns: List[Turn]) -> str:
        """Find the last assistant message that is a final answer (no tool calls)."""
        for turn in reversed(turns):
            if turn.assistant_message and not turn.tool_calls:
                return turn.assistant_message.strip()
            # Some trajectories have the final answer in a turn with tool calls
            # followed by tool responses and then a final text
            if turn.assistant_message and turn.tool_calls and turn.tool_responses:
                # Check if there's a subsequent assistant message
                continue
        # Fallback: last non-empty assistant message
        for turn in reversed(turns):
            if turn.assistant_message:
                return turn.assistant_message.strip()
        return ""

    def parse_batch(self, records: List[Dict[str, Any]]) -> List[AgenticTrajectory]:
        """Parse a batch of records, filtering out failures."""
        trajectories = []
        failed = 0
        for record in records:
            traj = self.parse(record)
            if traj is not None:
                trajectories.append(traj)
            else:
                failed += 1
        logger.info(f"Parsed {len(trajectories)} trajectories, {failed} failed")
        return trajectories
