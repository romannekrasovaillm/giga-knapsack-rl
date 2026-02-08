"""
Simulated tool environment for multi-turn agentic rollouts.

Replays ground truth tool responses from the dataset when the model
makes matching tool calls. For non-matching calls, returns error responses.
"""

import json
import logging
import re
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field

from .tool_registry import ToolRegistry

logger = logging.getLogger(__name__)


@dataclass
class EnvironmentState:
    """Tracks the state of a multi-turn rollout."""
    step: int = 0
    max_steps: int = 10
    tool_calls_made: List[Dict[str, Any]] = field(default_factory=list)
    tool_responses: List[str] = field(default_factory=list)
    conversation_history: List[Dict[str, str]] = field(default_factory=list)
    done: bool = False
    final_answer: Optional[str] = None


class SimulatedToolEnvironment:
    """
    Simulated tool-calling environment.

    Uses ground truth tool call/response pairs from the dataset.
    When the model makes a tool call that matches the ground truth
    (by function name), returns the corresponding ground truth response.
    For non-matching calls, returns a generic error.

    This allows training without actually executing tools while
    maintaining realistic tool interaction patterns.
    """

    def __init__(
        self,
        tools: List[Dict[str, Any]],
        tool_env_log: List[Dict[str, Any]],
        max_steps: int = 10,
        fuzzy_match: bool = True,
    ):
        self.registry = ToolRegistry(tools)
        self.tool_env_log = tool_env_log
        self.max_steps = max_steps
        self.fuzzy_match = fuzzy_match

        # Index ground truth responses by function name + step
        self._gt_responses: Dict[str, List[str]] = {}
        self._gt_call_index: Dict[str, int] = {}
        for entry in tool_env_log:
            call = entry.get("call", {})
            resp = entry.get("response", {})
            func_name = call.get("function", "")
            if func_name:
                if func_name not in self._gt_responses:
                    self._gt_responses[func_name] = []
                    self._gt_call_index[func_name] = 0
                self._gt_responses[func_name].append(resp.get("content", "{}"))

    def reset(self) -> EnvironmentState:
        """Reset environment for a new rollout."""
        # Reset ground truth indices
        for name in self._gt_call_index:
            self._gt_call_index[name] = 0
        return EnvironmentState(max_steps=self.max_steps)

    def step(
        self,
        state: EnvironmentState,
        model_output: str,
    ) -> Tuple[EnvironmentState, str]:
        """
        Process one step of model output.

        Parses tool calls from model output, looks up ground truth responses,
        returns the response text and updated state.

        Returns:
            (updated_state, response_text)
        """
        if state.done:
            return state, ""

        state.step += 1

        # Check if model produced a final answer (no tool calls)
        tool_calls = self._parse_tool_calls(model_output)

        if not tool_calls:
            # Model gave a final answer
            state.done = True
            state.final_answer = self._extract_final_answer(model_output)
            return state, ""

        # Process tool calls
        responses = []
        for tc in tool_calls:
            func_name = tc["name"]
            arguments = tc["arguments"]

            state.tool_calls_made.append({
                "function": func_name,
                "arguments": arguments,
            })

            # Validate call
            validation = self.registry.validate_call(func_name, arguments)

            if not validation["valid"]:
                error_resp = json.dumps({
                    "error": f"Invalid tool call: {'; '.join(validation['errors'])}"
                })
                responses.append(
                    f"<tool_response>\n{error_resp}\n</tool_response>"
                )
                state.tool_responses.append(error_resp)
                continue

            # Look up ground truth response
            gt_response = self._get_gt_response(func_name, arguments)
            responses.append(
                f"<tool_response>\n{gt_response}\n</tool_response>"
            )
            state.tool_responses.append(gt_response)

        # Check step limit
        if state.step >= state.max_steps:
            state.done = True

        response_text = "\n".join(responses)
        state.conversation_history.append({
            "role": "assistant",
            "content": model_output,
        })
        state.conversation_history.append({
            "role": "tool",
            "content": response_text,
        })

        return state, response_text

    def _parse_tool_calls(self, text: str) -> List[Dict[str, Any]]:
        """Extract tool calls from model output."""
        calls = []
        # Pattern: <tool_call>\n{"name": "...", "arguments": {...}}\n</tool_call>
        pattern = r'<tool_call>\s*(\{.*?\})\s*</tool_call>'
        matches = re.findall(pattern, text, re.DOTALL)

        for match in matches:
            try:
                parsed = json.loads(match)
                name = parsed.get("name", "")
                args = parsed.get("arguments", {})
                if isinstance(args, str):
                    args = json.loads(args)
                if name:
                    calls.append({"name": name, "arguments": args})
            except json.JSONDecodeError:
                continue

        return calls

    def _extract_final_answer(self, text: str) -> str:
        """Extract the final answer from model output (no tool calls)."""
        # Remove any stray tags
        clean = re.sub(r'<tool_call>.*?</tool_call>', '', text, flags=re.DOTALL)
        return clean.strip()

    def _get_gt_response(
        self,
        func_name: str,
        arguments: Dict[str, Any],
    ) -> str:
        """
        Get ground truth response for a tool call.

        Matching strategy:
          1. Exact function name match -> return next GT response for that function
          2. If no more GT responses -> return a generic success response
        """
        if func_name in self._gt_responses:
            idx = self._gt_call_index.get(func_name, 0)
            responses = self._gt_responses[func_name]
            if idx < len(responses):
                self._gt_call_index[func_name] = idx + 1
                return responses[idx]

        # Fallback: generic response
        return json.dumps({
            "status": "success",
            "message": f"Function {func_name} executed successfully",
            "result": None,
        })

    def get_available_tools_text(self) -> str:
        """Get formatted tools description for prompting."""
        return self.registry.format_for_prompt()
