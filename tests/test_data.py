"""Tests for data parsing and environment simulation."""

import json
import pytest

from src.data.parser import AgenticTrajectoryParser, AgenticTrajectory
from src.environment.tool_env import SimulatedToolEnvironment
from src.environment.tool_registry import ToolRegistry


SAMPLE_RECORD = {
    "uuid": "test-001",
    "messages": [
        {
            "role": "system",
            "content": "You are a helpful assistant.",
            "tool_calls": None,
            "tool_call_id": None,
            "reasoning_content": None,
        },
        {
            "role": "user",
            "content": "What's the weather in Moscow?",
            "tool_calls": None,
            "tool_call_id": None,
            "reasoning_content": None,
        },
        {
            "role": "assistant",
            "content": None,
            "reasoning_content": "I need to check the weather.",
            "tool_calls": [
                {
                    "id": "call_1",
                    "type": "function",
                    "function": {
                        "name": "get_weather",
                        "arguments": '{"city": "Moscow"}',
                    },
                }
            ],
            "tool_call_id": None,
        },
        {
            "role": "tool",
            "content": '{"temp": -5, "condition": "snow"}',
            "tool_call_id": "call_1",
            "name": "get_weather",
            "tool_calls": None,
            "reasoning_content": None,
        },
        {
            "role": "assistant",
            "content": "The weather in Moscow is -5C with snow.",
            "tool_calls": None,
            "tool_call_id": None,
            "reasoning_content": None,
        },
    ],
    "tools": [
        {
            "type": "function",
            "function": {
                "name": "get_weather",
                "description": "Get weather for a city",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "city": {"type": "string"},
                    },
                    "required": ["city"],
                },
            },
        }
    ],
    "reasoning": "",
    "license": "cc-by-4.0",
    "used_in": [],
}


class TestParser:
    def setup_method(self):
        self.parser = AgenticTrajectoryParser()

    def test_parse_basic(self):
        traj = self.parser.parse(SAMPLE_RECORD)
        assert traj is not None
        assert traj.uuid == "test-001"
        assert traj.system_prompt == "You are a helpful assistant."
        assert traj.final_answer == "The weather in Moscow is -5C with snow."
        assert traj.num_tool_calls == 1

    def test_parse_tools(self):
        traj = self.parser.parse(SAMPLE_RECORD)
        assert len(traj.tools) == 1
        assert traj.tool_names_used == ["get_weather"]

    def test_parse_env_log(self):
        traj = self.parser.parse(SAMPLE_RECORD)
        log = traj.build_environment_log()
        assert len(log) == 1
        assert log[0]["call"]["function"] == "get_weather"
        assert '"temp"' in log[0]["response"]["content"]

    def test_parse_batch(self):
        records = [SAMPLE_RECORD, SAMPLE_RECORD]
        trajs = self.parser.parse_batch(records)
        assert len(trajs) == 2

    def test_parse_empty(self):
        result = self.parser.parse({"uuid": "empty", "messages": []})
        assert result is None


class TestToolRegistry:
    def test_basic(self):
        reg = ToolRegistry(SAMPLE_RECORD["tools"])
        assert reg.num_tools == 1
        assert reg.has_tool("get_weather")
        assert not reg.has_tool("nonexistent")

    def test_validate_call(self):
        reg = ToolRegistry(SAMPLE_RECORD["tools"])
        result = reg.validate_call("get_weather", {"city": "Moscow"})
        assert result["valid"]

    def test_validate_missing_required(self):
        reg = ToolRegistry(SAMPLE_RECORD["tools"])
        result = reg.validate_call("get_weather", {})
        assert not result["valid"]


class TestEnvironment:
    def test_basic_interaction(self):
        traj = AgenticTrajectoryParser().parse(SAMPLE_RECORD)
        env = SimulatedToolEnvironment(
            tools=traj.tools,
            tool_env_log=traj.build_environment_log(),
        )
        state = env.reset()
        assert not state.done

        # Simulate model outputting a tool call
        model_output = '<tool_call>\n{"name": "get_weather", "arguments": {"city": "Moscow"}}\n</tool_call>'
        state, response = env.step(state, model_output)
        assert "temp" in response
        assert len(state.tool_calls_made) == 1

    def test_final_answer(self):
        traj = AgenticTrajectoryParser().parse(SAMPLE_RECORD)
        env = SimulatedToolEnvironment(
            tools=traj.tools,
            tool_env_log=traj.build_environment_log(),
        )
        state = env.reset()

        # Model gives final answer directly
        state, response = env.step(state, "The weather is cold.")
        assert state.done
        assert state.final_answer == "The weather is cold."
