"""
Tool registry: stores tool definitions and provides lookup for the simulated environment.
"""

import json
import logging
from typing import Dict, List, Any, Optional

logger = logging.getLogger(__name__)


class ToolRegistry:
    """
    Registry of available tools for a given trajectory.

    Loads tool definitions from the dataset and provides:
      - lookup by function name
      - parameter validation
      - tool description formatting
    """

    def __init__(self, tools: List[Dict[str, Any]]):
        self._tools: Dict[str, Dict[str, Any]] = {}
        for tool in tools:
            if tool.get("type") == "function":
                func = tool.get("function", {})
                name = func.get("name", "")
                if name:
                    self._tools[name] = func

    @property
    def tool_names(self) -> List[str]:
        return list(self._tools.keys())

    @property
    def num_tools(self) -> int:
        return len(self._tools)

    def has_tool(self, name: str) -> bool:
        return name in self._tools

    def get_tool(self, name: str) -> Optional[Dict[str, Any]]:
        return self._tools.get(name)

    def get_parameters(self, name: str) -> Optional[Dict[str, Any]]:
        tool = self._tools.get(name)
        if tool:
            return tool.get("parameters", {})
        return None

    def validate_call(self, function_name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate a tool call against the registry.

        Returns:
            dict with 'valid' (bool), 'errors' (list of strings)
        """
        errors = []

        if not self.has_tool(function_name):
            return {"valid": False, "errors": [f"Unknown function: {function_name}"]}

        params = self.get_parameters(function_name)
        if not params:
            return {"valid": True, "errors": []}

        # Check required parameters
        required = params.get("required", [])
        properties = params.get("properties", {})

        for req in required:
            if req not in arguments:
                errors.append(f"Missing required parameter: {req}")

        # Check for unknown parameters
        for key in arguments:
            if key not in properties:
                errors.append(f"Unknown parameter: {key}")

        # Basic type checking
        for key, value in arguments.items():
            if key in properties:
                expected_type = properties[key].get("type")
                if expected_type and not self._check_type(value, expected_type):
                    errors.append(
                        f"Parameter '{key}': expected {expected_type}, "
                        f"got {type(value).__name__}"
                    )

        return {"valid": len(errors) == 0, "errors": errors}

    def _check_type(self, value: Any, expected: str) -> bool:
        type_map = {
            "string": str,
            "integer": int,
            "number": (int, float),
            "boolean": bool,
            "array": list,
            "object": dict,
        }
        expected_type = type_map.get(expected)
        if expected_type is None:
            return True
        return isinstance(value, expected_type)

    def format_for_prompt(self) -> str:
        """Format all tools as JSON for inclusion in prompts."""
        tools_list = []
        for name, func in self._tools.items():
            tools_list.append({
                "type": "function",
                "function": func,
            })
        return json.dumps(tools_list, indent=2, ensure_ascii=False)
