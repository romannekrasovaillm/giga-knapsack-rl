"""
Rollout generation backends: vLLM server (batched) or HuggingFace fallback.

vLLM server mode generates rollouts via OpenAI-compatible API, enabling
batched inference with PagedAttention for much faster throughput.
"""

import json
import logging
import time
from typing import Dict, List, Any, Optional, Tuple

from src.environment.tool_env import SimulatedToolEnvironment

logger = logging.getLogger(__name__)

# Terminal colors
CYAN = "\033[96m"
GREEN = "\033[92m"
YELLOW = "\033[93m"
RED = "\033[91m"
BOLD = "\033[1m"
DIM = "\033[2m"
RESET = "\033[0m"


def _truncate(text: str, head: int = 80, tail: int = 60) -> str:
    """Truncate text showing head and tail."""
    text = text.replace("\n", "\\n")
    if len(text) <= head + tail + 10:
        return text
    return text[:head] + f" ...({len(text)} chars)... " + text[-tail:]


class VLLMRolloutGenerator:
    """
    Generate rollouts using a vLLM server (OpenAI-compatible API).

    Supports batched multi-turn generation: all rollouts for the same turn
    are batched together for maximum throughput.
    """

    def __init__(
        self,
        server_url: str = "http://localhost:8000/v1",
        model_name: str = "ai-sage/GigaChat3-10B-A1.8B-base",
        max_response_length: int = 2048,
        max_env_steps: int = 8,
        batch_chunk_size: int = 128,
    ):
        try:
            import openai
        except ImportError:
            raise ImportError("pip install openai  — required for vLLM server mode")

        self.client = openai.OpenAI(base_url=server_url, api_key="not-needed")
        self.model_name = model_name
        self.max_response_length = max_response_length
        self.max_env_steps = max_env_steps
        self.batch_chunk_size = batch_chunk_size
        self.server_url = server_url

        # Verify connection
        try:
            models = self.client.models.list()
            available = [m.id for m in models.data]
            logger.info(f"vLLM server connected at {server_url}")
            logger.info(f"Available models: {available}")
            if model_name not in available and available:
                self.model_name = available[0]
                logger.info(f"Using model: {self.model_name}")
        except Exception as e:
            raise ConnectionError(f"Cannot connect to vLLM at {server_url}: {e}")

    def generate_rollouts(
        self,
        prompt_texts: List[str],
        tools_jsons: List[str],
        tool_env_logs: List[str],
        budgets,
        temperature: float = 1.0,
        top_p: float = 0.95,
    ) -> Dict[str, Any]:
        """
        Generate all rollouts for a batch with batched multi-turn generation.

        Returns dict with:
            responses: List[str] — flat list of full response texts
            token_counts: List[int]
            gen_times: List[float]
            num_turns: List[int]
            prompt_indices: List[int] — which prompt each rollout belongs to
        """
        t_total = time.time()
        n_prompts = len(prompt_texts)
        n_rollouts = int(sum(budgets))

        # Expand everything to per-rollout level
        expanded = []
        for i, (prompt, tools_json, env_log_str) in enumerate(
            zip(prompt_texts, tools_jsons, tool_env_logs)
        ):
            N_i = int(budgets[i])
            tools = json.loads(tools_json) if isinstance(tools_json, str) else tools_json
            env_log = json.loads(env_log_str) if isinstance(env_log_str, str) else env_log_str
            for j in range(N_i):
                expanded.append({
                    "prompt_idx": i,
                    "rollout_idx": j,
                    "prompt": prompt,
                    "tools": tools,
                    "env_log": env_log,
                    "conversation": prompt,
                    "response": "",
                    "total_tokens": 0,
                    "num_turns": 0,
                    "done": False,
                    "env": SimulatedToolEnvironment(
                        tools=tools, tool_env_log=env_log,
                        max_steps=self.max_env_steps,
                    ),
                    "state": None,
                    "gen_time": 0.0,
                })
                expanded[-1]["state"] = expanded[-1]["env"].reset()

        print(
            f"  {CYAN}[vLLM]{RESET} Generating {n_rollouts} rollouts "
            f"for {n_prompts} prompts (budget={int(sum(budgets))})",
            flush=True,
        )

        # Multi-turn batched generation
        for turn in range(self.max_env_steps):
            active = [r for r in expanded if not r["done"]]
            if not active:
                break

            # Batch generate for all active rollouts
            prompts_batch = [r["conversation"] for r in active]
            max_new = min(512, self.max_response_length)

            t_turn = time.time()
            completions = self._batch_generate(
                prompts_batch, max_new, temperature, top_p,
            )
            turn_time = time.time() - t_turn

            # Process results
            n_tool_calls = 0
            n_final = 0
            turn_tokens = 0

            for r, text in zip(active, completions):
                r["num_turns"] += 1
                r["total_tokens"] += len(text.split())  # approx
                r["gen_time"] += turn_time / len(active)

                state, tool_response = r["env"].step(r["state"], text)
                r["state"] = state
                r["response"] += text

                if state.done:
                    r["done"] = True
                    n_final += 1
                else:
                    n_tool_calls += 1
                    r["conversation"] += "\n" + text + "\n" + tool_response
                    r["response"] += "\n" + tool_response + "\n"

            turn_tokens = sum(len(c.split()) for c in completions)
            still_active = sum(1 for r in expanded if not r["done"])

            print(
                f"    turn {turn+1}: {len(active)} active | "
                f"{n_tool_calls} tool_calls, {n_final} final | "
                f"{turn_tokens:,} tokens | {turn_time:.1f}s | "
                f"{still_active} remaining",
                flush=True,
            )

        # Mark any remaining as done
        for r in expanded:
            if not r["done"]:
                r["done"] = True

        total_time = time.time() - t_total

        # Log per-group summary
        self._log_group_summaries(expanded, budgets, prompt_texts)

        print(
            f"  {CYAN}[vLLM]{RESET} Done: {n_rollouts} rollouts in {total_time:.1f}s "
            f"({total_time/n_rollouts:.2f}s/rollout)",
            flush=True,
        )

        return {
            "responses": [r["response"] for r in expanded],
            "token_counts": [r["total_tokens"] for r in expanded],
            "gen_times": [r["gen_time"] for r in expanded],
            "num_turns": [r["num_turns"] for r in expanded],
            "prompt_indices": [r["prompt_idx"] for r in expanded],
        }

    def _batch_generate(
        self,
        prompts: List[str],
        max_tokens: int,
        temperature: float,
        top_p: float,
    ) -> List[str]:
        """Batch generate completions via vLLM server."""
        results = []
        chunk_size = self.batch_chunk_size

        for start in range(0, len(prompts), chunk_size):
            chunk = prompts[start:start + chunk_size]
            try:
                response = self.client.completions.create(
                    model=self.model_name,
                    prompt=chunk,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    top_p=top_p,
                )
                # Sort by index to maintain order
                sorted_choices = sorted(response.choices, key=lambda c: c.index)
                results.extend([c.text for c in sorted_choices])
            except Exception as e:
                logger.error(f"vLLM batch generation failed: {e}")
                # Return empty strings as fallback
                results.extend(["" for _ in chunk])

        return results

    def _log_group_summaries(
        self,
        expanded: List[Dict],
        budgets,
        prompt_texts: List[str],
    ):
        """Log per-group rollout summaries."""
        offset = 0
        for i, N_i in enumerate(budgets):
            N_i = int(N_i)
            group = expanded[offset:offset + N_i]
            offset += N_i

            token_counts = [r["total_tokens"] for r in group]
            turns = [r["num_turns"] for r in group]
            avg_tokens = sum(token_counts) / len(token_counts) if token_counts else 0
            avg_turns = sum(turns) / len(turns) if turns else 0

            prompt_preview = _truncate(prompt_texts[i], head=60, tail=0)

            # Show first and last rollout
            first = _truncate(group[0]["response"], head=80, tail=40)
            last = _truncate(group[-1]["response"], head=80, tail=40) if len(group) > 1 else ""

            print(
                f"    {DIM}[Group {i+1}/{len(budgets)}] N_i={N_i} | "
                f"avg_tokens={avg_tokens:.0f} | avg_turns={avg_turns:.1f}{RESET}",
                flush=True,
            )
            print(f"      {DIM}prompt: {prompt_preview}{RESET}", flush=True)
            print(f"      {DIM}roll[0]: {first}{RESET}", flush=True)
            if last:
                print(f"      {DIM}roll[-1]: {last}{RESET}", flush=True)
