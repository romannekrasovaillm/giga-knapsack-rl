"""
Rollout generation via vLLM server (OpenAI-compatible API).

Fully async: each rollout runs independently through multi-turn generation.
Fast rollouts don't wait for slow ones. Uses asyncio + aiohttp for concurrency.
"""

import asyncio
import json
import logging
import time
from typing import Dict, List, Any

from src.environment.tool_env import SimulatedToolEnvironment

logger = logging.getLogger(__name__)

CYAN = "\033[96m"
GREEN = "\033[92m"
YELLOW = "\033[93m"
RED = "\033[91m"
BOLD = "\033[1m"
DIM = "\033[2m"
RESET = "\033[0m"


def _truncate(text: str, head: int = 80, tail: int = 60) -> str:
    text = text.replace("\n", "\\n")
    if len(text) <= head + tail + 10:
        return text
    return text[:head] + f" ...({len(text)} chars)... " + text[-tail:]


class VLLMRolloutGenerator:
    """
    Async rollout generation via vLLM server.

    Each rollout runs its multi-turn loop independently — fast rollouts
    complete without waiting for slow ones. Concurrency is controlled
    by a semaphore to avoid overwhelming the server.
    """

    def __init__(
        self,
        server_url: str = "http://localhost:8000/v1",
        model_name: str = "ai-sage/GigaChat3-10B-A1.8B-base",
        max_response_length: int = 2048,
        max_env_steps: int = 8,
        max_concurrent: int = 64,
    ):
        import openai
        self.client = openai.OpenAI(base_url=server_url, api_key="not-needed")
        self.async_client = openai.AsyncOpenAI(base_url=server_url, api_key="not-needed")
        self.model_name = model_name
        self.max_response_length = max_response_length
        self.max_env_steps = max_env_steps
        self.max_concurrent = max_concurrent
        self.server_url = server_url

        # Verify connection & auto-detect model
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
        """Generate all rollouts async. Blocks until all are done."""
        return asyncio.run(self._generate_all(
            prompt_texts, tools_jsons, tool_env_logs,
            budgets, temperature, top_p,
        ))

    async def _generate_all(
        self,
        prompt_texts, tools_jsons, tool_env_logs,
        budgets, temperature, top_p,
    ) -> Dict[str, Any]:
        t_total = time.time()
        n_prompts = len(prompt_texts)
        n_rollouts = int(sum(budgets))
        sem = asyncio.Semaphore(self.max_concurrent)

        # Progress counter
        progress = {"done": 0, "total": n_rollouts, "successes": 0}

        print(
            f"  {CYAN}[vLLM async]{RESET} Generating {n_rollouts} rollouts "
            f"for {n_prompts} prompts (max_concurrent={self.max_concurrent})",
            flush=True,
        )

        # Build tasks
        tasks = []
        for i, (prompt, tools_json, env_log_str) in enumerate(
            zip(prompt_texts, tools_jsons, tool_env_logs)
        ):
            N_i = int(budgets[i])
            tools = json.loads(tools_json) if isinstance(tools_json, str) else tools_json
            env_log = json.loads(env_log_str) if isinstance(env_log_str, str) else env_log_str

            for j in range(N_i):
                tasks.append(self._run_single_rollout(
                    prompt_idx=i, rollout_idx=j,
                    prompt=prompt, tools=tools, env_log=env_log,
                    temperature=temperature, top_p=top_p,
                    sem=sem, progress=progress,
                ))

        results = await asyncio.gather(*tasks)

        total_time = time.time() - t_total
        total_tokens = sum(r["total_tokens"] for r in results)

        # Per-group summary
        self._log_group_summaries(results, budgets, prompt_texts)

        print(
            f"  {CYAN}[vLLM async]{RESET} Done: {n_rollouts} rollouts in "
            f"{total_time:.1f}s ({total_time/n_rollouts:.2f}s/rollout) | "
            f"{total_tokens:,} tokens",
            flush=True,
        )

        return {
            "responses": [r["response"] for r in results],
            "token_counts": [r["total_tokens"] for r in results],
            "gen_times": [r["gen_time"] for r in results],
            "num_turns": [r["num_turns"] for r in results],
            "prompt_indices": [r["prompt_idx"] for r in results],
        }

    async def _run_single_rollout(
        self, prompt_idx, rollout_idx, prompt, tools, env_log,
        temperature, top_p, sem, progress,
    ) -> Dict[str, Any]:
        """Run one multi-turn rollout independently."""
        env = SimulatedToolEnvironment(
            tools=tools, tool_env_log=env_log,
            max_steps=self.max_env_steps,
        )
        state = env.reset()
        conversation = prompt
        response = ""
        total_tokens = 0
        num_turns = 0
        t0 = time.time()

        for step in range(self.max_env_steps):
            max_new = min(512, self.max_response_length - total_tokens)
            if max_new <= 0:
                break

            try:
                async with sem:
                    completion = await self.async_client.completions.create(
                        model=self.model_name,
                        prompt=conversation,
                        max_tokens=max_new,
                        temperature=temperature,
                        top_p=top_p,
                    )
                text = completion.choices[0].text
                n_tok = completion.usage.completion_tokens if completion.usage else len(text.split())
            except Exception as e:
                logger.warning(
                    f"vLLM error (prompt={prompt_idx} roll={rollout_idx} "
                    f"step={step}): {e}"
                )
                break

            total_tokens += n_tok
            num_turns += 1

            state, tool_response = env.step(state, text)
            response += text

            if state.done:
                break

            conversation += "\n" + text + "\n" + tool_response
            response += "\n" + tool_response + "\n"

        gen_time = time.time() - t0

        # Progress update
        progress["done"] += 1
        done = progress["done"]
        total = progress["total"]
        if done % max(1, total // 20) == 0 or done == total:
            elapsed = time.time() - t0
            print(
                f"    {done}/{total} rollouts done "
                f"({done/total:.0%}) | this: turns={num_turns} "
                f"tok={total_tokens} {gen_time:.1f}s",
                flush=True,
            )

        return {
            "prompt_idx": prompt_idx,
            "rollout_idx": rollout_idx,
            "response": response,
            "total_tokens": total_tokens,
            "num_turns": num_turns,
            "gen_time": gen_time,
        }

    def _log_group_summaries(
        self,
        results: List[Dict],
        budgets,
        prompt_texts: List[str],
    ):
        offset = 0
        for i, N_i in enumerate(budgets):
            N_i = int(N_i)
            group = results[offset:offset + N_i]
            offset += N_i

            token_counts = [r["total_tokens"] for r in group]
            turns = [r["num_turns"] for r in group]
            avg_tokens = sum(token_counts) / len(token_counts) if token_counts else 0
            avg_turns = sum(turns) / len(turns) if turns else 0

            prompt_preview = _truncate(prompt_texts[i], head=60, tail=0)
            first = _truncate(group[0]["response"], head=80, tail=40)
            last = _truncate(group[-1]["response"], head=80, tail=40) if len(group) > 1 else ""

            print(
                f"    {DIM}[Group {i+1}/{len(budgets)}] N_i={N_i} | "
                f"avg_tok={avg_tokens:.0f} | avg_turns={avg_turns:.1f}{RESET}",
                flush=True,
            )
            print(f"      {DIM}prompt: {prompt_preview}{RESET}", flush=True)
            print(f"      {DIM}roll[0]: {first}{RESET}", flush=True)
            if last:
                print(f"      {DIM}roll[-1]: {last}{RESET}", flush=True)
