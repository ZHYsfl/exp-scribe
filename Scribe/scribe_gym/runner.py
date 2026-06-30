from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from .chat_template import render_messages
from .openai_bridge import GymBackedAgent


@dataclass
class TurnResult:
    stop_run: bool
    feedback: Optional[str] = None


class ScribeRunner:
    def __init__(
        self,
        agent: GymBackedAgent,
        system_prompt: Optional[str] = None,
        done_mode: str = "threshold",
        reward_threshold: float = 1.0,
        max_turns: int = 32,
    ):
        self.agent = agent
        self.system_prompt = system_prompt
        self.done_mode = done_mode
        self.reward_threshold = reward_threshold
        self.max_turns = max_turns
        self._best_reward = float("-inf")

    async def run(
        self,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        observations = await self.reset(seed=seed, options=options)
        for i in range(self.max_turns):
            observations = await self.agent.chat(observations)
            if not self.agent.trajectory:
                continue

            result = self._evaluate_turn(self.agent.trajectory[-1])
            if result is None:
                continue
            if result.stop_run:
                break
            if result.feedback is not None:
                observations.append({"role": "user", "content": result.feedback})
        return observations

    async def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        obs, info = self.agent.env.reset(seed=seed, options=options)
        self.agent.trajectory.clear()
        self._best_reward = float("-inf")
        return self._build_initial_observations(obs)

    def _build_system_prompt(self, task_description: Optional[str] = None) -> str:
        # Tools are NOT baked into the system content here. The server injects
        # the tool definitions into the prompt itself from the `tools=` API
        # field (verified against a local vllm server: the chat template renders
        # tools into the system block on its own). Baking them here too would
        # duplicate the <tools> block in the prompt the model actually sees.
        content = self.system_prompt or ""
        if task_description:
            content = f"{content}\n\n# Task\n{task_description}".strip()
        if content:
            messages = [{"role": "system", "content": content}]
        else:
            messages = [{"role": "user", "content": ""}]
        rendered = render_messages(messages, tools=[], add_generation_prompt=False)
        start = rendered.find("<|im_start|>system\n")
        if start == -1:
            return ""
        start += len("<|im_start|>system\n")
        end = rendered.find("<|im_end|>", start)
        if end == -1:
            return rendered[start:]
        return rendered[start:end]

    def _build_initial_observations(self, task_description: str) -> List[Dict[str, Any]]:
        messages: List[Dict[str, Any]] = []
        system_content = self._build_system_prompt(task_description)
        if system_content:
            messages.append({"role": "system", "content": system_content})
        # The task description is merged into the system prompt; the first user
        # message is a fixed kickoff. All later user messages are the feedback
        # produced by _evaluate_turn.
        messages.append({"role": "user", "content": "Please start solving the problem."})
        return messages

    def _evaluate_turn(self, step: Dict[str, Any]) -> Optional[TurnResult]:
        if not self._is_turn_done(step):
            return None

        if self._should_stop_run(step):
            return TurnResult(stop_run=True)

        reward = step["reward"]
        is_new_best = reward > self._best_reward
        self._best_reward = max(self._best_reward, reward)

        feedback = self._build_feedback(step, is_new_best)
        self._reset_env_step_count()
        return TurnResult(stop_run=False, feedback=feedback)

    def _is_turn_done(self, step: Dict[str, Any]) -> bool:
        if step["truncated"]:
            return True
        if step["terminated"] and step["info"].get("done_reason") == "no_tool_call":
            return True
        return False

    def _should_stop_run(self, step: Dict[str, Any]) -> bool:
        return self.done_mode == "threshold" and step["reward"] >= self.reward_threshold

    def _build_feedback(
        self, step: Dict[str, Any], is_new_best: bool
    ) -> Optional[str]:
        reward = step["reward"]
        answer = step["info"].get("answer")
        best = self._best_reward

        if step["truncated"]:
            return (
                f"Reached the step limit. Current reward: {reward}. "
                f"Please improve in the next turn."
            )

        if answer is None:
            return (
                "No submit tool call was produced. Please use the available tools to "
                "solve the task, and call the submit tool when you have the final answer."
            )

        if self.done_mode == "threshold":
            if is_new_best:
                return (
                    f"Reach new best reward: {reward}. The threshold is "
                    f"{self.reward_threshold}. Keep improving."
                )
            return (
                f"Reward: {reward}, best so far: {best}. The threshold is "
                f"{self.reward_threshold}. Try to reach it."
            )

        if reward >= self.reward_threshold:
            return (
                f"Reward {reward} has reached the threshold (best so far: {best}). "
                f"Continue optimizing for an even better answer."
            )
        if is_new_best:
            return (
                f"New best reward: {reward} (threshold: {self.reward_threshold}). "
                f"Keep optimizing."
            )
        return (
            f"Reward: {reward}, best so far: {best} (threshold: "
            f"{self.reward_threshold}). Keep optimizing."
        )

    def _reset_env_step_count(self) -> None:
        reset = getattr(self.agent.env, "reset_step_count", None)
        if reset is not None:
            reset()


__all__ = ["ScribeRunner"]
