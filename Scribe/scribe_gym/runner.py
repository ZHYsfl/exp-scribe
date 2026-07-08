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
        tokenizer: Optional[Any] = None,
    ):
        self.agent = agent
        self.system_prompt = system_prompt
        self.done_mode = done_mode
        self.reward_threshold = reward_threshold
        self.max_turns = max_turns
        self.tokenizer = tokenizer
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
                await self._finalize_current_turn(result)
                break
            await self._finalize_current_turn(result)
            if result.feedback is not None:
                # SCRIBE mode: feedback lives in the history_manager (set in
                # finalize); do NOT append to observations (would duplicate).
                # Plain ReAct: append to observations as before.
                hm = getattr(self.agent, "history_manager", None)
                if hm is None:
                    observations.append({"role": "user", "content": result.feedback})
        return observations

    async def _finalize_current_turn(self, result) -> None:
        """Hand the just-ended turn to the history_manager (no-op in ReAct).
        The 15-metric turn reward is computed INSIDE finalize_turn from the
        freshly-built TurnRecord (block contents) + the env trajectory's
        terminated/truncated/answer info, so the runner does NOT pass a scalar
        reward here — only the stop_run/feedback signaling."""
        finalize = getattr(self.agent, "finalize_turn", None)
        if finalize is None:
            return
        await finalize(result.feedback if result else None)

    async def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        obs, info = self.agent.env.reset(seed=seed, options=options)
        self.agent.trajectory.clear()
        self._best_reward = float("-inf")
        messages = self._build_initial_observations(obs, tokenizer=self.tokenizer)
        # Seed the history_manager prefix (system + kickoff) for a fresh episode.
        hm = getattr(self.agent, "history_manager", None)
        if hm is not None:
            hm.reset()
            system_msg = messages[0] if messages and messages[0].get("role") == "system" else None
            kickoff_msg = None
            for m in messages:
                if m.get("role") == "user":
                    kickoff_msg = m
                    break
            if system_msg is not None and kickoff_msg is not None:
                hm.set_prefix(system_msg, kickoff_msg)
        return messages

    def _build_system_prompt(
        self, task_description: Optional[str] = None, tokenizer: Optional[Any] = None
    ) -> str:
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
        rendered = render_messages(
            messages, tokenizer=tokenizer, tools=[], add_generation_prompt=False
        )
        start = rendered.find("<|im_start|>system\n")
        if start == -1:
            return ""
        start += len("<|im_start|>system\n")
        end = rendered.find("<|im_end|>", start)
        if end == -1:
            return rendered[start:]
        return rendered[start:end]

    def _build_initial_observations(
        self, task_description: str, tokenizer: Optional[Any] = None
    ) -> List[Dict[str, Any]]:
        messages: List[Dict[str, Any]] = []
        system_content = self._build_system_prompt(task_description, tokenizer=tokenizer)
        if system_content:
            messages.append({"role": "system", "content": system_content})
        # The task description is merged into the system prompt; the first user
        # message is the turn kickoff. We deliberately do NOT reveal the turn
        # number or max_turns: horizon awareness biases the model's behavior
        # (e.g. gambling when it thinks few turns remain) and contaminates the
        # training distribution. All later user messages are the feedback
        # produced by _evaluate_turn.
        kickoff = (
            "Please start solving the problem. Work within this turn: use tools, "
            "call submit exactly once with your final answer, then end with a final "
            "plain-text message containing <reflect> followed by <turn_summary>."
        )
        messages.append({"role": "user", "content": kickoff})
        return messages

    def _evaluate_turn(
        self, step: Dict[str, Any]
    ) -> Optional[TurnResult]:
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
        # Stop immediately when the submitted answer is correct, even if the
        # scalar reward is below the threshold because of repeated-submit or
        # other shaping penalties. The 15-metric turn reward is still computed
        # from the full record, but the episode should not continue once the
        # task is solved.
        info = step.get("info") or {}
        answer = info.get("answer")
        right_answer = info.get("right_answer")
        if answer is not None and right_answer is not None:
            if answer.strip() == right_answer.strip():
                return True
        return self.done_mode == "threshold" and step["reward"] >= self.reward_threshold

    def _build_feedback(
        self,
        step: Dict[str, Any],
        is_new_best: bool,
    ) -> Optional[str]:
        # NOTE: feedback must NOT reveal the turn number or max_turns. Horizon
        # awareness biases the model (gambling when it thinks few turns remain)
        # and contaminates the RL training distribution. Keep feedback to
        # reward/best/threshold + the per-turn submit rule only.
        reward = step["reward"]
        answer = step["info"].get("answer")
        right_answer = step["info"].get("right_answer")
        best = self._best_reward

        submit_note = (
            " Repeated submits in the same turn are penalized — verify with "
            "tools first, then submit once."
        )

        # Explicitly tell the model whether its submitted answer is correct so
        # it does not mistake a wrong-answer rollout for a formatting problem.
        answer_note = ""
        if answer is not None and right_answer is not None:
            if answer.strip() == right_answer.strip():
                answer_note = f" Your submitted answer '{answer}' is CORRECT."
            else:
                answer_note = f" Your submitted answer '{answer}' is INCORRECT."

        if step["truncated"]:
            return (
                f"Reached the step limit. Current reward: {reward}.{answer_note} Improve in "
                f"the next turn. Remember: a turn is one solving attempt from "
                f"kickoff to your final submit; call submit at most once per turn."
            )

        if answer is None:
            return (
                "No submit tool call was produced this turn. Use the available "
                "tools to solve the task, then call submit exactly once with the "
                "final answer."
            )

        if self.done_mode == "threshold":
            if is_new_best:
                return (
                    f"New best reward: {reward}.{answer_note} Threshold is "
                    f"{self.reward_threshold}. Keep improving.{submit_note}"
                )
            return (
                f"Reward: {reward}, best so far: {best}.{answer_note} Threshold is "
                f"{self.reward_threshold}. Try to reach it.{submit_note}"
            )

        if reward >= self.reward_threshold:
            return (
                f"Reward {reward} reached the threshold (best: {best}).{answer_note} "
                f"Continue optimizing.{submit_note}"
            )
        if is_new_best:
            return (
                f"New best reward: {reward} (threshold: "
                f"{self.reward_threshold}).{answer_note} Keep optimizing.{submit_note}"
            )
        return (
            f"Reward: {reward}, best so far: {best} (threshold: "
            f"{self.reward_threshold}).{answer_note} Keep optimizing.{submit_note}"
        )

    def _reset_env_step_count(self) -> None:
        reset = getattr(self.agent.env, "reset_step_count", None)
        if reset is not None:
            reset()


__all__ = ["ScribeRunner"]
