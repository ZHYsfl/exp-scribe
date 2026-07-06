from typing import Any, Dict, List, Optional, Tuple

from ..llm_runtime import Agent, LLMConfig, StructuredGenerator

from .helpers import (
    build_tool_messages,
    extract_tool_calls,
    message_to_scribe_blocks,
    strip_think_reflect,
)
from .history_manager import HistoryManager
from .rewards import (
    DEFAULT_TURN_REWARD_CONFIG,
    Judge,
    TurnOutcome,
    TurnRewardConfig,
    build_structured_judge,
    compute_turn_reward,
)
from .tool_calling_env import ToolCallingScribeEnv
from .turn_record import StepRecord, TurnRecord


def _count_submit_calls(step_records: List[StepRecord]) -> int:
    """Count submit tool calls across a turn's step records (by parsed name)."""
    n = 0
    for rec in step_records:
        for b in rec.output_blocks:
            p = getattr(b, "parsed", None)
            if isinstance(p, dict) and p.get("name") == "submit":
                n += 1
    return n


class GymBackedAgent(Agent):
    def __init__(
        self,
        config: LLMConfig,
        env: ToolCallingScribeEnv,
        max_tool_retries: int = 3,
        debug: bool = False,
        compressor: Any | None = None,
        history_manager: Optional[HistoryManager] = None,
        reward_config: Optional[TurnRewardConfig] = None,
        judge: Optional[Judge] = None,
        token_counter: Any = None,
        enable_judge: bool = False,
        judge_max_retries: int = 2,
        judge_temperature: float = 0.0,
        judge_semaphore: Optional[Any] = None,
    ):
        # SCRIBE mode does not use the base class's tool-error retry prompt.
        # Tool error content is already present in the tool response messages;
        # the model can adjust from there without an extra user message.
        super().__init__(
            config, max_tool_retries=0, debug=debug, compressor=compressor
        )
        self.env = env
        self.trajectory: List[Dict[str, Any]] = []
        # Hot-swap: history_manager=None -> plain ReAct (baseline, full history).
        # history_manager set -> SCRIBE per-turn cross-turn input rebuild.
        self.history_manager = history_manager
        # Per-turn step records (推理实况), populated by the SCRIBE chat loop.
        self._turn_step_records: List[StepRecord] = []
        # 13-metric turn reward config + token counter, injected so the runner/tests
        # can wire them without touching env correctness logic.
        self.reward_config = reward_config or DEFAULT_TURN_REWARD_CONFIG
        self.token_counter = token_counter
        # LLM-as-judge for metrics 7-10. Explicit judge wins; otherwise optionally
        # auto-build one from the agent's own client/model for parallel tasks.
        self.judge = judge
        if self.judge is None and enable_judge:
            judge_gen = StructuredGenerator(
                client=self.client,
                model=self.config.model,
                max_retries=judge_max_retries,
                temperature=judge_temperature,
                semaphore=judge_semaphore,
            )
            self.judge = build_structured_judge(judge_gen)

    def _get_tools(self) -> List[Dict[str, Any]]:
        return self.env.get_tools()

    async def chat(self, observations: list[dict]) -> list[dict]:
        # Hot-swap: no history_manager -> plain ReAct via the base Agent.chat.
        if self.history_manager is None:
            return await super().chat(observations)

        # CROSS-TURN: rebuild the input from prior turns (SCRIBE block retention).
        # The rebuilt input never contains T/R (hm.retained_messages are stripped).
        turn_input = self.history_manager.build_input()
        self._turn_step_records = []

        # WITHIN-TURN accumulation: we own the loop (do NOT delegate to
        # super().chat) so we can (a) strip T/R from each appended assistant
        # message (spec: T/R disposable next step — base class didn't strip),
        # and (b) capture the faithful per-step StepRecord (推理实况) so training
        # data has zero drift — step.input is exactly what the model saw.
        msgs = list(turn_input)
        while True:
            print("\033[94mThinking...\033[0m")
            response = await self.client.chat.completions.create(
                model=self.config.model,
                messages=msgs,
                tools=self._get_tools(),
                tool_choice="auto",
            )
            print(response.choices[0].message.content)
            message = response.choices[0].message.model_dump()
            _, output_blocks, _ = message_to_scribe_blocks(message)
            finish_reason = response.choices[0].finish_reason

            # Capture this step's faithful record (input = what model saw, which
            # is T/R-stripped; output = raw blocks the model generated, WITH T).
            # tool_messages is filled after execution for tool-call steps.
            rec = StepRecord(
                input_messages=[dict(m) for m in msgs],
                output_blocks=list(output_blocks),
                raw_assistant_message=dict(message),
            )
            self._turn_step_records.append(rec)

            if finish_reason != "tool_calls":
                # Final non-tool step: step the env (records trajectory), then
                # append the (T/R-stripped) final message — it carries S, which
                # future turns must see. End the turn.
                await self._step_env_for_message(message)
                msgs.append(strip_think_reflect(message))
                break

            # Tool-call step: execute tools, capture tool_messages on the record,
            # then append the (T/R-stripped) assistant message + tool responses
            # so the model sees results next step (T/R stripped: never re-read).
            tool_messages = await self._get_tool_response_observations(response)
            rec.tool_messages = list(tool_messages)
            msgs.append(strip_think_reflect(message))
            msgs.extend(tool_messages)

            if self._should_stop_after_tool_response(tool_messages):
                break

        return msgs

    async def finalize_turn(self, feedback: Optional[str]) -> Optional[TurnRecord]:
        """Package this turn into a TurnRecord, compute the 13-metric turn
        reward FROM the record, then hand it to the history_manager. Called by
        ScribeRunner._evaluate_turn when a turn ends. No-op when no
        history_manager is attached (plain ReAct).

        Order matters: the record is built FIRST (so its parsed blocks are
        available), THEN the 13-metric reward is computed from those blocks +
        the env trajectory's termination/answer info, and set onto
        record.reward. This is the inverse of the old (reward-then-record)
        order, which had nothing to compute the reward from.

        step_records is the single source of truth; input / new_blocks /
        rollout_blocks / retained_messages are all derived via properties."""
        if self.history_manager is None:
            return None
        record = TurnRecord(
            step_records=list(self._turn_step_records),
            reward=0.0,
            feedback=feedback,
        )
        record.reward = await self._compute_turn_reward(record)
        self.history_manager.add_turn(record)
        if feedback is not None:
            self.history_manager.set_feedback(self.history_manager.num_turns - 1, feedback)
        return record

    async def _compute_turn_reward(self, record: TurnRecord) -> float:
        """Build the TurnOutcome from the env trajectory + step records, compute
        the 13-metric reward, and return the scalar total."""
        oc = self._build_outcome(record)
        breakdown = await compute_turn_reward(
            record,
            self.reward_config,
            outcome=oc,
            judge=self.judge,
            token_counter=self.token_counter,
        )
        record.reward_breakdown = breakdown  # for debugging / ablation
        return breakdown.total

    def _build_outcome(self, record: TurnRecord) -> TurnOutcome:
        """Lift termination/answer info off the last env trajectory entry."""
        info: Dict[str, Any] = {}
        truncated = False
        terminated = False
        if self.trajectory:
            last = self.trajectory[-1]
            info = last.get("info", {}) or {}
            truncated = bool(last.get("truncated", False))
            terminated = bool(last.get("terminated", False))
        return TurnOutcome(
            truncated=truncated,
            terminated=terminated,
            done_reason=info.get("done_reason"),
            answer=info.get("answer"),
            right_answer=info.get("right_answer"),
            submit_count=_count_submit_calls(record.step_records),
        )

    async def _env_step(
        self, action: Any
    ) -> Tuple[str, float, bool, bool, Dict[str, Any]]:
        astep = getattr(self.env, "astep", None)
        if astep is not None:
            return await astep(action)
        return self.env.step(action)

    async def _step_env_for_message(
        self, message: Dict[str, Any]
    ) -> Tuple[str, List[Dict[str, Any]]]:
        action_text, blocks, is_valid = message_to_scribe_blocks(message)
        parsed_tool_calls = extract_tool_calls(blocks)

        action = parsed_tool_calls if parsed_tool_calls else action_text

        if self.debug:
            print("[Debug] env.step action:\n", action)

        obs, reward, terminated, truncated, info = await self._env_step(action)

        if self.debug:
            print(
                f"[Debug] reward={reward} terminated={terminated} truncated={truncated}"
            )

        self.trajectory.append({
            "action_text": action_text,
            "blocks": blocks,
            "is_valid": is_valid,
            "action": action,
            "observation": obs,
            "reward": reward,
            "terminated": terminated,
            "truncated": truncated,
            "info": info,
        })

        if not parsed_tool_calls:
            return obs, []

        message_tool_calls = message.get("tool_calls", [])
        tool_messages = build_tool_messages(message_tool_calls, info)
        return obs, tool_messages

    async def _get_tool_response_observations(self, response) -> List[Dict[str, Any]]:
        message = response.choices[0].message.model_dump()
        _, tool_messages = await self._step_env_for_message(message)
        return tool_messages

    def _should_stop_after_tool_response(
        self, tool_responses: List[Dict[str, Any]]
    ) -> bool:
        if not self.trajectory:
            return False
        last = self.trajectory[-1]
        return last["terminated"] or last["truncated"]


__all__ = ["GymBackedAgent"]
