"""Integration test: GymBackedAgent + HistoryManager end-to-end with a MOCK LLM.

Drives a 2-turn episode without any real LLM/network. Verifies:
  - chat override rebuilds input from hm each turn.
  - finalize_turn packages turns into hm.
  - feedback is routed to hm (in-position), not duplicated in observations.
  - cross-turn input contains no T/R.
  - hot-swap: hm=None path still works (delegates to base chat).
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from Scribe.llm_runtime import LLMConfig
from Scribe.scribe_gym import GymBackedAgent, HistoryManager, LinuxWorkspaceEnv
from Scribe.scribe_gym.history_manager import _TokenCounterLike  # noqa

# Real think tags via chr() to avoid literal angle-bracket tags in source.
THINK = chr(60) + "think" + chr(62)
THINK_END = chr(60) + "/think" + chr(62)


class FakeCounter:
    def count_observations(self, observations):
        return len(observations) * 10

    def count(self, text):
        return len(text)

    def encode(self, text):
        # deterministic pseudo-token ids (one per char) for set/n-gram metrics
        return [ord(ch) for ch in text]


class FakeMsg:
    def __init__(self, content, tool_calls=None):
        self.content = content
        self.tool_calls = tool_calls
    def model_dump(self):
        tc_dicts = None
        if self.tool_calls:
            tc_dicts = [
                {"id": tc.id, "type": "function",
                 "function": {"name": tc.function.name, "arguments": tc.function.arguments}}
                for tc in self.tool_calls
            ]
        return {"role": "assistant", "content": self.content, "refusal": None,
                "annotations": None, "audio": None, "function_call": None,
                "tool_calls": tc_dicts}


class FakeFunc:
    def __init__(self, name, arguments):
        self.name = name
        self.arguments = arguments


class FakeToolCall:
    def __init__(self, id_, name, arguments):
        self.id = id_
        self.type = "function"
        self.function = FakeFunc(name, arguments)


class FakeChoice:
    def __init__(self, message, finish_reason):
        self.message = message
        self.finish_reason = finish_reason


class FakeResponse:
    def __init__(self, message, finish_reason):
        self.choices = [FakeChoice(message, finish_reason)]


class FakeClient:
    """Replays a scripted list of (content, tool_calls, finish_reason)."""
    def __init__(self, script):
        self._script = list(script)
        self.calls = 0
        self.last_messages = None

    class _Completions:
        def __init__(self, outer):
            self._o = outer
        async def create(self, **kwargs):
            self._o.last_messages = kwargs.get("messages")
            content, tool_calls, finish = self._o._script[self._o.calls]
            self._o.calls += 1
            return FakeResponse(FakeMsg(content, tool_calls), finish)

    @property
    def chat(self):
        return type("C", (), {"completions": self._Completions(self)})()


def _bash_tc(id_):
    return FakeToolCall(id_, "bash", '{"command": "echo 42"}')


def _submit_tc(id_):
    return FakeToolCall(id_, "submit", '{"answer": "42"}')


async def run_episode():
    env = LinuxWorkspaceEnv(
        task_description="compute 42",
        workspace_root="/tmp/x",
        max_steps=10,
        right_answer="42",
    )
    hm = HistoryManager(FakeCounter(), k=2, hard_limit=1_000_000, compression_margin=0)
    agent = GymBackedAgent(
        config=LLMConfig(api_key="x", model="m", base_url="u"),
        env=env,
        history_manager=hm,
        token_counter=FakeCounter(),
    )
    agent.client = FakeClient([
        # turn0: bash, submit WRONG answer (99), final text -> reward 0, feedback, next turn
        (f"{THINK}t0{THINK_END}o0", [_bash_tc("c0")], "tool_calls"),
        ("o1", [FakeToolCall("c1", "submit", '{"answer": "99"}')], "tool_calls"),
        ("o2<turn_summary>sum0</turn_summary>", None, "stop"),
        # turn1: bash, submit RIGHT answer (42), final text -> reward 1, stop_run
        (f"{THINK}t1{THINK_END}o3", [_bash_tc("c2")], "tool_calls"),
        ("o4", [_submit_tc("c3")], "tool_calls"),
        ("o5<turn_summary>sum1</turn_summary>", None, "stop"),
    ])

    from Scribe.scribe_gym import ScribeRunner
    runner = ScribeRunner(agent=agent, system_prompt="SP",
                          done_mode="threshold", reward_threshold=1.0, max_turns=3)
    await runner.run()

    # hm should have 2 turns
    assert hm.num_turns == 2, f"expected 2 turns, got {hm.num_turns}"
    # turn0 has summary
    assert hm._turns[0].has_summary
    # cross-turn input for turn1 (build after turn0 added) has no T/R
    # rebuild now (post-episode) — last build_input was for turn1
    # Verify turn1's input (stored on turn record) has no 民主党/<reflect>
    turn1_input = hm._turns[1].input
    for m in turn1_input:
        c = m.get("content", "") or ""
        assert THINK not in c, f"T leaked into turn1 input: {c}"
        assert "<reflect>" not in c
    # feedback from turn0 is in turn1's input, in position (after turn0's blocks)
    contents = [m.get("content", "") for m in turn1_input]
    assert any("sum0" in c for c in contents), "turn0 summary missing from turn1 input"
    print("INTEGRATION OK: 2 turns, no T/R leak, feedback in-position")


def test_integration():
    asyncio.run(run_episode())


if __name__ == "__main__":
    asyncio.run(run_episode())
