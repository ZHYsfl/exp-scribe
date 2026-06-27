from dataclasses import dataclass, field
from enum import Enum


class Phase(str, Enum):
    REASON = "ReasonPhase"
    ACTION = "ActionPhase"
    OBSERVATION = "ObservationPhase"
    REFLECTION = "ReflectionPhase"
    SUMMARY = "SummaryPhase"


class Event(str, Enum):
    THINK = "Think"
    ACT = "Act"
    OBSERVE = "Observe"
    REFLECT = "Reflect"
    SUMMARIZE = "Summarize"


# SCRIBE state machine (README §"实验2"): each step is R A O Reflect S, and the
# only way back to Reason is Summary --Think--> Reason. This table is the
# authoritative SCRIBE definition and is kept exactly as the README specifies.
SCRIBE_TRANSITIONS = {
    (Phase.REASON, Event.ACT): Phase.ACTION,
    (Phase.REASON, Event.OBSERVE): Phase.OBSERVATION,
    (Phase.ACTION, Event.OBSERVE): Phase.OBSERVATION,
    (Phase.OBSERVATION, Event.ACT): Phase.ACTION,
    (Phase.OBSERVATION, Event.REFLECT): Phase.REFLECTION,
    (Phase.REFLECTION, Event.SUMMARIZE): Phase.SUMMARY,
    (Phase.SUMMARY, Event.THINK): Phase.REASON,
}

# Plain React state machine: each step is R A O, and Think closes the step and
# returns to the next Reason. No Reflect/Summary phases.
REACT_TRANSITIONS = {
    (Phase.REASON, Event.ACT): Phase.ACTION,
    (Phase.REASON, Event.OBSERVE): Phase.OBSERVATION,
    (Phase.ACTION, Event.OBSERVE): Phase.OBSERVATION,
    (Phase.OBSERVATION, Event.ACT): Phase.ACTION,
    (Phase.OBSERVATION, Event.THINK): Phase.REASON,
}

# Backwards-compatible default alias (SCRIBE is the full model).
TRANSITIONS = SCRIBE_TRANSITIONS


@dataclass(frozen=True)
class TokenUsage:
    prompt: int = 0
    completion: int = 0

    @property
    def total(self) -> int:
        return self.prompt + self.completion

    def __add__(self, other: "TokenUsage") -> "TokenUsage":
        return TokenUsage(
            prompt=self.prompt + other.prompt,
            completion=self.completion + other.completion,
        )


@dataclass(frozen=True)
class PhaseRecord:
    source_phase: Phase
    input_event: Event
    target_phase: Phase
    content: str = ""
    token_usage: TokenUsage = TokenUsage()


@dataclass
class Step:
    index: int
    phase_records: list[PhaseRecord] = field(default_factory=list)

    def add(self, phase_record: PhaseRecord) -> None:
        self.phase_records.append(phase_record)

    def token_usage(self, *phases: Phase) -> TokenUsage:
        selected = set(phases)
        total = TokenUsage()
        for phase_record in self.phase_records:
            # If no phase arguments are provided, include all phases by default.
            if not selected or phase_record.source_phase in selected:
                total += phase_record.token_usage
        return total

    @property
    def reflection_tokens(self) -> TokenUsage:
        return self.token_usage(Phase.REFLECTION)

    @property
    def summary_tokens(self) -> TokenUsage:
        return self.token_usage(Phase.SUMMARY)

    @property
    def reflection_summary_tokens(self) -> TokenUsage:
        return self.token_usage(Phase.REFLECTION, Phase.SUMMARY)


@dataclass
class Trajectory:
    steps: list[Step] = field(default_factory=list)

    def add_step(self, step: Step) -> None:
        self.steps.append(step)

    def token_usage(self, *phases: Phase) -> TokenUsage:
        total = TokenUsage()
        for step in self.steps:
            total += step.token_usage(*phases)
        return total

    @property
    def reflection_tokens(self) -> TokenUsage:
        return self.token_usage(Phase.REFLECTION)

    @property
    def summary_tokens(self) -> TokenUsage:
        return self.token_usage(Phase.SUMMARY)

    @property
    def reflection_summary_tokens(self) -> TokenUsage:
        return self.token_usage(Phase.REFLECTION, Phase.SUMMARY)


class InvalidTransition(ValueError):
    pass


class ScribeStateMachine:
    def __init__(
        self,
        phase: Phase = Phase.REASON,
        transitions: dict | None = None,
    ) -> None:
        self.phase = phase
        # Which transition table governs this machine. Defaults to the full
        # SCRIBE model; pass REACT_TRANSITIONS for a plain React agent.
        self.transitions = transitions if transitions is not None else TRANSITIONS
        self.trajectory = Trajectory()
        self.current_step = Step(index=0)

    def can_apply(self, event: Event) -> bool:
        return (self.phase, event) in self.transitions

    def next_phase(self, event: Event) -> Phase:
        if not self.can_apply(event):
            raise InvalidTransition(f"{self.phase.value} cannot handle {event.value}")
        return self.transitions[(self.phase, event)]

    def apply(
        self,
        event: Event,
        content: str = "",
        token_usage: TokenUsage | None = None,
    ) -> Phase:
        source_phase = self.phase
        target_phase = self.next_phase(event)
        self.current_step.add(PhaseRecord(
            source_phase=source_phase,
            input_event=event,
            target_phase=target_phase,
            content=content,
            token_usage=token_usage or TokenUsage(),
        ))
        self.phase = target_phase

        if event == Event.THINK:
            self.trajectory.add_step(self.current_step)
            self.current_step = Step(index=self.current_step.index + 1)

        return self.phase

    def act(self, content: str = "", token_usage: TokenUsage | None = None) -> Phase:
        return self.apply(Event.ACT, content, token_usage)

    def observe(self, content: str = "", token_usage: TokenUsage | None = None) -> Phase:
        return self.apply(Event.OBSERVE, content, token_usage)

    def reflect(self, content: str = "", token_usage: TokenUsage | None = None) -> Phase:
        return self.apply(Event.REFLECT, content, token_usage)

    def summarize(self, content: str = "", token_usage: TokenUsage | None = None) -> Phase:
        return self.apply(Event.SUMMARIZE, content, token_usage)

    def think(self, content: str = "", token_usage: TokenUsage | None = None) -> Phase:
        return self.apply(Event.THINK, content, token_usage)

    @property
    def last_completed_step(self) -> Step | None:
        if not self.trajectory.steps:
            return None
        return self.trajectory.steps[-1]

    @property
    def last_reflection_summary_tokens(self) -> TokenUsage:
        step = self.last_completed_step
        if step is None:
            return TokenUsage()
        return step.reflection_summary_tokens
