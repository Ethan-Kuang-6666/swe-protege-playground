import time

from minisweagent import Environment, Model
from minisweagent.agents.default import AgentConfig, DefaultAgent
from minisweagent.exceptions import LimitsExceeded, TimeExceeded

EXPERT_HEADER = "SENIOR ENGINEER"
"""Prepended to the expert's reply. mini stores it as an ordinary assistant turn, so without a label
the student reads the diagnosis as its own earlier thought. Asking the expert for the label instead
worked on 6 of 19 turns, so it is applied here. The student's prompt must use the same words."""


class UncertaintyAgentConfig(AgentConfig):
    expert_call_limit: int
    """Total steps the expert may take in an episode, across all handovers."""
    expert_steps: int = 1
    """Consecutive steps the expert takes each time it is handed control."""
    expert_system_prompt: str = ""
    """Replaces the system message on the expert's calls only, so it must restate the response format.
    Without it the expert is addressed by the student's prompt and answers as the student would,
    exploring rather than fixing."""
    cutoff: float
    confident_steps: int = 1
    """Consecutive steps at or above `cutoff` before handing over. One step is weak evidence: the model
    is legitimately certain about an obvious action, such as running a script it has just written."""
    last_n_observations: int
    """Only the last n observations are sent to the model in full; the content of older ones is replaced by a placeholder."""


class UncertaintyAgent(DefaultAgent):
    def __init__(
        self,
        model: Model,
        expert_model: Model,
        env: Environment,
        *,
        config_class: type = UncertaintyAgentConfig,
        **kwargs,
    ):
        super().__init__(model, env, config_class=config_class, **kwargs)
        self.expert_model = expert_model
        self.expert_calls_used = 0
        self.expert_steps_left = 0
        self.avglogp = None
        self.streak = 0

    def query(self) -> dict:
        """Query the model and return model messages. Override to add hooks."""
        if 0 < self.config.step_limit <= self.n_calls or 0 < self.config.cost_limit <= self.cost:
            raise LimitsExceeded(
                {
                    "role": "exit",
                    "content": "LimitsExceeded",
                    "extra": {"exit_status": "LimitsExceeded", "submission": ""},
                }
            )
        if 0 < self.config.wall_time_limit_seconds <= int(time.time() - self._start_time):
            raise TimeExceeded(
                {
                    "role": "exit",
                    "content": "TimeExceeded",
                    "extra": {"exit_status": "TimeExceeded", "submission": ""},
                }
            )
        self.n_calls += 1

        if self.expert_steps_left == 0 and self.streak >= self.config.confident_steps:
            self.expert_steps_left = self.config.expert_steps
        if self.expert_steps_left and self.expert_calls_used < self.config.expert_call_limit:
            message = self.expert_model.query(self.get_context(self.config.expert_system_prompt))
            message = {**message, "content": f"{EXPERT_HEADER}\n{message['content']}"}
            self.expert_calls_used += 1
            self.expert_steps_left -= 1
            # The expert's logprobs are its own, not the student's, and its step is the fresh start the
            # streak was counting towards, so the student begins again from the next step.
            self.avglogp, self.streak = None, 0
        else:
            self.expert_steps_left = 0  # the budget ran out mid-handover
            message = self.model.query(self.get_context())
            self.avglogp = compute_avglogp(message)
            self.streak = self.streak + 1 if self.avglogp is not None and self.avglogp >= self.config.cutoff else 0
        self.cost += message.get("extra", {}).get("cost", 0.0)
        self.add_messages(message)

        return message

    def get_context(self, system_prompt: str = "") -> list[dict]:
        """Messages as sent to the model: all but the last n observations have their content elided.

        `system_prompt` replaces the system message rather than extending it, so the expert is not
        simultaneously addressed as the agent. It must therefore restate the response format.
        """
        observations = [i for i, m in enumerate(self.messages) if i > 1 and m["role"] == "user"]
        elided = set(observations[: -self.config.last_n_observations])
        messages = [{**m, "content": "(lines omitted)"} if i in elided else m for i, m in enumerate(self.messages)]
        if system_prompt:
            messages[0] = {**messages[0], "content": system_prompt}
        return messages


def compute_avglogp(message: dict) -> float | None:
    tokens = (message["extra"]["response"]["choices"][0].get("logprobs") or {}).get("content") or []
    return sum(t["logprob"] for t in tokens) / len(tokens) if tokens else None
