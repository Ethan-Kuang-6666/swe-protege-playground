import time

from minisweagent import Environment, Model
from minisweagent.agents.default import AgentConfig, DefaultAgent
from minisweagent.exceptions import LimitsExceeded, TimeExceeded


class UncertaintyAgentConfig(AgentConfig):
    expert_call_limit: int
    cutoff: float
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
        self.avglogp = None

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

        if (
            self.avglogp is not None
            and self.avglogp >= self.config.cutoff
            and self.expert_calls_used < self.config.expert_call_limit
        ):
            message = self.expert_model.query(self.get_context())
            self.expert_calls_used += 1
        else:
            message = self.model.query(self.get_context())
        self.cost += message.get("extra", {}).get("cost", 0.0)
        self.add_messages(message)

        self.avglogp = compute_avglogp(message)

        return message

    def get_context(self) -> list[dict]:
        """Messages as sent to the model: all but the last n observations have their content elided."""
        observations = [i for i, m in enumerate(self.messages) if i > 1 and m["role"] == "user"]
        elided = set(observations[: -self.config.last_n_observations])
        return [{**m, "content": "(lines omitted)"} if i in elided else m for i, m in enumerate(self.messages)]


def compute_avglogp(message: dict) -> float | None:
    tokens = (message["extra"]["response"]["choices"][0].get("logprobs") or {}).get("content") or []
    return sum(t["logprob"] for t in tokens) / len(tokens) if tokens else None
