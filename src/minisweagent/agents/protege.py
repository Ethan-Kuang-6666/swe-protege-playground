"""The protege agent."""

import json

from minisweagent import Environment, Model
from minisweagent.agents.default import AgentConfig, DefaultAgent
from minisweagent.exceptions import ExpertCallLimitsExceeded, Submitted
from minisweagent.models.expert_model import ExpertModel


class ProtegeAgentConfig(AgentConfig):
    expert_call_limit: int = 6
    expert_context_window: int = 10
    expert_system_template: str
    """Template for the expert system message (First expert message)"""
    expert_review_system_template: str
    """Template for the expert system during the review phase"""
    review_limit: int = 2
    review_buffer_steps: int = 10
    review_buffer_cost: float = 0.1
    """Cost headroom kept free so a review cannot exhaust the budget before the agent can resubmit."""


class ProtegeAgent(DefaultAgent):
    def __init__(
        self,
        model: Model,
        expert_model: ExpertModel,
        env: Environment,
        *,
        config_class: type = ProtegeAgentConfig,
        **kwargs,
    ):
        super().__init__(model, env, config_class=config_class, **kwargs)
        self.expert_model = expert_model
        self.expert_calls_used = 0
        self.rejection_count = 0

    def get_expert_context(self) -> str:
        messages_tail = self.messages[-self.config.expert_context_window :]
        context = [{"role": m.get("role", ""), "content": m.get("content", "")} for m in messages_tail]
        context_json = json.dumps(context, indent=2)
        task = self.extra_template_vars.get("task", "")
        return f"Task: \n{task}\n\nRecent agent's conversation in JSON:\n{context_json}"

    def ask_expert(self, question: str) -> dict:
        if self.expert_calls_used >= self.config.expert_call_limit:
            raise ExpertCallLimitsExceeded(
                {
                    "role": "exit",
                    "content": "ExpertCallLimitsExceeded",
                    "extra": {"exit_status": "ExpertCallLimitsExceeded", "submission": ""},
                }
            )

        self.expert_calls_used += 1
        context = self.get_expert_context()
        ground_truth_patch = self.extra_template_vars.get("ground_truth_patch", "")
        if ground_truth_patch:
            context = context + f"\n\n<ground_truth_patch>\n{ground_truth_patch}\n</ground_truth_patch>"
        expert_messages = [
            {"role": "system", "content": self.config.expert_system_template},
            {"role": "user", "content": context + "\n\nQuestion:\n" + question},
        ]
        response = self.expert_model.query(expert_messages)
        self.cost += response.get("extra", {}).get("cost", 0.0)
        answer = response.get("content", "")
        return {
            "output": f"<expert_llm_guidance>\n{answer}\n</expert_llm_guidance>",
            "returncode": 0,
            "exception_info": "",
            "extra": {"expert_usage": response.get("extra", {}).get("response", {}).get("usage")},
        }

    def review_patch(self, submission: str):
        task = self.extra_template_vars.get("task", "")
        expert_messages = [
            {"role": "system", "content": self.config.expert_review_system_template},
            {"role": "user", "content": f"Task:\n{task}\n\n<submitted_patch>\n{submission}\n</submitted_patch>"},
        ]
        response = self.expert_model.query(expert_messages)
        self.cost += response.get("extra", {}).get("cost", 0.0)
        verdict, _, reason = response.get("content", "").strip().partition("\n")
        accepted = not verdict.upper().lstrip("*# ").startswith("REJECT")
        return accepted, (reason.strip() or verdict), response.get("extra", {}).get("response", {}).get("usage")

    def get_template_vars(self, **kwargs) -> dict:
        return super().get_template_vars(expert_calls_used=self.expert_calls_used, **kwargs)

    def execute_actions(self, message: dict) -> list[dict]:
        """Execute actions in message, add observation messages, return them."""
        actions = message.get("extra", {}).get("actions", [])
        outputs = []
        for action in actions:
            if action.get("tool_name") == "ask_expert_llm":
                outputs.append(self.ask_expert(action["question"]))
            else:
                try:
                    outputs.append(self.env.execute(action))
                except Submitted as e:
                    if (
                        self.rejection_count >= self.config.review_limit
                        or self.config.step_limit <= self.n_calls + self.config.review_buffer_steps
                        or self.config.cost_limit <= self.cost + self.config.review_buffer_cost
                    ):
                        raise
                    verdict, reason, usage = self.review_patch(e.messages[0]["extra"]["submission"])
                    if verdict:
                        e.messages[0]["extra"]["review_usage"] = usage
                        raise
                    self.rejection_count += 1
                    outputs.append(
                        {
                            "output": f"<expert_llm_guidance>\nYour submission was rejected by the expert and was NOT "
                            f"submitted.\nReason:\n{reason}\n\nAddress the issue above, regenerate patch.txt, then "
                            f"submit again.\n</expert_llm_guidance>",
                            "returncode": 0,
                            "exception_info": "",
                            "extra": {"review_usage": usage},
                        }
                    )

        return self.add_messages(*self.model.format_observation_messages(message, outputs, self.get_template_vars()))
