"""The protege agent."""

import json

from minisweagent import Environment, Model
from minisweagent.agents.default import AgentConfig, DefaultAgent
from minisweagent.models.expert_model import ExpertModel


class ProtegeAgentConfig(AgentConfig):
    expert_call_limit: int
    expert_context_window: int = 10
    expert_system_template: str
    """Template for the expert system message (First expert message)"""
    expert_quota_template: str = (
        "You have used all {{expert_call_limit}} of your expert calls, so no more are available for this task. "
        "Continue on your own and submit your best solution."
    )
    """Returned in place of guidance once the call limit is reached. The episode continues: spending the budget
    is not a failure, and ending the run here would make over-asking and under-asking look the same."""


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

    def get_expert_context(self) -> str:
        messages_tail = self.messages[-self.config.expert_context_window :]
        context = [{"role": m.get("role", ""), "content": m.get("content", "")} for m in messages_tail]
        context_json = json.dumps(context, indent=2)
        task = self.extra_template_vars.get("task", "")
        return f"Task: \n{task}\n\nRecent agent's conversation in JSON:\n{context_json}"

    def ask_expert(self, question: str) -> dict:
        if self.expert_calls_used >= self.config.expert_call_limit:
            notice = self._render_template(self.config.expert_quota_template)
            # Deliberately not <expert_llm_guidance>, so the student cannot read the notice as advice.
            return {
                "output": f"<expert_llm_unavailable>\n{notice}\n</expert_llm_unavailable>",
                "returncode": 0,
                "exception_info": "",
            }

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
        }

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
                outputs.append(self.env.execute(action))

        return self.add_messages(*self.model.format_observation_messages(message, outputs, self.get_template_vars()))
