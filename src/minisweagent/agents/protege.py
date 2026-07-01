"""The protege agent."""

import json

from minisweagent import Environment, Model
from minisweagent.agents.default import AgentConfig, DefaultAgent
from minisweagent.exceptions import ExpertCallLimitsExceeded
from minisweagent.models.expert_model import ExpertModel


class ProtegeAgentConfig(AgentConfig):
    expert_call_limit: int
    expert_context_window: int = 10
    expert_system_template: str
    """Template for the expert system message (First expert message)"""


class ProtegeAgent(DefaultAgent):
    def __init__(
        self,
        model: Model,
        experts: list[ExpertModel],
        env: Environment,
        *,
        config_class: type = ProtegeAgentConfig,
        **kwargs,
    ):
        super().__init__(model, env, config_class=config_class, **kwargs)
        self.experts = experts
        self.expert_calls_used = 0
        self.expert_index = 0
        self.current_failed_commands: set[str] = set()
        self.previous_failed_l1_commands: set[str] = set()

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
        response = self.experts[self.expert_index].query(expert_messages)
        self.cost += response.get("extra", {}).get("cost", 0.0)
        answer = response.get("content", "")
        if self.expert_index == 0:
            self.previous_failed_l1_commands = set(self.current_failed_commands)
        else:
            self.previous_failed_l1_commands = set()
            self.expert_index = 0
        self.current_failed_commands = set()

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
        escalate = False
        for action in actions:
            if action.get("tool_name") == "ask_expert_llm":
                outputs.append(self.ask_expert(action["question"]))
            else:
                output = self.env.execute(action)
                if output["returncode"] != 0:
                    self.current_failed_commands.add(action["command"])
                    if action["command"] in self.previous_failed_l1_commands:
                        escalate = True
                outputs.append(output)
        if escalate:
            self.expert_index = min(self.expert_index + 1, len(self.experts) - 1)
        return self.add_messages(*self.model.format_observation_messages(message, outputs, self.get_template_vars()))
