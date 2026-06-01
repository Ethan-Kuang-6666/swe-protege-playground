"""SWE-protege agent: DefaultAgent augmented with an ask_expert tool.

Reference: "SWE-protege: Teaching Novice AI Agents to Solve Real-World Software Engineering Tasks"
Budget defaults: 75 steps, $2 cost limit, 6 expert calls max.
"""

import json
import logging

import litellm
from jinja2 import StrictUndefined, Template
from pydantic import BaseModel

from minisweagent.agents.default import AgentConfig, DefaultAgent
from minisweagent.exceptions import FormatError
from minisweagent.models import GLOBAL_MODEL_STATS
from minisweagent.models.litellm_model import LitellmModel, LitellmModelConfig
from minisweagent.models.utils.actions_toolcall import ASK_EXPERT_TOOL, BASH_TOOL

logger = logging.getLogger("protege")


# ---------------------------------------------------------------------------
# Tool-call parsing for both bash + ask_expert
# ---------------------------------------------------------------------------


def parse_protege_actions(tool_calls: list, *, format_error_template: str) -> list[dict]:
    """Parse bash and ask_expert tool calls. Raises FormatError on unknown tool or bad args."""
    if not tool_calls:
        raise FormatError(
            {
                "role": "user",
                "content": Template(format_error_template, undefined=StrictUndefined).render(
                    error="No tool calls found in the response. Every response MUST include at least one tool call.",
                    actions=[],
                ),
                "extra": {"interrupt_type": "FormatError"},
            }
        )
    actions = []
    for tool_call in tool_calls:
        error_msg = ""
        args = {}
        try:
            args = json.loads(tool_call.function.arguments)
        except Exception as e:
            error_msg = f"Error parsing tool call arguments: {e}."

        name = tool_call.function.name
        if name == "bash":
            if not isinstance(args, dict) or "command" not in args:
                error_msg += "Missing 'command' argument in bash tool call."
            elif not error_msg:
                actions.append({"command": args["command"], "tool_call_id": tool_call.id})
        elif name == "ask_expert":
            if not isinstance(args, dict) or "question" not in args:
                error_msg += "Missing 'question' argument in ask_expert tool call."
            elif not error_msg:
                actions.append({"type": "ask_expert", "question": args["question"], "tool_call_id": tool_call.id})
        else:
            error_msg += f"Unknown tool '{name}'."

        if error_msg:
            raise FormatError(
                {
                    "role": "user",
                    "content": Template(format_error_template, undefined=StrictUndefined).render(
                        actions=[], error=error_msg.strip()
                    ),
                    "extra": {"interrupt_type": "FormatError"},
                }
            )
    return actions


# ---------------------------------------------------------------------------
# ProtegeModel — LitellmModel that exposes both bash and ask_expert
# ---------------------------------------------------------------------------


class ProtegeModel(LitellmModel):
    """LitellmModel variant that registers both bash and ask_expert as available tools."""

    def _query(self, messages, **kwargs):
        try:
            return litellm.completion(
                model=self.config.model_name,
                messages=messages,
                tools=[BASH_TOOL, ASK_EXPERT_TOOL],
                **(self.config.model_kwargs | kwargs),
            )
        except litellm.exceptions.AuthenticationError as e:
            e.message += " You can permanently set your API key with `mini-extra config set KEY VALUE`."
            raise e

    def _parse_actions(self, response) -> list[dict]:
        tool_calls = response.choices[0].message.tool_calls or []
        return parse_protege_actions(tool_calls, format_error_template=self.config.format_error_template)


# ---------------------------------------------------------------------------
# ProtegeAgent — DefaultAgent that routes ask_expert actions to an expert LLM
# ---------------------------------------------------------------------------


class ProtegeAgentConfig(AgentConfig):
    step_limit: int = 75
    cost_limit: float = 2.0
    expert_call_limit: int = 6
    expert_model_name: str = "anthropic/claude-opus-4-5"
    expert_model_kwargs: dict = {}


class ProtegeAgent(DefaultAgent):
    """DefaultAgent augmented with an ask_expert tool limited to 6 calls per run."""

    def __init__(self, model, env, *, config_class=ProtegeAgentConfig, **kwargs):
        super().__init__(model, env, config_class=config_class, **kwargs)
        self.expert_calls_used = 0
        self._expert_llm_config = LitellmModelConfig(
            model_name=self.config.expert_model_name,
            cost_tracking="ignore_errors",
            model_kwargs=self.config.expert_model_kwargs,
        )

    def _call_expert(self, question: str) -> dict:
        """Query the expert model. Returns an output dict compatible with the observation template."""
        if self.expert_calls_used >= self.config.expert_call_limit:
            return {
                "output": (
                    f"Expert call limit ({self.config.expert_call_limit}) has been reached. "
                    "No further expert calls are available."
                ),
                "returncode": 1,
                "exception_info": "",
            }
        self.expert_calls_used += 1
        task = self.extra_template_vars.get("task", "")
        messages = [
            {
                "role": "system",
                "content": (
                    "You are an expert software engineer. A junior AI agent is working on a software "
                    "engineering task and needs your guidance. Answer clearly and concisely. "
                    "You have no direct access to the environment, so give conceptual guidance."
                ),
            },
            {
                "role": "user",
                "content": f"Task:\n{task}\n\nAgent's question:\n{question}",
            },
        ]
        logger.info(
            "Expert call %d/%d: %s...",
            self.expert_calls_used,
            self.config.expert_call_limit,
            question[:80],
        )
        try:
            response = litellm.completion(
                model=self._expert_llm_config.model_name,
                messages=messages,
                **self._expert_llm_config.model_kwargs,
            )
            answer = response.choices[0].message.content or ""
            try:
                cost = litellm.cost_calculator.completion_cost(
                    response, model=self._expert_llm_config.model_name
                )
                self.cost += cost
                GLOBAL_MODEL_STATS.add(cost)
            except Exception:
                pass
        except Exception as e:
            answer = f"Expert call failed: {e}"
        return {"output": answer, "returncode": 0, "exception_info": ""}

    def execute_actions(self, message: dict) -> list[dict]:
        """Execute bash and ask_expert actions in order, return observation messages."""
        actions = message.get("extra", {}).get("actions", [])
        outputs = []
        for action in actions:
            if action.get("type") == "ask_expert":
                outputs.append(self._call_expert(action["question"]))
            else:
                outputs.append(self.env.execute(action))
        return self.add_messages(*self.model.format_observation_messages(message, outputs, self.get_template_vars()))

    def get_template_vars(self, **kwargs) -> dict:
        return super().get_template_vars(
            expert_calls_used=self.expert_calls_used,
            expert_call_limit=self.config.expert_call_limit,
            **kwargs,
        )

    def serialize(self, *extra_dicts) -> dict:
        data = super().serialize(*extra_dicts)
        data["info"]["model_stats"]["expert_calls_used"] = self.expert_calls_used
        data["info"]["model_stats"]["expert_call_limit"] = self.config.expert_call_limit
        return data
