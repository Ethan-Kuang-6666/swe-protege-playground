import re

import litellm
from jinja2 import StrictUndefined, Template

from minisweagent.exceptions import FormatError
from minisweagent.models.litellm_model import LitellmModel, LitellmModelConfig
from minisweagent.models.utils.actions_text import format_observation_messages


class LitellmTextbasedProtegeModelConfig(LitellmModelConfig):
    bash_regex: str = r"```mswea_bash_command\s*\n(.*?)\n```"
    """Regex to extract a bash action from the LM's output."""
    ask_expert_regex: str = r"```mswea_ask_expert\s*\n(.*?)\n```"
    """Regex to extract an ask_expert_llm question from the LM's output."""
    format_error_template: str = (
        "Please provide EXACTLY ONE action: either a bash command in a ```mswea_bash_command``` block, "
        "or a question for the expert in a ```mswea_ask_expert``` block. Found {{actions|length}} actions."
    )


class LitellmTextbasedProtegeModel(LitellmModel):
    def __init__(self, **kwargs):
        super().__init__(config_class=LitellmTextbasedProtegeModelConfig, **kwargs)

    def _query(self, messages: list[dict[str, str]], **kwargs):
        try:
            return litellm.completion(
                model=self.config.model_name, messages=messages, **(self.config.model_kwargs | kwargs)
            )
        except litellm.exceptions.AuthenticationError as e:
            e.message += " You can permanently set your API key with `mini-extra config set KEY VALUE`."
            raise e

    def _parse_actions(self, response) -> list[dict]:
        """Parse a bash or ask_expert_llm action from the model response. Raises FormatError if not exactly one."""
        content = response.choices[0].message.content or ""
        bash_matches = [m.strip() for m in re.findall(self.config.bash_regex, content, re.DOTALL)]
        expert_matches = [m.strip() for m in re.findall(self.config.ask_expert_regex, content, re.DOTALL)]
        actions = bash_matches + expert_matches
        if len(actions) != 1:
            error = f"Expected exactly 1 action, found {len(actions)}."
            raise FormatError(
                {
                    "role": "user",
                    "content": Template(self.config.format_error_template, undefined=StrictUndefined).render(
                        actions=actions, error=error, finish_reason=response.choices[0].finish_reason
                    ),
                    "extra": {
                        "interrupt_type": "FormatError",
                        "n_actions": len(actions),
                        "model_response": content,
                    },
                }
            )
        if bash_matches:
            return [{"tool_name": "bash", "command": bash_matches[0]}]
        return [{"tool_name": "ask_expert_llm", "question": expert_matches[0]}]

    def format_observation_messages(
        self, message: dict, outputs: list[dict], template_vars: dict | None = None
    ) -> list[dict]:
        """Format execution outputs into observation messages."""
        return format_observation_messages(
            outputs,
            observation_template=self.config.observation_template,
            template_vars=template_vars,
            multimodal_regex=self.config.multimodal_regex,
        )
