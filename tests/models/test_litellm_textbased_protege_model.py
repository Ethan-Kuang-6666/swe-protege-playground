from unittest.mock import Mock, patch

import pytest

from minisweagent.exceptions import FormatError
from minisweagent.models.litellm_textbased_protege_model import LitellmTextbasedProtegeModel


def _mock_response(content: str, finish_reason: str = "stop"):
    mock_response = Mock()
    mock_message = Mock()
    mock_message.content = content
    mock_message.model_dump.return_value = {"role": "assistant", "content": content}
    mock_response.choices = [Mock(message=mock_message, finish_reason=finish_reason)]
    mock_response.model_dump.return_value = {"test": "response"}
    return mock_response


def test_parses_bash_action():
    model = LitellmTextbasedProtegeModel(model_name="gpt-4o", cost_tracking="ignore_errors")
    response = _mock_response("THOUGHT: list files\n\n```mswea_bash_command\nls -la\n```")
    assert model._parse_actions(response) == [{"tool_name": "bash", "command": "ls -la"}]


def test_parses_ask_expert_action():
    model = LitellmTextbasedProtegeModel(model_name="gpt-4o", cost_tracking="ignore_errors")
    response = _mock_response("THOUGHT: I'm stuck\n\n```mswea_ask_expert\nWhere is the bug?\n```")
    assert model._parse_actions(response) == [{"tool_name": "ask_expert_llm", "question": "Where is the bug?"}]


def test_raises_format_error_when_no_action():
    model = LitellmTextbasedProtegeModel(model_name="gpt-4o", cost_tracking="ignore_errors")
    with pytest.raises(FormatError):
        model._parse_actions(_mock_response("I don't know what to do."))


def test_raises_format_error_when_both_actions_present():
    model = LitellmTextbasedProtegeModel(model_name="gpt-4o", cost_tracking="ignore_errors")
    content = "```mswea_bash_command\nls\n```\n```mswea_ask_expert\nWhat now?\n```"
    with pytest.raises(FormatError):
        model._parse_actions(_mock_response(content))


def test_query_end_to_end_with_bash_action():
    model = LitellmTextbasedProtegeModel(model_name="gpt-4o", cost_tracking="ignore_errors")
    response = _mock_response("```mswea_bash_command\necho hi\n```")
    with patch("litellm.completion", return_value=response):
        message = model.query([{"role": "user", "content": "do something"}])
        assert message["extra"]["actions"] == [{"tool_name": "bash", "command": "echo hi"}]
