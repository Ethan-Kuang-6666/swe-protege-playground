"""Parsing of the protege tool calls, and the expert tool's advertised parameters."""

import json
from types import SimpleNamespace

import pytest

from minisweagent.exceptions import FormatError
from minisweagent.models.utils.actions_toolcall import ASK_EXPERT_LLM_TOOL, parse_protege_toolcall_actions

TEMPLATE = "{{error}}"


def call(name: str, **args) -> SimpleNamespace:
    return SimpleNamespace(id="call_1", function=SimpleNamespace(name=name, arguments=json.dumps(args)))


def parse(*calls) -> list[dict]:
    return parse_protege_toolcall_actions(list(calls), format_error_template=TEMPLATE)


def test_the_expert_tool_advertises_only_a_question():
    assert list(ASK_EXPERT_LLM_TOOL["function"]["parameters"]["properties"]) == ["question"]
    assert ASK_EXPERT_LLM_TOOL["function"]["parameters"]["required"] == ["question"]


@pytest.mark.parametrize(
    ("tool_call", "expected"),
    [
        (call("bash", command="ls"), {"tool_name": "bash", "command": "ls", "tool_call_id": "call_1"}),
        (
            call("ask_expert_llm", question="why?"),
            {"tool_name": "ask_expert_llm", "question": "why?", "tool_call_id": "call_1"},
        ),
        # an argument the model invented is dropped, not rejected: it must not cost the student a step
        (
            call("ask_expert_llm", question="why?", budget_tokens=99999, nonsense=True),
            {"tool_name": "ask_expert_llm", "question": "why?", "tool_call_id": "call_1"},
        ),
    ],
)
def test_actions_parse_and_unknown_arguments_are_ignored(tool_call, expected):
    assert parse(tool_call) == [expected]


@pytest.mark.parametrize(
    ("tool_call", "message"),
    [
        (call("ask_expert_llm"), "Missing 'question' argument"),
        (call("bash"), "Missing 'command' argument"),
        (call("something_else", question="why?"), "Unknown tool 'something_else'"),
    ],
)
def test_malformed_calls_are_rejected(tool_call, message):
    with pytest.raises(FormatError) as excinfo:
        parse(tool_call)
    assert message in excinfo.value.messages[0]["content"]


def test_no_tool_call_at_all_is_rejected():
    with pytest.raises(FormatError) as excinfo:
        parse()
    assert "MUST include at least one tool call" in excinfo.value.messages[0]["content"]
