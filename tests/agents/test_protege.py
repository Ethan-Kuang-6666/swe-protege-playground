"""Expert calls in the protege agent: routing, and what happens once the budget is spent."""

from minisweagent.agents.protege import ProtegeAgent
from minisweagent.environments.local import LocalEnvironment
from minisweagent.models.test_models import DeterministicModel, make_output

GUIDANCE = [{"role": "assistant", "content": "look at loader.py", "extra": {"cost": 0.25}}] * 5


def agent(**overrides) -> ProtegeAgent:
    return ProtegeAgent(
        DeterministicModel(outputs=[make_output("s", [{"tool_name": "bash", "command": "true"}])] * 5),
        DeterministicModel(outputs=GUIDANCE),
        LocalEnvironment(),
        system_template="SYSTEM",
        instance_template="TASK",
        expert_system_template="EXPERT SYSTEM",
        **({"expert_call_limit": 2} | overrides),
    )


def test_guidance_comes_back_tagged_and_billed():
    a = agent()
    assert a.ask_expert("why?") == {
        "output": "<expert_llm_guidance>\nlook at loader.py\n</expert_llm_guidance>",
        "returncode": 0,
        "exception_info": "",
    }
    assert (a.expert_calls_used, a.cost) == (1, 0.25)


def test_the_episode_continues_once_the_budget_is_spent():
    a = agent(expert_call_limit=2)
    replies = [a.ask_expert("q")["output"] for _ in range(4)]

    # the first two are answered, the rest are refused without raising and without billing
    assert [r.startswith("<expert_llm_guidance>") for r in replies] == [True, True, False, False]
    # the refusal is not dressed as advice, and it says the budget is gone rather than that something broke
    assert replies[-1] == (
        "<expert_llm_unavailable>\nYou have used all 2 of your expert calls, so no more are available for "
        "this task. Continue on your own and submit your best solution.\n</expert_llm_unavailable>"
    )
    assert (a.expert_calls_used, a.cost) == (2, 0.5)


def test_a_refusal_does_not_end_the_run_or_set_an_exit_status():
    a = agent(expert_call_limit=0)
    a.messages = [{"role": "system", "content": "SYSTEM"}, {"role": "user", "content": "TASK"}]
    observations = a.execute_actions(
        {"role": "assistant", "content": "t", "extra": {"actions": [{"tool_name": "ask_expert_llm", "question": "q"}]}}
    )
    assert [m["role"] for m in observations] == ["user"]
    assert "<expert_llm_unavailable>" in observations[-1]["content"]


def test_bash_actions_still_reach_the_shell_and_the_expert_is_left_alone():
    a = agent()
    a.messages = []
    observations = a.execute_actions(
        {"role": "assistant", "content": "t", "extra": {"actions": [{"tool_name": "bash", "command": "echo hi"}]}}
    )
    assert "hi" in observations[-1]["content"]
    assert a.expert_calls_used == 0
