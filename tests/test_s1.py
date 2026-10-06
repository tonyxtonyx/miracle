import json

from miracle.agents.s0 import S0Runtime
from miracle.agents.s1 import LENGTH_NUDGE, S1Runtime
from miracle.llm.loop import run_tool_loop
from test_llm import Scripted
from test_s0 import LocalEnv, _task  # noqa: F401  (fixtures: env)
from test_s0 import env  # noqa: F401

EXACT = ("Your previous response was truncated by the output token limit. Continue the investigation from where "
         "you stopped. Use the available tools when appropriate.")


class Tools:
    specs = []

    def call(self, name, args):
        return "ok", False


def loop(turns, nudge=EXACT, max_steps=10):
    return run_tool_loop(Scripted(turns), [{"role": "user", "content": "task"}], Tools(), max_steps, {}, length_nudge=nudge)


def test_nudge_text_is_exactly_the_specified_message():
    assert LENGTH_NUDGE == EXACT


def test_truncated_reply_without_tool_call_is_not_the_end():
    trace, usage = loop([("long analysis...", [], "length"), (None, [("list_files", {"path": "."})]), ("done", [])])
    assert trace["stop_reason"] == "final_answer" and trace["final_answer"] == "done"
    assert [s.get("length_nudge", False) for s in trace["steps"]] == [True, False, False]
    assert trace["totals"]["length_nudges"] == 1 and usage.llm_calls == 3
    m = trace["messages"]
    assert [x["role"] for x in m[:4]] == ["user", "assistant", "user", "assistant"]
    assert m[1]["content"] == "long analysis..."         # the truncated reply stays in the conversation
    assert m[2] == {"role": "user", "content": EXACT}


def test_truncated_tool_call_gets_error_result_then_nudge_in_valid_order():
    trace, _ = loop([(None, [("write_file", '{"path": "a.py", "content": "def f(')], "length"), ("fixed", [])])
    r = trace["steps"][0]["tool_results"][0]
    assert not r["arguments_valid"] and r["is_error"]
    roles = [x["role"] for x in trace["messages"]]
    assert roles == ["user", "assistant", "tool", "user", "assistant"]          # tool result precedes the nudge
    assert trace["messages"][1]["tool_calls"][0]["id"] == trace["messages"][2]["tool_call_id"]
    assert trace["messages"][3]["content"] == EXACT and trace["stop_reason"] == "final_answer"


def test_consecutive_truncations_each_continue_and_count_as_steps():
    trace, usage = loop([("a", [], "length"), ("b", [], "length"), ("c", [], "length"), ("done", [])])
    assert trace["totals"]["length_nudges"] == 3 and usage.llm_calls == 4 and trace["stop_reason"] == "final_answer"


def test_nudged_turns_consume_the_step_budget_and_last_step_adds_no_dangling_nudge():
    trace, usage = loop([("a", [], "length")] * 5, max_steps=3)
    assert usage.llm_calls == 3 and trace["totals"]["length_nudges"] == 2
    assert trace["stop_reason"] == "length"                                   # budget gone while truncated
    assert trace["messages"][-1]["role"] == "assistant"                       # no unanswerable trailing nudge


def test_without_nudge_behaviour_is_exactly_s0():
    trace, _ = loop([("cut off", [], "length"), ("never reached", [])], nudge=None)
    assert trace["stop_reason"] == "length" and trace["final_answer"] == "cut off"
    assert trace["totals"]["length_nudges"] == 0 and len(trace["steps"]) == 1


def test_stop_finish_reason_still_ends_the_run():
    trace, _ = loop([("all done", [], "stop")])
    assert trace["stop_reason"] == "final_answer" and trace["totals"]["length_nudges"] == 0


def test_s1_is_s0_plus_only_the_nudge(monkeypatch):
    monkeypatch.setenv("DEEPINFRA_API_KEY", "sk-secret-123")
    s0, s1 = S0Runtime(), S1Runtime()
    assert s0.length_nudge is None and "length_nudge" not in s0.describe()      # S0 manifests/behaviour unchanged
    d0, d1 = s0.describe(), s1.describe()
    assert s1.name == "s1-qwen3.5-9b" and d1["length_nudge"] == EXACT and d1["class"] == "S1Runtime"
    same = ("max_steps", "system_prompt", "llm", "tools")
    assert all(d0[k] == d1[k] for k in same)                                      # nothing else differs
    assert s1.environment == "container" and "sk-secret-123" not in json.dumps(d1)


def test_s1_recovers_the_s0_failure_scenario_end_to_end(env, tmp_path_factory):
    """S0 ended here with an empty patch; S1 continues past the truncation and edits."""
    turns = [(None, [("read_file", {"path": "pkg/a.py"})]),
             ("The bug is in f, because ...(cut off by the token cap)", [], "length"),
             (None, [("edit_file", {"path": "pkg/a.py", "old_str": "def f():\n    return 1",
                                    "new_str": "def f():\n    return 2"})]),
             ("Fixed.", [])]
    s0 = S0Runtime(client=Scripted(list(turns)))
    r0 = s0.solve(_task(env))
    assert r0.metadata["stop_reason"] == "length" and r0.patch == ""
    env2 = LocalEnv(tmp_path_factory.mktemp("second"))
    s1 = S1Runtime(client=Scripted(list(turns)))
    r1 = s1.solve(_task(env2))
    assert r1.metadata["stop_reason"] == "final_answer" and "return 2" in r1.patch
    assert r1.metadata["stats"]["length_nudges"] == 1 and r1.metadata["trace"]["totals"]["length_nudges"] == 1
