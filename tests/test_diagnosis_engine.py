import json
from pathlib import Path

import pytest

from miracle.diagnosis.config import DiagnosisConfig
from miracle.diagnosis.engine import DiagnosisEngine
from miracle.diagnosis.graph import read_events, replay
from miracle.diagnosis.handoff import build_handoff
from miracle.diagnosis.runtime import S3Runtime
from miracle.diagnosis.toytask import INSTANCE_ID, ISSUE, instance_record, write_repo
from miracle.errors import InfraError
from miracle.host_env import HostEnvironment
from miracle.runtime import Task
from test_llm import Scripted

IID = INSTANCE_ID
REPO_TOOLS = {"list_files", "read_file", "search_code", "edit_file", "write_file", "run_command", "git_diff"}


def T(*calls, content=None):
    return (content, [(n, a) for n, a in calls])


H1 = dict(claim="total() applies the tax twice when a discount is given, once inside the discount branch and once at the end", type="root_cause",
          predictions=["total(100, discount=10) calls add_tax twice, giving 100*0.9*1.1*1.1 = 108.9"],
          falsification_criteria=["if add_tax runs exactly once for a discounted price, this hypothesis is false"])
H2 = dict(claim="apply_discount computes the discounted price incorrectly (percent handling)", type="mechanism",
          predictions=["apply_discount(100, 10) does not return 90.0"],
          falsification_criteria=["if apply_discount(100, 10) returns 90.0 this hypothesis is false"])
PLAN = [
    T(("create_hypothesis", H1), ("create_hypothesis", H2)),
    T(("create_relation", dict(from_id="H1", to_id="H2", type="alternative_to", rationale="they predict different intermediate values")),
      ("create_probe", dict(target_hypothesis_ids=["H1", "H2"], purpose="discriminate", planned_action="print the intermediate values inside total(100, discount=10)",
                            expected_outcomes=[dict(hypothesis_id="H1", expect="add_tax is applied twice, final value 108.9"),
                                               dict(hypothesis_id="H2", expect="the discounted price is not 90.0")]))),
    T(("done_planning", {})),
]
BRANCH_H1 = [   # T1 read, T2 instrumentation edit (opens IV1), T3 instrumented run
    T(("read_file", dict(path="shop/pricing.py"))),
    T(("record_evidence", dict(type="code_excerpt", summary="total() calls add_tax inside the discount branch and again on return", path="shop/pricing.py", start_line=16, end_line=18)),
      ("record_relation", dict(evidence_id="E2", hypothesis_id="H1", relation="supports", rationale="the discounted price goes through add_tax twice", strength=3))),
    T(("edit_file", dict(path="shop/pricing.py", old_str="    return add_tax(price)", new_str="    print('PRICE-BEFORE-FINAL-TAX', price)\n    return add_tax(price)"))),
    T(("run_command", dict(command="python -c \"from shop.pricing import total; print(total(100, discount=10))\""))),
    T(("record_evidence", dict(type="intervention_result", summary="with the debug print the price is already taxed (99.0) before the final add_tax", source_tool_call_id="T3", quote="108.9")),
      ("record_relation", dict(evidence_id="E3", hypothesis_id="H1", relation="supports", rationale="the second tax is applied on an already-taxed price", strength=3))),
    T(("finish_investigation", dict(summary="Both add_tax calls run for a discounted price; the result is 108.9 as in the issue.", outcome="supports"))),
]
REVIEW_1 = [T(("update_hypothesis_status", dict(hypothesis_id="H1", new_status="supported", cause_evidence_ids=["E2", "E3"])),
              ("create_probe", dict(target_hypothesis_ids=["H1", "H2"], purpose="discriminate", planned_action="evaluate apply_discount(100, 10) in isolation and compare with the value total() used",
                                    expected_outcomes=[dict(hypothesis_id="H1", expect="apply_discount returns 90.0, the extra tax comes later"),
                                                       dict(hypothesis_id="H2", expect="apply_discount does not return 90.0")])),
              ("select_branch", dict(hypothesis_id="H2", probe_ids=["P2"], rationale="try to falsify the competing hypothesis")))]
BRANCH_H2 = [   # T4 read, T5 command
    T(("read_file", dict(path="shop/pricing.py"))),
    T(("record_evidence", dict(type="code_excerpt", summary="apply_discount multiplies by (1 - percent/100)", path="shop/pricing.py", start_line=6, end_line=6)),
      ("record_relation", dict(evidence_id="E4", hypothesis_id="H2", relation="contradicts", rationale="the formula is correct for percent values", strength=2))),
    T(("run_command", dict(command="python -c \"from shop.pricing import apply_discount; print(apply_discount(100, 10))\""))),
    T(("record_evidence", dict(type="runtime_observation", summary="apply_discount(100, 10) prints 90.0", source_tool_call_id="T5", quote="90.0")),
      ("record_relation", dict(evidence_id="E5", hypothesis_id="H2", relation="contradicts", rationale="the discounted price is exactly 90.0 as predicted by the falsification criterion", strength=2))),
    T(("finish_investigation", dict(summary="apply_discount returns 90.0, so the discount itself is correct.", outcome="contradicts"))),
]
COMMIT = dict(selected_hypothesis_ids=["H1"], causal_explanation="total() applies add_tax inside the discount branch and again before returning, so a discounted price is taxed twice.",
              affected_components=["shop.pricing.total"], requirements_explained=["total(100, discount=10) returns 108.9 instead of 99.0"],
              strongest_evidence_ids=["E2", "E3"], rejected_alternatives=[dict(hypothesis_id="H2", reason="apply_discount(100, 10) returns 90.0 (E5), so the discount is correct")],
              unresolved_risks=[], implementation_direction="remove the add_tax call from the discount branch of total()")
REVIEW_2 = [T(("update_hypothesis_status", dict(hypothesis_id="H2", new_status="refuted", cause_evidence_ids=["E4", "E5"])), ("commit_diagnosis", COMMIT))]
IMPLEMENT = [
    T(("read_file", dict(path="shop/pricing.py"))),
    T(("edit_file", dict(path="shop/pricing.py", old_str="price = add_tax(apply_discount(price, discount))", new_str="price = apply_discount(price, discount)"))),
    T(("run_command", dict(command="python -m pytest -q tests/ 2>&1 | tail -n 3"))),
    ("The tax was applied twice for discounted prices; fixed total().", []),
]
HAPPY = PLAN + BRANCH_H1 + REVIEW_1 + BRANCH_H2 + REVIEW_2


@pytest.fixture
def env(tmp_path_factory):
    d = tmp_path_factory.mktemp("repo")
    write_repo(d)
    yield HostEnvironment(d)


def engine(env, turns, tmp_path, **cfg):
    c = Scripted(turns)
    return DiagnosisEngine(c, env, IID, ISSUE, DiagnosisConfig(**cfg), tmp_path / "diagnosis"), c


def test_happy_path_builds_a_valid_graph_commits_and_leaves_the_repo_clean(env, tmp_path):
    eng, c = engine(env, HAPPY, tmp_path)
    st = eng.run()
    assert st.diagnosis["status"] == "committed" and st.diagnosis["selected_hypothesis_ids"] == [f"{IID}:H1"]
    assert st.hypotheses[f"{IID}:H1"]["status"] == "selected" and st.hypotheses[f"{IID}:H2"]["status"] == "refuted"
    assert [h["new"] for h in st.hypotheses[f"{IID}:H2"]["history"]] == ["investigating", "refuted"]
    assert {r["from_id"].split(":")[1] for r in st.relations.values() if r["type"] == "supports"} == {"E2", "E3"}
    assert env.dirty_state()["clean"] and env.diff() == ""                       # nothing survives the diagnosis
    assert (env.host / "shop" / "pricing.py").read_text().count("PRICE-BEFORE-FINAL-TAX") == 0
    for f in ("events.jsonl", "graph.json", "diagnosis.json", "controller_trace.json", "reset_verification.json", "tool_calls.jsonl", "repo_map.txt", "usage.json"):
        assert (tmp_path / "diagnosis" / f).exists(), f
    assert sorted(p.name for p in (tmp_path / "diagnosis" / "investigators").iterdir()) == ["B1.json", "B2.json"]


def test_temporary_instrumentation_is_tagged_and_reset_verified(env, tmp_path):
    eng, _ = engine(env, HAPPY, tmp_path)
    st = eng.run()
    e2, e3, e5 = (st.evidence[f"{IID}:E{i}"] for i in (2, 3, 5))
    assert e2["observation_state"] == "original" and e2["intervention_id"] is None and e2["excerpt"].strip().startswith("if discount:")
    assert e3["observation_state"] == "post_intervention" and e3["intervention_id"] == f"{IID}:IV1" and e3["type"] == "intervention_result"
    assert e3["command"].startswith("python -c") and e3["exit_code"] == 0 and e3["duration_s"] is not None and e3["source_tool_call_id"] == f"{IID}:T3"
    assert e5["observation_state"] == "original" and e5["intervention_id"] is None
    iv = st.interventions[f"{IID}:IV1"]
    assert iv["branch_id"] == f"{IID}:B1" and iv["reset_verified"] is True and iv["ended_at_step"] is not None
    ctxs = [r["context"] for r in st.resets]
    assert ctxs == ["baseline", "before " + f"{IID}:B1", f"after {IID}:B1", "before " + f"{IID}:B2", f"after {IID}:B2", "after diagnosis (before implementation)"]
    assert all(r["ok"] for r in st.resets)
    rv = json.loads((tmp_path / "diagnosis" / "reset_verification.json").read_text())
    assert any(r["was_dirty"] and r["verified_clean"] for r in rv) and rv[-1]["verified_clean"]


def test_controller_and_investigators_are_separate_contexts_with_separate_tools(env, tmp_path):
    eng, c = engine(env, HAPPY, tmp_path)
    eng.run()
    ctl_idx = [i for i, t in enumerate(c.tools_seen) if t and not ({x["function"]["name"] for x in t} & REPO_TOOLS)]
    inv_idx = [i for i, t in enumerate(c.tools_seen) if t and ({x["function"]["name"] for x in t} & REPO_TOOLS)]
    assert len(ctl_idx) == 3 + 1 + 1 and len(inv_idx) == 6 + 5                       # plan(3) + 2 reviews(1 each) ; B1(6) + B2(5)
    inv_system = {c.calls[i][0]["content"][:40] for i in inv_idx}
    ctl_system = {c.calls[i][0]["content"][:40] for i in ctl_idx}
    assert inv_system.isdisjoint(ctl_system)
    for i in inv_idx:                                                                # investigators cannot touch hypotheses or the diagnosis
        names = {x["function"]["name"] for x in c.tools_seen[i]}
        assert not names & {"create_hypothesis", "update_hypothesis_status", "commit_diagnosis", "create_probe", "select_branch"}
    first_inv = c.calls[inv_idx[0]][1]["content"]
    assert "total() applies the tax twice" in first_inv and "apply_discount computes the discounted price incorrectly" in first_inv    # competitor: compact claim
    assert "apply_discount(100, 10) does not return 90.0" not in first_inv           # ...but not its predictions/falsification detail
    ctl_review = c.calls[ctl_idx[3]][1]["content"]
    assert "Result of the investigation of H1" in ctl_review and "T3" not in ctl_review      # compact evidence summaries, no raw tool output


def test_event_log_replays_to_the_final_graph_at_every_step(env, tmp_path):
    eng, _ = engine(env, HAPPY, tmp_path)
    st = eng.run()
    events = read_events(tmp_path / "diagnosis" / "events.jsonl")
    assert [e["seq"] for e in events] == list(range(1, len(events) + 1))
    assert replay(IID, events).to_dict() == st.to_dict()
    mid = next(e["seq"] for e in events if e["type"] == "hypothesis_status_updated" and e["data"]["new_status"] == "supported")
    before = replay(IID, events, upto=mid - 1)
    assert before.hypotheses[f"{IID}:H1"]["status"] == "investigating" and replay(IID, events, upto=mid).hypotheses[f"{IID}:H1"]["status"] == "supported"
    assert any(e["type"] == "operation_rejected" for e in events) is False                 # the happy path needed no corrections


def test_invalid_model_output_is_rejected_with_guidance_and_recorded(env, tmp_path):
    bad = [T(("create_hypothesis", dict(claim="a hypothesis with no predictions at all, which is not allowed", type="root_cause", predictions=[], falsification_criteria=[]))),
           T(("update_hypothesis_status", dict(hypothesis_id="H9", new_status="refuted", cause_evidence_ids=["E1"]))),
           *PLAN, *BRANCH_H1, *REVIEW_1, *BRANCH_H2, *REVIEW_2]
    # the rejected calls consume turns inside the plan phase (a different tool is not available there, so it is rejected as unknown)
    eng, c = engine(env, bad, tmp_path)
    st = eng.run()
    events = read_events(tmp_path / "diagnosis" / "events.jsonl")
    assert eng.ops.rejections >= 1 and any(e["type"] == "operation_rejected" and "observable prediction" in e["data"]["error"] for e in events)
    assert st.diagnosis and st.diagnosis["status"] == "committed"


def probe(n_expect_a="apply_discount returns 90.0", n_expect_b="apply_discount does not return 90.0"):
    return ("create_probe", dict(target_hypothesis_ids=["H1", "H2"], purpose="discriminate", planned_action=f"another experiment to tell the two apart ({n_expect_a})",
                                 expected_outcomes=[dict(hypothesis_id="H1", expect=n_expect_a), dict(hypothesis_id="H2", expect=n_expect_b)]))


def test_status_change_without_linked_evidence_is_refused(env, tmp_path):
    turns = PLAN + BRANCH_H1 + [T(("update_hypothesis_status", dict(hypothesis_id="H2", new_status="refuted", cause_evidence_ids=["E2"]))),      # E2 is not linked to H2
                                T(("update_hypothesis_status", dict(hypothesis_id="H1", new_status="supported", cause_evidence_ids=["E2", "E3"])), probe(),
                                  ("select_branch", dict(hypothesis_id="H2", probe_ids=["P2"])))] \
        + BRANCH_H2 + REVIEW_2
    eng, c = engine(env, turns, tmp_path)
    st = eng.run()
    assert any("refuting needs" in e["data"]["error"] for e in read_events(tmp_path / "diagnosis" / "events.jsonl") if e["type"] == "operation_rejected")
    assert [h["new"] for h in st.hypotheses[f"{IID}:H2"]["history"]][-1] == "refuted"      # only after the proper evidence was linked


def test_budget_exhaustion_yields_an_honest_inconclusive_diagnosis(env, tmp_path):
    turns = PLAN + BRANCH_H1 + [T(probe(), ("select_branch", dict(hypothesis_id="H2", probe_ids=["P2"])))] + BRANCH_H2 \
        + [T(probe("the value is 90.0 at the second step"), ("select_branch", dict(hypothesis_id="H1", probe_ids=["P3"])))]
    eng, _ = engine(env, turns, tmp_path, max_iterations=2)
    st = eng.run()
    assert st.diagnosis["status"] == "inconclusive" and st.diagnosis["causal_explanation"].startswith("INCONCLUSIVE")
    assert st.diagnosis["selected_hypothesis_ids"] == [] and all(h["status"] != "selected" for h in st.hypotheses.values())
    assert env.dirty_state()["clean"]


def test_synthesis_is_triggered_when_the_leading_hypothesis_is_contradicted(env, tmp_path):
    contra = [T(("read_file", dict(path="shop/config.py"))),
              T(("record_evidence", dict(type="code_excerpt", summary="the tax rate is a single constant 0.10", path="shop/config.py", start_line=1, end_line=1)),
                ("record_relation", dict(evidence_id="E4", hypothesis_id="H1", relation="contradicts", rationale="a single constant cannot explain a doubled tax by itself", strength=1))),
              T(("finish_investigation", dict(summary="The constant is single-valued, which does not by itself explain a doubled tax.", outcome="mixed")))]
    turns = PLAN + BRANCH_H1 + REVIEW_1 + contra + [T(("select_branch", dict(hypothesis_id="H1"))),                       # review after branch 2 (on H2)
                                                     T(("finish_synthesis", dict(summary="H1 remains supported; the contradiction is weak", inconclusive=True)))]
    eng, _ = engine(env, turns, tmp_path)
    st = eng.run()
    assert st.syntheses and "contradicts the leading hypothesis" in " ".join(next(iter(st.syntheses.values()))["reasons"])
    assert st.diagnosis["status"] == "inconclusive"                                     # the synthesis declared the investigation inconclusive


def test_reset_verification_failure_is_an_infrastructure_failure(env, tmp_path, monkeypatch):
    monkeypatch.setattr(HostEnvironment, "reset_to_baseline", lambda self: None)       # a reset that silently does nothing
    eng, _ = engine(env, PLAN + BRANCH_H1, tmp_path)
    with pytest.raises(InfraError, match="reset verification failed"):
        eng.run()


def test_llm_provider_failure_aborts_as_infrastructure(env, tmp_path):
    from miracle.llm.client import LLMError
    eng, _ = engine(env, [(LLMError("HTTP 503", 503), [])], tmp_path)
    with pytest.raises(InfraError, match="model provider failed"):
        eng.run()
    assert (tmp_path / "diagnosis" / "graph.json").exists()                              # partial artefacts are still persisted


# ---- S3 runtime: diagnosis -> reset -> handoff -> implementation ---------------------------------------------------------
def run_s3(env, tmp_path, instance=None, turns=None):
    c = Scripted(turns or (HAPPY + IMPLEMENT))
    rec = instance or instance_record()
    task = Task.from_instance(rec, env=env, artifact_dir=tmp_path / "agent")
    return S3Runtime(client=c).solve(task), c, task


def test_s3_runtime_end_to_end_patch_contains_only_the_implementation(env, tmp_path):
    res, c, _ = run_s3(env, tmp_path)
    assert "PRICE-BEFORE-FINAL-TAX" not in res.patch and "apply_discount(price, discount)" in res.patch and "-        price = add_tax(apply_discount(price, discount))" in res.patch
    a = tmp_path / "agent"
    for f in ("diagnosis/events.jsonl", "diagnosis/graph.json", "diagnosis/diagnosis.json", "diagnosis/reset_verification.json", "implementation/result.json",
              "implementation/patch.diff", "implementation/trace.json", "implementation/handoff.md"):
        assert (a / f).exists(), f
    m = res.metadata["diagnosis"]
    assert m["status"] == "committed" and m["resets_verified"] and m["branches"] == 2 and m["interventions"] == 1
    assert res.usage.llm_calls == len(c.calls) and res.usage.tool_calls > 0


def test_handoff_contains_only_the_allowed_material(env, tmp_path):
    res, c, _ = run_s3(env, tmp_path)
    handoff = (tmp_path / "agent" / "implementation" / "handoff.md").read_text()
    assert "## Diagnosis (committed)" in handoff and "taxed twice" in handoff and "## Alternatives that were rejected" in handoff and "## Repository map" in handoff
    assert "remove the add_tax call from the discount branch" in handoff and "if discount:" in handoff             # selected supporting evidence excerpt
    for forbidden in ("H1", "H2", "E3", "intervention", "PRICE-BEFORE-FINAL-TAX", "[T3]", "operation_rejected", "controller"):
        assert forbidden not in handoff, forbidden
    impl_first = c.calls[-4][1]["content"]                                                # first message of the implementation loop
    assert ISSUE.strip()[:40] in impl_first and "## Diagnosis (committed)" in impl_first


def test_candidate_never_sees_gold_or_hidden_fields(env, tmp_path):
    rec = instance_record()
    rec.update(patch="SENTINEL_GOLD_PATCH_91ab", test_patch="SENTINEL_TEST_PATCH_77cd", hints_text="SENTINEL_HINTS_55ef",
               FAIL_TO_PASS=["SENTINEL_F2P_13aa"], PASS_TO_PASS=["SENTINEL_P2P_24bb"])
    res, c, _ = run_s3(env, tmp_path, instance=rec)
    blob = json.dumps(c.calls) + json.dumps(res.metadata, default=str)
    for root, _, files in __import__("os").walk(tmp_path / "agent"):
        for f in files:
            blob += (Path(root) / f).read_text(errors="ignore")
    for s in ("SENTINEL_GOLD_PATCH_91ab", "SENTINEL_TEST_PATCH_77cd", "SENTINEL_HINTS_55ef", "SENTINEL_F2P_13aa", "SENTINEL_P2P_24bb"):
        assert s not in blob, s


def test_candidate_side_modules_never_reference_hidden_fields():
    """Static guard (AST: identifiers and string literals, not prose): nothing on the solving side may name the hidden dataset fields."""
    import ast
    root = Path(__import__("miracle").__file__).parent
    candidate = [*root.joinpath("diagnosis").glob("*.py"), *root.joinpath("agents").glob("*.py"), *root.joinpath("llm").glob("*.py"),
                 root / "runtime.py", root / "environment.py", root / "host_env.py"]
    tokens = ("test_patch", "fail_to_pass", "pass_to_pass", "hints_text", "gold")
    for f in candidate:
        tree = ast.parse(f.read_text())
        doc_nodes = {id(n.body[0].value) for n in ast.walk(tree) if isinstance(n, (ast.Module, ast.ClassDef, ast.FunctionDef)) and n.body
                     and isinstance(n.body[0], ast.Expr) and isinstance(getattr(n.body[0], "value", None), ast.Constant)}
        for n in ast.walk(tree):
            texts = []
            if isinstance(n, ast.Name):
                texts.append(n.id)
            elif isinstance(n, ast.Attribute):
                texts.append(n.attr)
            elif isinstance(n, ast.arg):
                texts.append(n.arg)
            elif isinstance(n, ast.Constant) and isinstance(n.value, str) and id(n) not in doc_nodes:
                texts.append(n.value)
            for t in texts:
                for tok in tokens:
                    assert tok not in t.lower(), f"{f.name}:{getattr(n, 'lineno', '?')} names a hidden field ({tok}): {t[:60]!r}"


def test_planning_phase_is_capped_and_every_result_restates_the_protocol(env, tmp_path):
    """Real-model finding: the controller kept creating hypotheses (12 in a row) and never planned a probe or stopped."""
    claims = ["total() applies add_tax inside the discount branch and again before returning, so discounted prices are taxed twice",
              "apply_discount mishandles the percent argument and returns a wrong intermediate price for ten percent",
              "the TAX_RATE constant in shop.config is loaded from the wrong module and is larger than documented",
              "floating point rounding of the intermediate price accumulates into a visible error on the final amount",
              "the tests call total() with a keyword argument that shadows a positional parameter of another helper",
              "add_tax is decorated or wrapped somewhere so that every call multiplies the amount twice",
              "the package __init__ re-exports a stale copy of total() from an old module that still has the bug"]
    many = [T(("create_hypothesis", dict(H1, claim=c_, predictions=[f"observation {i} differs from the expected value"], falsification_criteria=[f"if observation {i} is normal this is false"])))
            for i, c_ in enumerate(claims)]
    eng, c = engine(env, many + PLAN[1:] + BRANCH_H1 + [T(probe_stub(), ("select_branch", dict(hypothesis_id="H2")))] , tmp_path, max_iterations=1)
    eng.run()
    outs = [json.loads(json.dumps(m)) for m in c.calls[6]]                                    # the call after the 6th create_hypothesis
    errors = [e["data"]["error"] for e in read_events(tmp_path / "diagnosis" / "events.jsonl") if e["type"] == "operation_rejected"]
    assert any("limit for the planning phase" in e and "done_planning" in e for e in errors)
    assert len([h for h in eng.state.hypotheses.values()]) == 5                               # never more than the planning cap
    tool_msgs = [m["content"] for m in c.calls[2] if m["role"] == "tool"]
    assert tool_msgs and "[progress: 2 hypotheses so far (planning limit 5)" in tool_msgs[-1] and "done_planning" in tool_msgs[-1]


def probe_stub():
    return ("create_probe", dict(target_hypothesis_ids=["H1", "H2"], purpose="discriminate", planned_action="a different experiment for the same two hypotheses",
                                 expected_outcomes=[dict(hypothesis_id="H1", expect="outcome one is seen"), dict(hypothesis_id="H2", expect="outcome two is seen")]))
