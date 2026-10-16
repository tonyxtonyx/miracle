import copy

import pytest

from miracle.diagnosis.config import DiagnosisConfig
from miracle.diagnosis.graph import GraphError, GraphState, read_events, replay, short_id
from miracle.diagnosis.ops import GraphOps

IID = "proj__repo-1"
ISSUE = "Calling total() with a discount returns the wrong amount.\nThe tax seems to be applied twice when discount > 0."


class FakeSource:
    def __init__(self):
        self.files = {"pkg/pricing.py": "def total(p, d):\n    x = apply_discount(p, d)\n    return add_tax(add_tax(x))\n"}
        self.calls = {}
        self.intervention = None

    def read_excerpt(self, path, start, end):
        if path not in self.files:
            raise GraphError(f"no such file inside the repository: {path}")
        lines = self.files[path].splitlines()
        if not (1 <= start <= end <= len(lines)):
            raise GraphError(f"invalid line range {start}-{end} (file has {len(lines)} lines)")
        return {"path": path, "start_line": start, "end_line": end, "excerpt": "\n".join(lines[start - 1:end]),
                "repo_state": {"base_tree": "abc", "intervention_id": self.intervention}}

    def tool_call(self, call_id):
        return self.calls.get(call_id) or self.calls.get(f"{IID}:{call_id}")


@pytest.fixture
def ops(tmp_path):
    st = GraphState(IID)
    o = GraphOps(st, tmp_path / "events.jsonl", ISSUE, DiagnosisConfig(), FakeSource())
    return o


def hyp(ops, claim="tax is applied twice inside total() when a discount is given", **kw):
    args = dict(actor="controller", claim=claim, type="root_cause", predictions=["total(100, 10) returns 98.01 instead of 99.0"],
                falsification_criteria=["if add_tax is called once in the call chain this is false"])
    args.update(kw)
    return ops.create_hypothesis(**args)["id"]


# ---- hypotheses ----------------------------------------------------------------------------------------------------
def test_ids_are_runtime_generated_namespaced_and_short_ids_resolve(ops):
    a, b = hyp(ops), hyp(ops, claim="the discount percentage is parsed as a fraction before being applied")
    assert (a, b) == ("H1", "H2") and ops.state.hypotheses[f"{IID}:H1"]["id"] == f"{IID}:H1"
    assert ops.state.resolve("h2") == f"{IID}:H2" and ops.state.resolve(f"{IID}:H2") == f"{IID}:H2"


@pytest.mark.parametrize("kw,msg", [
    ({"predictions": []}, "observable prediction"),
    ({"falsification_criteria": []}, "falsification criterion"),
    ({"type": "vibes"}, "type must be"),
    ({"claim": "short"}, "claim"),
])
def test_hypothesis_requires_predictions_falsification_and_valid_type(ops, kw, msg):
    with pytest.raises(GraphError, match=msg):
        hyp(ops, **kw)
    assert not ops.state.hypotheses and any(e["type"] == "operation_rejected" for e in read_events(ops.log_path))


def test_duplicate_claims_depth_and_cap_are_rejected(ops):
    h1 = hyp(ops)
    with pytest.raises(GraphError, match="overlaps H1"):
        hyp(ops, claim="tax is applied twice inside total() when a discount is given!")
    h2 = hyp(ops, claim="the refinement: add_tax is called twice because of a merge conflict artefact", parent_ids=[h1])
    h3 = hyp(ops, claim="the second add_tax call comes from a stray decorator on the helper", parent_ids=[h2])
    h4 = hyp(ops, claim="the stray decorator was introduced by the caching refactor in the helper", parent_ids=[h3])   # depth 3 = the limit
    assert ops.state.depth(f"{IID}:{h4}") == 3
    with pytest.raises(GraphError, match="depth limit"):
        hyp(ops, claim="a deeper refinement of the caching refactor idea in the helper", parent_ids=[h4])
    assert {r["type"] for r in ops.state.relations.values()} == {"refines"}
    with pytest.raises(GraphError, match="unknown id"):
        hyp(ops, claim="a hypothesis with a dangling parent reference", parent_ids=["H99"])


def test_role_permissions(ops):
    with pytest.raises(GraphError, match="investigator is not allowed"):
        hyp(ops, actor="investigator:B1")
    with pytest.raises(GraphError, match="investigator is not allowed"):
        ops.update_hypothesis_status("investigator:B1", "H1", "refuted", [], "no")
    hyp(ops)
    assert ops.rejections == 2


# ---- evidence provenance ---------------------------------------------------------------------------------------------
def test_issue_evidence_must_quote_the_issue_verbatim(ops):
    e = ops.attach_evidence("runtime", "issue_statement", "the issue states the symptom", quote="tax seems to be applied twice")["id"]
    ev = ops.state.evidence[f"{IID}:{e}"]
    assert ISSUE[ev["char_start"]:ev["char_end"]] == "tax seems to be applied twice"
    with pytest.raises(GraphError, match="exact substring"):
        ops.attach_evidence("runtime", "issue_statement", "a made up quote about taxes", quote="taxes are doubled")


def test_code_evidence_is_read_by_the_runtime_and_hashed(ops):
    e = ops.attach_evidence("investigator:B1", "code_excerpt", "total() wraps add_tax twice", path="pkg/pricing.py", start_line=3, end_line=3, branch_id="B1")["id"]
    ev = ops.state.evidence[f"{IID}:{e}"]
    assert ev["excerpt"] == "    return add_tax(add_tax(x))" and len(ev["excerpt_hash"]) == 64
    assert (ev["path"], ev["start_line"], ev["end_line"], ev["observation_state"]) == ("pkg/pricing.py", 3, 3, "original")
    for bad, msg in ((dict(path="../etc/passwd", start_line=1, end_line=2), "no such file"), (dict(path="pkg/pricing.py", start_line=2, end_line=99), "invalid line range"),
                     (dict(path=None, start_line=1, end_line=2), "path, start_line")):
        with pytest.raises(GraphError, match=msg):
            ops.attach_evidence("investigator:B1", "code_excerpt", "evidence about the pricing function", **bad)


def test_tool_derived_evidence_needs_a_recorded_call_and_an_exact_quote(ops):
    src = ops.source
    src.calls[f"{IID}:T1"] = {"id": f"{IID}:T1", "branch_id": "B1", "name": "run_command", "command": "pytest -q", "exit_code": 1, "duration_s": 2.5,
                              "output": "FAILED test_total - assert 98.01 == 99.0", "intervention_id": None, "repo_state": {"base_tree": "abc"}}
    src.calls[f"{IID}:T2"] = {**src.calls[f"{IID}:T1"], "id": f"{IID}:T2", "intervention_id": f"{IID}:IV1"}
    src.calls[f"{IID}:T3"] = {**src.calls[f"{IID}:T1"], "id": f"{IID}:T3", "branch_id": "B2"}
    e = ops.attach_evidence("investigator:B1", "test_result", "the test fails with the doubled tax amount", source_tool_call_id="T1",
                            quote="assert 98.01 == 99.0", branch_id="B1")["id"]
    ev = ops.state.evidence[f"{IID}:{e}"]
    assert (ev["command"], ev["exit_code"], ev["duration_s"], ev["observation_state"], ev["intervention_id"]) == ("pytest -q", 1, 2.5, "original", None)
    cases = [(dict(source_tool_call_id="T9"), "unknown tool call"), (dict(source_tool_call_id="T1", quote="invented output"), "exact substring"),
             (dict(source_tool_call_id="T3"), "different investigation branch"), (dict(source_tool_call_id=None), "source_tool_call_id is required"),
             (dict(source_tool_call_id="T1", type="intervention_result"), "no temporary change")]
    for kw, msg in cases:
        with pytest.raises(GraphError, match=msg):
            ops.attach_evidence("investigator:B1", kw.pop("type", "test_result"), "an observation made by the probe", branch_id="B1", **kw)
    iv = ops.attach_evidence("investigator:B1", "intervention_result", "with debug print the second add_tax is reached", source_tool_call_id="T2", branch_id="B1")["id"]
    assert ops.state.evidence[f"{IID}:{iv}"]["observation_state"] == "post_intervention" and ops.state.evidence[f"{IID}:{iv}"]["intervention_id"] == f"{IID}:IV1"


# ---- relations -----------------------------------------------------------------------------------------------------
def _evidence(ops, actor="investigator:B1"):
    return ops.attach_evidence(actor, "code_excerpt", "total() wraps add_tax twice here", path="pkg/pricing.py", start_line=3, end_line=3, branch_id="B1")["id"]


def test_relation_kinds_duplicates_and_investigator_scope(ops):
    h1, h2 = hyp(ops), hyp(ops, claim="the discount percentage is parsed as a fraction before being applied")
    e = _evidence(ops)
    ops.visible["investigator:B1"] = {f"{IID}:{h1}"}
    r = ops.create_relation("investigator:B1", e, h1, "supports", "the doubled add_tax is on this line", 3)["id"]
    assert ops.state.relations[f"{IID}:{r}"]["strength"] == 3
    with pytest.raises(GraphError, match="already exists"):
        ops.create_relation("investigator:B1", e, h1, "supports", "same relation again for the same pair", 2)
    with pytest.raises(GraphError, match="not part of your investigation context"):
        ops.create_relation("investigator:B1", e, h2, "contradicts", "this is about another hypothesis entirely", 2)
    with pytest.raises(GraphError, match="links a"):
        ops.create_relation("controller", h1, e, "supports", "wrong direction: hypothesis to evidence", 2)
    with pytest.raises(GraphError, match="links a"):
        ops.create_relation("controller", e, h1, "alternative_to", "evidence cannot be an alternative to anything", 2)
    # the controller can cross-link the same evidence to a competing hypothesis
    ops.create_relation("controller", e, h2, "contradicts", "the doubled tax leaves no room for a parsing bug", 2)
    assert {r_["to_id"] for r_ in ops.state.relations_of(f"{IID}:{e}", direction="out")} == {f"{IID}:{h1}", f"{IID}:{h2}"}
    other = _evidence(ops, "investigator:B2")
    ops.visible["investigator:B1"].add(f"{IID}:{h2}")
    with pytest.raises(GraphError, match="only relate evidence you recorded"):
        ops.create_relation("investigator:B1", other, h2, "supports", "attempt to reuse another branch's evidence", 2)
    with pytest.raises(GraphError, match="cycle"):
        c = hyp(ops, claim="a child refinement of the tax-doubling hypothesis about the helper", parent_ids=[h1])
        ops.create_relation("controller", h1, c, "refines", "closing the loop creates a refinement cycle", 2)


def test_alternative_to_is_symmetric_and_probes_validate(ops):
    h1, h2 = hyp(ops), hyp(ops, claim="the discount percentage is parsed as a fraction before being applied")
    ops.create_relation("controller", h1, h2, "alternative_to", "they predict different totals for the same input", 2)
    with pytest.raises(GraphError, match="already exists"):
        ops.create_relation("controller", h2, h1, "alternative_to", "reverse direction of the same alternative", 2)
    ok = dict(actor="controller", target_hypothesis_ids=[h1, h2], purpose="discriminate", planned_action="run total(100, 10) and print the intermediate values",
              expected_outcomes=[{"hypothesis_id": h1, "expect": "add_tax is called twice"}, {"hypothesis_id": h2, "expect": "discount is applied as 0.1 percent"}])
    p = ops.create_probe(**ok)["id"]
    assert ops.state.probes[f"{IID}:{p}"]["status"] == "planned" and {r["type"] for r in ops.state.relations_of(f"{IID}:{p}")} == {"tests"}
    for kw, msg in ((dict(target_hypothesis_ids=[h1]), "at least two"), (dict(expected_outcomes=[ok["expected_outcomes"][0]]), "EACH target"),
                    (dict(expected_outcomes=[{"hypothesis_id": h1, "expect": "same thing seen"}, {"hypothesis_id": h2, "expect": "same thing seen"}]), "different expected outcomes"),
                    (dict(purpose="guess"), "purpose must be")):
        with pytest.raises(GraphError, match=msg):
            ops.create_probe(**{**ok, **kw})


# ---- status transitions & causes -----------------------------------------------------------------------------------------
def test_status_changes_require_recorded_causes(ops):
    h1, h2 = hyp(ops), hyp(ops, claim="the discount percentage is parsed as a fraction before being applied")
    e = _evidence(ops)
    ops.visible["investigator:B1"] = {f"{IID}:{h1}", f"{IID}:{h2}"}
    with pytest.raises(GraphError, match="'supported' needs"):
        ops.update_hypothesis_status("controller", h1, "supported", [e], "")
    ops.create_relation("investigator:B1", e, h1, "supports", "the doubled add_tax is on this line", 3)
    ops.update_hypothesis_status("controller", h1, "supported", [e], "")
    assert ops.state.hypotheses[f"{IID}:H1"]["support_level"] == "medium"
    with pytest.raises(GraphError, match="refuting needs"):
        ops.update_hypothesis_status("controller", h2, "refuted", [e], "looks wrong to me")        # no contradicts relation
    ops.create_relation("investigator:B1", e, h2, "contradicts", "the doubled tax leaves no room for a parsing bug", 1)
    with pytest.raises(GraphError, match="refuting needs"):
        ops.update_hypothesis_status("controller", h2, "refuted", [e], "")                         # strength 1 alone is not enough
    ops.update_hypothesis_status("controller", h2, "weakened", [e], "")
    e2 = _evidence(ops)
    ops.create_relation("investigator:B1", e2, h2, "contradicts", "a second observation also contradicts it", 1)
    ops.update_hypothesis_status("controller", h2, "refuted", [e, e2], "")
    h = ops.state.hypotheses[f"{IID}:H2"]
    assert [x["new"] for x in h["history"]] == ["weakened", "refuted"] and h["history"][-1]["cause"]["evidence_ids"] == [f"{IID}:{e}", f"{IID}:{e2}"]
    assert f"{IID}:H2" in ops.state.hypotheses and ops.state.hypotheses[f"{IID}:H2"]["status"] == "refuted"   # never deleted
    with pytest.raises(GraphError, match="cannot move"):
        ops.update_hypothesis_status("controller", h2, "supported", [e], "")
    with pytest.raises(GraphError, match="AFTER the refutation"):
        ops.update_hypothesis_status("controller", h2, "investigating", [e], "reopen because I changed my mind")
    e3 = _evidence(ops)
    ops.update_hypothesis_status("controller", h2, "investigating", [e3], "new observation makes this plausible again")
    with pytest.raises(GraphError, match="only by commit_diagnosis|status must be"):
        ops.update_hypothesis_status("controller", h1, "selected", [e], "just pick it")


# ---- diagnosis commit gate ------------------------------------------------------------------------------------------------
def test_commit_requires_supported_hypothesis_evidence_and_disposition_of_alternatives(ops):
    h1, h2 = hyp(ops), hyp(ops, claim="the discount percentage is parsed as a fraction before being applied")
    e = _evidence(ops)
    ops.visible["investigator:B1"] = {f"{IID}:{h1}", f"{IID}:{h2}"}
    ops.create_relation("investigator:B1", e, h1, "supports", "the doubled add_tax is on this line", 3)
    commit = dict(actor="controller", selected_hypothesis_ids=[h1], causal_explanation="total() calls add_tax twice when a discount is present, so the tax is doubled.",
                  affected_components=["pkg.pricing.total"], requirements_explained=["tax applied twice when discount > 0"], strongest_evidence_ids=[e],
                  rejected_alternatives=[], unresolved_risks=[], implementation_direction="remove the second add_tax call in total()")
    with pytest.raises(GraphError, match="only a 'supported' hypothesis"):
        ops.commit_diagnosis(**commit)
    ops.update_hypothesis_status("controller", h1, "supported", [e], "")
    with pytest.raises(GraphError, match="every other active hypothesis"):
        ops.commit_diagnosis(**commit)
    with pytest.raises(GraphError, match="non-issue evidence"):
        ops.commit_diagnosis(**{**commit, "strongest_evidence_ids": [ops.attach_evidence("runtime", "issue_statement", "the issue states the symptom", quote="tax seems to be applied twice")["id"]],
                                "rejected_alternatives": [{"hypothesis_id": h2, "reason": "parsing is correct per the excerpt"}]})
    ops.commit_diagnosis(**{**commit, "rejected_alternatives": [{"hypothesis_id": h2, "reason": "parsing is correct per the excerpt"}]})
    d = ops.state.diagnosis
    assert d["status"] == "committed" and ops.state.hypotheses[f"{IID}:H1"]["status"] == "selected"
    with pytest.raises(GraphError, match="already committed"):
        ops.commit_diagnosis(**commit)


def test_contradicted_selection_needs_stated_risks(ops):
    h1 = hyp(ops)
    e, e2 = _evidence(ops), _evidence(ops)
    ops.visible["investigator:B1"] = {f"{IID}:{h1}"}
    ops.create_relation("investigator:B1", e, h1, "supports", "the doubled add_tax is on this line", 3)
    ops.create_relation("investigator:B1", e2, h1, "contradicts", "one test still passes with a discount", 1)
    ops.update_hypothesis_status("controller", h1, "supported", [e], "")
    kw = dict(actor="controller", selected_hypothesis_ids=[h1], causal_explanation="total() calls add_tax twice when a discount is present, so the tax is doubled.",
              affected_components=["pkg.pricing.total"], requirements_explained=["tax applied twice"], strongest_evidence_ids=[e], rejected_alternatives=[],
              unresolved_risks=[], implementation_direction="remove the second add_tax call in total()")
    with pytest.raises(GraphError, match="contradicting evidence"):
        ops.commit_diagnosis(**kw)
    ops.commit_diagnosis(**{**kw, "unresolved_risks": ["one passing test with a discount is unexplained"]})


def test_inconclusive_commit_is_honest(ops):
    h1 = hyp(ops)
    ops.commit_inconclusive([f"{IID}:{h1}"], "budget exhausted before any hypothesis was supported")
    d = ops.state.diagnosis
    assert d["status"] == "inconclusive" and d["selected_hypothesis_ids"] == [] and d["causal_explanation"].startswith("INCONCLUSIVE")
    assert ops.state.hypotheses[f"{IID}:H1"]["status"] == "proposed"


# ---- event log & replay ---------------------------------------------------------------------------------------------------
def test_replay_reproduces_the_graph_at_every_step_and_the_log_is_append_only(ops):
    snaps = []
    h1 = hyp(ops); snaps.append((ops.state.seq, copy.deepcopy(ops.state.to_dict())))
    h2 = hyp(ops, claim="the discount percentage is parsed as a fraction before being applied"); snaps.append((ops.state.seq, copy.deepcopy(ops.state.to_dict())))
    e = _evidence(ops); ops.visible["investigator:B1"] = {f"{IID}:{h1}"}
    ops.create_relation("investigator:B1", e, h1, "supports", "the doubled add_tax is on this line", 3); snaps.append((ops.state.seq, copy.deepcopy(ops.state.to_dict())))
    ops.update_hypothesis_status("controller", h1, "supported", [e], ""); snaps.append((ops.state.seq, copy.deepcopy(ops.state.to_dict())))
    events = read_events(ops.log_path)
    assert [x["seq"] for x in events] == list(range(1, len(events) + 1))
    for seq, snap in snaps:
        assert replay(IID, events, upto=seq).to_dict() == snap
    full = replay(IID, events)
    assert full.to_dict() == ops.state.to_dict() and full.hypotheses[f"{IID}:H1"]["status"] == "supported"
    bad = copy.deepcopy(events)
    bad[2]["seq"] = 99
    with pytest.raises(GraphError, match="sequence gap"):
        replay(IID, bad)
    assert short_id(f"{IID}:H1") == "H1"


def test_same_evidence_cannot_both_support_and_contradict_one_hypothesis(ops):
    """Real-model finding: to refute a duplicate hypothesis the controller linked as 'contradicts' the very evidence the investigator had linked as 'supports'."""
    h1, h4 = hyp(ops), hyp(ops, claim="the discount path is taxed once and then taxed again on the already taxed amount")
    e = _evidence(ops)
    ops.visible["investigator:B1"] = {f"{IID}:{h1}", f"{IID}:{h4}"}
    ops.create_relation("investigator:B1", e, h4, "supports", "the doubled add_tax predicts exactly this double taxation", 3)
    with pytest.raises(GraphError, match="cannot both support and contradict.*merge_hypotheses"):
        ops.create_relation("controller", e, h4, "contradicts", "H1 is more specific, so H4 is wrong", 2)
    ops.merge_hypotheses("controller", h1, h4, "both claim the tax is applied twice for discounted prices")
    assert ops.state.hypotheses[f"{IID}:{h4}"]["merged_into"] == f"{IID}:{h1}" and ops.state.hypotheses[f"{IID}:{h4}"]["status"] == "inconclusive"
    assert len(ops.state.active()) == 1
