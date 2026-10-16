import json
from pathlib import Path

import pytest

from miracle.diagnosis import freeze as fz
from miracle.diagnosis.config import DiagnosisConfig
from miracle.diagnosis.judging.clients import MockJudge
from miracle.diagnosis.judging.mockjudge import deterministic_mock
from miracle.diagnosis.orchestrate import load_task_list, run_task_list, ui_check
from miracle.diagnosis.reports import aggregate, render_aggregate_md, snapshot_at, snapshots, spearman, task_report
from miracle.diagnosis.runtime import S3Runtime
from miracle.diagnosis.toy import mock_judge, run_toy
from miracle.diagnosis.toytask import INSTANCE_ID
from test_diagnosis_engine import HAPPY, IMPLEMENT, Scripted, env, run_s3  # noqa: F401
from test_diagnosis_judging import OFFICIAL, good_blind, good_oracle, mock, run_judges


# ---- task list handling -----------------------------------------------------------------------------------------------------
def test_task_list_preserves_order_ignores_comments_and_rejects_duplicates(tmp_path):
    f = tmp_path / "tasks.txt"
    f.write_text("# my ordered list\nz__z-3\n\na__a-1  # second\nm__m-2\n")
    assert load_task_list(f) == ["z__z-3", "a__a-1", "m__m-2"]                      # NOT sorted: the order is the experiment's order
    f.write_text("a__a-1\nb__b-2\na__a-1\n")
    with pytest.raises(ValueError, match="duplicate"):
        load_task_list(f)
    f.write_text("# nothing\n")
    with pytest.raises(ValueError, match="empty"):
        load_task_list(f)


def test_run_task_list_refuses_when_prompts_are_not_frozen(monkeypatch):
    monkeypatch.setattr("miracle.diagnosis.orchestrate.verify_frozen", lambda: (False, ["DiagnosisConfig changed since the freeze"]))
    with pytest.raises(RuntimeError, match="not frozen or differ"):
        run_task_list("x", ["a__a-1"], None)


def test_run_task_list_rejects_unknown_ids_before_doing_any_work(monkeypatch):
    import miracle.diagnosis.orchestrate as o
    monkeypatch.setattr(o, "verify_frozen", lambda: (True, []))
    monkeypatch.setattr(o, "load_snapshot", lambda: (None, [{"instance_id": "a__a-1"}]))
    with pytest.raises(KeyError, match="not in the dataset"):
        run_task_list("x", ["a__a-1", "ghost__ghost-9"], None)


# ---- freezing -----------------------------------------------------------------------------------------------------------------
def test_freeze_detects_any_prompt_or_config_change(tmp_path, monkeypatch):
    monkeypatch.setattr(fz, "FROZEN_PATH", tmp_path / "FROZEN.json")
    assert fz.verify_frozen() == (False, ["FROZEN.json is missing: run `miracle diagnosis freeze` first"])
    m = fz.write_frozen()
    assert set(m["diagnosis_prompts"]) >= {"CONTROLLER_SYSTEM", "INVESTIGATOR_SYSTEM"} and set(m["judge_prompts"]) == {"BLIND_SYSTEM", "REFERENCE_SYSTEM", "ORACLE_SYSTEM"}
    assert fz.verify_frozen() == (True, [])
    from miracle.diagnosis import prompts as P
    monkeypatch.setattr(P, "CONTROLLER_SYSTEM", P.CONTROLLER_SYSTEM + " tweak")
    ok, problems = fz.verify_frozen()
    assert not ok and any("CONTROLLER_SYSTEM" in p for p in problems)
    monkeypatch.undo()
    monkeypatch.setattr(fz, "FROZEN_PATH", tmp_path / "FROZEN.json")
    import miracle.diagnosis.freeze as f2
    monkeypatch.setattr(f2, "DiagnosisConfig", lambda: DiagnosisConfig(max_iterations=99))
    assert not f2.verify_frozen()[0]


def test_runtime_describe_records_prompts_config_and_freeze_state(monkeypatch):
    monkeypatch.setenv("DEEPINFRA_API_KEY", "sk-secret-123")
    d = S3Runtime().describe()
    assert d["class"] == "S3Runtime" and d["config"]["max_iterations"] == 8 and len(d["prompt_hashes"]) == 10 and "CONTROLLER_SYSTEM" in d["prompts"]
    assert d["llm"]["model"] == "Qwen/Qwen3.5-9B" and d["llm"]["reasoning_effort"] == "none" and "sk-secret-123" not in json.dumps(d)
    assert "frozen" in d


# ---- statistics ----------------------------------------------------------------------------------------------------------------
def test_spearman():
    assert spearman([1, 2, 3, 4], [10, 20, 30, 40]) == 1.0 and spearman([1, 2, 3, 4], [4, 3, 2, 1]) == -1.0
    assert spearman([1, 2], [1, 2]) is None and spearman([1, 1, 1], [1, 2, 3]) is None
    assert -1 <= spearman([1, 2, 2, 5, 3], [3, 1, 2, 5, 4]) <= 1


# ---- reports ------------------------------------------------------------------------------------------------------------------------
@pytest.fixture
def judged(env, tmp_path):
    res, c, task = run_s3(env, tmp_path)
    inst = tmp_path / "agent"
    run_judges(inst, mock())
    return inst


def test_task_report_keeps_official_and_judge_outcomes_separate_and_computes_post_sufficiency_cost(judged):
    rep = task_report(judged, INSTANCE_ID, OFFICIAL)
    assert rep["official"]["status"] == "resolved_MARK_9d2" and rep["official"]["resolved"] is True
    assert rep["judges"]["blind"]["mean_score"] == 3.0 and rep["judges"]["oracle"]["scores"]["causal_correctness"] == 4
    assert "official" not in rep["judges"]["blind"] and "resolved" not in json.dumps(rep["judges"])          # nothing official leaked into / merged with the judge scores
    g = rep["graph"]
    assert (g["hypotheses"], g["probes"], g["branches"], g["interventions"], g["max_depth"]) == (2, 2, 2, 1, 0) and g["evidence_by_type"]["intervention_result"] == 1
    assert g["post_intervention_evidence"] == 1 and g["resets_all_verified"] is True and g["rejected_operations"] == 0
    s = rep["sufficiency"]["oracle"]
    events = [json.loads(l) for l in (judged / "diagnosis" / "events.jsonl").read_text().splitlines()]
    snaps = snapshots(events)
    assert s["event_seq"] == len(events) // 2 and s["cumulative_at_that_point"] == snapshot_at(snaps, s["event_seq"])
    end = snaps[-1][1]
    assert s["spent_after"]["llm_calls"] == end["llm_calls"] - s["cumulative_at_that_point"]["llm_calls"] and 0 <= s["share_after"]["llm_calls"] <= 1
    assert (judged / "report.json").exists() and "Outcomes (kept separate)" in (judged / "report.md").read_text()


def test_aggregate_over_tasks(tmp_path):
    run_dir = tmp_path / "run"
    def fake(iid, resolved, blind, causal, seq_b, seq_o):
        d = run_dir / "agent" / iid
        d.mkdir(parents=True)
        zero = {k: 0 for k in ("prompt_tokens", "completion_tokens", "llm_calls", "tool_calls", "cost_usd", "wall_s")}
        rep = {"instance_id": iid, "official": {"status": "resolved" if resolved else "unresolved", "resolved": resolved}, "diagnosis": {"status": "committed" if blind > 1 else "inconclusive"},
               "graph": {"hypotheses": 3, "evidence": 5, "relations": 7, "probes": 2, "branches": 3, "max_depth": 1, "avg_children": 0.3, "syntheses": 0, "interventions": 1, "rejected_operations": 0},
               "cost": {"at_diagnosis_end": {**zero, "llm_calls": 40, "cost_usd": 0.2}, "judges_usd": 0.1},
               "sufficiency": {"blind": {"event_seq": seq_b, "spent_after": {**zero, "llm_calls": 10, "cost_usd": 0.05}}, "oracle": {"event_seq": seq_o, "spent_after": {**zero, "llm_calls": 20, "cost_usd": 0.1}}},
               "judges": {"blind": {"present": True, "mean_score": blind, "scores": {"falsifiability": blind, "audit_clarity": 3}, "major_error_texts": [f"{iid}: weak probes"], "missed_hypotheses": ["x"], "impl_followed": "yes"},
                          "oracle": {"present": True, "scores": {"causal_correctness": causal}, "hindsight_leakage": "none", "accepted_despite_contradiction": 1 if causal < 2 else 0, "major_error_texts": [], "impl_followed": "yes"}}}
        (d / "report.json").write_text(json.dumps(rep))
    fake("t1", True, 3.5, 4, 10, 12)
    fake("t2", False, 1.0, 3, 8, 8)           # blind says poor process, oracle says correct: a disagreement
    fake("t3", False, 3.2, 1, 5, 9)           # blind says good process, oracle says wrong: a disagreement
    agg = aggregate(run_dir, ["t1", "t2", "t3"])
    assert agg["n_tasks"] == 3 and agg["judge_scores"]["blind"]["falsifiability"]["mean"] == pytest.approx(2.57, abs=0.01)
    assert agg["official_vs_diagnosis_quality"]["by_resolution"]["True"]["n"] == 1 and agg["official_vs_diagnosis_quality"]["by_resolution"]["False"]["n"] == 2
    assert {d["instance_id"] for d in agg["judge_agreement"]["disagreements"]} == {"t2", "t3"}
    assert agg["resources_after_sufficient_support"]["oracle"]["llm_calls"]["mean"] == 20 and agg["common_diagnostic_failures"]["hypothesis accepted despite contradictory evidence (oracle)"] == 1
    md = render_aggregate_md(agg)
    assert "Judge scores by dimension" in md and "Official resolution vs diagnosis quality" in md and "Common diagnostic failures" in md


# ---- the full toy run through the real orchestration code ------------------------------------------------------------------------
def test_toy_run_produces_every_required_artifact_and_the_ui_can_replay_it(home):
    rt = S3Runtime(client=Scripted(HAPPY + IMPLEMENT))
    out = run_toy("toy-e2e", runtime=rt, judge=mock_judge())
    inst = home / "runs" / "toy-e2e" / "agent" / INSTANCE_ID
    required = ["diagnosis/events.jsonl", "diagnosis/graph.json", "diagnosis/diagnosis.json", "diagnosis/controller_trace.json", "diagnosis/investigators/B1.json",
                "diagnosis/reset_verification.json", "implementation/result.json", "implementation/patch.diff", "judge/blind.json", "judge/reference_diagnosis.json",
                "judge/oracle.json", "report.json", "report.md", "ui_check.json"]
    for rel in required:
        assert (inst / rel).exists(), rel
    assert out["official"]["resolved"] is True and out["ui"]["ok"] and out["ui"]["graph_grows_monotonically"] and out["ui"]["diagnosis_loaded"]
    assert all(out["ui"]["judges_loaded"].values())
    assert json.loads((inst / "judge" / "blind.json").read_text())["judge"]["backend"] == "mock"             # the stand-in is labelled as such
    assert (home / "runs" / "toy-e2e" / "aggregate_report.md").exists()
    assert ui_check(home / "runs", "toy-e2e", INSTANCE_ID)["ok"]


def test_repository_prompts_and_config_match_the_committed_freeze():
    """If this fails, a frozen prompt, rubric or the DiagnosisConfig changed. That is only legitimate BEFORE the target list is processed:
    re-run `miracle diagnosis freeze` deliberately (and record why). After processing has started it invalidates the experiment."""
    ok, problems = fz.verify_frozen()
    assert ok, problems
