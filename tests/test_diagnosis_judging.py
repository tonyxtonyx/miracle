import json
import re
import stat
from pathlib import Path

import pytest

from miracle.diagnosis.judging import prompts as JP
from miracle.diagnosis.judging.clients import (AnthropicJudge, ClaudeCLIJudge, FileJudge, JudgeError, JudgePending, MockJudge, extract_json, make_judge)
from miracle.diagnosis.judging.inputs import CandidateMaterial, base_excerpts, gold_hunks
from miracle.diagnosis.judging.pipeline import judge_instance
from miracle.diagnosis.judging.schema import BLIND_DIMS, ORACLE_DIMS
from miracle.diagnosis.toytask import GOLD_PATCH, instance_record, write_repo
from test_diagnosis_engine import HAPPY, IMPLEMENT, env, run_s3  # noqa: F401  (fixtures)

OFFICIAL = {"status": "resolved_MARK_9d2", "resolved": True, "f2p_passed": 1, "f2p_total": 1, "p2p_passed": 3, "p2p_total": 3}


def hidden_instance():
    rec = instance_record()
    rec.update(patch=GOLD_PATCH + "\n# SENTINEL_GOLD_PATCH_91ab\n", test_patch="SENTINEL_TEST_PATCH_77cd", hints_text="SENTINEL_HINTS_55ef",
               FAIL_TO_PASS=["SENTINEL_F2P_13aa"], PASS_TO_PASS=["SENTINEL_P2P_24bb"])
    return rec


def good_blind(user, ids=("H1",)):
    n = int(re.search(r"Event numbers run 1\.\.(\d+)", user).group(1))
    dim = lambda: {"score": 3, "justification": "adequate", "cited_ids": list(ids)}
    return {"scores": {d: dim() for d in BLIND_DIMS}, "major_diagnostic_errors": [], "unsupported_conclusions": [], "missed_hypotheses": ["a rounding hypothesis"],
            "strongest_part": {"text": "the discriminating probe", "cited_ids": ["P1"]}, "earliest_supported_event_seq": n // 2,
            "implementation_followed_diagnosis": {"value": "yes", "justification": "the patch removes the second tax"}, "overall_assessment": "solid"}


def good_oracle(user):
    n = int(re.search(r"Event numbers run 1\.\.(\d+)", user).group(1))
    sc = {d: {"score": 4, "justification": "matches the reference", "cited_ids": ["H1"]} for d in ORACLE_DIMS}
    return {"scores": sc, "accepted_despite_contradiction": [], "major_diagnostic_errors": [], "missed_hypotheses": [],
            "hindsight_leakage": {"level": "none", "text": "no sign", "cited_ids": []}, "earliest_sufficient_event_seq": n // 2,
            "investigation_after_sufficient": "one more falsification branch", "implementation_followed_diagnosis": {"value": "yes", "justification": "ok"},
            "overall_assessment": "correct"}


REF = {"causal_mechanism": "REFERENCE_MARKER_c0ffee: add_tax is applied inside and after the discount branch", "key_locations": ["shop/pricing.py:total"],
       "issue_requirements": ["total(100, 10) == 99.0"], "facts_established": ["tests/test_pricing.py::test_total_with_discount_applies_tax_once fails"],
       "inferences_from_gold_patch": ["the inner add_tax is removed"], "uncertainties": [], "alternative_valid_explanations": ["remove the final add_tax only when discounted"],
       "what_a_correct_implementation_must_do": "tax exactly once"}


def mock(blind=None, oracle=None, ref=REF):
    def fn(name, system, user, schema):
        if name == "blind":
            return (blind or good_blind)(user)
        if name == "reference":
            return ref
        return (oracle or good_oracle)(user)
    return MockJudge(fn)


@pytest.fixture
def produced(env, tmp_path):
    res, c, task = run_s3(env, tmp_path)
    return tmp_path / "agent"


def run_judges(inst_dir, judge, rec=None, repo=lambda: write_repo(Path(__import__("tempfile").mkdtemp()))):
    return judge_instance(inst_dir, rec or hidden_instance(), OFFICIAL, judge, repo, log=lambda *_: None)


# ---- data boundaries -------------------------------------------------------------------------------------------------------
def test_blind_judge_never_receives_gold_hints_or_the_official_result(produced):
    j = mock()
    run_judges(produced, j)
    blind = next(c for c in j.calls if c["name"] == "blind")
    text = blind["system"] + blind["user"]
    for s in ("SENTINEL_GOLD_PATCH_91ab", "SENTINEL_TEST_PATCH_77cd", "SENTINEL_HINTS_55ef", "SENTINEL_F2P_13aa", "SENTINEL_P2P_24bb", "resolved_MARK_9d2", "REFERENCE_MARKER_c0ffee"):
        assert s not in text, s
    assert "FAIL_TO_PASS" not in text and "Gold patch" not in text and "gold patch" not in blind["user"].lower()
    assert "total() applies the tax twice" in blind["user"] and "[T3]" not in blind["user"] and "T3" in blind["user"]          # the agent's own work is shown, with provenance ids


def test_reference_is_built_from_the_instance_alone_without_any_candidate_artifact(produced):
    j = mock()
    run_judges(produced, j)
    ref = next(c for c in j.calls if c["name"] == "reference")
    t = ref["user"]
    assert "SENTINEL_GOLD_PATCH_91ab" in t and "SENTINEL_F2P_13aa" in t and "def total(price, discount=0):" in t        # gold + base code are there
    for cand in ("total() applies the tax twice", "apply_discount computes the discounted price incorrectly", "H1", "E2", "PRICE-BEFORE-FINAL-TAX", "resolved_MARK_9d2",
                 "reasoning graph", "committed diagnosis", "The agent's final patch"):
        assert cand not in t, cand


def test_oracle_sees_the_persisted_reference_then_the_candidate_and_the_official_result(produced):
    j = mock()
    run_judges(produced, j)
    assert [c["name"] for c in j.calls] == ["blind", "reference", "oracle"]
    d = produced / "judge"
    ref = json.loads((d / "reference_diagnosis.json").read_text())
    orc = json.loads((d / "oracle.json").read_text())
    assert ref["persisted_at"] <= orc["oracle_prompt_built_at"] and orc["reference_sha256"] == ref["reference_sha256"]
    t = next(c for c in j.calls if c["name"] == "oracle")["user"]
    assert "REFERENCE_MARKER_c0ffee" in t and "SENTINEL_GOLD_PATCH_91ab" in t and "resolved_MARK_9d2" in t and "total() applies the tax twice" in t
    assert t.index("REFERENCE_MARKER_c0ffee") < t.index("THE AGENT'S WORK")                                    # reference first, candidate after
    assert "SENTINEL_HINTS_55ef" not in t


def test_oracle_uses_the_reference_on_disk_not_a_regenerated_one(produced):
    d = produced / "judge"
    d.mkdir(parents=True, exist_ok=True)
    persisted = {"persisted_at": "2000-01-01T00:00:00.000000+00:00", "reference_sha256": "x" * 64, "reference": {**REF, "causal_mechanism": "PERSISTED_BEFORE_THE_RUN_7731"}}
    (d / "reference_diagnosis.json").write_text(json.dumps(persisted))
    j = mock()
    run_judges(produced, j)
    assert "reference" not in [c["name"] for c in j.calls]                                                       # not regenerated
    assert "PERSISTED_BEFORE_THE_RUN_7731" in next(c for c in j.calls if c["name"] == "oracle")["user"]


def test_candidate_material_has_no_slot_for_hidden_fields():
    import dataclasses
    assert {f.name for f in dataclasses.fields(CandidateMaterial)} == {"issue", "state", "event_count", "tool_calls", "implementation_patch"}
    import inspect
    from miracle.diagnosis.judging import inputs
    assert list(inspect.signature(inputs.blind_user_prompt).parameters) == ["m"]
    assert list(inspect.signature(inputs.reference_user_prompt).parameters) == ["instance", "excerpts"]            # no candidate parameter
    assert "m" in inspect.signature(inputs.oracle_user_prompt).parameters


# ---- citations & validation -----------------------------------------------------------------------------------------------------
def test_blind_judge_must_cite_existing_ids_and_gets_one_repair_round(produced):
    calls = {"n": 0}

    def bad_then_good(user):
        calls["n"] += 1
        o = good_blind(user)
        if calls["n"] == 1:
            o["scores"]["falsifiability"] = {"score": 1, "justification": "weak", "cited_ids": ["H99"]}                 # does not exist
            o["major_diagnostic_errors"] = [{"text": "something is wrong", "cited_ids": []}]                           # cites nothing
        return o
    run_judges(produced, mock(blind=bad_then_good))
    rec = json.loads((produced / "judge" / "blind.json").read_text())
    assert calls["n"] == 2 and rec["valid"] and len(rec["attempts"]) == 2 and rec["problems"] == []
    assert rec["result"]["scores"]["hypothesis_quality"]["cited_ids"] == ["proj__missing:H1"] or all(":" in i for i in rec["result"]["scores"]["hypothesis_quality"]["cited_ids"])
    assert (produced / "judge" / "raw" / "blind.attempt1.json").exists() and (produced / "judge" / "raw" / "blind.attempt2.json").exists()


def test_unrepairable_judge_output_is_kept_but_flagged_invalid(produced):
    def always_bad(user):
        o = good_blind(user)
        o["scores"]["audit_clarity"] = {"score": 7, "justification": "x", "cited_ids": ["E404"]}
        return o
    run_judges(produced, mock(blind=always_bad))
    rec = json.loads((produced / "judge" / "blind.json").read_text())
    assert rec["valid"] is False and any("audit_clarity" in p for p in rec["problems"]) and len(rec["attempts"]) == 2


def test_low_scores_require_citations_and_event_numbers_are_checked(produced):
    def lowscore(user):
        o = good_blind(user)
        o["scores"]["probe_discrimination"] = {"score": 1, "justification": "probes did not discriminate", "cited_ids": []}
        o["earliest_supported_event_seq"] = 10_000
        return o
    run_judges(produced, mock(blind=lowscore))
    rec = json.loads((produced / "judge" / "blind.json").read_text())
    assert not rec["valid"] and any("probe_discrimination" in p and "at least one" in p for p in rec["problems"]) and any("earliest_supported_event_seq" in p for p in rec["problems"])


def test_judging_is_resumable_and_force_reruns(produced):
    j = mock()
    run_judges(produced, j)
    n = len(j.calls)
    run_judges(produced, j)
    assert len(j.calls) == n
    judge_instance(produced, hidden_instance(), OFFICIAL, j, lambda: write_repo(Path(__import__("tempfile").mkdtemp())), log=lambda *_: None, force=True)
    assert len(j.calls) == 2 * n


def test_gold_hunk_and_base_excerpt_helpers(tmp_path):
    assert gold_hunks(GOLD_PATCH) == {"shop/pricing.py": [(13, 18)]}
    repo = write_repo(tmp_path / "r")
    ex = base_excerpts(repo, GOLD_PATCH)
    assert "shop/pricing.py" in ex and "price = add_tax(apply_discount(price, discount))" in ex["shop/pricing.py"]


# ---- backends ---------------------------------------------------------------------------------------------------------------------
def test_extract_json_variants():
    assert extract_json('{"a": 1}') == {"a": 1} and extract_json('text ```json\n{"a": 2}\n``` more') == {"a": 2} and extract_json('preamble {"a": 3} tail') == {"a": 3}
    with pytest.raises(JudgeError):
        extract_json("no json here")
    with pytest.raises(JudgeError):
        extract_json("[1, 2]")


def test_file_judge_round_trip(tmp_path):
    fj = FileJudge(tmp_path)
    with pytest.raises(JudgePending) as e:
        fj.complete("sys", "user text", {"type": "object"}, "blind")
    assert json.loads((tmp_path / "blind.prompt.json").read_text())["user"] == "user text" and e.value.path == tmp_path / "blind.response.json"
    (tmp_path / "blind.response.json").write_text('```json\n{"scores": {}}\n```')
    assert fj.complete("sys", "user text", {}, "blind").data == {"scores": {}}


def _fake_claude(tmp_path, stdout, code=0):
    p = tmp_path / "claude"
    p.write_text(f"#!/bin/sh\ncat <<'EOF'\n{stdout}\nEOF\nexit {code}\n")
    p.chmod(p.stat().st_mode | stat.S_IEXEC)
    return str(p)


def test_claude_cli_judge_parses_structured_output_and_errors(tmp_path):
    ok = ClaudeCLIJudge(executable=_fake_claude(tmp_path, json.dumps({"type": "result", "is_error": False, "result": "x", "structured_output": {"v": 1},
                                                                       "total_cost_usd": 0.5, "usage": {"input_tokens": 3}})))
    r = ok.complete("s", "u", {}, "blind")
    assert r.data == {"v": 1} and r.cost_usd == 0.5 and r.backend == "claude-cli"
    txt = ClaudeCLIJudge(executable=_fake_claude(tmp_path, json.dumps({"is_error": False, "result": 'here: {"v": 2}'})))
    assert txt.complete("s", "u", {}, "blind").data == {"v": 2}
    bad = ClaudeCLIJudge(executable=_fake_claude(tmp_path, json.dumps({"is_error": True, "result": "Not logged in · Please run /login"})))
    with pytest.raises(JudgeError, match="Not logged in"):
        bad.complete("s", "u", {}, "blind")
    with pytest.raises(JudgeError, match="failed to run"):
        ClaudeCLIJudge(executable="/nonexistent/claude").complete("s", "u", {}, "blind")


class _Stream:
    def __init__(self, msg, boom=None):
        self.msg, self.boom = msg, boom

    def __enter__(self):
        if self.boom:
            raise self.boom
        return self

    def __exit__(self, *a):
        return False

    def get_final_message(self):
        return self.msg


class _Block:
    def __init__(self, type, text=""):
        self.type, self.text = type, text


class _Msg:
    def __init__(self, text='{"ok": true}', stop="end_turn"):
        self.content = [_Block("thinking"), _Block("text", text)]
        self.stop_reason, self.stop_details = stop, None
        self.usage = type("U", (), {"input_tokens": 1000, "output_tokens": 500})()
        self._request_id = "req_1"


class _FakeAnthropic:
    def __init__(self, beta_boom=None, msg=None):
        self.requests, self.msg, self.beta_boom = [], msg or _Msg(), beta_boom
        outer = self
        self.messages = type("M", (), {"stream": lambda s, **kw: outer._rec("plain", kw)})()
        self.beta = type("B", (), {"messages": type("M", (), {"stream": lambda s, **kw: outer._rec("beta", kw, outer.beta_boom)})()})()

    def _rec(self, kind, kw, boom=None):
        self.requests.append((kind, kw))
        return _Stream(self.msg, boom)


def test_anthropic_judge_request_shape_and_cost():
    fake = _FakeAnthropic()
    r = AnthropicJudge(client=fake).complete("SYS", "USER", {"type": "object"}, "blind")
    kind, kw = fake.requests[0]
    assert kind == "beta" and kw["model"] == "claude-opus-5-5" and kw["thinking"] == {"type": "adaptive"} and kw["fallbacks"] == "default"
    assert kw["betas"] == ["server-side-fallback-2026-07-01"] and kw["output_config"]["format"] == {"type": "json_schema", "schema": {"type": "object"}}
    assert kw["system"] == "SYS" and kw["messages"] == [{"role": "user", "content": "USER"}]
    assert r.data == {"ok": True} and r.cost_usd == pytest.approx(1000 * 4 / 1e6 + 500 * 20 / 1e6) and r.meta["request_id"] == "req_1"


def test_anthropic_judge_retries_without_fallback_beta_if_rejected_and_handles_refusal_and_truncation():
    fake = _FakeAnthropic(beta_boom=RuntimeError("unknown beta: server-side-fallback"))
    assert AnthropicJudge(client=fake).complete("S", "U", {}, "x").data == {"ok": True}
    assert [k for k, _ in fake.requests] == ["beta", "plain"] and "fallbacks" not in fake.requests[1][1]
    with pytest.raises(JudgeError, match="refused"):
        AnthropicJudge(client=_FakeAnthropic(msg=_Msg(stop="refusal"))).complete("S", "U", {}, "x")
    with pytest.raises(JudgeError, match="max_tokens"):
        AnthropicJudge(client=_FakeAnthropic(msg=_Msg(stop="max_tokens"))).complete("S", "U", {}, "x")
    with pytest.raises(JudgeError, match="RuntimeError"):
        AnthropicJudge(client=_FakeAnthropic(beta_boom=RuntimeError("network down")), fallbacks=True).complete("S", "U", {}, "x")


def test_make_judge_selection(tmp_path, monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    assert isinstance(make_judge("cli"), ClaudeCLIJudge) and isinstance(make_judge("files", tmp_path), FileJudge)
    monkeypatch.setattr("miracle.llm.config.load_dotenv", lambda *a, **k: None)
    assert isinstance(make_judge("auto"), ClaudeCLIJudge)                                         # no key -> CLI
    with pytest.raises(ValueError):
        make_judge("nope")


def test_judge_prompts_are_hashed_for_the_freeze():
    h = JP.judge_prompt_hashes()
    assert set(h) == {"BLIND_SYSTEM", "REFERENCE_SYSTEM", "ORACLE_SYSTEM"} and all(len(v) == 64 for v in h.values())
    from miracle.diagnosis.freeze import current_manifest
    assert current_manifest()["judge_prompts"] == h
