import json

from miracle import evaluator
from miracle.predictions import write_predictions
import pytest


def _logdir(eval_dir, run_id, model, iid):
    d = eval_dir / "logs" / "run_evaluation" / run_id / model / iid
    d.mkdir(parents=True)
    return d


def test_status_classification_from_harness_artefacts(tmp_path):
    run, ev = tmp_path, tmp_path / "eval"
    inst = {i: {"image": None} for i in ("ok", "bad", "apply", "slow", "crash", "empty", "never")}
    preds = {i: {"instance_id": i, "model_name_or_path": "m/x", "model_patch": "diff"} for i in inst}
    preds["empty"]["model_patch"] = "  \n"
    L = lambda i: _logdir(ev, "r", "m__x", i)

    d = L("ok"); (d / "run_instance.log").write_text("Test runtime: 1_234.50 seconds")
    (d / "report.json").write_text(json.dumps({"ok": {"resolved": True, "tests_status": {
        "FAIL_TO_PASS": {"success": ["a"], "failure": []}, "PASS_TO_PASS": {"success": ["b"], "failure": ["c"]}}}}))
    d = L("bad"); (d / "report.json").write_text(json.dumps({"bad": {"resolved": False}}))
    (L("apply") / "run_instance.log").write_text(f"{evaluator.APPLY_PATCH_FAIL}:\nerror")
    d = L("slow"); (d / "run_instance.log").write_text("x"); (d / "test_output.txt").write_text("Timeout error: 5 seconds exceeded.")
    (L("crash") / "run_instance.log").write_text("Traceback...")
    # "never": no log dir at all

    out = evaluator.collect(run, ev, "r", preds, inst, None)
    assert {i: e.status for i, e in out.items()} == {
        "ok": "resolved", "bad": "unresolved", "apply": "patch_apply_failed", "slow": "timeout",
        "crash": "eval_error", "empty": "empty_patch", "never": "not_evaluated"}
    assert out["ok"].test_runtime_s == 1234.5 and (out["ok"].p2p_passed, out["ok"].p2p_total) == (1, 2)


def test_predictions_are_official_format_only(tmp_path):
    p = tmp_path / "p.jsonl"
    write_predictions(p, [{"instance_id": "a", "model_name_or_path": "m", "model_patch": "x", "extra": 1}])
    assert list(json.loads(p.read_text())) == ["instance_id", "model_name_or_path", "model_patch"]
    with pytest.raises(ValueError):
        write_predictions(p, [{"instance_id": "a", "model_name_or_path": "m", "model_patch": ""}] * 2)
