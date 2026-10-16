"""Run the EXISTING completion checker on every sample and measure how safe a COMPLETE verdict is.

Nothing here modifies the checker: decisions come from `CompletionController._check` (the exact code path of the
online controller: same prompt builder, same call options, same parser). Positive = COMPLETE (the checker would
terminate the run); any other outcome (INCOMPLETE, INVALID, ERROR) continues the run, i.e. counts as not-COMPLETE.
"""
from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from ..agents.s2 import CompletionController, S2Runtime
from ..dataset import read_jsonl
from ..runner import Run
from ..stats import wilson
from .dataset import LABELLED


def run_checks(work: Path, run_names: list[str], repeats: int = 3, workers: int = 4, log=print) -> int:
    samples = read_jsonl(work / "samples.jsonl")
    issues = {}
    for n in run_names:
        issues.update({i: v["problem_statement"] for i, v in Run(n).instances().items()})
    dpath = work / "decisions.jsonl"
    done = {(d["sample_id"], d["rep"]) for d in read_jsonl(dpath)} if dpath.exists() else set()
    client = S2Runtime().client                      # the S2 model/decoding config, unchanged
    todo = [(s, r) for r in range(repeats) for s in samples if (s["sample_id"], r) not in done]

    def one(args):
        s, rep = args
        ctl = CompletionController(client, None, issues[s["instance_id"]])
        ctl.tracker._tests = [{**t, "summary": {}} for t in s["tests"]]    # the online inputs for this sample
        rec = ctl._check(s["step"] or 0, s["trigger"] or {"rule": "offline"}, s["diff"], s["diff_sha256"])
        assert rec["prompt"] == s["prompt"], "checker input differs from the stored sample input"
        return {"sample_id": s["sample_id"], "rep": rep, **{k: rec[k] for k in (
            "decision", "reason", "raw_response", "finish_reason", "prompt_tokens", "completion_tokens", "cost_usd", "latency_s")}}

    with ThreadPoolExecutor(workers) as ex, open(dpath, "a") as f:
        for i, d in enumerate(ex.map(one, todo), 1):
            f.write(json.dumps(d, ensure_ascii=False) + "\n")
            f.flush()
            if i % 20 == 0:
                log(f"  {i}/{len(todo)} checks done")
    return len(todo)


# ---- metrics (pure) ---------------------------------------------------------------------------------------------
def confusion(pairs: list[tuple[str, str]]) -> dict:
    """pairs: (label RESOLVED|UNRESOLVED, decision). Positive = COMPLETE."""
    tp = sum(l == "RESOLVED" and d == "COMPLETE" for l, d in pairs)
    fp = sum(l == "UNRESOLVED" and d == "COMPLETE" for l, d in pairs)
    fn = sum(l == "RESOLVED" and d != "COMPLETE" for l, d in pairs)
    tn = sum(l == "UNRESOLVED" and d != "COMPLETE" for l, d in pairs)
    div = lambda a, b: (a / b) if b else None
    ci = lambda k, n: tuple(round(x, 3) for x in wilson(k, n)) if n else None
    return {"n": len(pairs), "TP": tp, "FP": fp, "FN": fn, "TN": tn,
            "complete_precision": div(tp, tp + fp), "precision_ci95": ci(tp, tp + fp),
            "complete_recall": div(tp, tp + fn), "false_complete_rate": div(fp, fp + tn),
            "base_rate_resolved": div(tp + fn, len(pairs)),
            "invalid_or_error": sum(d in ("INVALID", "ERROR") for _, d in pairs)}


def subsets(samples: list[dict]) -> dict[str, list[dict]]:
    lab = [s for s in samples if s["label"] in LABELLED]
    return {"online_eligible (primary)": [s for s in lab if s["online_eligible"]],
            "all labelled": lab,
            "final, all": [s for s in lab if s["kind"] == "final"],
            "final, online-eligible": [s for s in lab if s["kind"] == "final" and s["online_eligible"]],
            "final, not eligible (stress test)": [s for s in lab if s["kind"] == "final" and not s["online_eligible"]],
            "intermediate": [s for s in lab if s["kind"] == "intermediate"]}


def report(work: Path) -> dict:
    samples = read_jsonl(work / "samples.jsonl")
    dec = read_jsonl(work / "decisions.jsonl")
    reps = sorted({d["rep"] for d in dec})
    by = {(d["sample_id"], d["rep"]): d for d in dec}
    out = {"repeats": len(reps), "samples": len(samples), "label_counts": {}, "by_subset": {}, "stability": {}, "cost": {}}
    for s in samples:
        out["label_counts"][f"{s['kind']}/{s['label']}"] = out["label_counts"].get(f"{s['kind']}/{s['label']}", 0) + 1
    for name, ss in subsets(samples).items():
        out["by_subset"][name] = {f"rep{r}": confusion([(s["label"], by[(s["sample_id"], r)]["decision"]) for s in ss
                                                         if (s["sample_id"], r) in by]) for r in reps}
        out["by_subset"][name]["samples"] = len(ss)
        votes = [(s["label"], "COMPLETE" if sum(by[(s["sample_id"], r)]["decision"] == "COMPLETE" for r in reps) * 2 > len(reps) else "INCOMPLETE")
                 for s in ss if all((s["sample_id"], r) in by for r in reps)]
        out["by_subset"][name]["majority_vote"] = confusion(votes)
    full = [s for s in samples if all((s["sample_id"], r) in by for r in reps)]
    uni = sum(len({by[(s["sample_id"], r)]["decision"] for r in reps}) == 1 for s in full)
    out["stability"] = {"samples_with_identical_decision_across_repeats": uni, "of": len(full)}
    cost = [d["cost_usd"] or 0 for d in dec]
    out["cost"] = {"calls": len(dec), "prompt_tokens": sum(d["prompt_tokens"] for d in dec),
                   "completion_tokens": sum(d["completion_tokens"] for d in dec), "usd": round(sum(cost), 6),
                   "usd_per_call": round(sum(cost) / max(len(dec), 1), 6),
                   "mean_latency_s": round(sum(d["latency_s"] for d in dec) / max(len(dec), 1), 2)}
    out["decision_counts"] = {k: sum(d["decision"] == k for d in dec) for k in ("COMPLETE", "INCOMPLETE", "INVALID", "ERROR")}
    (work / "report.json").write_text(json.dumps(out, indent=2, sort_keys=True))
    (work / "report.md").write_text(markdown(out, samples, by, reps))
    return out


def _fmt(x, pct=True):
    return "n/a" if x is None else (f"{x:.0%}" if pct else str(x))


def markdown(out: dict, samples: list[dict], by: dict, reps: list[int]) -> str:
    L = ["| subset | n | TP | FP | FN | TN | **COMPLETE precision** (95% CI) | COMPLETE recall | false-COMPLETE rate | base rate resolved |",
         "|---|---|---|---|---|---|---|---|---|---|"]
    for name, v in out["by_subset"].items():
        c = v["rep0"] if "rep0" in v else None
        if not c:
            continue
        ci = c["precision_ci95"]
        L.append(f"| {name} | {c['n']} | {c['TP']} | {c['FP']} | {c['FN']} | {c['TN']} | "
                 f"**{_fmt(c['complete_precision'])}**{' (%s–%s)' % (f'{ci[0]:.0%}', f'{ci[1]:.0%}') if ci else ''} | "
                 f"{_fmt(c['complete_recall'])} | {_fmt(c['false_complete_rate'])} | {_fmt(c['base_rate_resolved'])} |")
    L += ["", "Per-sample decisions (decisions of repeats 0.." + str(len(reps) - 1) + "; C = COMPLETE, I = INCOMPLETE, ? = INVALID/ERROR):", "",
          "| sample | kind | label | online-eligible | decisions | first INCOMPLETE reason |", "|---|---|---|---|---|---|"]
    for s in samples:
        ds = [by.get((s["sample_id"], r)) for r in reps]
        code = "".join({"COMPLETE": "C", "INCOMPLETE": "I"}.get(d["decision"], "?") if d else "-" for d in ds)
        why = next((d["reason"] for d in ds if d and d["decision"] == "INCOMPLETE"), "")
        L.append(f"| {s['sample_id']} | {s['kind']} | {s['label']} | {s['online_eligible']} | {code} | {why[:110]} |")
    return "\n".join(L)
