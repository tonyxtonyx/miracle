import json
import urllib.error
import urllib.request
from pathlib import Path

import pytest

from miracle.diagnosis.graph import read_events, replay
from miracle.diagnosis.web import api
from miracle.diagnosis.web.server import start_in_thread
from test_diagnosis_engine import env, run_s3  # noqa: F401


@pytest.fixture
def runs(env, tmp_path):
    run_s3(env, tmp_path)                                       # writes tmp_path/agent/{diagnosis,implementation,...}
    root = tmp_path / "runs"
    (root / "toyrun").mkdir(parents=True)
    (tmp_path / "agent").rename(root / "toyrun" / "agent")
    iid = next((root / "toyrun" / "agent").iterdir()).name if False else None
    inner = root / "toyrun" / "agent"
    # the runner stores each instance under agent/<instance_id>/; mimic that layout
    (inner / "toy__shop-1").mkdir()
    for child in list(inner.iterdir()):
        if child.name != "toy__shop-1":
            child.rename(inner / "toy__shop-1" / child.name)
    (root / "toyrun" / "results.jsonl").write_text(json.dumps({"instance_id": "toy__shop-1", "eval": {"status": "resolved", "resolved": True, "f2p_passed": 1, "f2p_total": 1}}) + "\n")
    return root


def get(url, expect=200):
    try:
        with urllib.request.urlopen(url, timeout=10) as r:
            return r.status, r.read(), r.headers
    except urllib.error.HTTPError as e:
        return e.code, e.read(), e.headers


@pytest.fixture
def server(runs):
    srv, base = start_in_thread(runs, default_run="toyrun", default_instance="toy__shop-1")
    yield base
    srv.shutdown()


def test_api_lists_runs_instances_and_the_graph_replays_at_any_step(runs):
    assert api.list_runs(runs) == [{"name": "toyrun", "instances": 1}]
    inst = api.list_instances(runs, "toyrun")[0]
    assert inst["id"] == "toy__shop-1" and inst["diagnosis_status"] == "committed" and inst["official"]["resolved"] is True and inst["phase"] == "committed"
    total = api.graph_at(runs, "toyrun", "toy__shop-1", None)["total_events"]
    full = api.graph_at(runs, "toyrun", "toy__shop-1", None)["graph"]
    events = read_events(runs / "toyrun" / "agent" / "toy__shop-1" / "diagnosis" / "events.jsonl")
    assert total == len(events) and full == replay("toy__shop-1", events).to_dict()
    for upto in (0, 3, total // 2, total):
        assert api.graph_at(runs, "toyrun", "toy__shop-1", upto)["graph"] == replay("toy__shop-1", events, upto).to_dict()
    assert api.graph_at(runs, "toyrun", "toy__shop-1", 0)["graph"]["hypotheses"] == {} and api.graph_at(runs, "toyrun", "toy__shop-1", 10_000)["upto"] == total


def test_http_endpoints_and_static_files(server):
    code, body, hdr = get(server + "/")
    assert code == 200 and b"Diagnosis graph" in body and "text/html" in hdr["Content-Type"]
    assert get(server + "/app.js")[0] == 200 and get(server + "/style.css")[0] == 200
    cfg = json.loads(get(server + "/api/config")[1])
    assert cfg == {"run": "toyrun", "instance": "toy__shop-1"}
    q = "run=toyrun&instance=toy__shop-1"
    g = json.loads(get(f"{server}/api/graph?{q}&upto=8")[1])
    assert g["upto"] == 8 and g["graph"]["seq"] == 8
    ev = json.loads(get(f"{server}/api/events?{q}")[1])
    assert ev[0]["seq"] == 1 and {e["type"] for e in ev} >= {"hypothesis_created", "evidence_attached", "diagnosis_committed"}
    art = json.loads(get(f"{server}/api/artifacts?{q}")[1])
    assert art["diagnosis"]["status"] == "committed" and art["official"]["resolved"] is True and art["resets"] and art["implementation"]["patch_bytes"] > 0
    calls = json.loads(get(f"{server}/api/tool_calls?{q}")[1])
    assert any(c["name"] == "run_command" and c["intervention_id"] for c in calls)
    m = json.loads(get(f"{server}/api/meta?{q}")[1])
    assert m["events"] == len(ev) and m["done"] is True


def test_server_is_read_only_and_rejects_traversal_and_unknown_ids(server):
    for bad in ("/api/graph?run=..&instance=x", "/api/graph?run=toyrun&instance=../../etc", "/api/graph?run=nope&instance=x", "/api/graph?run=toyrun&instance=toy__shop-1&upto=abc",
                "/../../etc/passwd", "/%2e%2e/%2e%2e/etc/passwd", "/api/nothing", "/static/../../server.py"):
        code, _, _ = get(server + bad)
        assert code in (400, 404), bad
    req = urllib.request.Request(server + "/api/graph?run=toyrun&instance=toy__shop-1", method="POST", data=b"{}")
    with pytest.raises(urllib.error.HTTPError) as e:
        urllib.request.urlopen(req, timeout=10)
    assert e.value.code == 405


def test_ui_needs_no_external_resources_and_does_not_use_innerhtml():
    static = Path(api.__file__).parent / "static"
    for f in static.iterdir():
        src = f.read_text()
        assert "innerHTML" not in src, f.name                                  # excerpts are untrusted text: textContent only
        assert "http://" not in src.replace("http://www.w3.org/2000/svg", "") and "https://" not in src and "cdn" not in src.lower(), f.name
    html = (static / "index.html").read_text()
    assert "<script src=\"app.js\">" in html and "stylesheet" in html


def test_ui_loads_for_a_run_in_progress(runs):
    """A partial event log (run still going) must be readable: the UI polls /api/meta and re-reads events."""
    p = runs / "toyrun" / "agent" / "toy__shop-1" / "diagnosis"
    (p / "graph.json").unlink()
    lines = (p / "events.jsonl").read_text().splitlines()[:25]
    (p / "events.jsonl").write_text("\n".join(lines) + "\n")
    m = api.meta(runs, "toyrun", "toy__shop-1")
    assert m["events"] == 25 and m["done"] is False
    g = api.graph_at(runs, "toyrun", "toy__shop-1", None)
    assert g["graph"]["diagnosis"] is None and g["total_events"] == 25
