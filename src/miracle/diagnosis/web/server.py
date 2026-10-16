"""`miracle graph serve`: a tiny local read-only server (stdlib only). Binds to 127.0.0.1; no writes; no network dependencies."""
from __future__ import annotations

import json
import mimetypes
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

from . import api

STATIC = Path(__file__).parent / "static"


def make_server(runs_dir: Path, port: int = 8765, host: str = "127.0.0.1", default_run: str | None = None, default_instance: str | None = None):
    runs_dir = Path(runs_dir)

    class H(BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def _send(self, code: int, body: bytes, ctype: str):
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def _json(self, obj, code=200):
            self._send(code, json.dumps(obj, ensure_ascii=False, default=str).encode(), "application/json; charset=utf-8")

        def do_GET(self):
            u = urlparse(self.path)
            q = {k: v[0] for k, v in parse_qs(u.query).items()}
            try:
                if u.path == "/api/config":
                    return self._json({"run": default_run, "instance": default_instance})
                if u.path == "/api/runs":
                    return self._json(api.list_runs(runs_dir))
                if u.path == "/api/instances":
                    return self._json(api.list_instances(runs_dir, q.get("run", "")))
                if u.path == "/api/meta":
                    return self._json(api.meta(runs_dir, q.get("run", ""), q.get("instance", "")))
                if u.path == "/api/graph":
                    return self._json(api.graph_at(runs_dir, q.get("run", ""), q.get("instance", ""), int(q["upto"]) if "upto" in q else None))
                if u.path == "/api/events":
                    return self._json(api.events(runs_dir, q.get("run", ""), q.get("instance", "")))
                if u.path == "/api/tool_calls":
                    return self._json(api.tool_calls(runs_dir, q.get("run", ""), q.get("instance", "")))
                if u.path == "/api/artifacts":
                    return self._json(api.artifacts(runs_dir, q.get("run", ""), q.get("instance", "")))
                if u.path.startswith("/api/"):
                    return self._json({"error": "unknown endpoint"}, 404)
                name = "index.html" if u.path in ("/", "") else u.path.lstrip("/")
                f = (STATIC / name).resolve()
                if STATIC.resolve() not in f.parents or not f.is_file():
                    return self._json({"error": "not found"}, 404)
                self._send(200, f.read_bytes(), (mimetypes.guess_type(name)[0] or "text/plain") + "; charset=utf-8")
            except api.NotFound as e:
                self._json({"error": str(e)}, 404)
            except (ValueError, KeyError) as e:
                self._json({"error": f"bad request: {e}"}, 400)

        def do_POST(self):          # read-only by design
            self._json({"error": "read-only"}, 405)

        do_PUT = do_DELETE = do_PATCH = do_POST

    return ThreadingHTTPServer((host, port), H)


def serve(runs_dir: Path, run: str | None, instance: str | None, port: int, open_browser: bool = False) -> None:
    srv = make_server(runs_dir, port, default_run=run, default_instance=instance)
    url = f"http://127.0.0.1:{srv.server_address[1]}/"
    print(f"serving {runs_dir} at {url}  (read-only; Ctrl-C to stop)")
    if open_browser:
        import webbrowser
        webbrowser.open(url)
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        srv.server_close()


def start_in_thread(runs_dir: Path, **kw):
    srv = make_server(runs_dir, 0, **kw)
    t = threading.Thread(target=srv.serve_forever, daemon=True)
    t.start()
    return srv, f"http://127.0.0.1:{srv.server_address[1]}"
