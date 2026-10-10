"""Local server for the blind manual review of dataset_noto2024_v0 (standard library only).

Serves the dataset folder read-only and stores every answer on disk at once:
  dataset_noto2024_v0/review/labels_<reviewer>.csv  (append-only log; the newest row per sample wins)

Usage:
  python road_v2/review_server.py [port]        (default 8770, listens on 127.0.0.1 only)
Then open http://127.0.0.1:8770/review/review.html
Build the page first with road_v2/make_noto_review.py.
"""
from __future__ import annotations

import csv
import json
import re
import sys
import threading
import time
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

ROOT = Path(__file__).resolve().parent
DS = ROOT / "dataset_noto2024_v0"
REVIEW = DS / "review"
LABELS = {"blocked", "partial", "clear", "unclear", "invalid", "snow", "occluded"}
LOCK = threading.Lock()


def reviewer_file(name: str) -> Path:
    safe = re.sub(r"[^\w\-]", "_", name.strip())[:40] or "anon"
    return REVIEW / f"labels_{safe}.csv"


def read_labels(name: str) -> dict:
    path, out = reviewer_file(name), {}
    if path.exists():
        with open(path, encoding="utf-8") as fh:
            for r in csv.DictReader(fh):
                out[r["sample_id"]] = {"label": r["label"], "note": r["note"]}
    return out


class Handler(SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(DS), **kwargs)

    def log_message(self, fmt, *args):  # keep the console quiet; answers are in the CSV
        pass

    def end_headers(self):
        self.send_header("Cache-Control", "no-store")
        super().end_headers()

    def _json(self, obj, code=200):
        body = json.dumps(obj, ensure_ascii=False).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        url = urlparse(self.path)
        if url.path == "/api/labels":
            return self._json(read_labels(parse_qs(url.query).get("reviewer", [""])[0]))
        return super().do_GET()

    def do_POST(self):
        if urlparse(self.path).path != "/api/label":
            return self._json({"error": "not found"}, 404)
        try:
            d = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))).decode("utf-8"))
            who, sid, label, note = str(d["reviewer"]), str(d["sample_id"]), str(d["label"]), str(d.get("note", ""))[:500]
            if not who.strip() or not re.fullmatch(r"noto_\d{5}", sid) or label not in LABELS:
                raise ValueError("reviewer, sample_id or label not valid")
        except Exception as e:  # malformed request from the page
            return self._json({"error": str(e)}, 400)
        with LOCK:
            path = reviewer_file(who)
            new = not path.exists()
            with open(path, "a", newline="", encoding="utf-8") as fh:
                w = csv.writer(fh)
                if new:
                    w.writerow(["time", "reviewer", "sample_id", "label", "note"])
                w.writerow([time.strftime("%Y-%m-%d %H:%M:%S"), who.strip(), sid, label, note])
        return self._json({"ok": True})


def main() -> int:
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 8770
    if not (REVIEW / "review.html").exists():
        print("review/review.html missing: run road_v2/make_noto_review.py first")
        return 1
    srv = ThreadingHTTPServer(("127.0.0.1", port), Handler)
    print(f"review server: http://127.0.0.1:{port}/review/review.html  (answers -> {REVIEW})", flush=True)
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
