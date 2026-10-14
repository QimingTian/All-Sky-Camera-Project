#!/usr/bin/env python3
"""Tiny local server for the star labeling page.

GET  /                -> static/index.html
GET  /api/labels      -> current labels.json
POST /api/labels      -> overwrite labels.json
"""

import json
import os
import sys
from http.server import HTTPServer, SimpleHTTPRequestHandler

ROOT = os.path.dirname(os.path.abspath(__file__))
STATIC = os.path.join(ROOT, "static")
LABELS = os.path.join(ROOT, "labels.json")


class Handler(SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=STATIC, **kwargs)

    def log_message(self, fmt, *args):
        if "/api/" in (self.path or ""):
            super().log_message(fmt, *args)

    def do_GET(self):
        if self.path.split("?")[0] == "/api/labels":
            data = b"{}"
            if os.path.exists(LABELS):
                with open(LABELS, "rb") as f:
                    data = f.read()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(data)
            return
        super().do_GET()

    def do_POST(self):
        if self.path.split("?")[0] != "/api/labels":
            self.send_error(404)
            return
        length = int(self.headers.get("Content-Length") or 0)
        raw = self.rfile.read(length)
        try:
            payload = json.loads(raw.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            self.send_error(400, f"bad json: {exc}")
            return
        tmp = LABELS + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
        os.replace(tmp, LABELS)
        body = json.dumps({"ok": True, "count": len(payload.get("stars", []))}).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


def main():
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 8777
    server = HTTPServer(("127.0.0.1", port), Handler)
    print(f"star labeler -> http://localhost:{port}")
    print(f"labels file   -> {LABELS}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nbye")


if __name__ == "__main__":
    main()
