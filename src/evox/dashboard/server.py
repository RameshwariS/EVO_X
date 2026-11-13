"""Local read-only HTTP server (standard library only; GET only; loopback by default)."""
from __future__ import annotations

import logging
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from evox.dashboard.api import handle
from evox.dashboard.store import RunStore

log = logging.getLogger("dashboard")


def make_server(runs_dir: Path, host: str = "127.0.0.1", port: int = 8000) -> ThreadingHTTPServer:
    store = RunStore(runs_dir)

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):  # noqa: N802 (stdlib naming)
            status, ctype, body = handle(store, self.path)
            self.send_response(status)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.send_header("Content-Security-Policy", "default-src 'self'; script-src 'unsafe-inline'; style-src 'unsafe-inline'")
            self.end_headers()
            self.wfile.write(body)

        def _refuse(self):
            self.send_error(405, "read-only dashboard")

        do_POST = do_PUT = do_DELETE = do_PATCH = _refuse

        def log_message(self, fmt, *args):
            log.debug(fmt, *args)

    return ThreadingHTTPServer((host, port), Handler)
