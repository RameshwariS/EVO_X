"""Routing shared by the HTTP server and the static exporter (one code path, so they cannot diverge)."""
from __future__ import annotations

import json
from urllib.parse import parse_qs, urlparse

from evox.dashboard.page import PAGE
from evox.dashboard.store import RunStore

# endpoint -> ordered argument names; the snapshot key is "endpoint|arg1|arg2" (also built by the page's JS)
ENDPOINTS = dict(runs=(), overview=("run",), registries=("run",), timeline=("run", "registry"),
                 decisions=("run",), results=("run",), batches=("run", "scenario", "system"))


def snapshot_key(endpoint: str, *args: str) -> str:
    return "|".join((endpoint, *args))


def call(store: RunStore, endpoint: str, *args: str):
    if endpoint not in ENDPOINTS or len(args) != len(ENDPOINTS[endpoint]):
        raise KeyError(f"Unknown endpoint {endpoint!r}")
    return getattr(store, endpoint)(*args)


def handle(store: RunStore, target: str) -> tuple[int, str, bytes]:
    """GET ``target`` -> (status, content type, body).  Never raises."""
    url = urlparse(target)
    if url.path in ("/", "/index.html"):
        return 200, "text/html; charset=utf-8", PAGE.replace("/*__SNAPSHOT__*/null", "null").encode("utf-8")
    if url.path == "/healthz":
        return 200, "text/plain; charset=utf-8", b"ok"
    if url.path.startswith("/api/"):
        endpoint = url.path[len("/api/"):]
        if endpoint not in ENDPOINTS:
            return 404, "text/plain; charset=utf-8", b"unknown endpoint"
        query = parse_qs(url.query, keep_blank_values=True)
        try:
            args = [query[name][0] for name in ENDPOINTS[endpoint]]
        except KeyError as missing:
            return 400, "text/plain; charset=utf-8", f"missing parameter {missing}".encode()
        try:
            store.refresh() if endpoint == "runs" else None
            return 200, "application/json", json.dumps(call(store, endpoint, *args), allow_nan=False).encode("utf-8")
        except KeyError as exc:
            return 404, "text/plain; charset=utf-8", str(exc).encode("utf-8")
        except (OSError, ValueError) as exc:
            return 500, "text/plain; charset=utf-8", f"cannot read run: {exc}".encode("utf-8")
    return 404, "text/plain; charset=utf-8", b"not found"
