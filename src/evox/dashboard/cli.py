"""``evox dashboard``: serve the dashboard locally, or export a single offline HTML file."""
from __future__ import annotations

import argparse
import logging
from pathlib import Path

from evox.config import RESULTS_DIR


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="evox dashboard", description=__doc__)
    sub = ap.add_subparsers(dest="command", required=True)
    runs_default = RESULTS_DIR / "runs"
    s = sub.add_parser("serve", help="serve the read-only dashboard (loopback only unless --host is given)")
    s.add_argument("--runs-dir", type=Path, default=runs_default)
    s.add_argument("--host", default="127.0.0.1")
    s.add_argument("--port", type=int, default=8000)
    e = sub.add_parser("export", help="write one self-contained HTML file (no server needed)")
    e.add_argument("--runs-dir", type=Path, default=runs_default)
    e.add_argument("--output", type=Path, default=RESULTS_DIR / "runs" / "dashboard.html")
    e.add_argument("--run", action="append", help="limit to these run ids (default: all)")
    e.add_argument("--max-registries", type=int, default=12)
    a = ap.parse_args(argv)
    if not a.runs_dir.is_dir():
        print(f"Runs directory not found: {a.runs_dir}")
        return 2
    if a.command == "export":
        from evox.dashboard.export import export_html
        out = export_html(a.runs_dir, a.output, a.run, a.max_registries)
        print(f"Wrote {out} ({out.stat().st_size // 1024} KiB)")
        return 0
    from evox.dashboard.server import make_server
    server = make_server(a.runs_dir, a.host, a.port)
    print(f"Dashboard on http://{a.host}:{server.server_address[1]}/  (Ctrl+C to stop)")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("stopped")
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    raise SystemExit(main())
