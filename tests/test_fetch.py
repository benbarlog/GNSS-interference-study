"""Downloader against a local Range-capable HTTP server: resume, size check, state transitions."""

import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from gnss.adsblol.config import Paths
from gnss.adsblol.fetch import Downloader, part_paths
from gnss.adsblol.http import Backoff, PoliteClient
from gnss.adsblol.manifest import Manifest
from gnss.adsblol.runtime import Events, Status

BLOBS = {"/d/x.tar.aa": bytes(range(256)) * 4000, "/d/x.tar.ab": b"tail" * 3000}
SEEN_RANGES: list[str | None] = []


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def do_GET(self):
        data = BLOBS.get(self.path)
        if data is None:
            self.send_response(404)
            self.end_headers()
            return
        rng = self.headers.get("Range")
        SEEN_RANGES.append(rng)
        start = int(rng.split("=")[1].rstrip("-")) if rng else 0
        body = data[start:]
        self.send_response(206 if rng else 200)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


@pytest.fixture
def server():
    srv = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    th = threading.Thread(target=srv.serve_forever, daemon=True)
    th.start()
    yield f"http://127.0.0.1:{srv.server_address[1]}"
    srv.shutdown()


def test_download_resume_and_complete(tmp_path, cfg, server):
    paths = Paths(tmp_path)
    paths.ensure()
    m = Manifest(paths.manifest)
    urls = [server + "/d/x.tar.aa", server + "/d/x.tar.ab"]
    sizes = [len(BLOBS["/d/x.tar.aa"]), len(BLOBS["/d/x.tar.ab"])]
    m.upsert_catalog([{"date": "2025-06-14", "repo": "r", "tag": "t", "variant": "prod", "urls": urls,
                       "part_sizes": sizes, "total_bytes": sum(sizes), "preferred_tag": "t",
                       "selection_note": "preferred"}])
    m.set_schedule([("2025-06-14", "P0", 0, True, "H")])
    # a previous run left half of part .aa behind
    row = m.get("2025-06-14")
    p_aa = part_paths(paths, row)[0]
    p_aa.parent.mkdir(parents=True)
    p_aa.with_name(p_aa.name + ".partial").write_bytes(BLOBS["/d/x.tar.aa"][:500_000])

    stop = threading.Event()
    events, status = Events(console=False), Status()
    client = PoliteClient(cfg, Backoff(cfg, events, status), stop=stop)
    dl = Downloader(cfg, paths, m, client, status, events, stop, lambda: (["P0"], None))
    dl.start()
    t0 = time.time()
    while m.get("2025-06-14")["status"] != "downloaded" and time.time() - t0 < 30:
        time.sleep(0.2)
    stop.set()
    dl.join(5)
    row = m.get("2025-06-14")
    assert row["status"] == "downloaded" and row["total_bytes"] == sum(sizes)
    assert [p.read_bytes() for p in part_paths(paths, row)] == [BLOBS["/d/x.tar.aa"], BLOBS["/d/x.tar.ab"]]
    assert "bytes=500000-" in SEEN_RANGES                    # resumed, not restarted
    assert not list(p_aa.parent.glob("*.partial"))
    m.close()
