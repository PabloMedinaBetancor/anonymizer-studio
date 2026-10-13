"""End to end through the real server: start `python -m anonymizer.ui`, import a clip, analyze it, review, render with
the comparison video and the reports, and check what came out. Public data only.

    python3 -s scripts/smoke_ui.py --clip path/to/clip.mp4        # a clip you have
    python3 -s scripts/smoke_ui.py --from-photo path/to/photo.jpg # a short clip is made from a public photo
    python3 -s scripts/smoke_ui.py --from-photo photo.jpg --make-only out.mp4   # only make the clip (to open it in Studio)

Uses the real detectors ($ANONYMIZER_MODELS) and a worker process. Prints each step with its time.
"""

from __future__ import annotations

import argparse
import fractions
import http.client
import json
import os
import platform
import re
import subprocess
import sys
import tempfile
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)


def clip_from_photo(photo: str, target: str, frames: int = 24, fps: int = 8) -> None:
    import av
    import cv2

    image = cv2.imread(photo)
    if image is None:
        raise SystemExit(f"cannot read {photo}")
    height, width = image.shape[:2]
    scale = 720 / width if width > 720 else 1.0
    image = cv2.resize(image, (int(width * scale) // 2 * 2, int(height * scale) // 2 * 2), interpolation=cv2.INTER_AREA)
    with av.open(target, "w") as out:
        stream = out.add_stream("libx264", rate=fps)
        stream.width, stream.height, stream.pix_fmt = image.shape[1], image.shape[0], "yuv420p"
        stream.options = {"crf": "18", "preset": "ultrafast"}
        for t in range(frames):
            frame = av.VideoFrame.from_ndarray(image, format="bgr24")
            frame.pts, frame.time_base = t, fractions.Fraction(1, fps)
            for packet in stream.encode(frame):
                out.mux(packet)
        for packet in stream.encode(None):
            out.mux(packet)


class Client:
    def __init__(self, url: str):
        match = re.match(r"http://127\.0\.0\.1:(\d+)/\?t=(\w+)", url)
        self.port, self.token = int(match.group(1)), match.group(2)

    def call(self, method: str, path: str, body=None):
        conn = http.client.HTTPConnection("127.0.0.1", self.port, timeout=120)
        headers = {"Cookie": f"anon_session_{self.port}={self.token}"}
        payload = None
        if body is not None:
            payload = json.dumps(body).encode()
            headers["Content-Type"] = "application/json"
        conn.request(method, path, body=payload, headers=headers)
        response = conn.getresponse()
        data = response.read()
        conn.close()
        if response.status >= 400:
            raise SystemExit(f"{method} {path} -> {response.status} {data[:300]!r}")
        return json.loads(data) if response.getheader("Content-Type", "").startswith("application/json") else data

    def wait_job(self, timeout: float = 600.0, say_every: float = 30.0) -> dict:
        """Wait for the running job to end, saying how far it is every `say_every` seconds (a Careful analysis of a few clips takes
        minutes; a script that prints nothing for that long looks as if it hung)."""
        started = time.monotonic()
        end, last = started + timeout, started
        while time.monotonic() < end:
            state = self.call("GET", "/api/state")
            if state["job"] is None:
                return state
            if say_every and time.monotonic() - last >= say_every:
                last = time.monotonic()
                parts = [
                    f"{c['name']} {round((c.get('progress') or {}).get('fraction', 0) * 100)}%"
                    for c in state["clips"]
                    if c["status"] in ("analyzing", "rendering")
                ]
                print(
                    f"{time.monotonic() - started:7.0f} s  still working: {', '.join(parts) or state['job']['state']}",
                    flush=True,
                )
            time.sleep(0.2)
        raise SystemExit("the job did not end")


def step(name: str, started: float) -> None:
    print(f"{time.monotonic() - started:7.1f} s  {name}", flush=True)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--clip")
    parser.add_argument("--from-photo")
    parser.add_argument("--make-only")
    args = parser.parse_args()
    work = tempfile.mkdtemp(prefix="smoke-ui-")
    clip = args.clip
    if args.from_photo:
        clip = args.make_only or os.path.join(work, "clip.mp4")
        clip_from_photo(args.from_photo, clip)
        if args.make_only:
            print(clip)
            return 0
    if not clip:
        parser.error("give --clip or --from-photo")
    began = time.monotonic()
    env = dict(
        os.environ, PYTHONPATH=os.pathsep.join(filter(None, [os.path.join(ROOT, "src"), os.environ.get("PYTHONPATH")]))
    )
    command = [sys.executable, *(["-s"] if sys.flags.no_user_site else []), "-m", "anonymizer.ui", "--no-browser"]
    server = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, env=env)
    try:
        line = server.stdout.readline()
        client = Client(line.split(": ", 1)[1].strip())
        step(f"server up on {platform.node()} ({line.strip()[:40]}...)", began)
        client.call("GET", "/")
        state = client.call("GET", "/api/state")
        print("languages", state["languages"], "pdf", state["capabilities"]["pdf_report"]["available"])
        imported = client.call("POST", "/api/clips", {"paths": [clip]})
        assert not imported["rejected"], imported["rejected"]
        clip_id = imported["clips"][0]["clip_id"]
        step(f"imported {imported['clips'][0]['name']} ({imported['clips'][0]['frames']} frames)", began)
        client.call("POST", "/api/job/analyze", {"clips": [clip_id]})
        state = client.wait_job()
        status = state["clips"][0]
        assert status["status"] == "analyzed", status
        step(f"analyzed: {status['analysis']}", began)
        analysis = client.call("GET", f"/api/clips/{clip_id}/analysis")
        for item in analysis["items"]:
            print(
                "  item",
                item["item_id"],
                item["kind"],
                item["judgement"]["verdict"],
                item["judgement"]["reason"],
                round(item["max_score"], 2),
            )
            client.call("PATCH", f"/api/clips/{clip_id}/items/{item['item_id']}", {"blur": True})
        last = analysis["frames"] - 1
        for view in ("original", "anonymized", "boxes", "split"):
            data = client.call("GET", f"/api/clips/{clip_id}/preview?frame={last // 2}&view={view}&w=480")
            assert data[:3] == b"\xff\xd8\xff", view
        step("previews of the four views", began)
        folder = os.path.join(work, "out")
        options = {"destination": folder, "comparison": True, "reports": {"csv": True, "json": True, "pdf": False}}
        plan = client.call("POST", "/api/render/plan", {"clips": [clip_id], "options": options})
        assert plan["clips"][0]["blocked"] is None, plan
        client.call("POST", "/api/render", {"clips": [clip_id], "options": options})
        state = client.wait_job()
        assert state["clips"][0]["status"] == "rendered", state["clips"][0]
        files = sorted(os.listdir(folder))
        step(f"rendered: {files}", began)
        assert any(f.endswith(".compare.mp4") for f in files) and any(f.endswith(".report.json") for f in files), files
        with open(os.path.join(folder, [f for f in files if f.endswith(".report.json")][0]), encoding="utf-8") as f:
            print("  report rows:", len(json.load(f)["items"]))
        client.call("POST", "/api/quit")
        server.wait(20)
        step("server ended by /api/quit", began)
        print("OK")
        return 0
    finally:
        if server.poll() is None:
            server.kill()  # this server, by its pid


if __name__ == "__main__":
    sys.exit(main())
