"""The hosted demo end to end, with the real detectors: start `python -m anonymizer.ui --demo`, import the public samples of demo/, analyze
them, review what the detectors were not sure about, render everything and fetch the results as downloads; then check the limits (10 photos,
20 seconds, 720 px) and that a path cannot be imported. Public data only; prints each step with its time.

    python3 -s scripts/smoke_demo.py [--detection fast]

The four samples are analysed in Careful (the demo's default) unless `--detection fast` is given. Both take minutes (on the development laptop, Fast:
219 s in all, 188 s of them the analysis of the two videos of 15 s and 10 s; Careful, which looks twice at the doubtful places, takes longer):
the script says how far it is every 30 s. It is not a unit test and nothing runs it by itself; a tool that waits for it for 150 s will time it out.
"""

from __future__ import annotations

import http.client
import importlib.util
import json
import os
import platform
import subprocess
import sys
import tempfile
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
spec = importlib.util.spec_from_file_location("smoke_ui", os.path.join(HERE, "smoke_ui.py"))
smoke = importlib.util.module_from_spec(spec)
spec.loader.exec_module(smoke)


def request(client, method: str, path: str, body=None, raw=None):
    """(status, parsed answer) without stopping on an error status: the limits are expected to refuse."""
    conn = http.client.HTTPConnection("127.0.0.1", client.port, timeout=120)
    headers = {"Cookie": f"anon_session_{client.port}={client.token}"}
    payload = raw
    if body is not None:
        payload, headers["Content-Type"] = json.dumps(body).encode(), "application/json"
    conn.request(method, path, body=payload, headers=headers)
    response = conn.getresponse()
    data = response.read()
    conn.close()
    kind = response.getheader("Content-Type", "")
    return response.status, (json.loads(data) if kind.startswith("application/json") else data), response


def rendered_outputs(client, timeout: float = 30.0) -> list:
    """The download addresses of every `clip.rendered` event so far (the stream replays what the server still remembers)."""
    conn = http.client.HTTPConnection("127.0.0.1", client.port, timeout=timeout)
    conn.request(
        "GET", "/api/events", headers={"Cookie": f"anon_session_{client.port}={client.token}", "Last-Event-ID": "0"}
    )
    response = conn.getresponse()
    found, finished, end = [], 0, time.monotonic() + timeout
    try:
        for raw in response:
            line = raw.decode("utf-8").rstrip("\n")
            if line.startswith("data: "):
                event = json.loads(line[6:])
                if event["type"] == "clip.rendered":
                    found += event["outputs"]
                finished += event["type"] == "job.finished"
            if finished >= 2 or time.monotonic() > end:
                break
    finally:
        conn.close()
    return found


def main() -> int:
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--detection", choices=("fast", "careful"), default="careful")
    args = parser.parse_args()
    began = time.monotonic()
    env = dict(
        os.environ, PYTHONPATH=os.pathsep.join(filter(None, [os.path.join(ROOT, "src"), os.environ.get("PYTHONPATH")]))
    )
    command = [
        sys.executable,
        *(["-s"] if sys.flags.no_user_site else []),
        "-m",
        "anonymizer.ui",
        "--demo",
        "--no-browser",
    ]
    server = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, env=env)
    try:
        line = server.stdout.readline()
        client = smoke.Client(line.split(": ", 1)[1].strip())
        smoke.step(f"demo server up on {platform.node()}", began)
        state = client.call("GET", "/api/state")
        assert state["mode"] == "demo", state["mode"]
        assert state["demo_limits"] == {
            "max_photos": 10,
            "max_clips": 10,
            "max_seconds": 20,
            "max_height": 720,
            "max_upload_mb": 100,
        }, state["demo_limits"]
        assert state["capabilities"]["browse"] is False
        samples = client.call("GET", "/api/samples")
        assert len(samples) == 4 and all(s["licence"].startswith("CC") and s["author"] for s in samples), samples
        print("samples:", [(s["sample_id"], s["media"], s["licence"]) for s in samples])
        status, answer, _ = request(client, "POST", "/api/clips", {"paths": ["/etc"]})
        assert (status, answer["code"]) == (403, "error.demo.disabled"), (status, answer)
        imported = client.call("POST", "/api/clips", {"samples": [s["sample_id"] for s in samples]})
        assert not imported["rejected"] and len(imported["clips"]) == 4, imported
        ids = [c["clip_id"] for c in imported["clips"]]
        smoke.step("the four samples imported (within the limits)", began)
        client.call("POST", "/api/job/analyze", {"clips": ids, "options": {"detection": args.detection}})
        state = client.wait_job(1500)
        found = {}
        for clip in state["clips"]:
            assert clip["status"] == "analyzed", clip
            analysis = client.call("GET", f"/api/clips/{clip['clip_id']}/analysis")
            found[clip["name"]] = [(i["kind"], i["judgement"]["verdict"]) for i in analysis["items"]]
            assert analysis["unchecked_frames"] == 0, clip["name"]
        smoke.step(f"analyzed: { {k: len(v) for k, v in found.items()} }", began)
        # Not sure items are blurred unless the person turns them off, and the export asks for no confirmation (UI-PROTOCOL, section 8): the
        # plan only says how many there are. (This script used to expect the 422 `error.review.pending` of v1, which no longer exists.)
        status, plan, _ = request(client, "POST", "/api/render/plan", {"clips": ids, "options": {}})
        pending = sum(1 for items in found.values() for _, verdict in items if verdict == "check")
        assert status == 200 and sum(c["to_check"] for c in plan["clips"]) == pending, (status, plan)
        print(f"  {pending} Not sure items; the plan counts them and asks nothing")
        for clip_id in ids:
            for item in client.call("GET", f"/api/clips/{clip_id}/analysis")["items"]:
                client.call("PATCH", f"/api/clips/{clip_id}/items/{item['item_id']}", {"blur": True})
        options = {"comparison": True, "reports": {"csv": True, "json": True, "pdf": False}}
        client.call("POST", "/api/render", {"clips": ids, "options": options})
        state = client.wait_job(1500)
        assert all(c["status"] == "rendered" for c in state["clips"]), [c["status"] for c in state["clips"]]
        smoke.step("rendered", began)
        downloads = rendered_outputs(client)
        assert len(downloads) == 4, downloads
        for url in downloads:
            status, data, response = request(client, "GET", url)
            assert (
                status == 200 and len(data) > 1000 and "attachment" in response.getheader("Content-Disposition", "")
            ), url
        smoke.step(f"{len(downloads)} results fetched as downloads", began)
        work = tempfile.mkdtemp(prefix="smoke-demo-")
        long_clip = os.path.join(work, "long.mp4")
        smoke.clip_from_photo(os.path.join(ROOT, "demo", "street-taipei.jpg"), long_clip, frames=44, fps=2)  # 22 s
        with open(long_clip, "rb") as f:
            status, answer, _ = request(client, "PUT", "/api/upload?name=long.mp4", raw=f.read())
        assert (status, answer.get("code"), answer.get("params", {}).get("limit")) == (
            422,
            "error.demo.limit",
            "seconds",
        ), (status, answer)
        from PIL import Image

        big = os.path.join(work, "big.jpg")
        Image.new("RGB", (1920, 1080), (90, 120, 60)).save(big)
        with open(big, "rb") as f:
            status, answer, _ = request(client, "PUT", "/api/upload?name=big.jpg", raw=f.read())
        assert (status, answer.get("code"), answer.get("params", {}).get("limit")) == (
            422,
            "error.demo.limit",
            "height",
        ), (status, answer)
        smoke.step("over the limits (22 s, 1080 px) refused", began)
        client.call("POST", "/api/quit")
        server.wait(20)
        print("OK")
        return 0
    finally:
        if server.poll() is None:
            server.kill()  # this server, by its pid


if __name__ == "__main__":
    sys.exit(main())
