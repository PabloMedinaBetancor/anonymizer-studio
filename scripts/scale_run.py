"""A few hundred public photos through the real Studio in one run: import, analyze (Careful or Fast), render with CSV and JSON, then open
every output and compare it with its original. Prints the time of each step, the peak memory of the server and of its worker, and what
did not match. Public data only ($ANONYMIZER_BENCH_DATA, or --from FOLDER).

    python3 -s scripts/scale_run.py --photos 300 --detection careful --json out.json

Heavy: run it on a machine with nothing else heavy running, with nice.
"""

from __future__ import annotations

import argparse
import glob
import http.client
import json
import os
import re
import shutil
import site
import subprocess
import sys
import tempfile
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)


class Client:
    def __init__(self, url: str):
        match = re.match(r"http://127\.0\.0\.1:(\d+)/\?t=(\w+)", url)
        self.port, self.token = int(match.group(1)), match.group(2)

    def call(self, method: str, path: str, body=None, timeout: float = 300):
        conn = http.client.HTTPConnection("127.0.0.1", self.port, timeout=timeout)
        headers = {"Cookie": f"anon_session_{self.port}={self.token}"}
        payload = None
        if body is not None:
            payload, headers["Content-Type"] = json.dumps(body).encode(), "application/json"
        conn.request(method, path, body=payload, headers=headers)
        response = conn.getresponse()
        data = response.read()
        conn.close()
        if response.status >= 400:
            raise RuntimeError(f"{method} {path} -> {response.status} {data[:300]!r}")
        return json.loads(data) if response.getheader("Content-Type", "").startswith("application/json") else data


def descendants(root_pid: int) -> list:
    """The process and everything below it, by parent id (read from /proc)."""
    parents = {}
    for name in os.listdir("/proc"):
        if name.isdigit():
            try:
                with open(f"/proc/{name}/stat") as f:
                    parents[int(name)] = int(f.read().rsplit(")", 1)[1].split()[1])
            except (OSError, ValueError, IndexError):
                pass
    found, todo = [], [root_pid]
    while todo:
        pid = todo.pop()
        found.append(pid)
        todo += [child for child, parent in parents.items() if parent == pid]
    return found


def rss_peaks(root_pid: int) -> int:
    """Highest resident memory (kB, VmHWM) of one process among a process and its descendants: the server or its worker."""
    best = 0
    for pid in descendants(root_pid):
        try:
            with open(f"/proc/{pid}/status") as f:
                for line in f:
                    if line.startswith("VmHWM:"):
                        best = max(best, int(line.split()[1]))
        except OSError:
            pass
    return best


def megapixels(path: str) -> float:
    from PIL import Image

    try:
        with Image.open(path) as picture:
            return picture.size[0] * picture.size[1] / 1e6
    except Exception:
        return 0.0


def photos_from(folder_roots, count, min_mp=0.0):
    found = []
    for root in folder_roots:
        for pattern in ("*.jpg", "*.jpeg", "*.png"):
            found += sorted(glob.glob(os.path.join(root, "**", pattern), recursive=True))
    seen, unique = set(), []
    for path in found:
        key = (os.path.basename(path), os.path.getsize(path))
        if key not in seen and os.path.getsize(path) < 25_000_000 and megapixels(path) >= min_mp:
            seen.add(key)
            unique.append(path)
    return unique[:count]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--photos", type=int, default=300)
    parser.add_argument("--detection", choices=("fast", "careful"), default="careful")
    parser.add_argument(
        "--min-mp",
        type=float,
        default=0.0,
        help="only photos of at least this many megapixels (the public sets hold many small crops)",
    )
    parser.add_argument("--from", dest="source", default=os.environ.get("ANONYMIZER_BENCH_DATA"))
    parser.add_argument("--json", help="write the results here")
    args = parser.parse_args()
    if not args.source:
        parser.error("give --from or set ANONYMIZER_BENCH_DATA")
    pool = photos_from([args.source], args.photos, args.min_mp)
    work = tempfile.mkdtemp(prefix="scale-run-")
    src, out = os.path.join(work, "photos"), os.path.join(work, "out")
    os.makedirs(src)
    for n, path in enumerate(pool):  # a copy under a unique name: two sets can hold the same file name
        shutil.copyfile(path, os.path.join(src, f"{n:04d}-{os.path.basename(path)}"))
    began = time.monotonic()

    def step(text):
        print(f"{time.monotonic() - began:8.1f} s  {text}", flush=True)

    env = dict(
        os.environ,
        HOME=work,
        PYTHONUSERBASE=site.getuserbase(),  # HOME moves, the user site (`pip install --user`) does not
        PYTHONPATH=os.pathsep.join(filter(None, [os.path.join(ROOT, "src"), os.environ.get("PYTHONPATH")])),
    )
    log = open(os.path.join(work, "server.log"), "w")
    server = subprocess.Popen(
        [sys.executable, *(["-s"] if sys.flags.no_user_site else []), "-m", "anonymizer.ui", "--no-browser"],
        stdout=subprocess.PIPE,
        stderr=log,
        text=True,
        env=env,
    )
    results: dict = {
        "photos": len(pool),
        "detection": args.detection,
        "megapixels": round(sum(megapixels(p) for p in pool), 1),
    }
    try:
        client = Client(server.stdout.readline().split(": ", 1)[1].strip())
        imported = client.call("POST", "/api/clips", {"paths": [src]})
        ids = [c["clip_id"] for c in imported["clips"]]
        results["rejected"] = imported["rejected"]
        step(f"imported {len(ids)} photos ({len(imported['rejected'])} rejected)")
        client.call("POST", "/api/job/analyze", {"clips": ids, "options": {"detection": args.detection}})
        peak, last = 0, 0.0
        while True:
            state = client.call("GET", "/api/state")
            peak = max(peak, rss_peaks(server.pid))
            done = sum(1 for c in state["clips"] if c["status"] == "analyzed")
            if time.monotonic() - last > 60:
                last = time.monotonic()
                step(f"analysed {done}/{len(ids)}, peak {peak // 1024} MB")
            if state["job"] is None:
                break
            time.sleep(2)
        results["analysis_s"] = round(time.monotonic() - began, 1)
        results["analysis_peak_mb"] = peak // 1024
        statuses = [c["status"] for c in state["clips"]]
        results["analysed"] = statuses.count("analyzed")
        results["failed"] = [(c["name"], c.get("failure")) for c in state["clips"] if c["status"] == "failed"]
        step(f"analysed {results['analysed']} (failed {len(results['failed'])}), peak {results['analysis_peak_mb']} MB")
        items = 0
        for clip in state["clips"]:
            if clip["status"] != "analyzed":
                continue
            analysis = client.call("GET", f"/api/clips/{clip['clip_id']}/analysis")
            items += len(analysis["items"])
            for item in analysis["items"]:
                if item["judgement"]["verdict"] == "check":
                    client.call("PATCH", f"/api/clips/{clip['clip_id']}/items/{item['item_id']}", {"blur": True})
        results["items"] = items
        step(f"{items} items, every doubtful one blurred")
        good = [c["clip_id"] for c in state["clips"] if c["status"] == "analyzed"]
        t = time.monotonic()
        client.call(
            "POST",
            "/api/render",
            {"clips": good, "options": {"destination": out, "reports": {"pdf": False, "csv": True, "json": True}}},
        )
        while client.call("GET", "/api/state")["job"] is not None:
            peak = max(peak, rss_peaks(server.pid))
            time.sleep(2)
        results["render_s"] = round(time.monotonic() - t, 1)
        results["peak_mb"] = max(peak, rss_peaks(server.pid)) // 1024
        step(f"rendered in {results['render_s']} s, peak {results['peak_mb']} MB")
        from PIL import Image

        files = sorted(os.listdir(out))
        pictures = [f for f in files if f.endswith((".jpg", ".jpeg", ".png"))]
        results["outputs"] = {
            "pictures": len(pictures),
            "csv": sum(f.endswith(".csv") for f in files),
            "json": sum(f.endswith(".json") for f in files),
        }
        mismatched = []
        for name in sorted(os.listdir(src)):
            stem, ext = os.path.splitext(name)
            made = os.path.join(out, f"{stem}_anonymized{ext.lower()}")
            if not os.path.exists(made):
                if name in [c["name"] for c in state["clips"] if c["status"] == "analyzed"]:
                    mismatched.append((name, "no output"))
                continue
            with Image.open(os.path.join(src, name)) as a, Image.open(made) as b:
                if a.size != b.size:
                    mismatched.append((name, f"size {a.size} -> {b.size}"))
                else:
                    b.load()
        results["mismatched"] = mismatched
        step(f"outputs {results['outputs']}, mismatched {len(mismatched)}")
        results["server_stderr"] = open(os.path.join(work, "server.log")).read()[-500:]
    finally:
        server.terminate()
        try:
            server.wait(30)
        except subprocess.TimeoutExpired:
            server.kill()
        log.close()
        shutil.rmtree(work, ignore_errors=True)
    print(json.dumps(results, indent=1))
    if args.json:
        with open(args.json, "w") as f:
            json.dump(results, f, indent=1)
    return 0


if __name__ == "__main__":
    sys.exit(main())
