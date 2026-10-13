"""The real Studio (server, engine and models) behind a headless Chrome, for screenshots and the demo video.

    with Studio(lang="en") as studio:
        studio.import_files(["demo/street-madrid.mp4"])
        studio.analyze()

Nothing here is a mock: the page talks to `python3 -m anonymizer.ui` with the real models, on the public clips of demo/.
The Python packages come from the active environment and the models from `models/` (see `scripts/fetch_models.py`).
Chrome runs with a temporary profile, never the user's, and only on a machine where a headless Chrome may run.
"""

import json
import os
import pathlib
import re
import shutil
import socket
import subprocess
import sys
import tempfile
import threading

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from drive import Page  # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parents[2]
DEMO = ROOT / "demo"
# Clips are copied here and imported from here, so the default destination the page shows is a neutral
# /tmp/studio-demo/Videos/anonymized and no screenshot or video carries a real home or work folder.
STAGE = pathlib.Path(tempfile.gettempdir()) / "studio-demo" / "Videos"
PRIVATE = re.compile(r"/home/|/Users/", re.I)
TABS = {"import": "Import", "analyze": "Analyze", "review": "Review", "export": "Export"}


def main_checkout():
    marker = ROOT / ".git"
    if marker.is_file():  # a worktree: "gitdir: <main>/.git/worktrees/<name>"
        gitdir = pathlib.Path(marker.read_text().split(":", 1)[1].strip())
        return gitdir.parents[2]
    return ROOT


def private_hits(text):
    """What in `text` would give away a real user name or work folder (it must find nothing before a capture)."""
    return sorted(set(m.group(0) for m in PRIVATE.finditer(text)))


def check_page(page):
    text = page.eval(
        "document.documentElement.outerHTML + ' ' + [...document.querySelectorAll('input, select, textarea')].map(e => e.value).join(' ')"
        " + ' ' + document.body.innerText"
    )
    hits = private_hits(text)
    if hits:
        raise RuntimeError(f"the page shows private strings {hits}: not captured")


def environment(config=None):
    main = main_checkout()
    env = dict(os.environ)
    if config:  # settings of this run only: a capture never inherits (or changes) the person's saved options
        env["XDG_CONFIG_HOME"] = str(config)
    pydeps = ""
    local = main / ".env.local"
    if local.exists():
        for line in local.read_text().splitlines():
            if line.startswith("PYDEPS="):
                pydeps = line.split("=", 1)[1].strip()
    # What the caller already put on PYTHONPATH stays (a computing node keeps its libraries there).
    env["PYTHONPATH"] = os.pathsep.join(filter(None, [str(ROOT / "src"), pydeps, os.environ.get("PYTHONPATH")]))
    env.setdefault("ANONYMIZER_MODELS", str(main / "models"))
    return env


class Studio:
    def __init__(self, lang="en", port=None, width=1600, height=1000):
        if port is None:  # a free one: another session may hold 9337
            with socket.socket() as probe:
                probe.bind(("127.0.0.1", 0))
                port = probe.getsockname()[1]
        self.lang, self.port, self.width, self.height = lang, port, width, height
        self.server = self.page = None
        self.errors = []

    def __enter__(self):
        self.config = tempfile.mkdtemp(prefix="studio-config-")
        self.server = subprocess.Popen(
            [sys.executable, *(["-s"] if sys.flags.no_user_site else []), "-m", "anonymizer.ui", "--no-browser"],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            cwd=ROOT,
            env=environment(self.config),
        )
        self.url = self.server.stdout.readline().split(": ", 1)[1].strip()
        threading.Thread(
            target=lambda: [self.errors.append(line.rstrip()) for line in self.server.stdout], daemon=True
        ).start()
        self.page = Page(self.url + f"&lang={self.lang}", width=self.width, height=self.height, port=self.port)
        self.page.call(
            "Page.addScriptToEvaluateOnNewDocument",
            source="window.__e=[];addEventListener('error',e=>__e.push(e.message));addEventListener('unhandledrejection',e=>__e.push('rej '+(e.reason&&e.reason.message||e.reason)))",
        )
        self.wait_loaded()
        return self

    def __exit__(self, *exc):
        try:
            self.page.close()
        finally:
            self.server.terminate()
            self.server.wait(timeout=20)
            shutil.rmtree(self.config, ignore_errors=True)

    def reload(self, lang=None):
        if lang:
            self.lang = lang
        self.page.eval(f"location.href = '/?lang={self.lang}'")  # the cookie of the first visit authorises it
        self.page.sleep(1)
        self.wait_loaded()

    def wait_loaded(self):
        self.page.wait_for(
            "document.querySelector('.tabs') && document.querySelectorAll('.tabs button').length === 4", 180
        )
        self.page.sleep(1)

    def import_files(self, names):
        STAGE.mkdir(parents=True, exist_ok=True)
        for name in names:
            shutil.copy2(DEMO / name, STAGE / name)
        paths = [str(STAGE / name) for name in names]
        reply = self.page.eval(
            f"fetch('api/clips', {{method: 'POST', headers: {{'Content-Type': 'application/json'}}, body: JSON.stringify({{paths: {json.dumps(paths)}}})}}).then(r => r.text())"
        )
        if '"rejected":[]' not in reply:
            raise RuntimeError(reply)
        self.reload()

    def tab(self, name):
        label = TABS[name]
        self.page.eval(f"document.querySelectorAll('.tabs button')[{list(TABS).index(name)}].click()")
        self.page.sleep(1.5)
        return label

    def analyze(self, wait=True, seconds=900):
        self.page.click(
            *self.page.center(".page:not([hidden]) .cta .btn-primary")
        )  # the empty Import screen has a button of its own
        if wait:
            self.wait_analyzed(seconds)

    def wait_analyzed(self, seconds=900):
        self.page.wait_for(
            "fetch('api/state').then(r => r.json()).then(s => s.clips.length > 0 && s.clips.every(c => ['analyzed', 'failed', 'stopped'].includes(c.status)))",
            seconds,
        )

    def assert_nothing_private(self):
        check_page(self.page)

    def shot(self, path):
        self.page.sleep(0.6)
        self.assert_nothing_private()
        self.page.screenshot(str(path))

    def js_errors(self):
        return self.page.eval("JSON.stringify(window.__e || [])")
