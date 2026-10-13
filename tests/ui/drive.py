"""Drives the Studio page in a headless Chrome over the DevTools protocol, for screenshots and click-through checks.

Needs `websocket-client`. Chrome runs with a temporary profile, never the user's.
"""

import base64
import json
import os
import shutil
import subprocess
import tempfile
import time
import urllib.request

import websocket


class Page:
    def __init__(self, url, width=1600, height=1000, port=9333):
        self.profile = tempfile.mkdtemp(prefix="anon-ui-")
        self.chrome = subprocess.Popen(
            [
                "google-chrome",
                "--headless=new",
                "--no-sandbox",
                "--disable-gpu",
                f"--user-data-dir={self.profile}",
                f"--remote-debugging-port={port}",
                "--remote-allow-origins=*",
                f"--window-size={width},{height}",
                url,
            ],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            # Chrome makes `<temp>/com.google.Chrome.XXXXXX` at start and does not remove it when it is ended with a signal: 180 of them
            # stayed after one round of the tests. Its temporary folder is the profile's, which `close` removes.
            env=dict(os.environ, TMPDIR=self.profile),
        )
        self.socket = None
        for _ in range(240):
            try:
                tabs = json.load(urllib.request.urlopen(f"http://127.0.0.1:{port}/json"))
                page = next(tab for tab in tabs if tab["type"] == "page")
                self.socket = websocket.create_connection(page["webSocketDebuggerUrl"], timeout=90)
                break
            except Exception:
                time.sleep(0.5)
        if self.socket is None:
            self.chrome.terminate()
            raise RuntimeError("Chrome did not start")
        self.counter = 0
        self.frames = []  # (timestamp seconds, jpeg bytes) of a running screencast

    def on_event(self, message):
        if message.get("method") == "Page.screencastFrame":
            params = message["params"]
            self.frames.append((params["metadata"]["timestamp"], base64.b64decode(params["data"])))
            self.acks = getattr(self, "acks", 1_000_000) + 1  # their own ids: a call in flight waits for its own
            self.socket.send(
                json.dumps(
                    {"id": self.acks, "method": "Page.screencastFrameAck", "params": {"sessionId": params["sessionId"]}}
                )
            )

    def sleep(self, seconds):
        """Wait while still receiving (and acknowledging) screencast frames."""
        end = time.time() + seconds
        while time.time() < end:
            self.socket.settimeout(max(0.05, min(0.25, end - time.time())))
            try:
                self.on_event(json.loads(self.socket.recv()))
            except websocket.WebSocketTimeoutException:
                pass
        self.socket.settimeout(90)

    def start_screencast(self, quality=80, max_width=1600, max_height=1000):
        self.frames = []
        self.call(
            "Page.startScreencast",
            format="jpeg",
            quality=quality,
            maxWidth=max_width,
            maxHeight=max_height,
            everyNthFrame=1,
        )

    def stop_screencast(self):
        self.call("Page.stopScreencast")
        frames, self.frames = self.frames, []
        return frames

    def call(self, method, **params):
        self.counter += 1
        self.socket.send(json.dumps({"id": self.counter, "method": method, "params": params}))
        while True:
            message = json.loads(self.socket.recv())
            if "method" in message:
                self.on_event(message)
            elif message.get("id") == self.counter:
                if "error" in message:
                    raise RuntimeError(message["error"])
                return message["result"]

    def eval(self, expression):
        result = self.call("Runtime.evaluate", expression=expression, returnByValue=True, awaitPromise=True)
        if "exceptionDetails" in result:
            raise RuntimeError(result["exceptionDetails"])
        return result["result"].get("value")

    def wait_for(self, expression, seconds=90):
        end = time.time() + seconds
        while time.time() < end:
            if self.eval(expression):
                return
            self.sleep(0.3)
        raise TimeoutError(expression)

    def mouse(self, kind, x, y):
        self.call("Input.dispatchMouseEvent", type=kind, x=x, y=y, button="left", buttons=1, clickCount=1)

    def click(self, x, y):
        self.mouse("mouseMoved", x, y)
        self.mouse("mousePressed", x, y)
        self.mouse("mouseReleased", x, y)

    def drag(self, x0, y0, x1, y1, steps=1, pause=0.0):
        self.mouse("mouseMoved", x0, y0)
        self.mouse("mousePressed", x0, y0)
        for i in range(1, steps + 1):
            self.mouse("mouseMoved", x0 + (x1 - x0) * i / (steps + 1), y0 + (y1 - y0) * i / (steps + 1))
            if pause:
                self.sleep(pause)
        self.mouse("mouseMoved", x1, y1)
        self.mouse("mouseReleased", x1, y1)

    def key(self, key, code=None, text=None):
        base = {"key": key, "code": code or key, "text": text or (key if len(key) == 1 else "")}
        self.call("Input.dispatchKeyEvent", type="keyDown", **base)
        self.call("Input.dispatchKeyEvent", type="keyUp", **base)

    def center(self, selector):
        box = self.eval(
            f"(() => {{ const r = document.querySelector({json.dumps(selector)}).getBoundingClientRect(); return [r.x + r.width / 2, r.y + r.height / 2]; }})()"
        )
        return box

    def screenshot(self, path):
        data = self.call("Page.captureScreenshot", format="png")["data"]
        with open(path, "wb") as out:
            out.write(base64.b64decode(data))

    def close(self):
        try:
            self.socket.close()
        finally:
            self.chrome.terminate()
            try:
                self.chrome.wait(timeout=10)
            except subprocess.TimeoutExpired:
                self.chrome.kill()
                self.chrome.wait()
            # 50 MB of profile per Chrome, left in /tmp by every caller that did not remove it itself. Its helper processes can still write
            # in it for a moment after the main process has gone (a `Default` folder came back after the first removal): removed again.
            for _ in range(3):
                shutil.rmtree(self.profile, ignore_errors=True)
                time.sleep(0.4)
