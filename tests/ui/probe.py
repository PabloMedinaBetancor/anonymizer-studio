"""Evaluate a JavaScript expression on a page of the mock Studio at a window size and print the result.

python3 tests/ui/probe.py 1280x800 review "document.documentElement.scrollHeight"
"""

import json
import pathlib
import sys
import threading

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from dev_server import serve  # noqa: E402
from drive import Page  # noqa: E402
from screens import PORT, settle  # noqa: E402


def main(argv):
    width, height = (int(v) for v in argv[1].split("x"))
    name, expression = argv[2], argv[3]
    threading.Thread(target=serve, args=(PORT,), daemon=True).start()
    start = {"import": "", "analyze": "&start=analyze", "review": "&start=review", "export": "&start=export"}[name]
    page = Page(f"http://127.0.0.1:{PORT}/?mock=fast&lang=en{start}", width=width, height=height, port=9340)
    try:
        page.call("Emulation.setDeviceMetricsOverride", width=width, height=height, deviceScaleFactor=1, mobile=False)
        page.call(
            "Page.addScriptToEvaluateOnNewDocument",
            source="window.__e=[];addEventListener('error',e=>__e.push(e.message+' '+e.filename+':'+e.lineno));"
            "addEventListener('unhandledrejection',e=>__e.push('rej '+(e.reason&&e.reason.stack||e.reason)))",
        )
        page.call("Page.reload")
        page.sleep(3)
        try:
            settle(page, name)
        except TimeoutError as e:
            print("settle timed out:", e, "errors:", page.eval("JSON.stringify(window.__e)"))
        print(json.dumps(page.eval(expression), indent=1))
    finally:
        page.close()
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
