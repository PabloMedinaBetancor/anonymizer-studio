"""The four pages of the mock Studio at chosen window sizes, as PNG files, and what sticks out of the window at each.

python3 tests/ui/shoot_sizes.py OUT 1200x1920 700x900 420x800
"""

import pathlib
import sys
import threading

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from dev_server import serve  # noqa: E402
from drive import Page  # noqa: E402
from screens import OVERFLOW, PAGES, PORT, settle  # noqa: E402


def main(argv):
    out = pathlib.Path(argv[1])
    out.mkdir(parents=True, exist_ok=True)
    threading.Thread(target=serve, args=(PORT,), daemon=True).start()
    for size in argv[2:]:
        width, height = (int(v) for v in size.split("x"))
        page = Page(f"http://127.0.0.1:{PORT}/?mock=fast&lang=en", width=width, height=height, port=9339)
        try:
            page.call(
                "Emulation.setDeviceMetricsOverride", width=width, height=height, deviceScaleFactor=1, mobile=False
            )
            for name in PAGES:
                start = {
                    "import": "",
                    "analyze": "&start=analyze",
                    "review": "&start=review",
                    "export": "&start=export",
                }[name]
                page.call("Page.navigate", url=f"http://127.0.0.1:{PORT}/?mock=fast&lang=en{start}")
                page.sleep(1)
                settle(page, name)
                page.screenshot(str(out / f"{name}-{size}.png"))
                sideways = page.eval("document.documentElement.scrollWidth > innerWidth + 1")
                print(size, name, "sideways scroll" if sideways else "fits", page.eval(OVERFLOW), flush=True)
        finally:
            page.close()
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
