"""Serves the Studio pages for development: index at /, files at /static/, language files at /i18n/."""

import http.server
import pathlib
import sys

UI = pathlib.Path(__file__).resolve().parents[2] / "src" / "anonymizer" / "ui"


class Handler(http.server.SimpleHTTPRequestHandler):
    def translate_path(self, path):
        path = path.split("?", 1)[0]
        if path in ("/", "/index.html"):
            return str(UI / "static" / "index.html")
        return str(UI / path.lstrip("/"))

    def log_message(self, *args):
        pass


def serve(port=8765):
    http.server.ThreadingHTTPServer(("127.0.0.1", port), Handler).serve_forever()


if __name__ == "__main__":
    serve(int(sys.argv[1]) if len(sys.argv) > 1 else 8765)
