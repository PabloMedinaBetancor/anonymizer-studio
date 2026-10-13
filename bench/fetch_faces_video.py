"""Fetch the two public street clips of the face video benchmark (F4) from Wikimedia Commons.

    python3 -s -m bench.fetch_faces_video

Both are CC BY-SA. A clip that is already in $ANONYMIZER_BENCH_DATA/faces/video/ is kept if its SHA-1 is the one
Commons publishes; otherwise it is downloaded (one request per second, a descriptive agent). Prints the SOURCES.csv
rows; they are written by hand to bench/SOURCES.csv once.
"""

from __future__ import annotations

import hashlib
import html
import json
import os
import re
import sys
import time
import urllib.parse
import urllib.request

DATA = os.path.join(os.environ.get("ANONYMIZER_BENCH_DATA", "bench/data"), "faces", "video")
API = "https://commons.wikimedia.org/w/api.php"
AGENT = "AnonymizerStudioBench/0.1 (benchmark)"
CLIPS = {"preciados.ogv": "File:Calle Preciados Madrid.ogv", "hongkong.webm": None}  # None: set below
HONG_KONG_SEARCH = "Pasejo en Hong Kong Island 2014"


def get(url: str) -> bytes:
    time.sleep(1.1)
    with urllib.request.urlopen(urllib.request.Request(url, headers={"User-Agent": AGENT}), timeout=120) as r:
        return r.read()


def info(title: str) -> dict:
    q = {"action": "query", "format": "json", "titles": title, "prop": "imageinfo", "iiprop": "url|sha1|extmetadata"}
    page = next(iter(json.loads(get(API + "?" + urllib.parse.urlencode(q)))["query"]["pages"].values()))
    return page["imageinfo"][0] | {"title": page["title"]}


def find_hong_kong() -> str:
    q = {"action": "query", "format": "json", "list": "search", "srsearch": HONG_KONG_SEARCH, "srnamespace": "6"}
    return json.loads(get(API + "?" + urllib.parse.urlencode(q)))["query"]["search"][0]["title"]


def sha(path: str, algo: str) -> str:
    h = hashlib.new(algo)
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> int:
    os.makedirs(DATA, exist_ok=True)
    for name, title in CLIPS.items():
        meta = info(title or find_hong_kong())
        path = os.path.join(DATA, name)
        if not (os.path.isfile(path) and sha(path, "sha1") == meta["sha1"]):
            with open(path, "wb") as f:
                f.write(get(meta["url"]))
        ext = meta["extmetadata"]
        author = html.unescape(re.sub(r"<[^>]+>", "", ext.get("Artist", {}).get("value", ""))).strip()
        print(
            f"{name} | {meta['descriptionurl']} | {author} | {ext['LicenseShortName']['value']} | "
            f"{ext.get('LicenseUrl', {}).get('value', '')} | sha256 {sha(path, 'sha256')}"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
