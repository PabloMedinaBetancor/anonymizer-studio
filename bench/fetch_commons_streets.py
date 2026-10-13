"""Fetch street, car park and facade photos from Wikimedia Commons for the plate false-blur benchmark.

    python3 -s -m bench.fetch_commons_streets search            # candidates -> <temp>/streets-candidates.json
    python3 -s -m bench.fetch_commons_streets download N        # first N, round robin over the queries

Only CC0, CC BY and CC BY-SA files, JPEG, at least 1600 px wide, downloaded at 1920 px wide (a size Commons serves
as is), one request per second. Each file gets a row in bench/SOURCES.csv with the page, author and licence.
"""

from __future__ import annotations

import csv
import hashlib
import html
import json
import os
import re
import sys
import tempfile
import time
import urllib.error
import urllib.parse
import urllib.request
from datetime import date
from typing import Dict, List, Optional

DATA = os.path.join(os.environ.get("ANONYMIZER_BENCH_DATA", "bench/data"), "plates")
SOURCES = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "bench", "SOURCES.csv")
CANDIDATES = os.path.join(tempfile.gettempdir(), "streets-candidates.json")
API = "https://commons.wikimedia.org/w/api.php"
AGENT = "AnonymizerStudioBench/0.1 (benchmark)"
WIDTH = 1920
QUERIES = [
    "street parked cars",
    "residential street cars",
    "car park parked cars",
    "parking lot cars",
    "street traffic cars",
    "calle coches aparcados",
    "Strasse parkende Autos",
    "rue voitures garees",
    "shop front street",
    "building facade street",
    "apartment building facade windows",
    "street signs shops",
    "motorcycles parked street",
    "garage entrance",
]
LICENCES = {  # LicenseShortName on Commons -> the id used in SOURCES.csv and in the truth files
    "CC0": "CC0-1.0",
    "CC BY 2.0": "CC-BY-2.0",
    "CC BY 2.5": "CC-BY-2.5",
    "CC BY 3.0": "CC-BY-3.0",
    "CC BY 4.0": "CC-BY-4.0",
    "CC BY-SA 2.0": "CC-BY-SA-2.0",
    "CC BY-SA 2.5": "CC-BY-SA-2.5",
    "CC BY-SA 3.0": "CC-BY-SA-3.0",
    "CC BY-SA 4.0": "CC-BY-SA-4.0",
}


def get(url: str) -> bytes:
    """One polite request: a descriptive agent, one per second, and a long pause on 429 before giving up."""
    for wait in (0, 30, 90):
        time.sleep(wait or 1.1)
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers={"User-Agent": AGENT}), timeout=60) as r:
                return r.read()
        except urllib.error.HTTPError as exc:
            if exc.code != 429:
                raise
    raise SystemExit("Commons keeps answering 429: stopping")


def search(query: str) -> List[Dict]:
    params = {
        "action": "query",
        "format": "json",
        "generator": "search",
        "gsrsearch": f"{query} filetype:bitmap",
        "gsrnamespace": "6",
        "gsrlimit": "30",
        "prop": "imageinfo",
        "iiprop": "url|size|mime|extmetadata",
        "iiurlwidth": str(WIDTH),
    }
    reply = json.loads(get(API + "?" + urllib.parse.urlencode(params)))
    found = []
    for page in reply.get("query", {}).get("pages", {}).values():
        info = page["imageinfo"][0]
        meta = info.get("extmetadata", {})
        licence = LICENCES.get(meta.get("LicenseShortName", {}).get("value", ""))
        if info["mime"] != "image/jpeg" or info["width"] < WIDTH - 320 or licence is None:
            continue
        author = html.unescape(re.sub(r"<[^>]+>", "", meta.get("Artist", {}).get("value", ""))).strip()
        found.append(
            {
                "title": page["title"],
                "query": query,
                "thumb": info["thumburl"],
                "page": info["descriptionurl"],
                "author": author or "unknown",
                "licence": licence,
                "licence_url": meta.get("LicenseUrl", {}).get("value", ""),
                "size": [info["width"], info["height"]],
            }
        )
    return sorted(found, key=lambda f: f["title"])


def run_search() -> None:
    candidates = [c for q in QUERIES for c in search(q)]
    with open(CANDIDATES, "w", encoding="utf-8") as f:
        json.dump(candidates, f, indent=1)
    print(len(candidates), "candidates")


def pick(candidates: List[Dict], n: int) -> List[Dict]:
    by_query: Dict[str, List[Dict]] = {}
    for c in candidates:
        by_query.setdefault(c["query"], []).append(c)
    chosen, seen = [], set()
    while len(chosen) < n and any(by_query.values()):
        for query in QUERIES:
            while by_query.get(query):
                c = by_query[query].pop(0)
                if c["title"] not in seen:
                    seen.add(c["title"])
                    chosen.append(c)
                    break
    return chosen[:n]


def run_download(n: int) -> None:
    """Idempotent: a file that is already there is kept, so a run stopped by a 429 can be repeated later. The
    SOURCES rows are written for whatever was fetched, also when the run stops early."""
    with open(CANDIDATES, encoding="utf-8") as f:
        chosen = pick(json.load(f), n)
    folder = os.path.join(DATA, "streets")
    os.makedirs(folder, exist_ok=True)
    rows = []
    try:
        for i, c in enumerate(chosen, 1):
            name = f"street_{i:03d}.jpg"
            path = os.path.join(folder, name)
            if not os.path.isfile(path):
                with open(path, "wb") as f:
                    f.write(get(c["thumb"]))
                time.sleep(1)
            with open(path, "rb") as f:
                digest = hashlib.sha256(f.read()).hexdigest()
            rows.append(
                [
                    f"plates/streets/{name}",
                    "image",
                    c["page"],
                    c["author"],
                    c["licence"],
                    c["licence_url"],
                    date.today().isoformat(),
                    digest,
                    "plate false-blur benchmark (P5)",
                    c["title"] + " (Wikimedia Commons, downloaded at 1920 px wide)",
                ]
            )
    finally:
        with open(SOURCES, encoding="utf-8", newline="") as f:
            existing = [r for r in csv.reader(f) if r and not r[0].startswith("plates/streets/")]
        with open(SOURCES, "w", encoding="utf-8", newline="") as f:
            csv.writer(f, lineterminator="\n").writerows(existing + rows)
        print(len(rows), "photos in", folder)


def main(argv: Optional[List[str]] = None) -> None:
    args = argv if argv is not None else sys.argv[1:]
    if args[:1] == ["search"]:
        run_search()
    elif args[:1] == ["download"] and len(args) == 2:
        run_download(int(args[1]))
    else:
        raise SystemExit(__doc__)


if __name__ == "__main__":
    main()
