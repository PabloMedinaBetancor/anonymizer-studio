"""Download the face benchmark photos listed in bench/faces_titles.csv from Wikimedia Commons.

Each line of the list is `set,title`: set "a" holds photos with faces, "b" photos without people. Files go to
$ANONYMIZER_BENCH_DATA/faces/set-<a|b>/ at 1920 px at most, one request per second, and every file gets a row
in bench/SOURCES.csv (page, author, licence, hash). Only CC0, CC BY and CC BY-SA files are kept.
"""

from __future__ import annotations

import csv
import datetime
import hashlib
import html
import json
import os
import re
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from typing import Dict, List

API = "https://commons.wikimedia.org/w/api.php"
USER_AGENT = "AnonymizerStudioBench/0.1 (benchmark)"
ALLOWED = re.compile(r"^(CC0|CC BY(-SA)? \d\.\d)")
FIELDS = ["file", "kind", "source_url", "author", "licence", "licence_url", "downloaded", "sha256", "used_for", "notes"]

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.environ.get("ANONYMIZER_BENCH_DATA") or os.path.join(HERE, "data")
_last_request = 0.0


def get(url: str) -> bytes:
    """One request per second at most; waits and retries when the server says 429."""
    global _last_request
    for attempt in range(6):
        time.sleep(max(0.0, _last_request + 1.1 - time.monotonic()))
        _last_request = time.monotonic()
        try:
            req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
            with urllib.request.urlopen(req, timeout=90) as response:
                return response.read()
        except urllib.error.HTTPError as e:
            if e.code != 429:
                raise
            print(f"429 on {url[:110]}, waiting {30 * (attempt + 1)} s", file=sys.stderr)
            time.sleep(30 * (attempt + 1))
    raise RuntimeError(f"rate limited: {url}")


def image_info(titles: List[str]) -> Dict[str, dict]:
    query = urllib.parse.urlencode(
        {
            "action": "query",
            "format": "json",
            "formatversion": "2",
            "titles": "|".join(titles),
            "prop": "imageinfo",
            "iiprop": "url|size|mime|extmetadata",
            "iiextmetadatafilter": "LicenseShortName|LicenseUrl|Artist",
        }
    )
    pages = json.loads(get(f"{API}?{query}"))["query"]["pages"]
    return {p["title"]: (p.get("imageinfo") or [{}])[0] for p in pages}


def scaled_url(url: str, width: int) -> str:
    """The biggest standard Commons thumbnail (/thumb/<a>/<ab>/<name>/<N>px-<name>, N <= 1920) that is not larger
    than the original. Originals are throttled much harder than thumbnails, and asking the API for thumbnails
    makes it render a whole batch at once and answer 429."""
    url = url.split("?")[0]
    standard = next((n for n in (1920, 1280, 960) if width >= n), 0)
    if not standard:
        return url
    head, tail = url.split("/commons/", 1)
    name = tail.rsplit("/", 1)[1]
    return f"{head}/commons/thumb/{tail}/{standard}px-{name}"


def plain(text: str) -> str:
    return re.sub(r"\s+", " ", html.unescape(re.sub(r"<[^>]+>", "", text))).strip()


def slug(title: str) -> str:
    stem = os.path.splitext(title.split(":", 1)[-1])[0]
    return re.sub(r"[^A-Za-z0-9._-]+", "-", stem).strip("-")[:70]


def main() -> int:
    wanted = []
    with open(os.path.join(HERE, "faces_titles.csv"), encoding="utf-8", newline="") as f:
        for row in csv.reader(f):
            if row and not row[0].startswith("#"):
                wanted.append((row[0].strip(), row[1].strip()))
    sources_path = os.path.join(HERE, "SOURCES.csv")
    with open(sources_path, encoding="utf-8", newline="") as f:
        have = {r["file"] for r in csv.DictReader(f)}
    for start in range(0, len(wanted), 40):
        batch = wanted[start : start + 40]
        info = image_info([t for _, t in batch])
        for tag, title in batch:
            meta = info.get(title) or info.get(title.replace("_", " ")) or {}
            ext = meta.get("extmetadata", {})
            licence = ext.get("LicenseShortName", {}).get("value", "")
            if not meta.get("url") or not ALLOWED.match(licence) or meta.get("mime") != "image/jpeg":
                print(f"skipped {title}: licence={licence!r} mime={meta.get('mime')!r}")
                continue
            rel = f"faces/set-{tag}/{slug(title)}.jpg"
            path = os.path.join(DATA, rel)
            if rel in have and os.path.isfile(path):
                continue
            os.makedirs(os.path.dirname(path), exist_ok=True)
            try:
                blob = get(scaled_url(meta["url"], meta.get("width") or 0))
            except urllib.error.HTTPError as e:
                print(f"failed {title}: HTTP {e.code} {meta['url']}")  # the rest of the list is still wanted
                continue
            with open(path, "wb") as out:
                out.write(blob)
            row = {
                "file": rel,
                "kind": "photo",
                "source_url": "https://commons.wikimedia.org/wiki/" + urllib.parse.quote(title.replace(" ", "_")),
                "author": plain(ext.get("Artist", {}).get("value", "")),
                "licence": licence,
                "licence_url": ext.get("LicenseUrl", {}).get("value", ""),
                "downloaded": datetime.date.today().isoformat(),
                "sha256": hashlib.sha256(blob).hexdigest(),
                "used_for": "faces benchmark, " + ("photos with faces" if tag == "a" else "photos without people"),
                "notes": f"{meta.get('width')}x{meta.get('height')} original; resized to <= 1920 px by Commons",
            }
            if rel not in have:
                with open(sources_path, "a", encoding="utf-8", newline="") as f:
                    csv.DictWriter(f, FIELDS, lineterminator="\n").writerow(row)
            print("saved", rel)
    return 0


if __name__ == "__main__":
    sys.exit(main())
