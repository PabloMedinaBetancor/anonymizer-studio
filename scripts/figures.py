"""Numbers in documents come from the benchmark results, never from a hand.

A document writes a metric as a marker, `{plates.photo.eu.recall_full_pct}`, and this module replaces it with the value
that `bench/results/*.json` holds under that name. `check` is the gate on the published numbers; the command line fills
the documents:

    scripts/fill_numbers.py REPO [--out DIR] [--check]

A metric name has at least three dot-separated parts. A results file may hold it as one flat key ("plates.photo.eu.x" at
the top level) or nested ("faces": {"photo": {...}}), and the two may be mixed. A hardware record
({"cpu": ..., "cores": ..., "ram_gb": ...}) is written as one phrase; a list of two figures (an interval) as "low-high". A results file may carry a top-level `_publication` map,
`{"metric.or.prefix": {"publishable": false, "why": "..."}}`: a marker under such an entry is refused (`not-publishable`) and stays
unfilled, however the figure was measured.

The constants DOCS (the documents with markers), RESULTS (the folder of JSON files, default "bench/results") and ALLOW (a line
that contains one of these strings may keep a number with a unit) hold the settings.
"""

from __future__ import annotations

import fnmatch
import glob
import json
import os
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

PASS, FAIL = "pass", "fail"


class ConfigError(Exception):
    """The results folder cannot be read."""


@dataclass
class Finding:
    path: str
    line: int
    rule: str
    text: str

    def __str__(self) -> str:
        where = f"{self.path}:{self.line}" if self.line else self.path
        return f"{where}  {self.rule}: {self.text}"


def excluded(path: str, patterns: Sequence[str]) -> bool:
    return any(fnmatch.fnmatchcase(path, p) for p in patterns)


def status_of(findings: Sequence[Finding]) -> str:
    return FAIL if findings else PASS


MARKER = re.compile(r"\{([a-z][a-z0-9_]*(?:\.[a-z0-9_]+){2,})\}")
FENCE = re.compile(r"^(```|~~~).*?^\1[^\n]*$", re.M | re.S)
INLINE_CODE = re.compile(r"`[^`\n]*`")
URL = re.compile(r"https?://\S+")
# A figure that reads as a measurement: a number with a unit of accuracy, size, time or speed.
LOOSE_NUMBER = re.compile(
    r"(?<![\w.{])\d[\d.,]*(?:\s?%|\s?(?:ms|MB|GB|fps)\b|\s?[x×]\s+faster|\s+times\s+(?:faster|slower)"
    r"|\s?s\s+per\b|\s+seconds\s+per\b|\s?s/\w)",
    re.I,
)


def blank(match: "re.Match[str]") -> str:
    return re.sub(r"[^\n]", " ", match.group(0))


def prose(text: str) -> str:
    """The text with code (fenced blocks, inline spans), URLs and markers blanked out, row for row."""
    text = FENCE.sub(blank, text)
    text = INLINE_CODE.sub(blank, text)
    text = URL.sub(blank, text)
    return MARKER.sub(blank, text)


def load_results(folder: str) -> Dict[str, Any]:
    """{file name: parsed JSON} for every *.json in the folder; a file that does not parse is a ConfigError."""
    results: Dict[str, Any] = {}
    for path in sorted(glob.glob(os.path.join(folder, "*.json"))):
        try:
            with open(path, encoding="utf-8") as f:
                results[os.path.basename(path)] = json.load(f)
        except ValueError as e:
            raise ConfigError(f"{path}: not valid JSON ({e})") from e
    return results


def descend(node: Any, parts: Sequence[str]) -> Tuple[bool, Any]:
    """Find `parts` in `node`: at each level the longest flat key made of the next parts wins, then the shorter ones."""
    if not isinstance(node, dict):
        return False, None
    for i in range(len(parts), 0, -1):
        key = ".".join(parts[:i])
        if key in node:
            if i == len(parts):
                return True, node[key]
            found, value = descend(node[key], parts[i:])
            if found:
                return True, value
    return False, None


def lookup(results: Dict[str, Any], name: str) -> List[Tuple[str, Any]]:
    """Every (file, value) that holds the metric."""
    parts = name.split(".")
    hits = []
    for file, data in results.items():
        found, value = descend(data, parts)
        if found:
            hits.append((file, value))
    return hits


def render(value: Any) -> Optional[str]:
    """The text a value becomes in a document, or None when it is not something a sentence can hold."""
    if isinstance(value, bool) or value is None:
        return None
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        return str(int(value)) if value.is_integer() else f"{value:.10g}"
    if isinstance(value, str):
        return value or None
    if isinstance(value, (list, tuple)) and len(value) == 2:
        low, high = (render(v) for v in value)  # an interval: "94.9-99.8"
        return f"{low}-{high}" if low is not None and high is not None else None
    if isinstance(value, dict) and value.get("cpu"):
        ram = value.get("ram_gb") or (value["ram_mb"] / 1024 if value.get("ram_mb") else None)
        parts = [str(value["cpu"])]
        if value.get("cores"):
            parts.append(f"{value['cores']} cores")
        if ram:
            parts.append(f"{render(round(ram, 1))} GB RAM")
        return ", ".join(parts)
    return None


def not_publishable(results: Dict[str, Any], name: str, files: Sequence[str]) -> Optional[str]:
    """Why a metric must not be quoted, from the `_publication` map of the results file that holds it: {name or prefix: {"publishable":
    false, "why": ...}}. A prefix covers every metric under it (at a dot)."""
    for file in files:
        notes = results[file].get("_publication") if isinstance(results[file], dict) else None
        if not isinstance(notes, dict):
            continue
        for prefix, entry in notes.items():
            if (name == prefix or name.startswith(prefix + ".")) and isinstance(entry, dict):
                if entry.get("publishable") is False:
                    return f"{file}: {entry.get('why') or 'marked as not publishable'}"
    return None


def resolve(results: Dict[str, Any], name: str) -> Tuple[Optional[str], Optional[Tuple[str, str]]]:
    """(text, None) for a metric that has one value; (None, (rule, why)) when it cannot be written."""
    hits = lookup(results, name)
    if not hits:
        return None, ("no-metric", f"{{{name}}} is in no results file")
    refused = not_publishable(results, name, [f for f, _ in hits])
    if refused is not None:
        return None, ("not-publishable", f"{{{name}}} is not to be quoted ({refused})")
    texts = {render(v) for _, v in hits}
    if len(texts) > 1:
        return None, ("ambiguous", f"{{{name}}} has different values in " + ", ".join(f for f, _ in hits))
    text = texts.pop()
    if text is None:
        return None, ("no-value", f"{{{name}}} in {hits[0][0]} is not a figure or text (null, a list or a record)")
    return text, None


def fill(rel: str, text: str, results: Dict[str, Any]) -> Tuple[str, List[Finding]]:
    """The document with every resolvable marker replaced, and a finding for each that was not."""
    findings: List[Finding] = []

    def sub(match: "re.Match[str]") -> str:
        value, problem = resolve(results, match.group(1))
        if problem is None:
            return value or ""
        line = text.count("\n", 0, match.start()) + 1
        findings.append(Finding(rel, line, *problem))
        return match.group(0)

    return MARKER.sub(sub, text), findings


def loose_numbers(rel: str, text: str, allow: Sequence[str]) -> List[Finding]:
    """Numbers with a unit that are written in the prose instead of being a marker."""
    lines = text.splitlines()
    found = []
    for i, row in enumerate(prose(text).splitlines(), 1):
        if allow and any(a in lines[i - 1] for a in allow):
            continue
        for m in LOOSE_NUMBER.finditer(row):
            found.append(Finding(rel, i, "loose-number", f"'{m.group(0).strip()}' is typed, not a {{metric}} marker"))
    return found


DOCS = (
    "README.template.md",
    "docs/accuracy.template.md",
    "docs/MEASURED.template.md",
    "docs/media/benchmark.template.html",
)
RESULTS = "bench/results"
ALLOW = ("100% offline", "at full size / 50 % / 30 %", "height: 100%", "width: 100%")


def settings(root: str):
    """(documents that carry markers, the results folder)."""
    return list(DOCS), os.path.join(root, RESULTS)


def check(root: str) -> Tuple[str, str, List[Finding]]:
    """Every marker resolves, and no figure with a unit is left typed in the prose of the documents."""
    docs, folder = settings(root)
    results = load_results(folder)
    findings: List[Finding] = []
    markers = 0
    for rel in docs:
        with open(os.path.join(root, rel), encoding="utf-8") as f:
            text = f.read()
        markers += len(MARKER.findall(text))
        findings += fill(rel, text, results)[1]
        findings += loose_numbers(rel, text, ALLOW)
    return status_of(findings), f"{len(docs)} documents, {markers} markers, {len(results)} results files", findings
