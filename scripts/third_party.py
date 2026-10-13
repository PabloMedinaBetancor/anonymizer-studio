"""Write or check THIRD-PARTY.md from scripts/third_party.json.

    python3 scripts/third_party.py --write    # regenerate THIRD-PARTY.md
    python3 scripts/third_party.py --check    # exit 1 if THIRD-PARTY.md is not what the data says

The data lists every model and library of the product with its licence; tests/test_licence_and_provenance.py checks the licence names against the allow-list and that
every model file of models/SHA256SUMS has a row, so a component cannot be added to the product without being listed here.
"""

from __future__ import annotations

import json
import os
import re
import sys
from typing import Dict, List

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
DATA = os.path.join(HERE, "third_party.json")
OUTPUT = os.path.join(ROOT, "THIRD-PARTY.md")


def load() -> Dict[str, object]:
    with open(DATA, encoding="utf-8") as f:
        return json.load(f)


def licence_cell(entry: Dict[str, str]) -> str:
    """The licence name and, after it, the note (a note that starts with ";" is joined to the name without a space)."""
    note = entry.get("licence_note", "")
    return f"{entry['licence']}{note}" if note.startswith(";") else f"{entry['licence']} {note}".strip()


APACHE_RF_DETR = re.compile(r"^RF-DETR-(N|S|M|L)$")  # the sizes Roboflow designates Apache 2.0; XL and 2XL are PML 1.0


def variant_problems(entry: Dict[str, str]) -> List[str]:
    """What is missing or wrong in the row of an RF-DETR model: the exact variant, the licence text and URL of its weights, the weights'
    URL and the hash of the file. A row that only says "RF-DETR" (or names XL / 2XL, which are PML 1.0) is a problem, not a row."""
    problems: List[str] = []
    variant = entry.get("variant", "")
    if not APACHE_RF_DETR.match(variant):
        problems.append(f"variant {variant!r} is not one of RF-DETR-N, -S, -M, -L (the Apache-designated sizes)")
    elif variant not in entry["name"]:
        problems.append(f"the name {entry['name']!r} does not carry the variant {variant}")
    for key in ("licence_url", "licence_quote", "weights_url", "sha256"):
        if not entry.get(key):
            problems.append(f"no {key}")
    if "Apache" not in entry.get("licence_quote", "Apache") or "Apache" not in entry["licence"]:
        problems.append("the quoted licence is not Apache")
    return problems


def render(data: Dict[str, object]) -> str:
    lines: List[str] = ["# Third-party components", "", str(data["intro"]), "", "## Models", ""]
    lines += ["| Model | Licence of the code and the weights | Trained on | Note |", "|---|---|---|---|"]
    for m in data["models"]:  # type: ignore[union-attr]
        cell = licence_cell(m)
        if "weights_url" in m:  # a row that names its weights: where they come from and the text of their licence
            cell += f' Weights: {m["weights_url"]} (SHA-256 `{m["sha256"]}`). Licence text: {m["licence_url"]}. "{m["licence_quote"]}"'
        lines.append(f"| {m['name']} | {cell} | {m['trained_on']} | {m['note']} |")
    lines += ["", "## Libraries", "", "| Component | Licence | Used for |", "|---|---|---|"]
    for lib in data["libraries"]:  # type: ignore[union-attr]
        lines.append(f"| {lib['name']} | {licence_cell(lib)} | {lib['used_for']} |")
    lines += ["", "## Test and benchmark data (not shipped)", "", str(data["data"]), ""]
    return "\n".join(lines)


def main(argv: List[str]) -> int:
    text = render(load())
    if "--write" in argv:
        with open(OUTPUT, "w", encoding="utf-8") as f:
            f.write(text)
        return 0
    with open(OUTPUT, encoding="utf-8") as f:
        current = f.read()
    if current != text:
        print(
            "THIRD-PARTY.md is not what scripts/third_party.json says: run scripts/third_party.py --write",
            file=sys.stderr,
        )
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
