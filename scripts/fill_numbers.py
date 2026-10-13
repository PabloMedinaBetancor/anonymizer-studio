"""Write the documents with their {metric} markers replaced by the values of bench/results/*.json.

    python3 -s scripts/fill_numbers.py --publish      # README.md, docs/accuracy.md and docs/MEASURED.md, written from their templates
    python3 -s scripts/fill_numbers.py --verify       # exit 1 if a published document is not what its template fills to
    python3 -s scripts/fill_numbers.py --out build    # the templates, filled, somewhere else
    python3 -s scripts/fill_numbers.py --check        # only say which markers cannot be filled, and which figures are typed in prose

The documents a reader opens (README.md, docs/accuracy.md, docs/MEASURED.md) are the committed, filled copies of the
`*.template.md` files: nobody edits them, `--publish` writes them, and `tests/test_numbers_published.py` fails when one is stale or
still carries a raw {metric} marker. The work is done by scripts/figures.py.
"""

from __future__ import annotations

import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import figures  # noqa: E402


def published_name(template: str) -> str:
    return template.replace(".template", "", 1)


def publish(verify: bool, out: str | None = None) -> int:
    docs, folder = figures.settings(ROOT)
    results = figures.load_results(folder)
    problems = 0
    for template in (d for d in docs if d.endswith(".md")):  # the .html template belongs to scripts/make_demo_video.py
        with open(os.path.join(ROOT, template), encoding="utf-8") as f:
            filled, findings = figures.fill(template, f.read(), results)
        for finding in findings:
            print(finding, file=sys.stderr)
        problems += len(findings)
        name = published_name(template)
        if out:
            target = os.path.join(out, name)
            os.makedirs(os.path.dirname(target), exist_ok=True)
        else:
            target = os.path.join(ROOT, name)
        if verify:
            try:
                with open(target, encoding="utf-8") as f:
                    current = f.read()
            except OSError:
                current = None
            if current != filled:
                print(f"{name} is not what {template} fills to: run --publish", file=sys.stderr)
                problems += 1
        elif not findings:
            with open(target, "w", encoding="utf-8") as f:
                f.write(filled)
            print(f"wrote {name}")
    return 1 if problems else 0


def main() -> int:
    args = sys.argv[1:]
    if args == ["--publish"]:
        return publish(verify=False)
    if args == ["--verify"]:
        return publish(verify=True)
    if len(args) == 2 and args[0] == "--out":
        return publish(verify=False, out=args[1])
    if args == ["--check"]:
        status, summary, findings = figures.check(ROOT)
        for finding in findings:
            print(finding, file=sys.stderr)
        print(summary)
        return 0 if status == figures.PASS else 1
    print(__doc__, file=sys.stderr)
    return 2


if __name__ == "__main__":
    sys.exit(main())
