"""The only reader of the sealed part of truth v2: a frozen product, one read, a log.

    python3 -s -m bench.read_sealed freeze          # once, when nothing is left to change: records the hash of every frozen file
    python3 -s -m bench.read_sealed status          # what is frozen, what was read
    python3 -s -m bench.read_sealed read            # the one read: refuses unless the freeze still holds and nothing was read before

Everything that touches a sealed photo goes through `read`. Code that only wants the dev photos asks `dev_keys`, which never returns a sealed
key; code that is handed a key list of unknown origin calls `refuse_sealed(keys)` first. A second read needs `--reason` (the entry of
DECISIONS-V2-B.md that allows it); the log keeps the reason next to the hashes the read ran with.

The measurements the read runs are registered in `MEASUREMENTS` (name -> function taking the list of sealed keys and returning a JSON-able
dict) by `bench/sealed_measure.py`, which is written when the claims are pre-registered (M3) and is itself frozen.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import os
import sys
import time
from typing import Callable, Dict, Iterable, List, Optional, Sequence

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
SPLIT = os.path.join(HERE, "truth", "split-v2.json")
FREEZE = os.path.join(HERE, "results", "sealed-freeze.json")
READS = os.path.join(HERE, "results", "sealed-reads.json")
RESULT = os.path.join(HERE, "results", "sealed-v2.json")
# What a read depends on: the product, the models, the truth and the measurement code. A change to any of them after the freeze voids it.
FROZEN_GLOBS = (
    "src/anonymizer/",
    "models/SHA256SUMS",
    "bench/truth/",
    "bench/bench_cells.py",
    "bench/cells_eval.py",
    "bench/bench_faces.py",
    "bench/bench_plates.py",
    "bench/bench_faces_video.py",
    "bench/results_meta.py",
    "bench/sealed_measure.py",
    "bench/read_sealed.py",
)
SKIP = ("__pycache__", ".pyc")

MEASUREMENTS: Dict[str, Callable[[Sequence[str]], dict]] = {}


class SealedError(RuntimeError):
    """A refusal: the sealed part is not opened."""


def sha256_file(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def frozen_files(root: str = ROOT, globs: Iterable[str] = FROZEN_GLOBS) -> List[str]:
    """Every file the freeze covers, as paths relative to `root`, sorted."""
    found = set()
    for entry in globs:
        full = os.path.join(root, entry)
        if os.path.isdir(full):
            for folder, _, names in os.walk(full):
                for name in names:
                    path = os.path.join(folder, name)
                    if not any(s in path for s in SKIP):
                        found.add(os.path.relpath(path, root))
        elif os.path.isfile(full):
            found.add(entry)
    return sorted(found)


def fingerprint(root: str = ROOT, globs: Iterable[str] = FROZEN_GLOBS) -> Dict[str, str]:
    return {rel: sha256_file(os.path.join(root, rel)) for rel in frozen_files(root, globs)}


def load_split(path: Optional[str] = None) -> Dict[str, str]:
    """{key: "dev" | "sealed"} from the fixed split file."""
    with open(path or SPLIT, encoding="utf-8") as f:
        items = json.load(f)["items"]
    return {key: entry["part"] for key, entry in items.items()}


def dev_keys(path: Optional[str] = None) -> List[str]:
    """The dev photos: the only keys that ordinary code (pilot cells, thresholds) may run on."""
    return sorted(k for k, part in load_split(path).items() if part == "dev")


def sealed_keys(path: Optional[str] = None) -> List[str]:
    return sorted(k for k, part in load_split(path).items() if part == "sealed")


def refuse_sealed(keys: Iterable[str], path: Optional[str] = None) -> None:
    """Raise `SealedError` if any of `keys` is a sealed photo (or is not in the split at all: an unknown photo is not assumed dev)."""
    parts = load_split(path)
    for key in keys:
        part = parts.get(key)
        if part == "sealed":
            raise SealedError(f"{key} is sealed: only bench.read_sealed opens it, once, after the freeze")
        if part is None:
            raise SealedError(f"{key} is not in the split file: it is neither dev nor sealed")


def _load(path: str, default):
    if not os.path.isfile(path):
        return default
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _write(path: str, data) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".part"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=1, sort_keys=True)
        f.write("\n")
    os.replace(tmp, path)


def freeze(root: Optional[str] = None, freeze_path: Optional[str] = None, split_path: Optional[str] = None) -> dict:
    """Record the hash of every frozen file and of the split. Refuses to freeze twice: a new freeze after a read would hide a change."""
    root, freeze_path, split_path = root or ROOT, freeze_path or FREEZE, split_path or SPLIT
    if os.path.isfile(freeze_path):
        raise SealedError(f"{freeze_path} exists: the product is already frozen")
    record = {
        "time": time.strftime("%Y-%m-%d %H:%M:%S"),
        "split_sha256": sha256_file(split_path),
        "files": fingerprint(root),
        "measurements": sorted(MEASUREMENTS),
    }
    _write(freeze_path, record)
    return record


def check_freeze(
    root: Optional[str] = None, freeze_path: Optional[str] = None, split_path: Optional[str] = None
) -> dict:
    """The freeze record, if every frozen file and the split are still as recorded; else `SealedError` naming what changed."""
    root, freeze_path, split_path = root or ROOT, freeze_path or FREEZE, split_path or SPLIT
    record = _load(freeze_path, None)
    if record is None:
        raise SealedError("nothing is frozen: run `freeze` first (the sealed part is opened only for a frozen product)")
    if sha256_file(split_path) != record["split_sha256"]:
        raise SealedError("the split file changed after the freeze")
    now = fingerprint(root)
    changed = sorted(k for k in set(now) | set(record["files"]) if now.get(k) != record["files"].get(k))
    if changed:
        raise SealedError(f"{len(changed)} frozen file(s) changed after the freeze, first: {changed[0]}")
    return record


def read(
    reason: Optional[str] = None,
    root: Optional[str] = None,
    freeze_path: Optional[str] = None,
    split_path: Optional[str] = None,
    reads_path: Optional[str] = None,
    result_path: Optional[str] = None,
) -> dict:
    """The one read of the sealed part. Refuses unless the freeze holds; a second read needs `reason`. Runs every registered measurement on the
    sealed keys, writes `result_path` and logs the read (time, reason, freeze and split hashes) before returning."""
    split_path, reads_path, result_path = split_path or SPLIT, reads_path or READS, result_path or RESULT
    record = check_freeze(root, freeze_path, split_path)
    reads = _load(reads_path, [])
    if reads and not reason:
        raise SealedError(
            "the sealed part was already read; a second read needs a reason (the DECISIONS-V2-B.md entry that allows it)"
        )
    if not MEASUREMENTS:
        raise SealedError(
            "no measurement is registered (bench/sealed_measure.py, written with the pre-registered claims)"
        )
    keys = sealed_keys(split_path)
    reads.append(
        {
            "time": time.strftime("%Y-%m-%d %H:%M:%S"),
            "reason": reason,
            "split_sha256": record["split_sha256"],
            "freeze_time": record["time"],
            "keys": len(keys),
        }
    )
    _write(reads_path, reads)  # logged before the measurements run: a read that crashes still counts
    result = {
        "read": len(reads),
        "freeze_time": record["time"],
        "split_sha256": record["split_sha256"],
        "keys": len(keys),
        "results": {},
    }
    result["failed"] = []
    for name in sorted(MEASUREMENTS):
        # One measurement that fails must not lose the others (a read cannot be repeated without a recorded reason): its error is recorded, the result
        # file is rewritten after every measurement, and only an interrupt (Ctrl-C) leaves the loop, after the partial result is on disk.
        try:
            result["results"][name] = MEASUREMENTS[name](keys)
        except (Exception, SystemExit) as err:
            result["results"][name] = {"error": f"{type(err).__name__}: {err}"}
            result["failed"].append(name)
        except BaseException:
            result["failed"].append(name)
            _write(result_path, result)
            raise
        _write(result_path, result)
    return result


def status(freeze_path: Optional[str] = None, reads_path: Optional[str] = None) -> dict:
    record = _load(freeze_path or FREEZE, None)
    return {
        "frozen": record is not None,
        "freeze_time": record and record["time"],
        "files": record and len(record["files"]),
        "reads": _load(reads_path or READS, []),
    }


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("command", choices=["freeze", "status", "read"])
    ap.add_argument("--reason")
    args = ap.parse_args(argv)
    try:
        if args.command == "freeze":
            record = freeze()
            print(f"frozen {len(record['files'])} files, split {record['split_sha256'][:12]}")
        elif args.command == "status":
            print(json.dumps(status(), indent=1))
        else:
            try:
                importlib.import_module("bench.sealed_measure")  # registers MEASUREMENTS
            except ImportError:
                pass
            result = read(args.reason)
            print(
                f"read {result['read']}: {result['keys']} sealed photos, {len(result['results'])} measurements -> {RESULT}"
            )
            if result["failed"]:
                print(f"FAILED measurements (see the result file): {', '.join(result['failed'])}", file=sys.stderr)
                return 3
    except SealedError as refusal:
        print(f"refused: {refusal}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
