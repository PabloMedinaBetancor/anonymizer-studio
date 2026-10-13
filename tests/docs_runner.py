"""Runs documentation examples in a sandbox folder, so a block behaves as it does for a reader at the repository root.

python tests/docs_runner.py page SPEC.json REPORT.json real     # run the blocks named in SPEC.json on the real models
python tests/docs_runner.py script SCRIPT.py real|fakes         # run one examples/*.py file in a fresh sandbox

Not meant to be run by hand: `tests/docs_support.py`'s callers (`test_docs_blocks.py`, `test_docs_examples.py`,
`test_examples_scripts.py`) start it, or call `run_page_in_process` (the fakes). Needs `anonymizer` importable
(`PYTHONPATH=src`).

The sandbox. Every page (and every script) runs in a new temporary folder that is the working directory and holds the
sample files under their repository paths: `demo/street-madrid.mp4`, `demo/street-hongkong.mp4`,
`demo/street-taipei.jpg`, `demo/street-westminster.jpg` and the 5-second clip `examples/data/street-madrid-5s.mp4`.
In `real` mode they are the repository's own files and the detectors are
the real models; in `fakes` mode they are small synthetic stand-ins (a white plate and a red face, found by the
detectors of `tests/fakes.py`). Nothing else is there: a block that writes files makes its folders itself, and a second
page never sees what the first wrote.

The blocks of a page share one namespace and run in order. Markers, on the line before the fence:

    <!-- docs-test: skip - the reason -->        never run (needs a Windows machine, a second process, ...)
    <!-- docs-test: real-only - the reason -->   run only on the real models (the fakes find other things)

A ```text block that follows a python block and starts with the line `# output` is the output the page shows; on the
real models the printed output must equal it (a line may contain `...` for any text, for times and the like).
"""

from __future__ import annotations

import contextlib
import io
import json
import os
import re
import shutil
import sys
import tempfile
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EXAMPLE_FILES = (
    "examples/data/street-madrid-5s.mp4",
)  # the 5 s clip, outside demo/ so the Studio's sample list is unchanged
DEMO_FILES = (
    "street-madrid.mp4",
    "street-hongkong.mp4",
    "street-taipei.jpg",
    "street-westminster.jpg",
)


def make_fake_demo(folder: Path) -> None:
    """Synthetic stand-ins with the demo files' names: photos with a white plate and a red face, clips with a moving plate."""
    import cv2
    import numpy as np

    from tests import media

    folder.mkdir(parents=True, exist_ok=True)
    for name in ("street-taipei.jpg", "street-westminster.jpg"):
        img = np.full((240, 320, 3), (90, 120, 60), np.uint8)
        cv2.rectangle(img, (40, 150), (140, 180), (250, 250, 250), -1)
        cv2.circle(img, (230, 80), 22, (0, 0, 255), -1)
        cv2.imwrite(str(folder / name), img, [cv2.IMWRITE_JPEG_QUALITY, 98])
    for name in ("street-madrid.mp4", "street-hongkong.mp4"):
        media.moving_plate_clip(str(folder / name), n=36, audio=False)
    clip = folder.parent / EXAMPLE_FILES[0]
    clip.parent.mkdir(parents=True, exist_ok=True)
    media.moving_plate_clip(str(clip), n=36, audio=False)


def make_sandbox(mode: str) -> Path:
    """A new temporary working folder holding `demo/`; the caller removes it."""
    work = Path(tempfile.mkdtemp(prefix="anon-docs-"))
    if mode == "real":
        shutil.copytree(ROOT / "demo", work / "demo", ignore=shutil.ignore_patterns("SOURCES.csv"))
        for name in EXAMPLE_FILES:
            (work / name).parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(ROOT / name, work / name)
    else:
        make_fake_demo(work / "demo")
    return work


def output_matches(expected: str, actual: str) -> bool:
    """Equal after stripping trailing spaces; a `...` in an expected line stands for any text on that line."""
    want = [ln.rstrip() for ln in expected.strip().splitlines()]
    got = [ln.rstrip() for ln in actual.strip().splitlines()]
    if len(want) != len(got):
        return False
    for w, g in zip(want, got, strict=True):
        if "..." in w:
            pattern = ".*".join(re.escape(part) for part in w.split("..."))
            if not re.fullmatch(pattern, g):
                return False
        elif w != g:
            return False
    return True


def fake_loader():
    """What `anonymizer.classes.load` returns on the fakes: a white-plate finder and a red-face finder."""
    from tests.fakes import face_module, plate_module

    def load(kinds, threads, models, **kw):
        modules = []
        if "face" in kinds:
            modules.append(face_module())
        if "plate" in kinds:
            modules.append(plate_module())
        return modules

    return load


def run_blocks_here(spec: dict, mode: str) -> list[dict]:
    """Run the blocks of `spec` in this process, in one namespace, in the current folder. Stops at the first failure."""
    namespace: dict = {"__name__": "__main__"}
    results = []
    failed = False
    for block in spec["blocks"]:
        entry = {
            "number": block["number"],
            "line": block["line"],
            "status": "ok",
            "error": None,
            "actual_output": None,
            "output_mismatch": None,
        }
        if failed:
            entry["status"] = "not-run"
        elif block["skip_reason"]:
            entry["status"] = "skipped"
            entry["error"] = block["skip_reason"]
        elif block.get("real_only") and mode != "real":
            entry["status"] = "skipped"
            entry["error"] = "real models only: " + block["real_only"]
        else:
            captured = io.StringIO()
            try:
                with contextlib.redirect_stdout(captured):
                    exec(compile(block["code"], f"{spec['page']} block {block['number']}", "exec"), namespace)  # noqa: S102
            except BaseException:  # noqa: BLE001 - an example may fail in any way, and the report says how
                failed = True
                entry["status"] = "fail"
                entry["error"] = "".join(traceback.format_exception_only(*sys.exc_info()[:2])).strip()
                entry["traceback"] = traceback.format_exc()[-1500:]
            entry["actual_output"] = captured.getvalue()
            expected = block["expected_output"]
            if (
                not failed
                and mode == "real"
                and expected is not None
                and not output_matches(expected, captured.getvalue())
            ):
                failed = True
                entry["status"] = "fail"
                entry["error"] = "the printed output differs from the `# output` block that follows"
                entry["output_mismatch"] = {"expected": expected.strip(), "actual": captured.getvalue().strip()}
        results.append(entry)
    return results


def run_page_in_process(spec: dict, mode: str) -> list[dict]:
    """The blocks of one page in a new sandbox, in this process (the default suite, on the fakes)."""
    import socket
    from unittest import mock

    import anonymizer.classes

    work = make_sandbox(mode)
    here = os.getcwd()
    os.chdir(work)
    # A page may switch the network off for its own process (the offline check): put the sockets back afterwards,
    # so that the tests that follow in this process can open theirs.
    sockets = (socket.socket.connect, socket.getaddrinfo)
    try:
        if mode == "fakes":
            with mock.patch.object(anonymizer.classes, "load", fake_loader()):
                return run_blocks_here(spec, mode)
        return run_blocks_here(spec, mode)
    finally:
        socket.socket.connect, socket.getaddrinfo = sockets
        os.chdir(here)
        shutil.rmtree(work, True)


def run_page(spec_path: Path, report_path: Path, mode: str) -> int:
    spec = json.loads(spec_path.read_text(encoding="utf-8"))
    results = run_page_in_process(spec, mode)
    report_path.write_text(json.dumps({"page": spec["page"], "results": results}, indent=1), encoding="utf-8")
    return 1 if any(r["status"] == "fail" for r in results) else 0


def run_script(script: Path, mode: str) -> int:
    """Run one examples/*.py file as a program in a new sandbox; its exit code is returned."""
    import subprocess

    work = make_sandbox(mode)
    try:
        shutil.copy2(script, work / script.name)
        if mode == "fakes":
            wrapper = (
                "import runpy, sys\nfrom unittest import mock\nimport anonymizer.classes\nfrom tests import docs_runner\n"
                f"with mock.patch.object(anonymizer.classes, 'load', docs_runner.fake_loader()):\n"
                f"    runpy.run_path({script.name!r}, run_name='__main__')\n"
            )
            command = [sys.executable, "-c", wrapper]
        else:
            command = [sys.executable, script.name]
        env = dict(os.environ)
        paths = [str(ROOT / "src"), str(ROOT)] if mode == "fakes" else [str(ROOT / "src")]
        env["PYTHONPATH"] = os.pathsep.join(paths + ([env["PYTHONPATH"]] if env.get("PYTHONPATH") else []))
        done = subprocess.run(command, cwd=work, env=env, capture_output=True, text=True, timeout=2400)
        sys.stdout.write(done.stdout)
        sys.stderr.write(done.stderr)
        return done.returncode
    finally:
        shutil.rmtree(work, True)


def main(argv: list[str]) -> int:
    if len(argv) == 4 and argv[0] == "page":
        return run_page(Path(argv[1]), Path(argv[2]), argv[3])
    if len(argv) == 3 and argv[0] == "script":
        return run_script(Path(argv[1]), argv[2])
    print(__doc__)
    return 0 if any(a in ("-h", "--help") for a in argv) else 2


if __name__ == "__main__":
    sys.path.insert(0, str(ROOT))
    sys.exit(main(sys.argv[1:]))
