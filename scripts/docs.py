"""Build, serve and check the documentation site, and print the PDF manual.

python scripts/docs.py build           # regenerate every generated file, then build site/ with mkdocs --strict
python scripts/docs.py serve           # regenerate, then serve the site at http://127.0.0.1:8000 for editing
python scripts/docs.py check           # change nothing: generated files are current, the site builds, every example runs
python scripts/docs.py run PAGE.md     # run the python blocks of one page and print what each one printed
python scripts/docs.py pdf [OUT.pdf] [--manual user|developer]   # print the documentation to one PDF manual with headless Chrome

`check` is the gate: it exits 0 when the generated files equal the committed ones, `mkdocs build --strict` has no
warning, the built site asks no other host for anything, every ```python block of the pages, of `AGENTS.md` and of
`README.md`, and every `examples/*.py`, runs on the detectors of `tests/fakes.py` and again on the real models, the
output a page shows equals what the real run printed, every internal link and anchor resolves, and no banned phrase or
personal detail is in a page. It prints a short summary and writes the full output to `work/docs-check/`
(`outputs/` holds what every block printed on the real models). The real-model part takes minutes: run it niced and
one at a time. `--skip-real` leaves it out.

Files that are still the 2.2.0 text (`LEGACY` in `tests/docs_support.py`) are scanned too, but what is found in them is
listed as a warning until `--final`, which turns the warnings into failures.

MkDocs and the PDF's Markdown come from `requirements/docs.txt`; install it in a separate environment and name that
interpreter in `DOCS_PYTHON` (default: the interpreter running this script). The examples run with the interpreter
running this script, so that one needs the Anonymizer's dependencies (and `PYTHONPATH=src` or an install), the models
(`ANONYMIZER_MODELS`) and `pytest`.

`--help` prints this text and does nothing else.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import signal
import subprocess
import sys
import tempfile
import time
import xml.etree.ElementTree as ET
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DOCS = ROOT / "docs"
LLMS_FULL = ROOT / "llms-full.txt"
MKDOCS_YML = ROOT / "mkdocs.yml"
OUT = ROOT / "work" / "docs-check"

FAST_TESTS = [
    "tests/test_docs_blocks.py",
    "tests/test_docs_generated.py",
    "tests/test_docs_style.py",
    "tests/test_docs_links.py",
    "tests/test_docs_reference.py",
    "tests/test_docs_pdf.py",
]
REAL_TESTS = ["tests/test_docs_examples.py", "tests/test_examples_scripts.py"]


# --- generated files ----------------------------------------------------------------------------------------------


def _name(path: Path) -> str:
    return path.relative_to(ROOT).as_posix() if path.is_relative_to(ROOT) else path.name


def nav_pages() -> list[str]:
    """The page paths of `mkdocs.yml`'s `nav`, in order (every `- Title: path.md` line)."""
    if not MKDOCS_YML.is_file():
        return []
    text = MKDOCS_YML.read_text(encoding="utf-8")
    nav = text.split("\nnav:", 1)[1] if "\nnav:" in text else ""
    return re.findall(r":\s+([\w./-]+\.md)\s*$", nav, re.M)


def site_pages() -> list[str]:
    """The pages of the site under `docs/`: the navigation's, in order, then any other page that is not a development document."""
    sys.path.insert(0, str(ROOT))
    try:
        from tests import docs_support
    finally:
        sys.path.remove(str(ROOT))
    pages = []
    for name in nav_pages():
        if (DOCS / name).is_file() and name not in pages:
            pages.append(name)
    for path in sorted(DOCS.rglob("*.md")) if DOCS.is_dir() else []:
        name = path.relative_to(DOCS).as_posix()
        if name not in pages and not docs_support.is_excluded(path):
            pages.append(name)
    return pages


def render_llms_full() -> str:
    """`llms-full.txt`: every page of the site as one plain file, in navigation order and then any page the navigation
    leaves out (development documents excluded). Each page follows a line `# docs/path.md`; the page text is as written."""
    parts = []
    for name in site_pages():
        body = (DOCS / name).read_text(encoding="utf-8")
        parts.append(f"# docs/{name}\n\n{body.rstrip()}\n")
    return "\n".join(parts)


def script_env() -> dict:
    env = dict(os.environ)
    src = str(ROOT / "src")
    env["PYTHONPATH"] = src + os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else src
    return env


def run_script(*args: str) -> subprocess.CompletedProcess:
    return subprocess.run([sys.executable, *args], cwd=ROOT, env=script_env(), capture_output=True, text=True)


def numbers_state() -> tuple[bool | None, str]:
    """(current?, note) for `fill_numbers.py --verify`."""
    done = run_script("scripts/fill_numbers.py", "--verify")
    lines = (done.stdout + done.stderr).strip().splitlines()
    return done.returncode == 0, lines[-1] if lines else ""


def regenerate() -> list[str]:
    """Rewrite every generated file (the API reference through `scripts/api_docs.py`, then `llms-full.txt`); return the
    names of those this script wrote that changed. The published numbers come from `scripts/fill_numbers.py`."""
    changed = []
    api = run_script("scripts/api_docs.py")
    if api.returncode != 0:
        raise SystemExit("scripts/api_docs.py failed:\n" + (api.stderr or api.stdout)[-1500:])
    full = render_llms_full()  # after the reference pages it joins
    if not LLMS_FULL.is_file() or LLMS_FULL.read_text(encoding="utf-8") != full:
        LLMS_FULL.write_text(full, encoding="utf-8")
        changed.append("llms-full.txt")
    return changed


def stale_files() -> list[str]:
    """Names of generated files that differ from what their script writes now. Changes nothing."""
    stale = []
    if not LLMS_FULL.is_file() or LLMS_FULL.read_text(encoding="utf-8") != render_llms_full():
        stale.append("llms-full.txt")
    if run_script("scripts/api_docs.py", "--check").returncode != 0:
        stale.append("the API reference (scripts/api_docs.py --check)")
    current, _note = numbers_state()
    if current is False:
        stale.append("the published numbers (scripts/fill_numbers.py --verify)")
    return stale


# --- the site -----------------------------------------------------------------------------------------------------


def docs_python() -> str:
    return os.environ.get("DOCS_PYTHON", sys.executable)


def mkdocs_command(*args: str) -> list[str]:
    return [docs_python(), "-m", "mkdocs", *args]


def mkdocs_env() -> dict:
    env = dict(os.environ)
    env["NO_MKDOCS_2_WARNING"] = "true"  # the theme's notice about the next MkDocs version is not about this site
    return env


def build_site(destination: Path) -> tuple[bool, str]:
    done = subprocess.run(
        mkdocs_command("build", "--strict", "--clean", "-d", str(destination)),
        cwd=ROOT,
        env=mkdocs_env(),
        capture_output=True,
        text=True,
    )
    log = done.stdout + done.stderr
    if done.returncode != 0 and "No module named mkdocs" in log:
        log += "\nInstall the toolchain: python -m pip install --require-hashes -r requirements/docs.txt, and set DOCS_PYTHON."
    return done.returncode == 0, log


def outside_requests(site: Path) -> list[str]:
    sys.path.insert(0, str(ROOT))
    try:
        from tests import docs_support
    finally:
        sys.path.remove(str(ROOT))
    return docs_support.outside_requests(site)


def cmd_build(_args) -> int:
    changed = regenerate()
    print("regenerated:", ", ".join(changed) if changed else "nothing to change")
    ok, log = build_site(ROOT / "site")
    print(log.rstrip())
    if not ok:
        return 1
    problems = outside_requests(ROOT / "site")
    for problem in problems:
        print("outside request:", problem)
    return 1 if problems else 0


def cmd_serve(_args) -> int:
    regenerate()
    return subprocess.call(mkdocs_command("serve"), cwd=ROOT, env=mkdocs_env())


# --- run one page ---------------------------------------------------------------------------------------------------


def cmd_run(args) -> int:
    """Run the python blocks of one page and print, block by block, the status and what it printed."""
    sys.path.insert(0, str(ROOT))
    try:
        from tests import docs_support as ds
    finally:
        sys.path.remove(str(ROOT))
    path = Path(args.page).resolve()
    if not path.is_file():
        print(f"no such page: {args.page}")
        return 2
    blocks = ds.python_blocks(path)
    if not blocks:
        print("no python blocks")
        return 0
    if args.real:
        try:
            os.nice(19)
        except OSError:
            pass
    results = ds.run_blocks(ds.rel(path), blocks, "real" if args.real else "fakes")
    failed = 0
    for r in results:
        print(f"--- block {r['number']} (line {r['line']}): {r['status']}" + (f": {r['error']}" if r["error"] else ""))
        shown = r.get("actual_output")
        if shown and shown.strip():
            print(shown.rstrip())
        if r.get("output_mismatch"):
            print("--- the page shows:\n" + r["output_mismatch"]["expected"])
        failed += r["status"] == "fail"
    return 1 if failed else 0


# --- check --------------------------------------------------------------------------------------------------------


PRIVATE_NAMES_GLOB = "docs-private-names*.txt"


def private_names_paths() -> str | None:
    """The private internal-names lists: ANON_DOCS_PRIVATE_NAMES, else the files beside the docs venv's folder."""
    named = os.environ.get("ANON_DOCS_PRIVATE_NAMES")
    if named:
        return named if any(Path(p).is_file() for p in named.split(os.pathsep)) else None
    python = os.environ.get("DOCS_PYTHON")
    if python:
        beside = sorted(Path(python).absolute().parents[2].glob(PRIVATE_NAMES_GLOB))
        if beside:
            return os.pathsep.join(str(p) for p in beside)
    return None


def run_tests(tests: list[str], env_extra: dict, junit: Path, log: Path) -> int:
    env = dict(os.environ)
    env.update(env_extra)
    private = private_names_paths()
    if private:
        env["ANON_DOCS_PRIVATE_NAMES"] = private
    else:
        env.pop("ANON_DOCS_PRIVATE_NAMES", None)
    env["PYTHONPATH"] = script_env()["PYTHONPATH"]
    command = [sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider", f"--junitxml={junit}", *tests]
    with open(log, "w", encoding="utf-8") as out:
        return subprocess.run(command, cwd=ROOT, env=env, stdout=out, stderr=subprocess.STDOUT).returncode


def example_summary(report: Path) -> tuple[int, list[str], list[str], dict[str, str]]:
    """(blocks run, failures as `page block N: error`, skips as `page block N: reason`, outputs by `page block N`)."""
    run, failures, skips, outputs = 0, [], [], {}
    if report.is_file():
        for line in report.read_text(encoding="utf-8").splitlines():
            entry = json.loads(line)
            for r in entry["results"]:
                name = (
                    f"{entry['page']} block {r['number']}"
                    if not entry["page"].startswith("examples/")
                    else entry["page"]
                )
                if r["status"] in ("ok", "fail"):
                    run += 1
                if r["status"] == "fail":
                    detail = str(r["error"]).splitlines()[-1] if r["error"] else "failed"
                    if r.get("output_mismatch"):
                        shown, printed = r["output_mismatch"]["expected"], r["output_mismatch"]["actual"]
                        detail += f" (the page shows {shown[:60]!r}, the run printed {printed[:60]!r})"
                    failures.append(f"{name}: {detail}")
                if r["status"] == "skipped":
                    skips.append(f"{name}: {r['error']}")
                if r.get("actual_output"):
                    outputs[name] = r["actual_output"]
    return run, failures, skips, outputs


def junit_failures(junit: Path) -> dict[str, list[str]]:
    """Failed test cases by test file, each as `test name: first line of the message`."""
    found: dict[str, list[str]] = {}
    if not junit.is_file():
        return found
    for case in ET.parse(junit).getroot().iter("testcase"):
        bad = case.find("failure")
        bad = bad if bad is not None else case.find("error")
        if bad is not None:
            classname = case.get("classname", "")
            module = classname.split(".")[1] if "." in classname else classname
            message = (bad.get("message") or "").strip().splitlines()
            found.setdefault(module, []).append(f"{case.get('name')}: {message[0] if message else 'failed'}")
    return found


def legacy_summary(report: Path) -> dict[str, int]:
    counts: dict[str, int] = {}
    if report.is_file():
        for line in report.read_text(encoding="utf-8").splitlines():
            entry = json.loads(line)
            counts[entry["file"]] = counts.get(entry["file"], 0) + 1
    return counts


def cmd_check(args) -> int:
    if not args.no_nice:
        try:
            os.nice(19)
        except OSError:
            pass
    OUT.mkdir(parents=True, exist_ok=True)
    shutil.rmtree(OUT / "outputs", True)
    failed = False

    stale = stale_files()
    print(f"generated files: {'all current' if not stale else 'STALE: ' + ', '.join(stale)}")
    failed |= bool(stale)
    current, numbers_note = numbers_state()
    print(f"published numbers: {'current' if current else numbers_note or 'STALE'}")

    site = Path(tempfile.mkdtemp(prefix="anonymizer-site-"))
    try:
        ok, log = build_site(site)
        (OUT / "mkdocs.log").write_text(log, encoding="utf-8")
        warnings = [ln for ln in log.splitlines() if ln.startswith(("WARNING", "ERROR", "Aborted"))]
        print(
            f"mkdocs build --strict: {'ok' if ok else 'FAILED'}"
            + ("" if ok else f" ({len(warnings)} lines; see {OUT / 'mkdocs.log'})")
        )
        for line in warnings[:8]:
            print("   ", line)
        failed |= not ok
        if ok:
            problems = outside_requests(site)
            print(f"requests outside the site: {'none' if not problems else len(problems)}")
            for problem in problems[:8]:
                print("   ", problem)
            failed |= bool(problems)
    finally:
        shutil.rmtree(site, True)

    report, legacy = OUT / "examples.jsonl", OUT / "legacy.jsonl"
    for stale_output in (report, legacy):
        stale_output.unlink(missing_ok=True)
    extra = {"ANON_DOCS_LEGACY_REPORT": str(legacy), "ANON_DOCS_FINAL": "1" if args.final else "0"}
    junit_fast, junit_real = OUT / "junit-fast.xml", OUT / "junit-real.xml"
    code = run_tests(FAST_TESTS, extra, junit_fast, OUT / "pytest-fast.log")
    code_real = 0
    if not args.skip_real:
        code_real = run_tests(
            REAL_TESTS,
            {**extra, "ANON_DOCS": "1", "ANON_DOCS_REPORT": str(report)},
            junit_real,
            OUT / "pytest-real.log",
        )
    print(f"private-names scan: {'on' if private_names_paths() else 'off'}")

    by_file = junit_failures(junit_fast)
    for label, module in (
        ("blocks on the fakes", "test_docs_blocks"),
        ("generated files", "test_docs_generated"),
        ("style and privacy", "test_docs_style"),
        ("links and anchors", "test_docs_links"),
        ("hand-written reference parts", "test_docs_reference"),
        ("PDF print page", "test_docs_pdf"),
    ):
        bad = by_file.get(module, [])
        print(f"{label}: {'ok' if not bad else str(len(bad)) + ' fail'}")
        for line in bad[:6]:
            print("   ", line)
    if args.skip_real:
        print("real models: skipped (--skip-real)")
    else:
        run, failures, skips, outputs = example_summary(report)
        for name, text in outputs.items():
            target = OUT / "outputs" / (re.sub(r"[^\w.-]+", "_", name) + ".txt")
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(text, encoding="utf-8")
        print(
            f"real models: {run} blocks and scripts run, {len(failures)} fail"
            + (": " + "; ".join(failures[:6]) if failures else "")
        )
        for skip in skips:
            print("   skipped", skip)
        by_real = junit_failures(junit_real)
        failed |= bool(failures) or bool(by_real)
        if code_real not in (0, 5) and not (
            failures or by_real
        ):  # 5: pytest found no test to run (no block, no script yet)
            print(f"pytest exit {code_real} with no failing test reported: see {OUT / 'pytest-real.log'}")
            failed = True
    failed |= code not in (0, 5)
    if code != 0 and not any(by_file.values()):
        print(f"pytest exit {code} with no failing test reported: see {OUT / 'pytest-fast.log'}")

    old = legacy_summary(legacy)
    if old:
        print(
            f"files still in the 2.2.0 text: {len(old)} with {sum(old.values())} findings ("
            + ", ".join(f"{k} {v}" for k, v in sorted(old.items()))
            + ")"
        )
        print(
            "   "
            + ("these fail the check (--final)" if args.final else "warnings until --final; full list: " + str(legacy))
        )
    print(f"full output: {OUT}")
    return 1 if failed else 0


# --- the PDF manual -------------------------------------------------------------------------------------------------

OFFLINE_FLAGS = [
    # The same flags the Studio's own PDF report uses: a fresh Chrome profile otherwise talks to its makers' servers.
    "--proxy-server=socks5://127.0.0.1:1",
    "--host-resolver-rules=MAP * ~NOTFOUND , EXCLUDE localhost",
    "--disable-background-networking",
    "--disable-component-update",
    "--disable-sync",
    "--disable-default-apps",
    "--disable-domain-reliability",
    "--disable-client-side-phishing-detection",
    "--no-pings",
    "--disable-breakpad",
]
MIN_FREE_MB = 900
CHROME_TIMEOUT = 300


def free_mb() -> int:
    for line in Path("/proc/meminfo").read_text().splitlines():
        if line.startswith("MemAvailable:"):
            return int(line.split()[1]) // 1024
    return 10**6  # not Linux: no check


def find_chrome() -> str | None:
    for name in ("google-chrome", "google-chrome-stable", "chromium", "chromium-browser", "microsoft-edge", "msedge"):
        found = shutil.which(name)
        if found:
            return found
    return None


def page_count(pdf: Path) -> int | None:
    if shutil.which("pdfinfo"):
        done = subprocess.run(["pdfinfo", str(pdf)], capture_output=True, text=True)
        found = re.search(r"^Pages:\s+(\d+)", done.stdout, re.M)
        return int(found[1]) if found else None
    return len(re.findall(rb"/Type\s*/Page[^s]", pdf.read_bytes())) or None


def chrome_print(chrome: str, work: Path, page: Path, target: Path) -> bool:
    """Print `page` to `target` with a headless browser at low priority; False when it fails or times out."""
    command = [
        "nice",
        "-n",
        "19",
        chrome,
        "--headless=new",
        "--disable-gpu",
        "--no-first-run",
        "--hide-scrollbars",
        f"--user-data-dir={work / 'profile'}",
        f"--print-to-pdf={target}",
        "--no-pdf-header-footer",
        # the PDF's Creator field is the browser's user agent: say what the file is, not which machine made it
        "--user-agent=Anonymizer Studio manual",
        *OFFLINE_FLAGS,
        page.as_uri(),
    ]
    proc = subprocess.Popen(command, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, start_new_session=True)
    try:
        proc.wait(timeout=CHROME_TIMEOUT)
    except subprocess.TimeoutExpired:
        print(f"Chrome did not finish in {CHROME_TIMEOUT} s; stopping it")
        return False
    finally:
        # Chrome closes by itself after printing; whatever it left (helper processes) belongs to the group this script started.
        try:
            os.killpg(proc.pid, signal.SIGTERM)
        except (ProcessLookupError, PermissionError):
            pass
    return True


def sheet_numbers(pdf: Path, titles_file: Path) -> dict[str, int]:
    """The sheet each page of the manual starts on, read from the printed PDF: every page begins on a new sheet, right
    under the running header, with its first heading. Pages are matched in navigation order; a page not found is left out."""
    try:
        titles = json.loads(titles_file.read_text(encoding="utf-8"))
        text = subprocess.run(["pdftotext", str(pdf), "-"], capture_output=True, text=True, timeout=120).stdout
    except (OSError, ValueError, subprocess.SubprocessError):
        return {}
    sheets = [
        re.sub(r"^Anonymizer Studio \S+, (?:user manual|developer reference|manual)\s+", "", " ".join(chunk.split()))
        for chunk in text.split("\f")
    ]
    numbers: dict[str, int] = {}
    at = 1  # the cover is sheet 1 and carries no running header
    for path, title in titles.items():
        wanted = " ".join(title.split())[:50]
        for index in range(at, len(sheets)):
            if wanted and sheets[index].startswith(wanted):
                numbers[path] = index + 1
                at = index + 1
                break
    return numbers


def cmd_pdf(args) -> int:
    chrome = find_chrome()
    if not chrome:
        print("no Chrome, Chromium or Edge found on this computer (needed to print the PDF)")
        return 1
    if free_mb() < MIN_FREE_MB and not args.force:
        print(
            f"only {free_mb()} MB of memory is free; the printing needs about {MIN_FREE_MB} MB. Close something, or use --force."
        )
        return 1
    regenerate()
    work = Path(tempfile.mkdtemp(prefix="anonymizer-pdf-"))
    manual = getattr(args, "manual", None)
    extra = ["--manual", manual] if manual else []
    default_name = {
        None: "anonymizer-studio-manual.pdf",
        "user": "anonymizer-studio-user-manual.pdf",
        "developer": "anonymizer-studio-developer-reference.pdf",
    }[manual]
    target = Path(args.out).resolve() if args.out else ROOT / "work" / default_name
    target.parent.mkdir(parents=True, exist_ok=True)
    try:
        page = work / "manual.html"
        done = subprocess.run(
            [docs_python(), str(ROOT / "scripts" / "docs_pdf.py"), str(page), *extra],
            cwd=ROOT,
            capture_output=True,
            text=True,
        )
        if done.returncode != 0:
            print("could not assemble the print page:\n" + (done.stderr or done.stdout)[-1500:])
            return 1
        started = time.time()
        if not chrome_print(chrome, work, page, target):
            return 1
        # A second printing puts the page numbers into the contents list: the first one finds where each page starts.
        numbers = sheet_numbers(target, work / "manual.titles.json")
        if numbers:
            numbered = work / "manual-numbered.json"
            numbered.write_text(json.dumps(numbers), encoding="utf-8")
            done = subprocess.run(
                [docs_python(), str(ROOT / "scripts" / "docs_pdf.py"), str(page), str(numbered), *extra],
                cwd=ROOT,
                capture_output=True,
                text=True,
            )
            if done.returncode == 0 and not chrome_print(chrome, work, page, target):
                return 1
        if not target.is_file() or target.stat().st_size < 1000:
            print("Chrome wrote no PDF")
            return 1
        pages = page_count(target)
        print(
            f"{target}: {pages if pages else '?'} pages, {target.stat().st_size // 1024} kB, {time.time() - started:.0f} s"
        )
        return 0
    finally:
        shutil.rmtree(work, True)


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(
        prog="python scripts/docs.py", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("build", help="regenerate, then build site/ with mkdocs --strict").set_defaults(run=cmd_build)
    sub.add_parser("serve", help="regenerate, then serve the site for editing").set_defaults(run=cmd_serve)
    check = sub.add_parser(
        "check", help="verify generated files, the build, the examples, links and style; changes nothing"
    )
    check.add_argument("--no-nice", action="store_true", help="keep the normal priority (the default lowers it to 19)")
    check.add_argument("--skip-real", action="store_true", help="leave out the blocks and scripts on the real models")
    check.add_argument(
        "--final", action="store_true", help="the files still in the 2.2.0 text and the repository gates fail the check"
    )
    check.set_defaults(run=cmd_check)
    run = sub.add_parser("run", help="run the python blocks of one page and print what each printed")
    run.add_argument("page", help="a Markdown page, for example docs/how-to/progress-and-errors.md")
    run.add_argument(
        "--real",
        action="store_true",
        help="on the real models, comparing each output with the page (default: the fakes)",
    )
    run.set_defaults(run=cmd_run)
    pdf = sub.add_parser("pdf", help="print the documentation to one PDF manual with headless Chrome")
    pdf.add_argument("out", nargs="?", help="where to write it (default: work/anonymizer-studio-manual.pdf)")
    pdf.add_argument(
        "--manual",
        choices=("user", "developer"),
        help="print only the user manual (no code) or the developer reference, each a shorter PDF",
    )
    pdf.add_argument("--force", action="store_true", help="print even when little memory is free")
    pdf.set_defaults(run=cmd_pdf)
    args = parser.parse_args(argv)
    return args.run(args)


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
